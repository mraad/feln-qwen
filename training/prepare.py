"""Build immutable FELN task data; run with the pinned feln-lora on PYTHONPATH."""

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import sqlglot
from feln import parse_relation
from feln.model import canon_kind
from sqlglot import exp
from src.feln_data import Schema, fingerprint, literal, render, sample_meta, target_key


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def family(meta):
    """Keep literal substitutions, paraphrases and reordered secondary filters together."""
    clauses = []
    for clause in meta["where"]:
        if not clause:
            clauses.append("")
            continue
        tree = sqlglot.parse_one(clause)
        for node in list(tree.walk()):
            if isinstance(node, exp.Cast):
                node.replace(node.this.copy())
        for node in list(tree.walk()):
            if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
                node.replace(exp.Literal.number("0"))
        for node in list(tree.find_all(exp.Literal)):
            # Preserve wildcard shape and literal type, erase the value.
            value = (
                ("%" if node.this.startswith("%") else "")
                + "VALUE"
                + ("%" if node.this.endswith("%") else "")
            )
            node.replace(
                exp.Literal.string(value) if node.is_string else exp.Literal.number("0")
            )
        from feln import normalize_where

        clauses.append(normalize_where(tree.sql()))
    secondary = sorted(
        (name, clause, canon_kind(parse_relation(rel).kind))
        for name, clause, rel in zip(meta["layers"][1:], clauses[1:], meta["relations"])
    )
    return json.dumps([meta["layers"][0], clauses[0], secondary], sort_keys=True)


def read_okf(directory, schema):
    """Read this bundle's Markdown schema/domains/hints; fail on catalog disagreement."""
    index = (directory / "index.md").read_text()
    links = re.findall(r"\]\(([^)]+\.md)\)", index)
    if set(links) != {f"{name}.md" for name in schema.layers}:
        raise ValueError("OKF index does not cover precisely the spatial catalog")
    result = {}
    for link in links:
        name = Path(link).stem
        section, field = None, None
        columns, domains, hints = {}, defaultdict(dict), defaultdict(list)
        for line_number, line in enumerate(
            (directory / link).read_text().splitlines(), 1
        ):
            if line.startswith("# "):
                section = line[2:]
            elif line.startswith("## `"):
                field = line.split("`")[1]
            elif section == "Schema" and line.startswith("| `"):
                cells = [s.strip() for s in line.strip("|").split("|")]
                columns[cells[0].strip("`")] = {
                    "alias": cells[1],
                    "dtype": cells[2],
                    "values": re.findall(r"`([^`]*)`", cells[3]),
                    "line": line_number,
                }
            elif section == "Domains" and line.startswith("| `"):
                cells = [s.strip() for s in line.strip("|").split("|")]
                domains[field][cells[0].strip("`")] = cells[1]
            elif section == "Query hints" and line.startswith("- "):
                hints[field].append(line[2:])
        expected = {c["name"]: c for c in schema.layers[name]["columns"]}
        if not set(expected) <= set(columns):
            raise ValueError(f"OKF missing catalog columns: {name}")
        # OKF exposes seven extra Wells fields hidden in the application catalog.
        # Record them in the manifest; never teach unsupported output fields.
        for field, column in columns.items():
            if field not in expected:
                column["excluded"] = "field absent from application catalog"
                continue
            spec = expected[field]
            for key in ("dtype",):
                if column[key] != spec[key]:
                    raise ValueError(f"OKF/catalog mismatch: {name}.{field} {key}")
            if column["alias"] != spec["alias"]:
                column["catalog_alias"] = spec["alias"]
            if column["values"] != [str(v) for v in spec["values"]]:
                raise ValueError(f"OKF/catalog mismatch: {name}.{field} values")
            if domains[field] != spec["keyval"]:
                raise ValueError(f"OKF/catalog mismatch: {name}.{field} domains")
            column["okf_extra_hints"] = [
                h for h in hints[field] if h not in spec["hints"]
            ]
            column["catalog_extra_hints"] = [
                h for h in spec["hints"] if h not in hints[field]
            ]
            column.update(keyval=dict(domains[field]), hints=hints[field])
        result[name] = columns
    return result


def build(source, output):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("FELN.json", "Layers.json"):
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / "okf", output / "okf")
    schema = Schema(output / "Layers.json")
    okf = read_okf(output / "okf", schema)
    records, excluded, transformations = [], [], []

    def add(text, meta, provenance):
        schema.validate(meta)
        compiled = schema.compile(meta)
        if compiled != meta:
            transformations.append(
                {"provenance": provenance, "before": meta, "after": compiled}
            )
        records.append(
            {
                "text": text,
                "meta": compiled,
                "provenance": [provenance],
                "group": family(compiled),
            }
        )

    gold = json.loads((source / "FELN.json").read_text())
    for i, row in enumerate(gold):
        if any(name not in schema.layers for name in row["meta"]["layers"]):
            excluded.append(
                {
                    "index": i,
                    "reason": "geometry-less table outside application FELN contract",
                    "record": row,
                }
            )
            continue
        for key in ("text", "source_text"):
            if row.get(key):
                add(
                    row[key],
                    row["meta"],
                    {"file": "FELN.json", "index": i, "field": key},
                )
    rng = random.Random(20260917)
    for i in range(3600):
        meta = sample_meta(schema, rng, i)
        add(
            render(meta, schema, i),
            meta,
            {"file": "Layers.json", "generator_index": i, "seed": 20260917},
        )
    for layer, columns in okf.items():
        for field, column in columns.items():
            if column.get("excluded"):
                continue
            values = list(column["keyval"]) or column["values"]
            for value in values:
                op = schema.like.get((layer, field), "=")
                typed = literal(
                    "%" + value + "%" if op in {"LIKE", "ILIKE"} else value,
                    column["dtype"],
                )
                meta = {
                    "layers": [layer],
                    "where": [f'"{field}" {op} {typed}'],
                    "relations": [],
                }
                display = column["keyval"].get(value, value)
                phrase = "contains" if op in {"LIKE", "ILIKE"} else "is"
                add(
                    f"Show {layer.lower()} whose {column['alias']} {phrase} {display}.",
                    meta,
                    {
                        "file": f"okf/{layer}.md",
                        "line": column["line"],
                        "field": field,
                        "value": value,
                    },
                )
    unique = {}
    for row in records:
        text = " ".join(row["text"].casefold().split())
        if text in unique:
            other = unique[text]
            if target_key(other["meta"]) != target_key(row["meta"]):
                from feln import FELN

                if not FELN.model_validate(other["meta"]).same(
                    FELN.model_validate(row["meta"])
                ):
                    raise ValueError(f"Conflicting duplicate: {text}")
            other["provenance"].extend(row["provenance"])
        else:
            unique[text] = row
    groups = defaultdict(list)
    for row in unique.values():
        groups[row["group"]].append(row)
    # Stratify on primary layer and layer count; split whole query families.
    strata = defaultdict(list)
    for key, rows in sorted(groups.items()):
        strata[(rows[0]["meta"]["layers"][0], len(rows[0]["meta"]["layers"]))].append(
            key
        )
    splits = {name: [] for name in ("train", "val", "test")}
    for keys in strata.values():
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            split = (
                "train"
                if i < int(len(keys) * 0.8)
                else "val"
                if i < int(len(keys) * 0.9)
                else "test"
            )
            splits[split].extend(groups[key])
    for name, rows in splits.items():
        rng.shuffle(rows)
        dump(output / f"{name}.json", rows)
    dump(output / "excluded.json", excluded)
    dump(output / "transformations.json", transformations)
    manifest = {
        "seed": 20260917,
        "source_records": len(gold),
        "excluded_records": len(excluded),
        "okf_excluded_fields": {
            layer: [field for field, c in cols.items() if c.get("excluded")]
            for layer, cols in okf.items()
        },
        "catalog_extra_hints": {
            layer: {
                f: c["catalog_extra_hints"]
                for f, c in cols.items()
                if c.get("catalog_extra_hints")
            }
            for layer, cols in okf.items()
        },
        "okf_extra_hints": {
            layer: {
                f: c["okf_extra_hints"]
                for f, c in cols.items()
                if c.get("okf_extra_hints")
            }
            for layer, cols in okf.items()
        },
        "alias_differences": {
            layer: {
                f: {"okf": c["alias"], "catalog": c["catalog_alias"]}
                for f, c in cols.items()
                if "catalog_alias" in c
            }
            for layer, cols in okf.items()
        },
        "deduplicated": len(records) - len(unique),
        "groups": len(groups),
        "counts": {k: len(v) for k, v in splits.items()},
        "slices": {
            k: dict(
                Counter(
                    f"{r['meta']['layers'][0]}:{len(r['meta']['layers'])}" for r in v
                )
            )
            for k, v in splits.items()
        },
        "sources": {
            str(p.relative_to(source)): fingerprint(p)
            for p in [
                source / "FELN.json",
                source / "Layers.json",
                *sorted((source / "okf").glob("*.md")),
            ]
        },
        "artifacts": {
            str(p.relative_to(output)): fingerprint(p)
            for p in output.rglob("*")
            if p.is_file()
        },
        "split_policy": "literal-masked SQL families, primary preserved, secondary order canonicalized; stratified by primary and layer count",
        "limitations": [
            "Synthetic language, no independent real-user test",
            "Static catalog shared across splits, query families are disjoint",
            "No filtering by returned row count",
        ],
    }
    dump(output / "manifest.json", manifest)
    verify(output)
    print(json.dumps(manifest, indent=2))


def verify(directory):
    from feln import FELN
    from feln.compare import canonical_text

    schema = Schema(directory / "Layers.json")
    seen_groups, seen_text, seen_targets = set(), set(), set()
    for name in ("train", "val", "test"):
        rows = json.loads((directory / f"{name}.json").read_text())
        if not rows:
            raise ValueError(f"Empty split: {name}")
        groups = {family(r["meta"]) for r in rows}
        texts = {" ".join(r["text"].casefold().split()) for r in rows}
        targets = {canonical_text(FELN.model_validate(r["meta"])) for r in rows}
        if groups & seen_groups or texts & seen_text:
            raise ValueError(f"Query families or text cross splits: {name}")
        if targets & seen_targets:
            raise ValueError(f"Equivalent FELN targets cross splits: {name}")
        if len(texts) != len(rows):
            raise ValueError(f"Duplicate text within split: {name}")
        for row in rows:
            schema.validate(row["meta"])
            if not row["provenance"] or row["group"] != family(row["meta"]):
                raise ValueError(f"Missing provenance or incorrect family: {name}")
        seen_groups |= groups
        seen_text |= texts
        seen_targets |= targets
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, expected_hash in manifest["artifacts"].items():
        if fingerprint(directory / path) != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    else:
        build(args.source, args.output)
