"""Strict raw/compiled FELN scoring through HF or the actual deployment HTTP path."""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from feln import FELN
from src.edge_client import complete
from src.feln_data import Schema, fingerprint
from src.prompt import make_prompt, parse_feln_json


def score(raw, expected, schema):
    parsed = parse_feln_json(raw)
    raw_exact = valid = compiled_exact = False
    if parsed is not None:
        try:
            schema.validate(parsed.model_dump())
            valid = True
            raw_exact = parsed.same(FELN.model_validate(expected))
        except ValueError:
            pass
        try:
            compiled = FELN.model_validate(schema.compile(parsed.model_dump()))
            compiled_exact = compiled.same(
                FELN.model_validate(schema.compile(expected))
            )
        except ValueError:
            pass
    return {
        "raw_exact": bool(raw_exact),
        "valid": valid,
        "compiled_exact": bool(compiled_exact),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--url")
    mode.add_argument("--model", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--selection",
        action="store_true",
        help="Validation only: first ten distinct query families per primary/layer-count stratum",
    )
    a = p.parse_args()
    if a.adapter and a.model is None:
        p.error("--adapter requires --model")
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    if a.selection and a.records.name != "val.json":
        p.error("Model selection must use val.json")
    if a.output.exists():
        raise SystemExit("Refusing to overwrite evaluation output")
    records = json.loads(a.records.read_text())
    if a.selection:
        strata = defaultdict(dict)
        for row in records:
            key = (row["meta"]["layers"][0], len(row["meta"]["layers"]))
            strata[key].setdefault(row["group"], row)
        records = [
            row for key in sorted(strata) for row in list(strata[key].values())[:10]
        ]
    if not records:
        p.error("At least one evaluation record is required")
    a.output.mkdir(parents=True)
    with (a.output / "records.json").open("x") as out:
        json.dump(records, out, ensure_ascii=False)
    schema = Schema(a.bundle / "Layers.json")
    if a.url:
        config = json.loads((a.bundle / "inference_config.json").read_text())
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(a.model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = Qwen3_5ForCausalLM.from_pretrained(
            a.model,
            dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        if a.adapter:
            model = PeftModel.from_pretrained(model, a.adapter)
        model.eval()
    predictions, latencies = [], []
    started = time.monotonic()
    size = 1 if a.url else a.batch_size
    with (a.output / "predictions.jsonl").open("x") as out:
        for offset in range(0, len(records), size):
            batch = records[offset : offset + size]
            before = time.monotonic()
            errors, timing = [None] * len(batch), None
            if a.url:
                try:
                    response = complete(config, batch[0]["text"], a.url)
                    generated = [response["content"]]
                    timing = response.get("timings")
                except (OSError, ValueError) as exc:
                    generated, errors = [""], [str(exc)]
            else:
                prompts = [
                    make_prompt(r["text"], tokenizer, schema.context()) for r in batch
                ]
                inputs = tokenizer(
                    prompts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(model.device)
                # Match training: PEFT keeps adapter weights in FP32, so inference
                # also needs BF16 autocast for their linear operations.
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    outputs = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=512,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                generated = tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            elapsed = time.monotonic() - before
            latencies.append(elapsed)
            for row, raw, error in zip(batch, generated, errors):
                result = {
                    "text": row["text"],
                    "expected": row["meta"],
                    "raw": raw,
                    **score(raw, row["meta"], schema),
                    "error": error,
                    "batch_seconds": elapsed,
                    "timings": timing,
                    "provenance": row.get("provenance"),
                    "group": row.get("group"),
                }
                predictions.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if len(predictions) % 25 < size or len(predictions) == len(records):
                print(
                    f"{len(predictions)}/{len(records)} raw={sum(x['raw_exact'] for x in predictions) / len(predictions):.4f}",
                    flush=True,
                )
    warm = sorted(latencies[1:] or latencies)
    slices = defaultdict(list)
    for row in predictions:
        paths = [p["file"] for p in row.get("provenance") or []]
        origin = (
            "FELN.json"
            if "FELN.json" in paths
            else "OKF"
            if any(p.startswith("okf/") for p in paths)
            else "Layers.json"
        )
        slices[origin].append(row)
        slices[
            f"{row['expected']['layers'][0]}:{len(row['expected']['layers'])}"
        ].append(row)
    report = {
        "n": len(records),
        "records_sha256": fingerprint(a.records),
        "evaluated_records_sha256": fingerprint(a.output / "records.json"),
        "selection_subset": a.selection,
        "model": str(a.model),
        "adapter": str(a.adapter),
        "url": a.url,
        "elapsed_seconds": time.monotonic() - started,
        "raw_exact": sum(x["raw_exact"] for x in predictions),
        "compiled_exact": sum(x["compiled_exact"] for x in predictions),
        "valid": sum(x["valid"] for x in predictions),
        "slices": {
            key: {
                "n": len(rows),
                "raw_exact": sum(r["raw_exact"] for r in rows),
                "compiled_exact": sum(r["compiled_exact"] for r in rows),
            }
            for key, rows in slices.items()
        },
        "raw_exact_definition": "schema-valid raw output compared by FELN.same; not byte equality",
        "first_batch_seconds": latencies[0],
        "warm_p50_seconds": statistics.median(warm),
        "warm_p95_seconds": warm[math.ceil(0.95 * len(warm)) - 1],
        "requests_per_second": len(records) / (time.monotonic() - started),
        "timing_scope": "sequential HTTP"
        if a.url
        else f"HF batches of {size}; not single-request latency",
    }
    with (a.output / "report.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
