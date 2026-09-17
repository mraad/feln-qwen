"""CPU-only validation regressions; also run with python -O -m unittest."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.feln_data import Schema, fingerprint

from serving import benchmark
from training import evaluate, prepare
from training.train import base_identity


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def rejects(self, entry, args, message):
        stderr = io.StringIO()
        with (
            patch("sys.argv", ["test", *map(str, args)]),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as error,
        ):
            entry()
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_evaluation_modes_fail_before_creating_output(self):
        records = self.write("records.json", [])
        output = self.root / "output"
        common = ["--records", records, "--bundle", self.root, "--output", output]
        for flags, message in [
            ([], "is required"),
            (["--url", "http://localhost", "--model", "base"], "not allowed"),
            (["--url", "http://localhost", "--adapter", "adapter"], "requires --model"),
            (["--adapter", "adapter"], "is required"),
            (["--url", "http://localhost", "--batch-size", "0"], "must be positive"),
            (["--url", "http://localhost"], "At least one"),
        ]:
            with self.subTest(flags=flags):
                self.rejects(evaluate.main, common + flags, message)
                self.assertFalse(output.exists())

    def test_benchmark_rejects_short_samples_before_memory_or_http(self):
        row = {"text": "wells", "meta": {"layers": ["Wells"]}}
        for rows in ([], [row]):
            with patch.object(benchmark, "memory") as memory:
                self.rejects(
                    benchmark.main,
                    [
                        "--records",
                        self.write("records.json", rows),
                        "--output",
                        self.root / "out",
                    ],
                    "At least two",
                )
                memory.assert_not_called()

    def test_benchmark_two_failed_requests_still_reports(self):
        rows = [{"text": "wells", "meta": {"layers": ["Wells"]}}] * 2
        output = self.root / "out.json"
        memory = {"MemTotal": 100, "MemAvailable": 50, "SwapTotal": 0, "SwapFree": 0}
        with (
            patch(
                "sys.argv",
                [
                    "test",
                    "--records",
                    str(self.write("records.json", rows)),
                    "--output",
                    str(output),
                ],
            ),
            patch.object(benchmark, "memory", return_value=memory),
            patch.object(
                benchmark.urllib.request, "urlopen", side_effect=OSError("unavailable")
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            benchmark.main()
        report = json.loads(output.read_text())
        self.assertEqual((report["n"], report["errors"]), (2, 2))

    def dataset(self):
        layers = ["Wells", "Pipelines", "Discoveries"]
        self.write(
            "Layers.json", {"layers": [{"name": n, "columns": []} for n in layers]}
        )
        for split, layer in zip(("train", "val", "test"), layers):
            meta = {"layers": [layer], "where": [""], "relations": []}
            self.write(
                split + ".json",
                [
                    {
                        "text": layer,
                        "meta": meta,
                        "group": prepare.family(meta),
                        "provenance": [{"file": "fixture"}],
                    }
                ],
            )
        self.write(
            "manifest.json",
            {
                "artifacts": {
                    p.name: fingerprint(p)
                    for p in self.root.glob("*.json")
                    if p.name != "manifest.json"
                }
            },
        )

    def test_split_integrity_survives_optimization(self):
        for case in ("empty", "leak", "duplicate", "provenance", "group", "hash"):
            with self.subTest(case=case):
                self.dataset()
                prepare.verify(self.root)
                rows = json.loads((self.root / "train.json").read_text())
                if case == "empty":
                    self.write("train.json", [])
                elif case == "leak":
                    self.write("val.json", rows)
                elif case == "duplicate":
                    self.write("train.json", rows * 2)
                elif case in ("provenance", "group"):
                    rows[0][case] = [] if case == "provenance" else "wrong"
                    self.write("train.json", rows)
                else:
                    manifest = json.loads((self.root / "manifest.json").read_text())
                    manifest["artifacts"]["train.json"] = "incorrect"
                    self.write("manifest.json", manifest)
                with self.assertRaises(ValueError):
                    prepare.verify(self.root)

    def test_catalog_disagreement_survives_optimization(self):
        catalog = {
            "layers": [
                {
                    "name": "Wells",
                    "columns": [
                        {
                            "name": "depth",
                            "alias": "Depth",
                            "dtype": "DOUBLE",
                            "values": [1],
                            "keyval": {"1": "one"},
                            "hints": [],
                        }
                    ],
                }
            ]
        }
        schema = Schema(self.write("Layers.json", catalog))
        (self.root / "index.md").write_text("[Wells](Wells.md)")
        valid = "# Schema\n| `depth` | Depth | DOUBLE | `1` |\n# Domains\n## `depth`\n| `1` | one |\n"
        (self.root / "Wells.md").write_text(valid)
        prepare.read_okf(self.root, schema)
        for old, new in [
            ("DOUBLE", "INTEGER"),
            ("DOUBLE | `1`", "DOUBLE | `2`"),
            ("one", "two"),
        ]:
            (self.root / "Wells.md").write_text(valid.replace(old, new))
            with self.assertRaises(ValueError):
                prepare.read_okf(self.root, schema)

    def test_actual_base_identity_and_weight_hash(self):
        self.write("config.json", {"_commit_hash": "fixture-revision"})
        self.write("tokenizer_config.json", {})
        weights = self.root / "model.safetensors"
        weights.write_bytes(b"fixture weights")
        before = base_identity(self.root)
        self.assertEqual(before["model_id"], str(self.root))
        self.assertEqual(before["revision"], "fixture-revision")
        self.assertIsNone(before["tokenizer_revision"])
        weights.write_bytes(b"changed weights")
        after = base_identity(self.root)
        self.assertNotEqual(
            before["base_artifacts_sha256"][weights.name],
            after["base_artifacts_sha256"][weights.name],
        )


if __name__ == "__main__":
    unittest.main()
