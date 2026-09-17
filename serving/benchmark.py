"""Measure local app round-trips and sampled shared-memory pressure on Orin."""

import argparse
import json
import math
import statistics
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from feln import FELN
from src.prompt import parse_feln_json


def memory():
    fields = {
        k: int(v.split()[0]) * 1024
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    return {k: fields[k] for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite benchmark")
    strata = defaultdict(list)
    for row in json.loads(args.records.read_text()):
        key = (row["meta"]["layers"][0], len(row["meta"]["layers"]))
        if len(strata[key]) < 2:
            strata[key].append(row)
    records = [row for key in sorted(strata) for row in strata[key]]
    if len(records) < 2:
        parser.error("At least two selected records are required")
    samples = [memory()]
    done = threading.Event()

    def sample():
        while not done.wait(0.05):
            samples.append(memory())

    monitor = threading.Thread(target=sample)
    monitor.start()
    results = []
    start = time.monotonic()
    try:
        for row in records:
            before = time.monotonic()
            request = urllib.request.Request(
                "http://127.0.0.1:18091/query",
                data=json.dumps({"text": row["text"], "limit": 10}).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
            except (OSError, ValueError) as exc:
                results.append(
                    {
                        "text": row["text"],
                        "seconds": time.monotonic() - before,
                        "error": str(exc),
                        "raw_exact": False,
                        "compiled_exact": False,
                    }
                )
                continue
            parsed = parse_feln_json(result["raw"])
            expected = FELN.model_validate(row["meta"])
            results.append(
                {
                    "text": row["text"],
                    "seconds": time.monotonic() - before,
                    "raw_exact": parsed is not None and parsed.same(expected),
                    "compiled_exact": FELN.model_validate(result["feln"]).same(
                        expected
                    ),
                    "feln": result["feln"],
                    "expected": row["meta"],
                    "count": result["execution"]["count"],
                    "model_timings": result["model_timings"],
                    "query_seconds": result["execution"]["query_seconds"],
                }
            )
    finally:
        done.set()
        monitor.join()
    elapsed = time.monotonic() - start
    warm = sorted(r["seconds"] for r in results[1:])
    report = {
        "n": len(results),
        "raw_exact": sum(r["raw_exact"] for r in results),
        "compiled_exact": sum(r["compiled_exact"] for r in results),
        "errors": sum("error" in r for r in results),
        "selection": "First two test questions per primary-layer/layer-count stratum",
        "first_seconds": results[0]["seconds"],
        "warm_p50_seconds": statistics.median(warm),
        "warm_p95_seconds": warm[math.ceil(0.95 * len(warm)) - 1],
        "elapsed_seconds": elapsed,
        "requests_per_second": len(results) / elapsed,
        "memory_before_bytes": samples[0],
        "minimum_available_bytes": min(s["MemAvailable"] for s in samples),
        "peak_system_used_bytes": max(
            s["MemTotal"] - s["MemAvailable"] for s in samples
        ),
        "maximum_swap_used_bytes": max(s["SwapTotal"] - s["SwapFree"] for s in samples),
        "memory_samples": len(samples),
        "sampling_interval_seconds": 0.05,
        "scope": "Orin localhost HTTP -> model -> schema compiler -> read-only DuckDB -> GeoJSON; one request at a time; memory is sampled system-wide shared RAM, not CUDA allocator peak",
        "results": results,
    }
    with args.output.open("x") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
