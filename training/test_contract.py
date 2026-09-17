"""Small runnable checks for leakage grouping and the app's strict evaluation contract."""

import json
from pathlib import Path

from src.feln_data import Schema

from training.evaluate import score
from training.prepare import family


def run():
    a = {
        "layers": ["Wells", "Pipelines", "Discoveries"],
        "where": ["water_depth > 100", "PipelinesType = 2", ""],
        "relations": ["withinDistance 5 kilometers", "intersects"],
    }
    b = {
        "layers": ["Wells", "Discoveries", "Pipelines"],
        "where": ["water_depth > 200", "", "PipelinesType = 4"],
        "relations": ["intersects", "withinDistance 10 kilometers"],
    }
    assert family(a) == family(b), (
        "literal and secondary-order variants must stay together"
    )
    converted = {**a, "relations": ["withinDistance 5000 meters", "intersects"]}
    assert family(a) == family(converted), "unit variants must stay together"
    inside = {**a, "relations": ["inside", "intersects"]}
    within = {**a, "relations": ["within", "intersects"]}
    assert family(inside) == family(within), "relation aliases must stay together"
    c = {**a, "relations": ["notWithinDistance 5 kilometers", "intersects"]}
    assert family(a) != family(c), "negative spatial predicate must not disappear"
    schema = Schema(Path("runs/data-20260917-v6/Layers.json"))
    expected = {
        "layers": ["Pipelines"],
        "where": ["PipelinesType = 2"],
        "relations": [],
    }
    assert score(json.dumps(expected), expected, schema)["raw_exact"]
    wrong = {**expected, "where": ["PipelinesType = 4"]}
    assert not score(json.dumps(wrong), expected, schema)["compiled_exact"]
    assert not score("```json\n" + json.dumps(expected) + "\n```", expected, schema)[
        "valid"
    ]
    print("PASS: family grouping, direction, enum exactness, JSON-only contract")


if __name__ == "__main__":
    run()
