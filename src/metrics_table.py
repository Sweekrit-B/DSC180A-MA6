"""Print a Markdown results table from a metrics JSON, digits unchanged.

REPORT.md tables must match reports/metrics.json to the printed digit, so
numbers are copied as stored, never rounded or retyped.

Run: python -m src.metrics_table [metrics.json] [--valid-only]
"""
import argparse
import json
from pathlib import Path

from src.config import METRICS


def table(metrics: dict, valid_only: bool = False) -> str:
    splits = ["valid"] if valid_only else ["valid", "test"]
    head = "| Model | " + " | ".join(f"{s} R² | {s} MAE" for s in splits) + " |"
    rule = "|---|" + "---|---|" * len(splits)
    rows = [
        f"| {name} | " + " | ".join(f"{scores[s]['r2']} | {scores[s]['mae']}" for s in splits) + " |"
        for name, scores in metrics.items()
        if isinstance(scores, dict) and "valid" in scores
    ]
    return "\n".join([head, rule, *rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(METRICS))
    parser.add_argument("--valid-only", action="store_true",
                        help="omit test columns, for use while still iterating")
    args = parser.parse_args()
    metrics = json.loads(Path(args.path).read_text())
    print(table(metrics, args.valid_only))
    n = sum(isinstance(v, dict) and "valid" in v for v in metrics.values())
    print(f"\nOK metrics_table: {n} models from {Path(args.path).name}, target={metrics.get('target', 'none')}")
