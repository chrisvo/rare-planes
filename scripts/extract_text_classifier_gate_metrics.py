#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--eval-csv", default="data/eval/gold_rarity_eval.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    evaluation = payload["evaluations"][args.eval_csv]
    gate_metrics = {
        "examples": evaluation["examples"],
        "accuracy": evaluation["accuracy"],
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "f1": evaluation["f1"],
        "invalid_json": 0,
        "invalid_json_rate": 0,
        "confusion": evaluation["confusion"],
        "source_metrics": str(args.metrics),
        "eval_csv": args.eval_csv,
        "model": payload["model"],
        "threshold": payload["threshold"],
        "estimated_serialized_model_bytes": payload["estimated_serialized_model_bytes"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate_metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
