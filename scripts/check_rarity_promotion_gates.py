#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_GATES = {
    "accuracy": 0.75,
    "invalid_json_rate": 0.02,
    "precision": 0.70,
    "recall": 0.80,
    "f1": 0.75,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--accuracy", type=float, default=DEFAULT_GATES["accuracy"])
    parser.add_argument("--invalid-json-rate", type=float, default=DEFAULT_GATES["invalid_json_rate"])
    parser.add_argument("--precision", type=float, default=DEFAULT_GATES["precision"])
    parser.add_argument("--recall", type=float, default=DEFAULT_GATES["recall"])
    parser.add_argument("--f1", type=float, default=DEFAULT_GATES["f1"])
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    gates = {
        "accuracy": args.accuracy,
        "invalid_json_rate": args.invalid_json_rate,
        "precision": args.precision,
        "recall": args.recall,
        "f1": args.f1,
    }
    checks = {
        "accuracy": float(metrics.get("accuracy", 0)) >= gates["accuracy"],
        "invalid_json_rate": float(metrics.get("invalid_json_rate", 1)) <= gates["invalid_json_rate"],
        "precision": float(metrics.get("precision", 0)) >= gates["precision"],
        "recall": float(metrics.get("recall", 0)) >= gates["recall"],
        "f1": float(metrics.get("f1", 0)) >= gates["f1"],
    }
    report = {
        "metrics": str(args.metrics),
        "passed": all(checks.values()),
        "checks": checks,
        "gates": gates,
        "observed": {key: metrics.get(key) for key in gates},
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
