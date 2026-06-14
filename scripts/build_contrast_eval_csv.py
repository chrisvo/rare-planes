#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from collect_socal_aircraft_dataset import make_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/eval/regional_contrast_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/eval/regional_contrast_cases.csv"))
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            response = {
                "is_rare": bool(case["expected"]),
                "confidence": 0.9,
                "reason": case["name"],
            }
            rows.append(
                {
                    "prompt": make_prompt(case["aircraft"], observer_context=case.get("observer_context")),
                    "response": json.dumps(response, sort_keys=True, separators=(",", ":")),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
