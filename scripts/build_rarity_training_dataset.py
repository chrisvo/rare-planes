#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from collect_socal_aircraft_dataset import make_prompt


DEFAULT_LIVE = Path("data/datasets/socal-aircraft/train.jsonl")
DEFAULT_SEED = Path("data/datasets/rarity-seed/examples.jsonl")
DEFAULT_OUT_DIR = Path("data/datasets/rarity-training")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def example_to_training_row(example: dict) -> dict[str, str]:
    if "prompt" in example and "response" in example:
        return {"prompt": example["prompt"], "response": example["response"]}

    aircraft = example["aircraft"]
    label = example["label"]
    response = {
        "is_rare": label["is_rare"],
        "confidence": label["confidence"],
        "reason": label["reason"],
    }
    return {
        "prompt": make_prompt(aircraft),
        "response": json.dumps(response, sort_keys=True, separators=(",", ":")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument(
        "--extra-seed",
        type=Path,
        action="append",
        default=[],
        help="Additional seed-style JSONL files to append after --seed.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    seed_paths = [args.seed, *args.extra_seed]
    seed_rows = [row for path in seed_paths for row in read_jsonl(path)]
    examples = read_jsonl(args.live) + seed_rows
    rows = [example_to_training_row(example) for example in examples]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(rows)

    with (args.out_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "live_examples": len(read_jsonl(args.live)),
        "seed_examples": len(seed_rows),
        "seed_files": [str(path) for path in seed_paths],
        "training_examples": len(rows),
        "train_csv": str(args.out_dir / "train.csv"),
        "train_jsonl": str(args.out_dir / "train.jsonl"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
