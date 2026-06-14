#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def label(row: dict[str, str]) -> bool:
    return bool(json.loads(row["response"])["is_rare"])


def stratification_key(row: dict[str, str]) -> tuple[bool, str]:
    prompt = json.loads(row["prompt"])
    aircraft = prompt.get("aircraft") or {}
    type_designator = (aircraft.get("type_designator") or "unknown").upper()
    hard_types = {
        "BE33", "BE36", "C150", "C152", "C172", "C182", "C130", "C17",
        "H47", "H53", "H60", "KC135", "KC46", "P28A", "P28R", "PA28",
        "SR20", "SR22", "T38", "V22",
    }
    if type_designator not in hard_types:
        type_designator = "other"
    context = prompt.get("observer_context") or {}
    pattern = context.get("military_pattern") if type_designator != "other" else "any"
    return label(row), f"{type_designator}:{pattern}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/datasets/rarity-quick-1000/train.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/datasets/rarity-quick-1000-split"))
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    strata: dict[tuple[bool, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.input):
        strata[stratification_key(row)].append(row)

    eval_rows: list[dict[str, str]] = []
    train_rows: list[dict[str, str]] = []
    for rows in strata.values():
        random.shuffle(rows)
        eval_count = round(len(rows) * args.eval_ratio)
        if len(rows) >= 4:
            eval_count = max(1, eval_count)
        eval_rows.extend(rows[:eval_count])
        train_rows.extend(rows[eval_count:])
    random.shuffle(eval_rows)
    random.shuffle(train_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "train.csv", train_rows)
    write_csv(args.out_dir / "eval.csv", eval_rows)
    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "eval.jsonl", eval_rows)

    summary = {
        "source": str(args.input),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_rare": sum(1 for row in train_rows if label(row)),
        "train_not_rare": sum(1 for row in train_rows if not label(row)),
        "eval_rare": sum(1 for row in eval_rows if label(row)),
        "eval_not_rare": sum(1 for row in eval_rows if not label(row)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
