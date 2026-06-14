#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> str:
    try:
        prompt = json.loads(row["prompt"])
    except json.JSONDecodeError:
        return hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
    aircraft = prompt.get("aircraft") or {}
    natural = "|".join(
        str(aircraft.get(key) or "")
        for key in ["provider", "icao_hex", "registration", "callsign", "type_designator", "source_url"]
    )
    if natural.strip("|"):
        return natural
    return hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()


def merge_rows(paths: list[Path]) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates = 0
    for path in paths:
        for row in read_csv(path):
            key = row_key(row)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append(row)
    return rows, duplicates


def label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        try:
            counts[str(json.loads(row["response"])["is_rare"])] += 1
        except Exception:
            counts["invalid"] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/datasets/rarity-training-v7-combined-split"))
    args = parser.parse_args()

    train_rows, train_duplicates = merge_rows(args.train)
    eval_rows, eval_duplicates = merge_rows(args.eval)
    write_csv(args.out_dir / "train.csv", train_rows)
    write_csv(args.out_dir / "eval.csv", eval_rows)
    summary = {
        "source_train": [str(path) for path in args.train],
        "source_eval": [str(path) for path in args.eval],
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_duplicates_dropped": train_duplicates,
        "eval_duplicates_dropped": eval_duplicates,
        "labels_train": label_counts(train_rows),
        "labels_eval": label_counts(eval_rows),
        "train_csv": str(args.out_dir / "train.csv"),
        "eval_csv": str(args.out_dir / "eval.csv"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
