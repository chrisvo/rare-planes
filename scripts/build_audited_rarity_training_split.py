#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DROP_SEVERITIES = {"critical", "high"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--flags", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/datasets/rarity-training-community-audited-split"))
    args = parser.parse_args()

    train_rows = read_csv(args.train)
    eval_rows = read_csv(args.eval)
    flags = read_csv(args.flags)
    drop_rows = {
        int(flag["row"])
        for flag in flags
        if flag.get("severity") in DROP_SEVERITIES
    }
    review_rows = [
        {**flag, "drop_from_training_v1": str(int(flag["row"]) in drop_rows).lower()}
        for flag in flags
    ]
    kept_train = [row for index, row in enumerate(train_rows) if index not in drop_rows]
    dropped_train = [row for index, row in enumerate(train_rows) if index in drop_rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "train.csv", kept_train, ["prompt", "response"])
    write_csv(args.out_dir / "eval.csv", eval_rows, ["prompt", "response"])
    write_csv(args.out_dir / "dropped_train_rows.csv", dropped_train, ["prompt", "response"])
    if review_rows:
        write_csv(args.out_dir / "review_queue.csv", review_rows, list(review_rows[0].keys()))
    else:
        write_csv(args.out_dir / "review_queue.csv", [], ["row", "severity", "code"])

    summary = {
        "source_train": str(args.train),
        "source_eval": str(args.eval),
        "source_flags": str(args.flags),
        "drop_severities": sorted(DROP_SEVERITIES),
        "train_original": len(train_rows),
        "train_kept": len(kept_train),
        "train_dropped": len(dropped_train),
        "eval_kept": len(eval_rows),
        "labels_original_train": label_counts(train_rows),
        "labels_kept_train": label_counts(kept_train),
        "labels_eval": label_counts(eval_rows),
        "train_csv": str(args.out_dir / "train.csv"),
        "eval_csv": str(args.out_dir / "eval.csv"),
        "review_queue_csv": str(args.out_dir / "review_queue.csv"),
        "dropped_train_rows_csv": str(args.out_dir / "dropped_train_rows.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
