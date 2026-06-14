#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate_rarity_text_classifier import MultinomialNB, read_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/datasets/rarity-training-v7-combined-split/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("ios/RareBird/Sources/RareBirdApp/Resources/RarityTextClassifier.json"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    rows = read_csv(args.train_csv)
    model = MultinomialNB(alpha=args.alpha, min_count=args.min_count)
    model.fit(rows)
    payload = {
        "version": 1,
        "model": "rarebirds_multinomial_nb_text_features",
        "train_csv": str(args.train_csv),
        "train_examples": len(rows),
        "threshold": args.threshold,
        "alpha": args.alpha,
        "min_count": args.min_count,
        "vocabulary": sorted(model.vocabulary),
        "class_document_counts": {
            "false": model.class_doc_counts[False],
            "true": model.class_doc_counts[True],
        },
        "class_total_features": {
            "false": model.class_total_features[False],
            "true": model.class_total_features[True],
        },
        "class_feature_counts": {
            "false": dict(model.class_feature_counts[False]),
            "true": dict(model.class_feature_counts[True]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "mb": round(args.output.stat().st_size / 1_000_000, 3),
        "vocabulary_size": len(model.vocabulary),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
