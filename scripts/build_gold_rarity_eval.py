#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from audit_rarity_dataset import COMMUNITY_PROVIDERS, category_for, community_signals, operational_signals, source_text


DEFAULT_SOURCES = [
    Path("data/eval/gold_rarity_supplemental_cases.csv"),
    Path("data/eval/regional_contrast_cases.csv"),
    Path("data/eval/ga_hard_cases.csv"),
    Path("data/eval/policy_regression_cases.csv"),
    Path("data/datasets/rarity-oc-la-socal-hard-v6-3600-split/eval.csv"),
    Path("data/datasets/rarity-training-community-balanced-split/eval.csv"),
]

TARGETS = {
    "common_local": 24,
    "community_signal": 20,
    "community_no_signal": 8,
    "emergency": 8,
    "special_mission": 8,
    "747_contextual": 12,
    "military_away_from_base": 14,
    "military_near_base": 14,
    "rare_type_or_special": 24,
    "routine_other": 16,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> str:
    prompt = json.loads(row["prompt"])
    aircraft = prompt.get("aircraft") or {}
    natural_key = "|".join(
        str(aircraft.get(key) or "")
        for key in ["provider", "icao_hex", "registration", "callsign", "type_designator", "source_url"]
    )
    if natural_key.strip("|"):
        return natural_key
    return hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()


def normalize_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "is_rare": bool(response["is_rare"]),
        "confidence": float(response.get("confidence", 0.9)),
        "reason": str(response.get("reason") or "").strip()[:220],
    }


def normalize_row(row: dict[str, str], source: Path) -> dict[str, str]:
    prompt = json.loads(row["prompt"])
    response = normalize_response(json.loads(row["response"]))
    category = category_for(prompt, response)
    prompt.setdefault("eval_metadata", {})
    prompt["eval_metadata"].update(
        {
            "category": category,
            "source_file": str(source),
            "gold_status": "repo_curated",
        }
    )
    return {
        "prompt": json.dumps(prompt, sort_keys=True, separators=(",", ":")),
        "response": json.dumps(response, sort_keys=True, separators=(",", ":")),
    }


def is_eligible_gold_row(row: dict[str, str]) -> bool:
    prompt = json.loads(row["prompt"])
    response = json.loads(row["response"])
    aircraft = prompt.get("aircraft") or {}
    provider = aircraft.get("provider")
    signals = [*community_signals(source_text(aircraft)), *operational_signals(aircraft)]
    if provider in COMMUNITY_PROVIDERS and response.get("is_rare") is True and not signals:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--out-csv", type=Path, default=Path("data/eval/gold_rarity_eval.csv"))
    parser.add_argument("--out-jsonl", type=Path, default=Path("data/eval/gold_rarity_eval.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/eval/gold_rarity_eval.summary.json"))
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()

    sources = args.source or [path for path in DEFAULT_SOURCES if path.exists()]
    rng = random.Random(args.seed)
    buckets: dict[str, list[dict[str, str]]] = {category: [] for category in TARGETS}
    seen: set[str] = set()

    for source in sources:
        for raw_row in read_rows(source):
            if not is_eligible_gold_row(raw_row):
                continue
            key = row_key(raw_row)
            if key in seen:
                continue
            seen.add(key)
            row = normalize_row(raw_row, source)
            category = json.loads(row["prompt"]).get("eval_metadata", {}).get("category", "routine_other")
            buckets.setdefault(category, []).append(row)

    selected: list[dict[str, str]] = []
    shortfalls: dict[str, int] = {}
    for category, target in TARGETS.items():
        rows = buckets.get(category, [])
        rng.shuffle(rows)
        selected.extend(rows[:target])
        if len(rows) < target:
            shortfalls[category] = target - len(rows)

    selected.sort(key=lambda row: (json.loads(row["prompt"])["eval_metadata"]["category"], row_key(row)))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(selected)
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    labels = Counter(json.loads(row["response"])["is_rare"] for row in selected)
    categories = Counter(json.loads(row["prompt"])["eval_metadata"]["category"] for row in selected)
    manifest = {
        "eval_csv": str(args.out_csv),
        "eval_jsonl": str(args.out_jsonl),
        "examples": len(selected),
        "labels": {"rare": labels[True], "not_rare": labels[False]},
        "categories": dict(categories),
        "sources": [str(path) for path in sources],
        "targets": TARGETS,
        "shortfalls": shortfalls,
        "status": "repo-curated candidate gold set; hand-review flagged rows before final model promotion",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
