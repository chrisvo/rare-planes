#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def estimated_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-examples", type=int, default=100)
    parser.add_argument("--max-duplicate-ratio", type=float, default=0.02)
    args = parser.parse_args()

    rows = load_rows(args.input)
    errors: list[str] = []
    warnings: list[str] = []
    labels: Counter[bool] = Counter()
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    total_lengths: list[int] = []
    prompt_seen: set[str] = set()
    duplicate_prompts = 0

    if len(rows) < args.min_examples:
        warnings.append(f"dataset has {len(rows)} examples, below recommended minimum {args.min_examples}")

    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"row {index}: missing non-empty prompt")
            continue
        if not isinstance(response, str) or not response.strip():
            errors.append(f"row {index}: missing non-empty response")
            continue

        if prompt in prompt_seen:
            duplicate_prompts += 1
        prompt_seen.add(prompt)

        try:
            prompt_json = json.loads(prompt)
        except json.JSONDecodeError as exc:
            errors.append(f"row {index}: prompt is not valid JSON: {exc}")
            continue
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError as exc:
            errors.append(f"row {index}: response is not valid JSON: {exc}")
            continue

        if "aircraft" not in prompt_json:
            errors.append(f"row {index}: prompt missing aircraft object")
        if "reference" not in prompt_json:
            errors.append(f"row {index}: prompt missing reference object")
        if set(response_json) != {"is_rare", "confidence", "reason"}:
            errors.append(f"row {index}: response must contain exactly is_rare, confidence, reason")
        if not isinstance(response_json.get("is_rare"), bool):
            errors.append(f"row {index}: response.is_rare must be boolean")
        else:
            labels[response_json["is_rare"]] += 1
        confidence = response_json.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"row {index}: response.confidence must be a number from 0 to 1")
        if not isinstance(response_json.get("reason"), str) or not response_json["reason"].strip():
            errors.append(f"row {index}: response.reason must be non-empty")

        prompt_len = estimated_tokens(prompt)
        response_len = estimated_tokens(response)
        total_len = prompt_len + response_len
        prompt_lengths.append(prompt_len)
        response_lengths.append(response_len)
        total_lengths.append(total_len)
        if total_len > args.max_tokens:
            warnings.append(f"row {index}: estimated token length {total_len} exceeds {args.max_tokens}")

    duplicate_ratio = duplicate_prompts / len(rows) if rows else 0
    if duplicate_ratio > args.max_duplicate_ratio:
        warnings.append(f"duplicate prompt ratio {duplicate_ratio:.3f} exceeds {args.max_duplicate_ratio:.3f}")

    rare = labels[True]
    not_rare = labels[False]
    if rare and not_rare:
        minority_ratio = min(rare, not_rare) / (rare + not_rare)
        if minority_ratio < 0.15:
            warnings.append(f"class balance is skewed: rare={rare}, not_rare={not_rare}")
    elif rows:
        errors.append("dataset must contain both rare and not-rare examples")

    summary = {
        "input": str(args.input),
        "examples": len(rows),
        "rare_examples": rare,
        "not_rare_examples": not_rare,
        "duplicate_prompts": duplicate_prompts,
        "duplicate_ratio": duplicate_ratio,
        "estimated_total_tokens": {
            "min": min(total_lengths) if total_lengths else 0,
            "mean": statistics.mean(total_lengths) if total_lengths else 0,
            "p50": percentile(total_lengths, 0.50),
            "p95": percentile(total_lengths, 0.95),
            "max": max(total_lengths) if total_lengths else 0,
        },
        "estimated_prompt_tokens_mean": statistics.mean(prompt_lengths) if prompt_lengths else 0,
        "estimated_response_tokens_mean": statistics.mean(response_lengths) if response_lengths else 0,
        "warnings": warnings[:100],
        "errors": errors[:100],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

