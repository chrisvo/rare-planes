#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from mlx_lm import generate, load


SYSTEM_PROMPT = (
    "You are rarebirds, a strict aircraft rarity classifier for plane spotters. "
    "You must output exactly one JSON object with keys is_rare, confidence, reason. "
    "No markdown, no metadata, no extra keys."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_prompt(prompt: str) -> str:
    return f"### System\n{SYSTEM_PROMPT}\n\n### Input JSON\n{prompt}\n\n### Output JSON\n"


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model/output/rarity-gemma4-oc-la-hard-v3-mlx-4bit"))
    parser.add_argument("--eval-csv", type=Path, default=Path("data/eval/regional_contrast_cases.csv"))
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = read_csv(args.eval_csv)
    if args.max_examples:
        rows = rows[: args.max_examples]

    started = time.perf_counter()
    model, tokenizer = load(str(args.model))

    correct = 0
    tp = fp = tn = fn = 0
    invalid = 0
    samples = []
    for row in rows:
        truth = json.loads(row["response"])["is_rare"]
        raw = generate(
            model,
            tokenizer,
            prompt=format_prompt(row["prompt"]),
            max_tokens=args.max_tokens,
            verbose=False,
        ).strip()
        parsed = extract_json(raw)
        prediction = parsed.get("is_rare") if isinstance(parsed, dict) else None
        valid = isinstance(prediction, bool)
        if not valid:
            invalid += 1
        if valid and prediction == truth:
            correct += 1
        if prediction is True and truth is True:
            tp += 1
        elif prediction is True and truth is False:
            fp += 1
        elif prediction is False and truth is True:
            fn += 1
        else:
            tn += 1
        if len(samples) < 10:
            samples.append({"truth": truth, "parsed": parsed, "raw": raw})

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    metrics = {
        "model": str(args.model),
        "eval_csv": str(args.eval_csv),
        "examples": len(rows),
        "accuracy": correct / len(rows) if rows else 0,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "invalid_json": invalid,
        "invalid_json_rate": invalid / len(rows) if rows else 0,
        "latency_seconds_total": time.perf_counter() - started,
        "samples": samples,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
