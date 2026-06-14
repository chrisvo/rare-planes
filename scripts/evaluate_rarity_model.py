#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from train_rarity_lora import format_training_text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id: str, adapter_dir: Path | None):
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def bool_label(row: dict[str, str]) -> bool:
    return bool(json.loads(row["response"])["is_rare"])


def calculate_perplexity(model, tokenizer, rows: list[dict[str, str]], max_seq_length: int, batch_size: int) -> float:
    texts = [format_training_text(row["prompt"], row["response"]) for row in rows]
    encodings = tokenizer(texts, truncation=True, max_length=max_seq_length, padding=True, return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            tokens = int(attention_mask.sum().item())
            total_loss += float(outputs.loss.item()) * tokens
            total_tokens += tokens
    return math.exp(total_loss / total_tokens) if total_tokens else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--eval-csv", type=Path, default=Path("data/datasets/rarity-quick-1000-split/eval.csv"))
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("model/output/rarity-eval.json"))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_id)
    model = load_model(args.model_id, args.adapter_dir)
    rows = read_csv(args.eval_csv)[: args.max_examples]

    tp = fp = tn = fn = invalid = 0
    strict_correct = 0
    lenient_correct = 0
    latencies: list[float] = []
    samples = []
    invalid_samples = []
    for row in rows:
        prompt_text = format_training_text(row["prompt"])
        encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latencies.append(time.perf_counter() - started)
        generated = tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        parsed = extract_json(generated)
        truth = bool_label(row)
        if parsed is None or not isinstance(parsed.get("is_rare"), bool):
            invalid += 1
            prediction = False
            valid_prediction = False
            if len(invalid_samples) < 10:
                invalid_samples.append({"truth": truth, "generated": generated})
        else:
            prediction = parsed["is_rare"]
            valid_prediction = True
        if prediction == truth:
            lenient_correct += 1
        if valid_prediction and prediction == truth:
            strict_correct += 1
        if prediction and truth:
            tp += 1
        elif prediction and not truth:
            fp += 1
        elif not prediction and truth:
            fn += 1
        else:
            tn += 1
        if len(samples) < 10:
            samples.append({"truth": truth, "generated": generated})

    accuracy = strict_correct / len(rows) if rows else 0
    lenient_accuracy = lenient_correct / len(rows) if rows else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    metrics = {
        "examples": len(rows),
        "accuracy": accuracy,
        "lenient_accuracy": lenient_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "perplexity": None,
        "invalid_json": invalid,
        "invalid_json_rate": invalid / len(rows) if rows else 0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies) if latencies else 0,
            "max": max(latencies) if latencies else 0,
        },
        "samples": samples,
        "invalid_samples": invalid_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if not args.skip_perplexity:
        metrics["perplexity"] = calculate_perplexity(model, tokenizer, rows, args.max_seq_length, args.batch_size)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
