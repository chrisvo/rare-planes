#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are Rare Bird, a strict aircraft rarity classifier for plane spotters. "
    "You must output exactly one JSON object with keys is_rare, confidence, reason. "
    "No markdown, no metadata, no extra keys."
)

CANDIDATES: dict[str, dict[str, str]] = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "output_dir": "model/output/rarity-qwen3-4b-unsloth-qlora",
        "note": "Smallest Qwen candidate; best mobile/edge experiment.",
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "output_dir": "model/output/rarity-qwen3-8b-unsloth-qlora",
        "note": "Likely small-model quality sweet spot for Gradio/server and high-end local inference.",
    },
    "gemma4-e4b": {
        "model_id": "google/gemma-4-E4B-it",
        "output_dir": "model/output/rarity-gemma4-e4b-unsloth-qlora",
        "note": "Gemma-family small reasoning baseline.",
    },
    "phi4-mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "output_dir": "model/output/rarity-phi4-mini-unsloth-qlora",
        "note": "Structured-classification baseline; train only if available/compatible.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_training_text(prompt: str, response: str | None = None, eos_token: str = "") -> str:
    text = (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Input JSON\n{prompt}\n\n"
        "### Output JSON\n"
    )
    if response is not None:
        text += response
        if eos_token:
            text += eos_token
    return text


def validate_response(row: dict[str, str]) -> bool:
    try:
        parsed = json.loads(row["response"])
    except Exception:
        return False
    # Existing datasets use is_rare/confidence/reason. New explanation datasets may use adjudication.
    return (
        isinstance(parsed, dict)
        and (
            isinstance(parsed.get("is_rare"), bool)
            or parsed.get("adjudication") in {"show", "review", "suppress"}
        )
    )


def label_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid = 0
    rare = 0
    show = review = suppress = 0
    for row in rows:
        if not validate_response(row):
            continue
        valid += 1
        parsed = json.loads(row["response"])
        if isinstance(parsed.get("is_rare"), bool):
            rare += int(parsed["is_rare"])
        adjudication = parsed.get("adjudication")
        show += int(adjudication == "show")
        review += int(adjudication == "review")
        suppress += int(adjudication == "suppress")
    return {
        "rows": len(rows),
        "valid_response_rows": valid,
        "invalid_response_rows": len(rows) - valid,
        "rare_bool_true": rare,
        "rare_bool_false": valid - rare if show + review + suppress == 0 else None,
        "adjudication_show": show,
        "adjudication_review": review,
        "adjudication_suppress": suppress,
    }


@dataclass
class ResponseOnlyCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature["labels"] for feature in features]
        model_features = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = [label + [-100] * (max_length - len(label)) for label in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def dataset_from_rows(rows: list[dict[str, str]], tokenizer: Any, max_seq_length: int):
    from datasets import Dataset

    examples = []
    eos_token = tokenizer.eos_token or ""
    label_counts: list[int] = []
    for row in rows:
        prompt_text = format_training_text(row["prompt"])
        full_text = format_training_text(row["prompt"], row["response"], eos_token=eos_token)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        labels = list(full["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        label_count = sum(1 for label in labels if label != -100)
        label_counts.append(label_count)
        examples.append({
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        })
    if any(count == 0 for count in label_counts):
        raise RuntimeError(f"{sum(count == 0 for count in label_counts)} rows lost all response labels after truncation")
    return Dataset.from_list(examples)


def resolve_candidate(name: str, model_id: str | None, output_dir: Path | None) -> tuple[str, Path, str]:
    if name not in CANDIDATES and not model_id:
        raise KeyError(f"unknown candidate {name!r}; choose one of {sorted(CANDIDATES)} or pass --model-id")
    config = CANDIDATES.get(name, {})
    resolved_model_id = model_id or config["model_id"]
    resolved_output = output_dir or Path(config.get("output_dir") or f"model/output/rarity-{name}-unsloth-qlora")
    return resolved_model_id, resolved_output, config.get("note", "custom candidate")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune RareBirds JSON classifier/explainer candidates with Unsloth QLoRA.")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="qwen3-4b")
    parser.add_argument("--model-id", help="Override HF model ID.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/datasets/rarity-oc-la-socal-hard-v2-3000-split/train.csv"))
    parser.add_argument("--eval-csv", type=Path, default=Path("data/datasets/rarity-oc-la-socal-hard-v2-3000-split/eval.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_id, output_dir, note = resolve_candidate(args.candidate, args.model_id, args.output_dir)
    train_rows = read_csv(args.train_csv)
    eval_rows = read_csv(args.eval_csv)
    if args.max_train_examples:
        train_rows = train_rows[: args.max_train_examples]
    if args.max_eval_examples:
        eval_rows = eval_rows[: args.max_eval_examples]
    summary = {
        "candidate": args.candidate,
        "model_id": model_id,
        "note": note,
        "output_dir": str(output_dir),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "max_seq_length": args.max_seq_length,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
        "train_summary": label_summary(train_rows),
        "eval_summary": label_summary(eval_rows),
    }
    if summary["train_summary"]["invalid_response_rows"] or summary["eval_summary"]["invalid_response_rows"]:
        print(json.dumps({"error": "invalid response rows found", **summary}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    from unsloth import FastLanguageModel
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    train_dataset = dataset_from_rows(train_rows, tokenizer, args.max_seq_length)
    eval_dataset = dataset_from_rows(eval_rows, tokenizer, args.max_seq_length)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=ResponseOnlyCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    print(json.dumps({**summary, "eval_metrics": metrics, "adapter_dir": str(output_dir / "adapter")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
