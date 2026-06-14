#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = (
    "You are rarebirds, a strict aircraft rarity classifier for plane spotters. "
    "You must output exactly one JSON object with keys is_rare, confidence, reason. "
    "No markdown, no metadata, no extra keys."
)


def format_training_text(prompt: str, response: str | None = None) -> str:
    text = (
        f"### System\n{SYSTEM_PROMPT}\n\n"
        f"### Input JSON\n{prompt}\n\n"
        "### Output JSON\n"
    )
    if response is not None:
        text += response
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id: str, qlora: bool):
    common_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if qlora:
        common_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **common_kwargs)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def dataset_from_rows(rows: list[dict[str, str]], tokenizer, max_seq_length: int) -> Dataset:
    examples = []
    for row in rows:
        prompt_text = format_training_text(row["prompt"])
        full_text = format_training_text(row["prompt"], row["response"] + tokenizer.eos_token)
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
        examples.append({
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        })
    return Dataset.from_list(examples)


def label_token_summary(dataset: Dataset) -> dict[str, float | int]:
    counts = [sum(1 for label in row["labels"] if label != -100) for row in dataset]
    zero_count = sum(1 for count in counts if count == 0)
    sorted_counts = sorted(counts)
    return {
        "min_response_label_tokens": min(counts) if counts else 0,
        "p50_response_label_tokens": sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0,
        "max_response_label_tokens": max(counts) if counts else 0,
        "zero_response_label_rows": zero_count,
        "zero_response_label_ratio": zero_count / len(counts) if counts else 0,
    }


@dataclass
class ResponseOnlyCollator:
    tokenizer: object

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_features = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_length - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument("--train-csv", type=Path, default=Path("data/datasets/rarity-quick-1000-split/train.csv"))
    parser.add_argument("--eval-csv", type=Path, default=Path("data/datasets/rarity-quick-1000-split/eval.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("model/output/rarity-gemma4-lora"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora-target-modules",
        default=r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$",
        help="Regex for PEFT target modules. Default limits LoRA to Gemma 4 language-model Linear layers.",
    )
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_id)
    train_rows = read_csv(args.train_csv)
    eval_rows = read_csv(args.eval_csv)
    train_dataset = dataset_from_rows(train_rows, tokenizer, args.max_seq_length)
    eval_dataset = dataset_from_rows(eval_rows, tokenizer, args.max_seq_length)
    train_label_summary = label_token_summary(train_dataset)
    eval_label_summary = label_token_summary(eval_dataset)

    if train_label_summary["zero_response_label_rows"] or eval_label_summary["zero_response_label_rows"]:
        print(json.dumps({
            "error": "one or more rows lost all response labels after truncation",
            "model_id": args.model_id,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "max_seq_length": args.max_seq_length,
            "train_label_summary": train_label_summary,
            "eval_label_summary": eval_label_summary,
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps({
            "model_id": args.model_id,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "max_seq_length": args.max_seq_length,
            "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
            "response_only_loss": True,
            "train_label_summary": train_label_summary,
            "eval_label_summary": eval_label_summary,
        }, indent=2, sort_keys=True))
        return 0

    model = load_model(args.model_id, args.qlora)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )
    data_collator = ResponseOnlyCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
