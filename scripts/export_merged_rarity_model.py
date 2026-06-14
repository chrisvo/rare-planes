#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


def load_base_model(model_id: str):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except ValueError:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def copy_if_exists(source: Path, target_dir: Path) -> None:
    if not source.exists():
        return
    target = target_dir / source.name
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge the rarebirds LoRA adapter into the Gemma base model for deployment conversion."
    )
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("model/output/rarity-gemma4-e2b-hard-v5-qlora/adapter"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/output/rarity-gemma4-e2b-hard-v5-merged"),
    )
    parser.add_argument(
        "--eval-artifact-dir",
        type=Path,
        default=Path("model/output/rarity-gemma4-e2b-hard-v5-qlora"),
        help="Directory containing eval JSON files to copy into the merged release folder.",
    )
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        print(f"adapter directory not found: {args.adapter_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = load_base_model(args.model_id)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()
    merged.eval()

    merged.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(args.output_dir)

    for artifact_name in (
        "eval_metrics.json",
        "generation_eval_150_strict.json",
        "regional_contrast_eval_strict.json",
    ):
        copy_if_exists(args.eval_artifact_dir / artifact_name, args.output_dir)

    manifest = {
        "base_model": args.model_id,
        "adapter_dir": str(args.adapter_dir),
        "output_dir": str(args.output_dir),
        "dtype": "bfloat16",
        "safe_serialization": args.safe_serialization,
        "next_step": "Convert or quantize this merged checkpoint to a phone runtime such as LiteRT-LM.",
    }
    (args.output_dir / "rare_bird_release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
