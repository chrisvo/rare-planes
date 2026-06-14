# Training Rationale

## Method

Use LoRA, not full fine-tuning. `google/gemma-4-E2B-it` is a foundation model and the task is narrow structured instruction tuning. LoRA gives enough capacity for this classifier while keeping training cheap and reversible.

Use QLoRA for the first run. The remote RTX 5090 has about 31 GB usable VRAM, but Gemma 4 loads multimodal towers even for this text-only task. Plain bf16 LoRA reached the first training step and ran out of memory.

## Dataset

The current pipeline-validation dataset is `data/datasets/rarity-quick-1000-split`.

It contains:

- 850 train examples.
- 150 eval examples.
- Stratified rare/not-rare labels.
- A mix of real ADS-B examples, curated seed examples, and synthetic factor-based examples.

This dataset validates the training path. It is not production-grade because most of the rare positives are synthetic.

## Hyperparameters

Current defaults in `scripts/train_rarity_lora.py`:

- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.1`
- Learning rate: `1e-4`
- Scheduler: cosine
- Warmup ratio: `0.05`
- Weight decay: `0.01`
- Epochs: `3`
- Per-device batch size: `1`
- Gradient accumulation: `16`
- Effective batch size: `16`
- Max sequence length: `1024`
- Early stopping patience: `3` eval checks
- Target modules: language-model attention and MLP projections only

These are conservative because the dataset is only 1,000 examples and synthetic-heavy. A higher-rank adapter or higher learning rate would increase overfitting risk.

Gemma 4 includes vision and audio towers whose projection wrappers are not PEFT-compatible LoRA targets. The training script therefore uses a regex that restricts LoRA to `model.language_model.layers.*` `Linear` modules.

The GPU must be mostly free before the QLoRA run starts. A concurrent ComfyUI process using about 16 GB VRAM prevented QLoRA initialization.

## Required Gates

Before training:

```bash
python3 scripts/validate_rarity_dataset.py --input data/datasets/rarity-quick-1000/train.csv
python3 scripts/split_rarity_dataset.py
python3 scripts/train_rarity_lora.py --dry-run --qlora --per-device-batch-size 1 --gradient-accumulation-steps 16 --max-seq-length 1024
```

After training:

```bash
python3 scripts/evaluate_rarity_model.py --adapter-dir model/output/rarity-gemma4-lora/adapter --max-examples 150
```

Do not treat the adapter as usable until evaluation reports valid JSON rate, precision, recall, F1, perplexity, and latency.
