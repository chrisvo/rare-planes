#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-model/output/rarity-gemma4-e2b-hard-v5-merged}"
OUTPUT_DIR="${2:-model/output/rarity-gemma4-e2b-hard-v5-mlx-4bit}"

rm -rf "$OUTPUT_DIR"

mlx_lm.convert \
  --hf-path "$MODEL_DIR" \
  --mlx-path "$OUTPUT_DIR" \
  --quantize \
  --q-bits 4 \
  --q-group-size 64 \
  --trust-remote-code
