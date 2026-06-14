#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-model/output/rarity-gemma4-e2b-hard-v5-merged}"
OUTPUT_DIR="${2:-model/output/rarity-gemma4-e2b-hard-v5-litert}"

rm -rf "$OUTPUT_DIR"

litert-torch export_hf \
  "$MODEL_DIR" \
  "$OUTPUT_DIR" \
  --task=text_generation \
  --bundle_litert_lm=True \
  --quantization_recipe=dynamic_int8 \
  --cache_length=256 \
  --prefill_lengths=256,512,1024 \
  --use_jinja_template=True \
  --trust_remote_code=True
