#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-}"
MODEL_DIR="${2:-model/output/rarity-gemma4-oc-la-hard-v3-mlx-4bit}"

if [[ -z "$REPO_ID" ]]; then
  echo "usage: $0 <user-or-org/repo-name> [model-dir]" >&2
  echo "example: $0 yourname/rare-bird-gemma4-e2b-mlx-4bit" >&2
  exit 2
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model directory not found: $MODEL_DIR" >&2
  exit 1
fi

if [[ -x ".venv-mlx/bin/hf" ]]; then
  HF_BIN=".venv-mlx/bin/hf"
else
  HF_BIN="${HF_BIN:-hf}"
fi

"$HF_BIN" upload "$REPO_ID" "$MODEL_DIR" . \
  --repo-type model \
  --private \
  --commit-message "Upload rarebirds Gemma 4 E2B MLX 4-bit"
