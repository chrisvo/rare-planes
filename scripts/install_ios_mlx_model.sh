#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-model/output/rarity-gemma4-oc-la-hard-v3-mlx-4bit}"
BUNDLE_ID="${2:-com.rarebird.app}"
SIMULATOR="${3:-booted}"
APP_MODEL_NAME="RareBirdsRarityModel"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model directory not found: $MODEL_DIR" >&2
  exit 1
fi

DATA_CONTAINER="$(xcrun simctl get_app_container "$SIMULATOR" "$BUNDLE_ID" data)"
TARGET_DIR="$DATA_CONTAINER/Library/Application Support/$APP_MODEL_NAME"

rm -rf "$TARGET_DIR"
mkdir -p "$(dirname "$TARGET_DIR")"
ditto "$MODEL_DIR" "$TARGET_DIR"

echo "Installed $MODEL_DIR"
echo "To $TARGET_DIR"
