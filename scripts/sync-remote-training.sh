#!/usr/bin/env bash
set -euo pipefail

REMOTE="${RARE_BIRD_REMOTE:-cvo@192.168.1.159}"
REMOTE_DIR="${RARE_BIRD_REMOTE_DIR:-~/rare-bird}"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude 'model/output/' \
  --exclude 'model/.cache/' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

echo "Synced to ${REMOTE}:${REMOTE_DIR}"

