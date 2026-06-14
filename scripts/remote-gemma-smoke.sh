#!/usr/bin/env bash
set -euo pipefail

REMOTE="${RARE_BIRD_REMOTE:-cvo@192.168.1.159}"
REMOTE_DIR="${RARE_BIRD_REMOTE_DIR:-~/rare-bird}"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && . .venv/bin/activate && python model/smoke_gemma_cuda.py"

