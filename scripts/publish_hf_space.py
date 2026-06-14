#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_REPO = "build-small-hackathon/RareBirds"

SPACE_FILES = [
    "README.md",
    "app.py",
    "requirements.txt",
    "scripts/__init__.py",
    "scripts/collect_socal_aircraft_dataset.py",
    "scripts/gradio_rarity_tester.py",
    "scripts/rarity_engine.py",
]


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def build_payload(destination: Path) -> None:
    for relative_path in SPACE_FILES:
        source = ROOT / relative_path
        target = destination / relative_path
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    token = require_env("HF_TOKEN")
    repo_id = os.getenv("HF_SPACE_REPO", DEFAULT_SPACE_REPO).strip() or DEFAULT_SPACE_REPO

    with tempfile.TemporaryDirectory(prefix="rarebirds-hf-space-") as tmp:
        payload_dir = Path(tmp)
        build_payload(payload_dir)
        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(payload_dir),
            path_in_repo=".",
            commit_message="Deploy rarebirds Gradio app",
        )
        print(f"Published {repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
