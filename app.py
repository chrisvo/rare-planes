#!/usr/bin/env python3
from __future__ import annotations

from scripts.gradio_rarity_tester import (
    APP_CSS,
    APP_THEME,
    DEFAULT_ADAPTER_DIR,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_ID,
    build_app,
)


app = build_app(
    model_id=DEFAULT_MODEL_ID,
    adapter_dir=DEFAULT_ADAPTER_DIR,
    load_in_4bit=DEFAULT_LOAD_IN_4BIT,
    max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
)


if __name__ == "__main__":
    app.launch(theme=APP_THEME, css=APP_CSS)
