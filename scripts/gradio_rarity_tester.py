#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
import urllib.request

import gradio as gr
import folium
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesFallback()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_socal_aircraft_dataset import (
    CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS,
    CONTEXTUAL_TYPE_DESIGNATORS,
    DEFAULT_LAT,
    DEFAULT_LON,
    adsbfi_url,
    adsblol_url,
    fetch_json,
    make_prompt as make_rarity_prompt,
    normalize_adsb_aircraft,
)
from rarity_engine import explanation_from_score, score_aircraft


DEFAULT_QWEN3_4B_ADAPTER_DIR = ROOT / "model" / "output" / "rarity-qwen3-4b-unsloth-qlora" / "adapter"


def default_adapter_dir() -> str:
    configured = os.getenv("RAREBIRD_ADAPTER_DIR")
    if configured is not None:
        return configured.strip()
    if DEFAULT_QWEN3_4B_ADAPTER_DIR.exists():
        return str(DEFAULT_QWEN3_4B_ADAPTER_DIR)
    return ""


DEFAULT_MODEL_ID = os.getenv("RAREBIRD_MODEL_ID", "Qwen/Qwen3-4B")
DEFAULT_ADAPTER_DIR = default_adapter_dir()
DEFAULT_LOAD_IN_4BIT = os.getenv("RAREBIRD_LOAD_IN_4BIT", "1").lower() not in {"0", "false", "no"}
DEFAULT_MAX_SEQ_LENGTH = int(os.getenv("RAREBIRD_MAX_SEQ_LENGTH", "2048"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("RAREBIRD_MAX_NEW_TOKENS", "160"))
DEFAULT_WATCH_RADIUS_NM = int(os.getenv("RAREBIRD_WATCH_RADIUS_NM", "15"))
DEFAULT_LIVE_MODEL_CANDIDATE_LIMIT = int(os.getenv("RAREBIRD_LIVE_MODEL_CANDIDATE_LIMIT", "8"))
SCAN_REFRESH_SECONDS = int(os.getenv("RAREBIRD_SCAN_REFRESH_SECONDS", "30"))
SCAN_CACHE_SECONDS = int(os.getenv("RAREBIRD_SCAN_CACHE_SECONDS", "25"))
SCAN_MIN_INTERVAL_SECONDS = int(os.getenv("RAREBIRD_SCAN_MIN_INTERVAL_SECONDS", str(SCAN_REFRESH_SECONDS)))
PLANESPOTTERS_CACHE_SECONDS = min(int(os.getenv("PLANESPOTTERS_CACHE_SECONDS", "86400")), 86400)
PLANESPOTTERS_USER_AGENT = os.getenv(
    "PLANESPOTTERS_USER_AGENT",
    "RareBird/0.1 (+https://github.com/chrisvo/rare-bird)",
)
HERO_IMAGE_PATH = ROOT / "assets" / "rarebirds-hero.png"
DEFAULT_OBSERVER_LABEL = f"Default watch: LA/Orange County ({DEFAULT_LAT:.4f}, {DEFAULT_LON:.4f})"

CITY_PRESETS = {
    "Los Angeles": (34.0522, -118.2437),
    "New York City": (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
}

MODEL_CHOICES = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "google/gemma-3-27b-it",
    "google/gemma-4-E2B-it",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap');

:root {
  --rb-bg: #050807;
  --rb-scope: #07110d;
  --rb-scope-2: #0a1711;
  --rb-ink: #d7ffe7;
  --rb-soft-text: #9bd8b0;
  --rb-muted: #5f9272;
  --rb-panel: #07110d;
  --rb-panel-soft: #0b1b13;
  --rb-line: rgba(91, 255, 151, 0.22);
  --rb-line-hot: rgba(91, 255, 151, 0.48);
  --rb-radar: #5bff97;
  --rb-warn: #ffd166;
  --rb-danger: #ff5c7a;
  --rb-display: "IBM Plex Sans Condensed", "Avenir Next Condensed", "DIN Condensed", sans-serif;
  --rb-mono: "IBM Plex Mono", "SFMono-Regular", "Menlo", "Consolas", monospace;
}
html { scrollbar-gutter: stable; overflow-y: scroll; }
* { box-sizing: border-box; }
body,
.gradio-container {
  background:
    linear-gradient(rgba(91,255,151,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(91,255,151,.035) 1px, transparent 1px),
    #050807 !important;
  background-size: 44px 44px, 44px 44px, auto !important;
  color: var(--rb-ink) !important;
}
body { overflow-x: hidden; }
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  margin: 0 auto !important;
  padding: 0 24px 36px !important;
  font-family: var(--rb-display) !important;
  font-feature-settings: "ss01", "tnum";
}
.gradio-container,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container label,
.gradio-container p,
.gradio-container span,
.gradio-container div {
  color: var(--rb-ink);
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  color: var(--rb-ink) !important;
  font-family: var(--rb-display) !important;
  font-weight: 700 !important;
  letter-spacing: .01em !important;
  text-shadow: 0 0 16px rgba(91,255,151,.18);
}
.gradio-container .prose a,
.gradio-container .markdown a { color: var(--rb-radar) !important; }
.rb-landing {
  width: min(1320px, calc(100vw - 48px)) !important;
  min-width: 0 !important;
  max-width: none !important;
  margin: 0 auto !important;
  box-sizing: border-box !important;
  flex: 0 0 auto !important;
  gap: 18px !important;
}
.rb-landing .rb-section,
.rb-settings {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  margin-left: auto !important;
  margin-right: auto !important;
  box-sizing: border-box !important;
  flex: 0 0 auto !important;
}
.rb-settings { width: min(1320px, calc(100vw - 48px)) !important; overflow: hidden !important; }
.rb-settings *, .rb-landing * { min-width: 0 !important; }
.rb-hero-frame { padding: 0 !important; overflow: hidden !important; }
.rb-hero-frame > div,
.rb-hero-frame .gradio-html,
.rb-hero-shell,
.rb-hero-shell > div {
  width: 100% !important; max-width: none !important; margin: 0 !important; padding: 0 !important; box-sizing: border-box !important;
}
#rb-hero {
  border: 1px solid var(--rb-line-hot);
  border-radius: 4px;
  margin: 0;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  overflow: hidden;
  background: #050807;
  color: var(--rb-ink);
  box-shadow: inset 0 0 0 1px rgba(91,255,151,.08), inset 0 0 80px rgba(91,255,151,.06), 0 22px 80px rgba(0,0,0,.50);
}
#rb-hero .hero-inner {
  min-height: 390px;
  padding: 34px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr);
  gap: 28px;
  align-items: end;
  background:
    linear-gradient(rgba(91,255,151,.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(91,255,151,.045) 1px, transparent 1px),
    radial-gradient(circle at 74% 48%, transparent 0 92px, rgba(91,255,151,.16) 93px 94px, transparent 95px 155px, rgba(91,255,151,.10) 156px 157px, transparent 158px 228px, rgba(91,255,151,.08) 229px 230px, transparent 231px),
    #050807;
  background-size: 28px 28px, 28px 28px, auto, auto;
}
#rb-hero .eyebrow {
  color: var(--rb-radar);
  font: 600 12px/1.4 var(--rb-mono);
  letter-spacing: .16em;
  text-transform: uppercase;
  margin-bottom: 13px;
}
#rb-hero h1 {
  margin: 0;
  font-size: clamp(48px, 7vw, 84px);
  line-height: .95;
  letter-spacing: -1.2px;
  font-weight: 650;
  font-family: var(--rb-display);
  color: var(--rb-ink);
  text-shadow: 0 0 18px rgba(91,255,151,.20);
}
#rb-hero p {
  margin: 14px 0 0;
  max-width: 670px;
  color: var(--rb-soft-text);
  font-size: 18px;
  line-height: 1.6;
}
#rb-hero .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }
#rb-hero .chip {
  border: 1px solid var(--rb-line);
  border-radius: 2px;
  padding: 7px 10px;
  color: var(--rb-ink);
  background: rgba(91,255,151,.045);
  font: 600 12px/1.25 var(--rb-mono);
  text-transform: uppercase;
  letter-spacing: .04em;
}
#rb-hero .hero-panel {
  align-self: stretch;
  min-height: 270px;
  border: 1px solid var(--rb-line-hot);
  border-radius: 3px;
  background: rgba(3, 12, 8, .88);
  padding: 18px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: inset 0 0 34px rgba(91,255,151,.06);
}
#rb-hero .scope-row,
#rb-hero .signal-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 11px 0;
  border-bottom: 1px solid rgba(91,255,151,.16);
}
#rb-hero .signal-row:last-child { border-bottom: 0; }
#rb-hero .metric { color: var(--rb-radar); font: 700 30px/1 var(--rb-mono); letter-spacing: -.6px; }
#rb-hero .label,
#rb-hero .scope-row span:first-child,
#rb-hero .signal-row span:first-child { color: var(--rb-muted); font: 600 12px/1.4 var(--rb-mono); text-transform: uppercase; letter-spacing: .08em; }
#rb-hero .scope-row strong,
#rb-hero .signal-row strong { color: var(--rb-ink); font: 600 13px/1.4 var(--rb-mono); }
.rb-section,
.rb-output,
.rb-settings {
  border: 1px solid var(--rb-line) !important;
  border-radius: 4px !important;
  background: rgba(7,17,13,.90) !important;
  box-shadow: 0 18px 58px rgba(0,0,0,.32), inset 0 0 24px rgba(91,255,151,.035) !important;
  box-sizing: border-box !important;
}
.rb-landing .rb-section { padding: 18px !important; }
.rb-landing .rb-section h2,
.rb-landing .rb-section h3,
.rb-landing .rb-section .prose h2,
.rb-landing .rb-section .prose h3,
.rb-landing .rb-section .markdown h2,
.rb-landing .rb-section .markdown h3 {
  margin-top: 0 !important;
  color: var(--rb-ink) !important;
  opacity: 1 !important;
  font-family: var(--rb-display) !important;
  font-size: 28px !important;
  font-weight: 700 !important;
  letter-spacing: .01em !important;
  text-transform: uppercase !important;
  text-shadow: 0 0 18px rgba(91,255,151,.18) !important;
}
.rb-settings { margin-top: 16px !important; margin-bottom: 14px !important; }
.rb-result {
  border-left: 4px solid var(--rb-radar) !important;
  border-radius: 3px !important;
  background: #eafff0 !important;
  color: #062410 !important;
  padding: 12px 14px !important;
  min-height: 128px;
}
.rb-result,
.rb-result * {
  color: #062410 !important;
}
.rb-result h3,
.rb-result h2,
.rb-result p:first-child strong {
  margin-top: 0 !important;
  color: #003d16 !important;
  opacity: 1 !important;
  font-size: 24px !important;
  line-height: 1.15 !important;
  letter-spacing: -.2px !important;
  text-shadow: none !important;
}
.rb-result strong { color: #003d16 !important; }
.rb-result code { color: #003d16 !important; background: rgba(3, 66, 24, 0.10) !important; }
.rb-output textarea,
.rb-output code,
.rb-output pre,
.rb-section textarea,
.rb-section code,
.rb-section pre { font-family: var(--rb-mono) !important; }
.rb-section label,
.rb-settings label {
  color: var(--rb-radar) !important;
  font-family: var(--rb-mono) !important;
  font-size: 12px !important;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.rb-section input,
.rb-section textarea,
.rb-section select,
.rb-settings input,
.rb-settings textarea,
.rb-settings select {
  border-color: var(--rb-line) !important;
  border-radius: 3px !important;
  background: rgba(3,12,8,.72) !important;
  color: var(--rb-ink) !important;
}
.rb-table { border-radius: 3px !important; overflow: hidden; min-height: 180px !important; }
.rb-table,
.rb-table * {
  color: var(--rb-ink) !important;
}
.rb-table table,
.rb-table .wrap,
.rb-table .table-container,
.rb-table .table-wrap,
.rb-table [role="grid"],
.rb-table [role="rowgroup"] {
  font-size: 13px !important;
  background: #07110d !important;
}
.rb-table th,
.rb-table .header-cell,
.rb-table [role="columnheader"] {
  background: #0b1b13 !important;
  color: var(--rb-radar) !important;
  border-color: rgba(91,255,151,.18) !important;
  font-family: var(--rb-mono) !important;
  text-shadow: none !important;
}
.rb-table td,
.rb-table .cell,
.rb-table [role="gridcell"] {
  background: rgba(3,12,8,.88) !important;
  color: var(--rb-ink) !important;
  border-color: rgba(91,255,151,.12) !important;
}
.rb-table .toolbar,
.rb-table .toolbar-buttons,
.rb-table button {
  background: rgba(3,12,8,.82) !important;
  color: var(--rb-ink) !important;
}
.rb-location-panel {
  border: 1px solid var(--rb-line-hot) !important;
  border-radius: 4px !important;
  background:
    linear-gradient(90deg, rgba(91,255,151,.08), rgba(91,255,151,.02)),
    rgba(3,12,8,.82) !important;
  padding: 13px !important;
  margin: 4px 0 14px !important;
  box-shadow: inset 0 0 22px rgba(91,255,151,.045) !important;
}
.rb-location-panel label { color: var(--rb-radar) !important; }
.rb-location-panel label[data-testid$="-radio-label"] {
  background: #f3f6f1 !important;
  border-color: rgba(91,255,151,.38) !important;
  color: #132018 !important;
  font-family: var(--rb-mono) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
}
.rb-location-panel label[data-testid$="-radio-label"] span {
  color: inherit !important;
}
.rb-location-panel label.selected[data-testid$="-radio-label"] {
  background: var(--rb-radar) !important;
  color: #031108 !important;
}
.rb-location-panel input { font-family: var(--rb-mono) !important; font-weight: 650 !important; }
.rb-location-status {
  border-left: 3px solid var(--rb-warn) !important;
  border-radius: 3px !important;
  background: rgba(255, 209, 102, .08) !important;
  padding: 8px 10px !important;
  min-height: 40px !important;
}
.rb-location-status,
.rb-location-status * {
  color: var(--rb-ink) !important;
  font: 650 12px/1.45 var(--rb-mono) !important;
}
.rb-map {
  border: 1px solid var(--rb-line-hot);
  border-radius: 4px;
  background: #07110d;
  box-shadow: 0 20px 70px rgba(0,0,0,.40), inset 0 0 22px rgba(91,255,151,.035);
  overflow: hidden;
  width: 100%; max-width: 100%; flex: 0 0 auto; min-height: 620px; box-sizing: border-box;
}
.rb-map iframe { height: 560px !important; min-height: 560px !important; filter: grayscale(.15) saturate(.75) contrast(1.05) brightness(.88); }
.rb-map .map-title {
  display: flex; justify-content: space-between; gap: 12px; padding: 12px 14px;
  background: #07110d; color: var(--rb-ink) !important; font: 650 14px/1.3 var(--rb-mono);
  text-transform: uppercase; letter-spacing: .05em;
}
.rb-map .map-title span:first-child { color: var(--rb-ink) !important; }
.rb-map .map-title span:last-child { color: var(--rb-radar); font-weight: 600; font-size: 12px; }
.rb-map svg { display: block; width: 100%; height: auto; background: #050807; }
.rb-map .legend {
  display: flex; flex-wrap: wrap; gap: 12px; padding: 10px 14px; border-top: 1px solid var(--rb-line);
  color: var(--rb-soft-text) !important; font-size: 13px; font-weight: 600;
}
.rb-map .legend span { color: var(--rb-soft-text) !important; }
.rb-map .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 6px; vertical-align: -1px; }
.rb-json { border-radius: 3px !important; }
.rb-status { border-radius: 3px !important; background: rgba(91,255,151,.045) !important; }
button.primary,
.gradio-container button.primary {
  background: rgba(91,255,151,.12) !important;
  border: 1px solid var(--rb-line-hot) !important;
  color: var(--rb-ink) !important;
  box-shadow: inset 0 0 18px rgba(91,255,151,.10), 0 0 18px rgba(91,255,151,.08) !important;
}
.gradio-container button.secondary,
.gradio-container button { border-radius: 3px !important; border-color: var(--rb-line) !important; }
.tabs button { font-weight: 600 !important; }
.rb-photo-strip {
  border: 1px solid var(--rb-line);
  border-radius: 4px;
  background: rgba(7,17,13,.76);
  padding: 14px;
  margin: 12px 0 18px;
}
.rb-photo-title {
  color: var(--rb-radar) !important;
  font: 700 12px/1.3 var(--rb-mono);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-bottom: 11px;
}
.rb-photo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 12px;
}
.rb-photo-card {
  display: grid;
  grid-template-columns: 142px minmax(0, 1fr);
  gap: 12px;
  min-height: 116px;
  border: 1px solid rgba(91,255,151,.16);
  border-radius: 4px;
  padding: 10px;
  background: rgba(11,27,19,.72);
}
.rb-photo-card a {
  display: block;
  color: var(--rb-ink);
  text-decoration: underline;
}
.rb-photo-card img {
  display: block;
  width: 142px;
  height: 95px;
  object-fit: cover;
  border-radius: 3px;
  border: 1px solid rgba(91,255,151,.24);
  background: #050807;
}
.rb-photo-meta {
  min-width: 0;
  color: var(--rb-soft-text) !important;
  font-size: 12px;
  line-height: 1.35;
}
.rb-photo-meta strong {
  display: block;
  color: var(--rb-ink) !important;
  font-size: 13px;
  margin-bottom: 3px;
}
.rb-photo-meta span {
  display: block;
  overflow-wrap: anywhere;
}
.rb-photo-meta p {
  margin: 7px 0 0;
  color: var(--rb-muted) !important;
}
.rb-photo-empty p {
  margin: 0;
  color: var(--rb-soft-text) !important;
  font-size: 15px !important;
}
.rb-photo-strip,
.rb-photo-strip * {
  color: var(--rb-soft-text) !important;
}
.rb-photo-strip .rb-photo-title {
  color: var(--rb-radar) !important;
}
@media (max-width: 780px) {
  .gradio-container { padding-left: 12px !important; padding-right: 12px !important; }
  #rb-hero .hero-inner { min-height: 320px; padding: 22px; grid-template-columns: 1fr; }
  #rb-hero h1 { font-size: 44px; }
  #rb-hero .hero-panel { min-height: 220px; }
  .rb-photo-grid { grid-template-columns: 1fr; }
  .rb-photo-card { grid-template-columns: 116px minmax(0, 1fr); }
  .rb-photo-card img { width: 116px; height: 78px; }
}
"""

APP_THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")

def hero_image_data_uri() -> str:
    if not HERO_IMAGE_PATH.exists():
        return ""
    encoded = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def hero_html() -> str:
    image_uri = hero_image_data_uri()
    style = f" style=\"--rb-hero-image: url('{image_uri}')\"" if image_uri else ""
    return f"""
<section id="rb-hero"{style}>
  <div class="hero-inner">
    <div>
      <div class="eyebrow">live ads-b rarity radar</div>
      <h1>rarebirds</h1>
      <p>Track the flights that make plane spotters look up twice — emergency squawks, special-use aircraft, rare types, and weird one-offs surfaced from the live sky.</p>
      <div class="chips">
        <span class="chip">Qwen3-4B ready</span>
        <span class="chip">deterministic fallback</span>
        <span class="chip">auditable rule scores</span>
      </div>
    </div>
    <aside class="hero-panel" aria-label="RareBirds signal stack">
      <div>
        <div class="label">Signal stack</div>
        <div class="scope-row"><span>Live feed</span><strong>ADS-B radius scan</strong></div>
        <div class="scope-row"><span>First pass</span><strong>rules engine</strong></div>
        <div class="scope-row"><span>Adjudicator</span><strong>Qwen3-4B candidate</strong></div>
      </div>
      <div>
        <div class="signal-row"><span>Held-out eval</span><strong class="metric">150/150</strong></div>
        <div class="signal-row"><span>Invalid JSON</span><strong>0</strong></div>
        <div class="signal-row"><span>Fallback mode</span><strong>preserve evidence</strong></div>
      </div>
    </aside>
  </div>
</section>
"""


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    adapter_dir: str
    load_in_4bit: bool
    max_seq_length: int

MODEL = None
TOKENIZER = None
MODEL_CONFIG: ModelConfig | None = None
SCAN_CACHE: dict[tuple[str, float, float, int], dict[str, Any]] = {}
PLANESPOTTERS_PHOTO_CACHE: dict[tuple[str, str], dict[str, Any]] = {}

SYSTEM_PROMPT = (
    "You are rarebirds, a strict aircraft rarity classifier for plane spotters. "
    "You must output exactly one JSON object with keys is_rare, confidence, reason. "
    "No markdown, no metadata, no extra keys."
)


EXAMPLES = [
    {
        "aircraft": {
            "icao_hex": "ae6031",
            "callsign": "KNIFE07",
            "registration": "17-20962",
            "type_designator": "H60",
            "description": "SIKORSKY UH-60 Black Hawk",
            "operator": None,
            "altitude_ft": 1750,
            "ground_speed_kt": 89.4,
            "distance_nm": 10.5,
            "squawk": "1206",
        },
        "observer_context": {
            "region": "Orange County and Los Angeles County, Southern California",
            "observer_lat": 33.8121,
            "observer_lon": -117.919,
            "current_local_area": "downtown Los Angeles",
            "nearest_airport": "LAX",
            "nearest_military_area": "Joint Forces Training Base Los Alamitos",
            "distance_to_nearest_military_nm": 18,
            "military_pattern": "away_from_base_pattern",
        },
    },
    {
        "aircraft": {
            "icao_hex": "synthetic001",
            "callsign": "GTI456",
            "registration": None,
            "type_designator": "BLCF",
            "description": "BOEING 747-400 Dreamlifter",
            "operator": "Atlas Air",
            "altitude_ft": 28000,
            "ground_speed_kt": 410,
            "distance_nm": 65,
            "squawk": "1200",
        },
        "observer_context": {
            "region": "Orange County and Los Angeles County, Southern California",
            "observer_lat": 33.6757,
            "observer_lon": -117.8682,
            "current_local_area": "central Orange County near SNA",
            "nearest_airport": "SNA",
            "nearest_military_area": "Joint Forces Training Base Los Alamitos",
            "distance_to_nearest_military_nm": 15,
            "military_pattern": "away_from_base_pattern",
        },
    },
    {
        "aircraft": {
            "icao_hex": "a74505",
            "callsign": "DAL41",
            "registration": "N568DZ",
            "type_designator": "A359",
            "description": "AIRBUS A-350-900",
            "operator": "Delta Air Lines",
            "altitude_ft": 0,
            "ground_speed_kt": 0,
            "distance_nm": 0.17,
            "squawk": None,
        },
        "observer_context": {
            "region": "Orange County and Los Angeles County, Southern California",
            "observer_lat": 33.9425,
            "observer_lon": -118.4081,
            "current_local_area": "LAX corridor",
            "nearest_airport": "LAX",
            "nearest_military_area": "Joint Forces Training Base Los Alamitos",
            "distance_to_nearest_military_nm": 22,
            "military_pattern": "away_from_base_pattern",
        },
    },
    {
        "aircraft": {
            "icao_hex": "ae0001",
            "callsign": "EDWARDS12",
            "registration": None,
            "type_designator": "T38",
            "description": "NORTHROP T-38 Talon",
            "operator": "USAF",
            "altitude_ft": 6200,
            "ground_speed_kt": 285,
            "distance_nm": 8,
            "squawk": "1200",
        },
        "observer_context": {
            "region": "Orange County and Los Angeles County, Southern California",
            "observer_lat": 34.9054,
            "observer_lon": -117.8837,
            "current_local_area": "Edwards Air Force Base pattern",
            "nearest_airport": "EDW",
            "nearest_military_area": "Edwards Air Force Base",
            "distance_to_nearest_military_nm": 3,
            "military_pattern": "base_pattern",
        },
    },
]


def format_training_text(prompt: str) -> str:
    return f"### System\n{SYSTEM_PROMPT}\n\n### Input JSON\n{prompt}\n\n### Output JSON\n"


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def device_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda_available": False}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "device": torch.cuda.get_device_name(device),
        "cuda_available": True,
        "vram_gb": round(props.total_memory / 1_000_000_000, 2),
    }


def preferred_torch_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def model_input_device(model):
    return next(model.parameters()).device


def load_model(config: ModelConfig):
    dtype = preferred_torch_dtype()
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    try:
        model = AutoModelForImageTextToText.from_pretrained(config.model_id, **kwargs)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    if config.adapter_dir:
        model = PeftModel.from_pretrained(model, config.adapter_dir)
    model.eval()
    return model


def get_model(config: ModelConfig):
    global MODEL, TOKENIZER, MODEL_CONFIG
    if MODEL is None or TOKENIZER is None or MODEL_CONFIG != config:
        TOKENIZER = load_tokenizer(config.model_id)
        MODEL = load_model(config)
        MODEL_CONFIG = config
    return MODEL, TOKENIZER


def extract_first_json(text: str) -> dict[str, Any] | None:
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def split_input(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if "aircraft" in payload:
        aircraft = payload["aircraft"]
        observer_context = payload.get("observer_context")
    else:
        aircraft = payload
        observer_context = payload.get("observer_context")
    if not isinstance(aircraft, dict):
        raise ValueError("Input must be an aircraft object or an object with an 'aircraft' object.")
    if observer_context is not None and not isinstance(observer_context, dict):
        raise ValueError("'observer_context' must be an object when provided.")
    return aircraft, observer_context


@spaces.GPU(duration=120)
def classify(
    aircraft_json: str,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    model_id: str,
    adapter_dir: str,
    load_in_4bit: bool,
    max_seq_length: int,
):
    try:
        payload = json.loads(aircraft_json)
        aircraft, observer_context = split_input(payload)
    except json.JSONDecodeError as exc:
        return f"Invalid input JSON: {exc}", "", {}
    except ValueError as exc:
        return str(exc), "", {}

    config = ModelConfig(
        model_id=model_id.strip(),
        adapter_dir=adapter_dir.strip(),
        load_in_4bit=bool(load_in_4bit),
        max_seq_length=int(max_seq_length),
    )
    if not config.model_id:
        return "Model ID is required.", "", {}
    model, tokenizer = get_model(config)
    prompt = make_rarity_prompt(aircraft, observer_context)
    text = format_training_text(prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_seq_length).to(model_input_device(model))
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            do_sample=temperature > 0,
            temperature=float(temperature) if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - started
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    parsed = extract_first_json(raw)
    status = {
        "model_id": config.model_id,
        "adapter_dir": config.adapter_dir or None,
        "load_in_4bit": config.load_in_4bit,
        "torch_dtype": str(preferred_torch_dtype()).replace("torch.", ""),
        "max_seq_length": config.max_seq_length,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "latency_seconds": round(latency, 3),
        **device_summary(),
        "valid_target_schema": isinstance(parsed, dict)
        and set(parsed) == {"is_rare", "confidence", "reason"}
        and isinstance(parsed.get("is_rare"), bool),
    }
    return raw, json.dumps(parsed, indent=2, sort_keys=True) if parsed is not None else "", status


def result_markdown(parsed_json: str, status: dict[str, Any]) -> str:
    try:
        parsed = json.loads(parsed_json) if parsed_json else None
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return "### Unable to parse rarity JSON"
    verdict = "Rare" if parsed.get("is_rare") else "Not rare"
    confidence = parsed.get("confidence")
    confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "unknown"
    latency = status.get("latency_seconds", "unknown")
    reason = parsed.get("reason") or "No reason returned."
    return f"### {verdict}\n\n**Confidence:** {confidence_text}  \n**Latency:** {latency}s\n\n{reason}"


def classify_for_demo(*args):
    raw, parsed, status = classify(*args)
    return result_markdown(parsed, status), raw, parsed, status


def live_model_config() -> ModelConfig:
    return ModelConfig(
        model_id=DEFAULT_MODEL_ID,
        adapter_dir=DEFAULT_ADAPTER_DIR,
        load_in_4bit=DEFAULT_LOAD_IN_4BIT,
        max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
    )


def live_observer_context(lat: float, lon: float, radius_nm: int) -> dict[str, Any]:
    return {
        "region": "live_map",
        "observer_lat": float(lat),
        "observer_lon": float(lon),
        "watch_radius_nm": int(radius_nm),
        "local_frequency_context": {
            "class": "unknown_or_contextual",
            "alert_policy": "use_aircraft_rarity_signals",
        },
    }


def should_model_score_aircraft(aircraft: dict[str, Any], label: dict[str, Any]) -> bool:
    type_designator = (aircraft.get("type_designator") or "").upper()
    squawk = str(aircraft.get("squawk") or "")
    return (
        bool(label.get("is_rare"))
        or type_designator in CONTEXTUAL_TYPE_DESIGNATORS
        or type_designator in CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS
        or squawk in {"7500", "7600", "7700"}
    )


@spaces.GPU(duration=120)
def model_label_aircraft(aircraft: dict[str, Any], weak: dict[str, Any], lat: float, lon: float, radius_nm: int) -> dict[str, Any]:
    config = live_model_config()
    if not config.model_id:
        return {**weak, "label_source": "deterministic_rules_no_model"}
    if not torch.cuda.is_available():
        return {
            **weak,
            "is_rare": False,
            "confidence": 0,
            "reason": "Model verdict unavailable; GPU model runtime is not active.",
            "label_source": "model_unavailable",
            "model_error": "No CUDA GPU is available in this Space runtime.",
            "prefilter_reason": weak.get("reason"),
        }
    try:
        model, tokenizer = get_model(config)
        prompt = make_rarity_prompt(aircraft, live_observer_context(lat, lon, radius_nm))
        text = format_training_text(prompt)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.max_seq_length).to(model_input_device(model))
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        parsed = extract_first_json(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("is_rare"), bool):
            raise ValueError(f"model did not return valid rarity JSON: {raw[:180]}")
        return {
            "is_rare": bool(parsed["is_rare"]),
            "confidence": float(parsed.get("confidence") or 0),
            "reason": str(parsed.get("reason") or "Model returned no reason."),
            "label_source": "model",
            "model_id": config.model_id,
            "adapter_dir": config.adapter_dir or None,
            "latency_seconds": round(time.perf_counter() - started, 3),
            "prefilter_reason": weak.get("reason"),
            "rule_score": weak.get("rarity_score"),
            "rule_recommendation": weak.get("recommendation"),
            "rule_reason_codes": weak.get("reason_codes"),
            "rule_factors": weak.get("factors"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **weak,
            "reason": explanation_from_score(weak),
            "label_source": "deterministic_rules_model_fallback",
            "model_error": str(exc),
            "prefilter_reason": weak.get("reason"),
        }


def adjudicate_live_labels(
    normalized: list[tuple[dict[str, Any], dict[str, Any]]],
    lat: float,
    lon: float,
    radius_nm: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = [
        item for item in normalized
        if should_model_score_aircraft(item[0], item[1])
    ]
    candidates.sort(key=lambda item: interesting_score(item[0], item[1]))
    candidate_ids = {
        aircraft.get("icao_hex")
        for aircraft, _label in candidates[:DEFAULT_LIVE_MODEL_CANDIDATE_LIMIT]
    }
    adjudicated = []
    for aircraft, label in normalized:
        if aircraft.get("icao_hex") in candidate_ids:
            label = model_label_aircraft(aircraft, label, lat, lon, radius_nm)
        else:
            label = {
                **label,
                "label_source": "deterministic_rules_prefilter",
            }
        adjudicated.append((aircraft, label))
    return adjudicated


def payload_from_form(
    icao_hex: str,
    callsign: str,
    registration: str,
    type_designator: str,
    description: str,
    operator: str,
    altitude_ft: float | int | None,
    ground_speed_kt: float | int | None,
    distance_nm: float | int | None,
    squawk: str,
    current_local_area: str,
    nearest_airport: str,
    nearest_military_area: str,
    military_pattern: str,
) -> str:
    aircraft = {
        "icao_hex": icao_hex.strip() or None,
        "callsign": callsign.strip() or None,
        "registration": registration.strip() or None,
        "type_designator": type_designator.strip().upper() or None,
        "description": description.strip() or None,
        "operator": operator.strip() or None,
        "altitude_ft": altitude_ft,
        "ground_speed_kt": ground_speed_kt,
        "distance_nm": distance_nm,
        "squawk": squawk.strip() or None,
    }
    observer_context = {
        "region": "Orange County and Los Angeles County, Southern California",
        "observer_lat": DEFAULT_LAT,
        "observer_lon": DEFAULT_LON,
        "current_local_area": current_local_area.strip() or None,
        "nearest_airport": nearest_airport.strip().upper() or None,
        "nearest_military_area": nearest_military_area.strip() or None,
        "military_pattern": military_pattern.strip() or None,
    }
    return json.dumps({"aircraft": aircraft, "observer_context": observer_context}, indent=2, sort_keys=True)


def interesting_score(aircraft: dict[str, Any], label: dict[str, Any]) -> tuple[int, float, str]:
    rare_bonus = 0 if label.get("is_rare") else 1
    distance = aircraft.get("distance_nm")
    distance_value = float(distance) if isinstance(distance, (int, float)) else 9999.0
    return (rare_bonus, distance_value, aircraft.get("icao_hex") or "")


AIRPORT_MARKERS = [
    ("SNA", 33.6757, -117.8682),
    ("LAX", 33.9425, -118.4081),
    ("LGB", 33.8177, -118.1516),
    ("ONT", 34.0560, -117.6012),
    ("BUR", 34.2007, -118.3587),
]


def airport_icon_html(code: str) -> str:
    safe_code = html.escape(code)
    return (
        "<div style=\"position:relative;width:58px;height:28px;display:flex;align-items:center;gap:5px;"
        "font:800 12px 'IBM Plex Sans Condensed','Avenir Next Condensed',sans-serif;color:#1F6F5F;filter:drop-shadow(0 2px 4px rgba(31,111,95,.18));\">"
        "<span style=\"width:22px;height:22px;border-radius:50% 50% 50% 0;background:#EEEEEE;"
        "border:2px solid #1F6F5F;display:flex;align-items:center;justify-content:center;"
        "transform:rotate(-45deg);flex:0 0 auto;\">"
        "<svg width=\"12\" height=\"12\" viewBox=\"0 0 24 24\" aria-hidden=\"true\" "
        "style=\"display:block;transform:rotate(45deg);\">"
        "<path fill=\"#1F6F5F\" d=\"M12 2 4 20h16L12 2Zm0 5.2 3.8 9H8.2l3.8-9Z\"/>"
        "</svg>"
        "</span>"
        "<span style=\"padding:2px 5px;border-radius:4px;background:rgba(238,238,238,.78);"
        "border:1px solid rgba(31,111,95,.18);line-height:1;\">"
        f"{safe_code}</span>"
        "</div>"
    )


def airplane_icon_html(background: str, heading_deg: float, size: int, *, rare: bool = False, label: str = "") -> str:
    plane_fill = "#1F6F5F" if rare else "#EEEEEE"
    glow = (
        "box-shadow:0 0 0 5px rgba(242,201,76,.55),0 0 0 13px rgba(242,201,76,.24),0 12px 30px rgba(31,111,95,.46);"
        if rare
        else "box-shadow:0 2px 8px rgba(31,111,95,.35);"
    )
    border = "4px solid #EEEEEE" if rare else "2px solid #EEEEEE"
    label_html = ""
    if rare and label:
        label_html = (
            f"<div style=\"position:absolute;left:50%;top:{size + 8}px;transform:translateX(-50%);"
            "white-space:nowrap;background:#1F6F5F;color:#F2C94C;border:1px solid #F2C94C;"
            "border-radius:999px;padding:4px 9px;font:800 11px 'IBM Plex Mono',monospace;"
            "letter-spacing:.03em;box-shadow:0 4px 12px rgba(31,111,95,.3);\">"
            f"{html.escape(label[:12])}</div>"
        )
    return (
        f"<div style=\"position:relative;width:{size}px;height:{size}px;display:flex;align-items:center;justify-content:center;"
        f"border-radius:999px;background:{background};border:{border};{glow}\">"
        f"<svg width=\"{size - 8}\" height=\"{size - 8}\" viewBox=\"0 0 24 24\" aria-hidden=\"true\" "
        f"style=\"display:block;transform:rotate({heading_deg:.0f}deg);\">"
        f"<path fill=\"{plane_fill}\" d=\"M21.8 13.4 14.3 10.9 12.1 2.8C12 2.3 11.6 2 11.1 2s-.9.3-1 .8L7.9 10.9.6 13.4c-.4.1-.6.5-.6.9v1c0 .6.5 1 1.1.9l6.7-1.2-1.5 5.1-2.2 1.3c-.3.2-.5.5-.5.8v.7c0 .4.4.7.8.6l3.8-.9 2.2-3.3c.3-.4.9-.4 1.2 0l2.2 3.3 3.8.9c.4.1.8-.2.8-.6v-.7c0-.3-.2-.6-.5-.8l-2.2-1.3-1.5-5.1 6.7 1.2c.6.1 1.1-.3 1.1-.9v-1c0-.4-.2-.8-.6-.9Z\"/>"
        "</svg>"
        f"{label_html}"
        "</div>"
    )


def preserve_map_view_script(map_name: str, lat: float, lon: float, radius_nm: int) -> str:
    view_key = f"rarebirds-map-view:{round(float(lat), 4)}:{round(float(lon), 4)}:{int(radius_nm)}"
    return f"""
(function() {{
  const map = {map_name};
  const key = {json.dumps(view_key)};
  const storage = (() => {{
    try {{
      if (window.parent && window.parent.localStorage) return window.parent.localStorage;
    }} catch (_err) {{}}
    try {{
      return window.localStorage;
    }} catch (_err) {{
      return null;
    }}
  }})();
  const restore = () => {{
    if (!storage) return;
    try {{
      const saved = JSON.parse(storage.getItem(key) || "null");
      if (
        saved &&
        Number.isFinite(saved.lat) &&
        Number.isFinite(saved.lng) &&
        Number.isFinite(saved.zoom)
      ) {{
        map.setView([saved.lat, saved.lng], saved.zoom, {{ animate: false }});
      }}
    }} catch (_err) {{}}
  }};
  const save = () => {{
    if (!storage) return;
    const center = map.getCenter();
    storage.setItem(key, JSON.stringify({{
      lat: center.lat,
      lng: center.lng,
      zoom: map.getZoom()
    }}));
  }};
  setTimeout(restore, 0);
  map.on("moveend zoomend", save);
}})();
"""


def map_popup_style() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600;700&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap');
.leaflet-popup-content-wrapper,
.leaflet-popup-tip {
  background: #f3f6f1;
  color: #132018;
}
.leaflet-popup-content {
  color: #132018;
  font-family: "IBM Plex Sans Condensed", "Avenir Next Condensed", sans-serif;
  font-size: 16px;
  line-height: 1.34;
}
.leaflet-popup-content b {
  color: #07110d;
  display: block;
  font-size: 20px;
  line-height: 1.1;
  margin-bottom: 6px;
}
.leaflet-popup-content a {
  color: #175b46;
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  font-weight: 700;
}
.leaflet-popup-close-button {
  color: #175b46 !important;
  font: 700 24px/1 "IBM Plex Mono", monospace !important;
}
</style>
"""


def aircraft_key(aircraft: dict[str, Any]) -> str:
    return str(aircraft.get("icao_hex") or aircraft.get("registration") or aircraft.get("callsign") or "")


def aircraft_map_html(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    lat: float,
    lon: float,
    radius_nm: int,
    photos_by_aircraft: dict[str, dict[str, Any]] | None = None,
) -> str:
    photos_by_aircraft = photos_by_aircraft or {}
    points = [
        (aircraft, label)
        for aircraft, label in rows
        if isinstance(aircraft.get("lat"), (int, float)) and isinstance(aircraft.get("lon"), (int, float))
    ]
    map_obj = folium.Map(
        location=[float(lat), float(lon)],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
        zoom_control=True,
    )
    folium.Circle(
        location=[float(lat), float(lon)],
        radius=radius_nm * 1852,
        color="#2FA084",
        weight=1,
        fill=True,
        fill_color="#2FA084",
        fill_opacity=0.06,
        interactive=False,
    ).add_to(map_obj)
    folium.CircleMarker(
        location=[float(lat), float(lon)],
        radius=7,
        color="#EEEEEE",
        weight=2,
        fill=True,
        fill_color="#6FCF97",
        fill_opacity=1,
        tooltip="Observer",
    ).add_to(map_obj)

    bounds = [[float(lat), float(lon)]]
    for code, airport_lat, airport_lon in AIRPORT_MARKERS:
        folium.Marker(
            [airport_lat, airport_lon],
            tooltip=code,
            icon=folium.DivIcon(
                icon_size=(58, 28),
                icon_anchor=(11, 22),
                html=airport_icon_html(code),
            ),
        ).add_to(map_obj)
        bounds.append([airport_lat, airport_lon])

    for aircraft, label in points[:500]:
        is_rare = bool(label.get("is_rare"))
        color = "#F2C94C" if is_rare else "#2FA084"
        heading = aircraft.get("heading_deg")
        heading_deg = float(heading) if isinstance(heading, (int, float)) else 45.0
        callsign = aircraft.get("callsign") or "unknown"
        type_designator = aircraft.get("type_designator") or "unknown type"
        distance = aircraft.get("distance_nm")
        label_source = str(label.get("label_source") or "")
        verdict_title = "Model verdict" if label_source == "model" else "Prefilter verdict"
        verdict_value = "rare" if is_rare else "routine/unknown"
        popup_rows = [
            f"<b>{html.escape(str(callsign))}</b>",
            f"Type: {html.escape(str(type_designator))}",
            f"Registration: {html.escape(str(aircraft.get('registration') or 'unknown'))}",
            f"Operator: {html.escape(str(aircraft.get('operator') or 'unknown'))}",
            f"Distance: {html.escape(str(distance if distance is not None else 'unknown'))} nm",
            f"{verdict_title}: {verdict_value}",
            f"Rule score: {html.escape(str(label.get('rarity_score', label.get('rule_score', 'unknown'))))}/100",
            f"Recommendation: {html.escape(str(label.get('recommendation', label.get('rule_recommendation', 'unknown'))))}",
            f"Rule factors: {html.escape(', '.join(str(code) for code in (label.get('reason_codes') or label.get('rule_reason_codes') or [])[:4]) or 'none')}",
            html.escape(str(label.get("reason") or "")),
        ]
        if label.get("model_error"):
            popup_rows.append(f"Model fallback: {html.escape(str(label.get('model_error'))[:180])}")
        photo = photos_by_aircraft.get(aircraft_key(aircraft))
        if photo:
            image = photo.get("thumbnail_large") or photo.get("thumbnail") or {}
            image_src = image.get("src")
            link = photo.get("link")
            photographer = photo.get("photographer") or "Planespotters.net photographer"
            if image_src and link:
                popup_rows.insert(
                    1,
                    "<a "
                    f"href='{html.escape(str(link), quote=True)}' "
                    "target='_blank' "
                    "style='display:block;margin:8px 0;color:#1F6F5F;text-decoration:underline;'>"
                    f"<img src='{html.escape(str(image_src), quote=True)}' "
                    f"alt='{html.escape(str(aircraft.get('registration') or callsign), quote=True)} aircraft photo' "
                    "style='display:block;width:240px;max-width:100%;height:auto;border-radius:4px;margin-bottom:4px;'>"
                    f"Photo: {html.escape(str(photographer))} / Planespotters.net"
                    "</a>",
                )
        popup_html = (
            "<div style='min-width:260px;max-width:360px;font:16px/1.34 \"IBM Plex Sans Condensed\",\"Avenir Next Condensed\",sans-serif;color:#132018;'>"
            + "<br>".join(popup_rows)
            + "</div>"
        )
        size = 29 if is_rare else 18
        icon_box_size = size + 72 if is_rare else size
        folium.Marker(
            location=[float(aircraft["lat"]), float(aircraft["lon"])],
            tooltip=f"{callsign} {type_designator}",
            popup=folium.Popup(popup_html, max_width=390),
            icon=folium.DivIcon(
                icon_size=(icon_box_size, icon_box_size),
                icon_anchor=(size / 2, size / 2),
                html=airplane_icon_html(color, heading_deg, size, rare=is_rare, label=str(callsign)),
            ),
        ).add_to(map_obj)
        bounds.append([float(aircraft["lat"]), float(aircraft["lon"])])

    map_html = map_obj.get_root().render()
    preserve_script = f"<script>{preserve_map_view_script(map_obj.get_name(), float(lat), float(lon), radius_nm)}</script>"
    map_html = map_html.replace("</head>", f"{map_popup_style()}</head>")
    map_html = map_html.replace("</html>", f"{preserve_script}</html>")
    iframe = (
        "<iframe "
        f"srcdoc=\"{html.escape(map_html, quote=True)}\" "
        "width=\"100%\" height=\"560\" "
        "style=\"border:0;display:block;width:100%;height:560px;\" "
        "loading=\"lazy\"></iframe>"
    )
    rare_count = sum(1 for _aircraft, label in points if label.get("is_rare"))
    model_count = sum(1 for _aircraft, label in points if label.get("label_source") == "model")
    return (
        "<div class='rb-map'>"
        f"<div class='map-title'><span>Live aircraft map</span><span>{len(points)} plotted / {rare_count} rare results / {model_count} model verdicts / {radius_nm} nm</span></div>"
        + iframe
        + "<div class='legend'><span><i style='background:#F2C94C'></i>rare flight</span><span><i style='background:#2FA084'></i>routine/unknown</span><span><i style='background:#1F6F5F'></i>observer</span></div>"
        + "</div>"
    )


def scan_cache_key(provider: str, lat: float, lon: float, radius_nm: int) -> tuple[str, float, float, int]:
    return (provider, round(float(lat), 4), round(float(lon), 4), int(radius_nm))


def clean_planespotters_key(value: Any, *, uppercase: bool = False) -> str:
    cleaned = "".join(char for char in str(value or "").strip() if char.isalnum() or char == "-")
    return cleaned.upper() if uppercase else cleaned.lower()


def fetch_planespotters_photo(kind: str, value: str) -> dict[str, Any] | None:
    if kind not in {"hex", "reg"} or not value:
        return None
    cache_key = (kind, value.lower())
    cached = PLANESPOTTERS_PHOTO_CACHE.get(cache_key)
    now = time.time()
    if cached and now - float(cached.get("fetched_at") or 0) < PLANESPOTTERS_CACHE_SECONDS:
        return cached.get("photo")

    request = urllib.request.Request(
        f"https://api.planespotters.net/pub/photos/{kind}/{value}",
        headers={"User-Agent": PLANESPOTTERS_USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        PLANESPOTTERS_PHOTO_CACHE[cache_key] = {
            "fetched_at": now,
            "photo": None,
            "error": str(exc),
        }
        return None

    photos = payload.get("photos") if isinstance(payload, dict) else None
    photo = photos[0] if isinstance(photos, list) and photos and isinstance(photos[0], dict) else None
    PLANESPOTTERS_PHOTO_CACHE[cache_key] = {"fetched_at": now, "photo": photo}
    return photo


def planespotters_photo_for_aircraft(aircraft: dict[str, Any]) -> dict[str, Any] | None:
    registration = clean_planespotters_key(aircraft.get("registration"), uppercase=True)
    if registration:
        photo = fetch_planespotters_photo("reg", registration)
        if photo:
            return photo
    hex_code = clean_planespotters_key(aircraft.get("icao_hex"))
    if hex_code:
        return fetch_planespotters_photo("hex", hex_code)
    return None


def planespotters_photos_for_rows(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    photos = {}
    for aircraft, _label in rows:
        photo = planespotters_photo_for_aircraft(aircraft)
        if photo:
            photos[aircraft_key(aircraft)] = photo
    return photos


def planespotters_photo_gallery(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
    photos_by_aircraft: dict[str, dict[str, Any]] | None = None,
) -> str:
    photos_by_aircraft = photos_by_aircraft or {}
    cards = []
    for aircraft, label in rows:
        photo = photos_by_aircraft.get(aircraft_key(aircraft))
        if photo is None:
            photo = planespotters_photo_for_aircraft(aircraft)
        if not photo:
            continue
        image = photo.get("thumbnail_large") or photo.get("thumbnail") or {}
        image_src = image.get("src")
        link = photo.get("link")
        if not image_src or not link:
            continue
        callsign = aircraft.get("callsign") or "unknown"
        type_designator = aircraft.get("type_designator") or "unknown type"
        registration = aircraft.get("registration") or aircraft.get("icao_hex") or "unknown airframe"
        photographer = photo.get("photographer") or "Planespotters.net photographer"
        reason = label.get("reason") or ""
        cards.append(
            "<article class='rb-photo-card'>"
            f"<a href='{html.escape(str(link), quote=True)}' target='_blank'>"
            f"<img src='{html.escape(str(image_src), quote=True)}' alt='{html.escape(str(registration), quote=True)} aircraft photo'>"
            "</a>"
            "<div class='rb-photo-meta'>"
            f"<strong>{html.escape(str(callsign))} · {html.escape(str(type_designator))}</strong>"
            f"<span>{html.escape(str(registration))}</span>"
            f"<span>Photo: {html.escape(str(photographer))} / Planespotters.net</span>"
            f"<p>{html.escape(str(reason))}</p>"
            "</div>"
            "</article>"
        )
    if not cards:
        return (
            "<section class='rb-photo-strip rb-photo-empty'>"
            "<div class='rb-photo-title'>Aircraft photos</div>"
            "<p>No PlaneSpotters photo match for the rare aircraft returned in this scan.</p>"
            "</section>"
        )
    return (
        "<section class='rb-photo-strip'>"
        "<div class='rb-photo-title'>Aircraft photos</div>"
        "<div class='rb-photo-grid'>"
        + "".join(cards)
        + "</div>"
        "</section>"
    )


def scan_outputs_from_normalized(
    normalized: list[tuple[dict[str, Any], dict[str, Any]]],
    lat: float,
    lon: float,
    radius_nm: int,
    limit: int,
    status: dict[str, Any],
):
    rare_rows = [(aircraft, label) for aircraft, label in normalized if label.get("is_rare")]
    rare_rows.sort(key=lambda item: interesting_score(item[0], item[1]))
    selected = rare_rows[: int(limit)]
    table = [
        [
            aircraft.get("callsign") or "unknown",
            aircraft.get("type_designator") or "unknown",
            aircraft.get("registration") or "",
            aircraft.get("operator") or aircraft.get("description") or "",
            aircraft.get("distance_nm"),
            label.get("rarity_score", label.get("rule_score", "")),
            label.get("recommendation", label.get("rule_recommendation", "")),
            label.get("reason"),
        ]
        for aircraft, label in selected
    ]
    top_payload = ""
    if selected:
        top_payload = json.dumps({"aircraft": selected[0][0]}, indent=2, sort_keys=True)
    config = live_model_config()
    status = {
        **status,
        "rare_aircraft": len(rare_rows),
        "rows_returned": len(table),
        "deterministic_prefiltered": sum(
            1 for _aircraft, label in normalized
            if label.get("label_source") == "deterministic_rules_prefilter"
        ),
        "deterministic_no_model": sum(
            1 for _aircraft, label in normalized
            if label.get("label_source") == "deterministic_rules_no_model"
        ),
        "model_scored": sum(1 for _aircraft, label in normalized if label.get("label_source") == "model"),
        "model_fallbacks": sum(1 for _aircraft, label in normalized if label.get("label_source") == "deterministic_rules_model_fallback"),
        "model_id": config.model_id or None,
        "adapter_dir": config.adapter_dir or None,
        "model_candidate_limit": DEFAULT_LIVE_MODEL_CANDIDATE_LIMIT,
    }
    photo_rows = rare_rows[: min(len(rare_rows), max(int(limit), 24))]
    photos_by_aircraft = planespotters_photos_for_rows(photo_rows)
    return (
        table,
        top_payload,
        status,
        aircraft_map_html(normalized, float(lat), float(lon), radius_nm, photos_by_aircraft),
        planespotters_photo_gallery(selected, photos_by_aircraft),
    )


def scan_aircraft(provider: str, lat: float, lon: float, dist_nm: int, limit: int):
    radius_nm = int(dist_nm)
    url = adsbfi_url(float(lat), float(lon), radius_nm) if provider == "adsb.fi" else adsblol_url(float(lat), float(lon), radius_nm)
    cache_key = scan_cache_key(provider, float(lat), float(lon), radius_nm)
    cached = SCAN_CACHE.get(cache_key)
    now = time.time()

    if cached and now - cached["fetched_at"] < SCAN_CACHE_SECONDS:
        status = {
            **cached["status"],
            "ok": True,
            "cached": True,
            "cache_age_seconds": round(now - cached["fetched_at"], 1),
            "message": "Using cached live snapshot.",
        }
        return scan_outputs_from_normalized(cached["normalized"], float(lat), float(lon), radius_nm, int(limit), status)

    started = time.perf_counter()
    raw = None
    fetch_error: Exception | None = None
    try:
        raw = fetch_json(url)
    except HTTPError as exc:
        fetch_error = exc
        if exc.code == 429 and provider == "adsb.fi":
            fallback_url = adsblol_url(float(lat), float(lon), radius_nm)
            try:
                raw = fetch_json(fallback_url)
                url = fallback_url
                fetch_error = None
            except Exception as fallback_exc:  # noqa: BLE001
                fetch_error = fallback_exc
    except Exception as exc:  # noqa: BLE001
        fetch_error = exc

    if raw is None:
        if cached:
            status = {
                **cached["status"],
                "ok": True,
                "cached": True,
                "cache_age_seconds": round(now - cached["fetched_at"], 1),
                "warning": f"Refresh failed; using cached live snapshot. {fetch_error}",
                "url": url,
            }
            return scan_outputs_from_normalized(cached["normalized"], float(lat), float(lon), radius_nm, int(limit), status)
        error = "ADS-B provider is rate-limiting requests. Wait a minute and refresh."
        if not isinstance(fetch_error, HTTPError) or fetch_error.code != 429:
            error = str(fetch_error or "No ADS-B data available.")
        return [], "", {"ok": False, "error": error, "url": url}, aircraft_map_html([], float(lat), float(lon), radius_nm), planespotters_photo_gallery([])

    rows = raw.get("aircraft") or raw.get("ac") or []
    if not isinstance(rows, list):
        return [], "", {"ok": False, "error": "API response did not include an aircraft list.", "url": url}, aircraft_map_html([], float(lat), float(lon), radius_nm), planespotters_photo_gallery([])
    normalized = []
    collected_at = int(time.time())
    for row in rows:
        if not isinstance(row, dict):
            continue
        aircraft = normalize_adsb_aircraft(row, collected_at)
        if not aircraft.get("icao_hex"):
            continue
        label = score_aircraft(aircraft, live_observer_context(float(lat), float(lon), radius_nm))
        normalized.append((aircraft, label))
    normalized = adjudicate_live_labels(normalized, float(lat), float(lon), radius_nm)
    normalized.sort(key=lambda item: interesting_score(item[0], item[1]))
    status = {
        "ok": True,
        "url": url,
        "cached": False,
        "aircraft_seen": len(normalized),
        "latency_seconds": round(time.perf_counter() - started, 3),
    }
    SCAN_CACHE[cache_key] = {
        "fetched_at": time.time(),
        "normalized": normalized,
        "status": status,
    }
    return scan_outputs_from_normalized(normalized, float(lat), float(lon), radius_nm, int(limit), status)


def plural(count: int, singular: str, plural_text: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural_text or singular + 's')}"


def scan_status_markdown(status: dict[str, Any]) -> str:
    if not status or not status.get("ok"):
        error = html.escape(str(status.get("error") or "Unknown scan error.")) if status else "Unknown scan error."
        return f"**Scan failed**\n\n{error}\n\nNo aircraft decisions were made."

    aircraft_seen = int(status.get("aircraft_seen") or 0)
    rare_aircraft = int(status.get("rare_aircraft") or 0)
    model_scored = int(status.get("model_scored") or 0)
    model_fallbacks = int(status.get("model_fallbacks") or 0)
    rules_only = int(status.get("deterministic_prefiltered") or 0) + int(status.get("deterministic_no_model") or 0)
    candidate_limit = int(status.get("model_candidate_limit") or DEFAULT_LIVE_MODEL_CANDIDATE_LIMIT)
    cached = "cached snapshot" if status.get("cached") else "fresh scan"
    latency = status.get("latency_seconds", "unknown")
    model_id = status.get("model_id") or "model disabled"
    adapter = status.get("adapter_dir") or "base model"
    adapter_label = "base model" if adapter == "base model" else f"adapter: `{adapter}`"

    if model_scored:
        headline = f"Model is deciding on {plural(model_scored, 'aircraft')}."
        model_status = f"healthy — adjudicated {plural(model_scored, 'aircraft')}"
    elif model_fallbacks:
        headline = "Model was attempted, but the rules engine preserved the decision after fallback."
        model_status = "unavailable — using deterministic rules fallback"
    else:
        headline = "Rules engine is deciding; no aircraft reached model adjudication on this scan."
        model_status = "not needed this scan — all aircraft handled by deterministic rules"

    return (
        "**Decision log**\n\n"
        f"**{headline}**\n\n"
        f"- Seen: **{aircraft_seen}** aircraft / rare results: **{rare_aircraft}**\n"
        f"- Model status: {model_status}\n"
        f"- Model: `{model_id}` ({adapter_label}); candidate limit: **{candidate_limit}**\n"
        f"- Decisions this scan: **{plural(model_scored, 'model verdict')}**, "
        f"**{plural(model_fallbacks, 'model fallback')}**, **{rules_only} rules-only**\n"
        f"- Source: **{cached}**; scan latency: **{latency}s**\n\n"
        "A scan can show 0 model verdicts and still be healthy when all visible traffic is routine enough "
        "for deterministic rules."
    )


def scan_public(provider: str, lat: float, lon: float, dist_nm: int, limit: int):
    table, _top_payload, status, map_html, photo_html = scan_aircraft(provider, lat, lon, dist_nm, limit)
    return table, map_html, photo_html, scan_status_markdown(status)


def scan_location(provider: str, lat: float, lon: float, dist_nm: int, limit: int, location_message: str):
    table, _top_payload, status, map_html, photo_html = scan_aircraft(provider, lat, lon, dist_nm, limit)
    return lat, lon, location_message, table, map_html, photo_html, scan_status_markdown(status)


def city_preset(name: str):
    lat, lon = CITY_PRESETS[name]
    return lat, lon, f"Observer fix: {name} ({lat:.4f}, {lon:.4f})"


def scan_city(name: str, provider: str, dist_nm: int, limit: int):
    lat, lon = CITY_PRESETS[name]
    return scan_location(provider, lat, lon, dist_nm, limit, f"Observer fix: {name} ({lat:.4f}, {lon:.4f})")


BROWSER_LOCATION_JS = """
(provider, _lat, _lon, radius, limit, _status) => new Promise((resolve) => {
  const fallbackLat = 33.7175;
  const fallbackLon = -117.8311;
  if (!navigator.geolocation) {
    resolve([provider, fallbackLat, fallbackLon, radius, limit, "Browser location unavailable. Using LA/Orange County default."]);
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (position) => {
      const lat = Number(position.coords.latitude.toFixed(6));
      const lon = Number(position.coords.longitude.toFixed(6));
      resolve([provider, lat, lon, radius, limit, `Observer fix: browser location (${lat.toFixed(4)}, ${lon.toFixed(4)})`]);
    },
    (error) => {
      resolve([provider, fallbackLat, fallbackLon, radius, limit, `Location permission failed: ${error.message || "unknown error"}. Using LA/Orange County default.`]);
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
  );
})
"""


def example_json(index: int) -> str:
    return json.dumps(EXAMPLES[index], indent=2, sort_keys=True)


def build_app(model_id: str, adapter_dir: str, load_in_4bit: bool, max_seq_length: int):
    with gr.Blocks(title="rarebirds") as app:
        scan_timer = gr.Timer(value=SCAN_REFRESH_SECONDS)
        with gr.Column(elem_classes="rb-landing"):
            with gr.Column(elem_classes="rb-section rb-hero-frame"):
                gr.HTML(hero_html(), elem_classes="rb-hero-shell", container=False, padding=False)
            with gr.Column(elem_classes="rb-section"):
                gr.Markdown("## Live Map")
                with gr.Column(elem_classes="rb-location-panel"):
                    with gr.Row():
                        provider = gr.Radio(["adsb.fi", "adsb.lol"], value="adsb.fi", label="ADS-B provider")
                        scan_lat = gr.Number(label="Observer latitude", value=DEFAULT_LAT, precision=6)
                        scan_lon = gr.Number(label="Observer longitude", value=DEFAULT_LON, precision=6)
                        scan_radius = gr.Slider(5, 100, value=DEFAULT_WATCH_RADIUS_NM, step=5, label="Watch radius nm")
                    with gr.Row():
                        scan_button = gr.Button("Refresh Map", variant="primary")
                        use_browser_location = gr.Button("Use Browser Location", variant="secondary")
                        watch_los_angeles = gr.Button("Los Angeles", variant="secondary")
                        watch_new_york = gr.Button("New York City", variant="secondary")
                        watch_chicago = gr.Button("Chicago", variant="secondary")
                    browser_location_status = gr.Markdown(value=DEFAULT_OBSERVER_LABEL, elem_classes="rb-location-status")
                live_map = gr.HTML(value=aircraft_map_html([], DEFAULT_LAT, DEFAULT_LON, DEFAULT_WATCH_RADIUS_NM), elem_classes="rb-map", container=False, padding=False)
                gr.Markdown("### Rare aircraft nearby")
                scan_table = gr.Dataframe(
                    headers=["callsign", "type", "registration", "operator/description", "distance_nm", "score", "recommendation", "why it is rare"],
                    interactive=False,
                    wrap=True,
                    elem_classes="rb-table",
                )
                scan_photos = gr.HTML(
                    value=planespotters_photo_gallery([]),
                    elem_classes="rb-photos",
                    container=False,
                    padding=False,
                )
                scan_status = gr.Markdown(
                    value="**Decision log**\n\nWaiting for the first scan. The map can show aircraft before any model adjudication happens.",
                    elem_classes="rb-result",
                )
                scan_limit = gr.State(8)
                scan_inputs = [provider, scan_lat, scan_lon, scan_radius, scan_limit]
                scan_outputs = [scan_table, live_map, scan_photos, scan_status]
                location_scan_outputs = [scan_lat, scan_lon, browser_location_status, scan_table, live_map, scan_photos, scan_status]
                scan_button.click(scan_public, scan_inputs, scan_outputs)
                use_browser_location.click(
                    fn=scan_location,
                    inputs=[provider, scan_lat, scan_lon, scan_radius, scan_limit, browser_location_status],
                    outputs=location_scan_outputs,
                    js=BROWSER_LOCATION_JS,
                )
                for city_name, button in [
                    ("Los Angeles", watch_los_angeles),
                    ("New York City", watch_new_york),
                    ("Chicago", watch_chicago),
                ]:
                    button.click(
                        fn=lambda provider, dist_nm, limit, name=city_name: scan_city(name, provider, dist_nm, limit),
                        inputs=[provider, scan_radius, scan_limit],
                        outputs=location_scan_outputs,
                    )
                scan_timer.tick(scan_public, scan_inputs, scan_outputs, show_progress="hidden")

        with gr.Accordion("Developer tools", open=False, elem_classes="rb-settings"):
            with gr.Accordion("Model settings", open=False):
                with gr.Row():
                    model_box = gr.Dropdown(choices=MODEL_CHOICES, value=model_id, allow_custom_value=True, label="Model ID")
                    adapter_box = gr.Textbox(value=adapter_dir, label="Adapter directory or HF adapter repo")
                with gr.Row():
                    load_4bit = gr.Checkbox(value=load_in_4bit, label="Load in 4-bit")
                    max_seq_length_box = gr.Slider(512, 4096, value=max_seq_length, step=128, label="Max sequence length")
                    max_new_tokens = gr.Slider(32, 256, value=DEFAULT_MAX_NEW_TOKENS, step=8, label="Max new tokens")
                    min_new_tokens = gr.Slider(0, 16, value=1, step=1, label="Min new tokens")
                    temperature = gr.Slider(0, 1, value=0, step=0.05, label="Temperature")

            shared_inputs = [max_new_tokens, min_new_tokens, temperature, model_box, adapter_box, load_4bit, max_seq_length_box]

            with gr.Accordion("Manual sighting", open=False):
                with gr.Row():
                    with gr.Column(elem_classes="rb-section"):
                        with gr.Row():
                            icao_hex = gr.Textbox(label="ICAO hex", value="ae6031")
                            callsign = gr.Textbox(label="Callsign", value="KNIFE07")
                            registration = gr.Textbox(label="Registration", value="17-20962")
                        with gr.Row():
                            type_designator = gr.Textbox(label="Type", value="H60")
                            description = gr.Textbox(label="Description", value="SIKORSKY UH-60 Black Hawk")
                        operator = gr.Textbox(label="Operator", value="")
                        with gr.Row():
                            altitude_ft = gr.Number(label="Altitude ft", value=1750)
                            ground_speed_kt = gr.Number(label="Speed kt", value=89.4)
                            distance_nm = gr.Number(label="Distance nm", value=10.5)
                            squawk = gr.Textbox(label="Squawk", value="1206")
                        with gr.Row():
                            current_local_area = gr.Textbox(label="Local area", value="downtown Los Angeles")
                            nearest_airport = gr.Textbox(label="Nearest airport", value="LAX")
                        nearest_military_area = gr.Textbox(label="Nearest military area", value="Joint Forces Training Base Los Alamitos")
                        military_pattern = gr.Dropdown(
                            ["away_from_base_pattern", "base_pattern", "not_base_pattern", ""],
                            value="away_from_base_pattern",
                            label="Military pattern",
                        )
                        build_payload = gr.Button("Build JSON")
                    with gr.Column(elem_classes="rb-output"):
                        form_aircraft = gr.Code(label="Generated aircraft JSON", language="json", lines=18, elem_classes="rb-json")
                        form_run = gr.Button("Classify Sighting", variant="primary")
                        form_result = gr.Markdown(elem_classes="rb-result")
                form_raw = gr.Textbox(label="Raw model output", lines=8, elem_classes="rb-output")
                form_parsed = gr.Code(label="Parsed JSON", language="json", lines=7, elem_classes="rb-json")
                form_status = gr.JSON(label="Status", elem_classes="rb-status")

                form_fields = [
                    icao_hex,
                    callsign,
                    registration,
                    type_designator,
                    description,
                    operator,
                    altitude_ft,
                    ground_speed_kt,
                    distance_nm,
                    squawk,
                    current_local_area,
                    nearest_airport,
                    nearest_military_area,
                    military_pattern,
                ]
                build_payload.click(payload_from_form, form_fields, form_aircraft)
                form_run.click(classify_for_demo, [form_aircraft, *shared_inputs], [form_result, form_raw, form_parsed, form_status])

            with gr.Accordion("JSON inspector", open=False):
                with gr.Row():
                    aircraft = gr.Code(value=example_json(0), language="json", label="Aircraft JSON", lines=18, elem_classes="rb-json")
                    with gr.Column(elem_classes="rb-section"):
                        run = gr.Button("Classify JSON", variant="primary")
                        with gr.Row():
                            ex0 = gr.Button("H-60")
                            ex1 = gr.Button("Dreamlifter")
                            ex2 = gr.Button("A350 ordinary")
                            ex3 = gr.Button("T-38 Edwards")
                inspector_result = gr.Markdown(elem_classes="rb-result")
                raw = gr.Textbox(label="Raw model output", lines=10, elem_classes="rb-output")
                parsed = gr.Code(label="First parsed JSON object", language="json", lines=8, elem_classes="rb-json")
                status = gr.JSON(label="Status", elem_classes="rb-status")

                run.click(classify_for_demo, [aircraft, *shared_inputs], [inspector_result, raw, parsed, status])
                ex0.click(lambda: example_json(0), outputs=aircraft)
                ex1.click(lambda: example_json(1), outputs=aircraft)
                ex2.click(lambda: example_json(2), outputs=aircraft)
                ex3.click(lambda: example_json(3), outputs=aircraft)

        app.load(
            scan_public,
            scan_inputs,
            scan_outputs,
        )
        app.load(
            payload_from_form,
            [
                icao_hex,
                callsign,
                registration,
                type_designator,
                description,
                operator,
                altitude_ft,
                ground_speed_kt,
                distance_nm,
                squawk,
                current_local_area,
                nearest_airport,
                nearest_military_area,
                military_pattern,
            ],
            form_aircraft,
        )
        app.queue(max_size=8)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=DEFAULT_LOAD_IN_4BIT)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_app(args.model_id, args.adapter_dir, args.load_in_4bit, args.max_seq_length)
    app.launch(server_name=args.host, server_port=args.port, theme=APP_THEME, css=APP_CSS)


if __name__ == "__main__":
    main()
