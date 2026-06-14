from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


gr_stub = module("gradio", themes=types.SimpleNamespace(Soft=lambda **kwargs: kwargs))
sys.modules.setdefault("gradio", gr_stub)
sys.modules.setdefault("folium", module("folium"))
sys.modules.setdefault(
    "torch",
    module(
        "torch",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        no_grad=lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *args: None),
    ),
)
sys.modules.setdefault("peft", module("peft", PeftModel=types.SimpleNamespace(from_pretrained=lambda model, adapter: model)))
sys.modules.setdefault(
    "transformers",
    module(
        "transformers",
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoModelForImageTextToText=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        BitsAndBytesConfig=lambda **kwargs: kwargs,
    ),
)

from scripts import gradio_rarity_tester as grt


def test_default_demo_model_points_to_winning_qwen3_4b_adapter():
    adapter_path = Path(grt.DEFAULT_ADAPTER_DIR)

    assert grt.DEFAULT_MODEL_ID == "Qwen/Qwen3-4B"
    assert adapter_path.name == "adapter"
    assert adapter_path.parent.name == "rarity-qwen3-4b-unsloth-qlora"
    assert (adapter_path / "adapter_config.json").exists()
    assert (adapter_path / "adapter_model.safetensors").exists()


def test_model_fallback_preserves_rule_evidence_when_live_model_fails(monkeypatch):
    def broken_get_model(_config):
        raise RuntimeError("boom while loading model")

    monkeypatch.setattr(grt.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(grt, "get_model", broken_get_model)
    monkeypatch.setattr(
        grt,
        "live_model_config",
        lambda: grt.ModelConfig(
            model_id="Qwen/Qwen3-4B",
            adapter_dir=str(Path(grt.ROOT) / "model/output/rarity-qwen3-4b-unsloth-qlora/adapter"),
            load_in_4bit=True,
            max_seq_length=2048,
        ),
    )
    weak = {
        "is_rare": True,
        "confidence": 0.82,
        "reason": "rule prefilter reason",
        "rarity_score": 82,
        "recommendation": "show",
        "reason_codes": ["coast_guard"],
        "factors": [{"code": "coast_guard", "label": "Coast Guard operator", "points": 35}],
        "aircraft_label": "USCG MH-60T Jayhawk C6043",
    }

    label = grt.model_label_aircraft(
        {"callsign": "C6043", "type_designator": "H60"},
        weak,
        lat=33.8,
        lon=-117.9,
        radius_nm=15,
    )

    assert label["is_rare"] is True
    assert label["rarity_score"] == 82
    assert label["recommendation"] == "show"
    assert label["reason_codes"] == ["coast_guard"]
    assert label["label_source"] == "deterministic_rules_model_fallback"
    assert "boom while loading model" in label["model_error"]
    assert "USCG MH-60T Jayhawk C6043" in label["reason"]


def test_scan_status_markdown_explains_model_and_rules_activity():
    status = {
        "ok": True,
        "aircraft_seen": 42,
        "rare_aircraft": 3,
        "rows_returned": 3,
        "model_scored": 2,
        "model_fallbacks": 1,
        "deterministic_prefiltered": 39,
        "cached": False,
        "latency_seconds": 1.23,
        "model_id": "Qwen/Qwen3-4B",
        "adapter_dir": None,
        "model_candidate_limit": 8,
    }

    markdown = grt.scan_status_markdown(status)

    assert "Model is deciding" in markdown
    assert "Model status: healthy" in markdown
    assert "2 aircraft" in markdown
    assert "rare results" in markdown
    assert "rare candidates" not in markdown
    assert "1 model fallback" in markdown
    assert "39 rules-only" in markdown
    assert "2 model verdicts" in markdown
    assert "Qwen/Qwen3-4B" in markdown
    assert "base model" in markdown


def test_scan_status_markdown_explains_model_fallback_state():
    markdown = grt.scan_status_markdown(
        {
            "ok": True,
            "aircraft_seen": 5,
            "rare_aircraft": 1,
            "model_scored": 0,
            "model_fallbacks": 1,
            "deterministic_prefiltered": 4,
            "model_id": "Qwen/Qwen3-4B",
            "adapter_dir": "vochris/rarebirds-adapter",
        }
    )

    assert "Model status: unavailable" in markdown
    assert "using deterministic rules fallback" in markdown


def test_scan_status_markdown_explains_model_not_needed_state():
    markdown = grt.scan_status_markdown(
        {
            "ok": True,
            "aircraft_seen": 14,
            "rare_aircraft": 0,
            "model_scored": 0,
            "model_fallbacks": 0,
            "deterministic_prefiltered": 14,
            "model_id": "Qwen/Qwen3-4B",
        }
    )

    assert "Model status: not needed this scan" in markdown


def test_scan_status_markdown_makes_fetch_errors_obvious():
    markdown = grt.scan_status_markdown({"ok": False, "error": "ADS-B provider unavailable"})

    assert "Scan failed" in markdown
    assert "ADS-B provider unavailable" in markdown
