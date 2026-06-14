#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_ml_stubs_if_needed() -> None:
    if importlib.util.find_spec("torch") is None:
        torch = types.ModuleType("torch")
        torch.float32 = "float32"
        torch.float16 = "float16"
        torch.bfloat16 = "bfloat16"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch
    if importlib.util.find_spec("peft") is None:
        peft = types.ModuleType("peft")
        peft.PeftModel = object
        sys.modules["peft"] = peft
    if importlib.util.find_spec("transformers") is None:
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = object
        transformers.AutoModelForImageTextToText = object
        transformers.AutoTokenizer = object
        transformers.BitsAndBytesConfig = object
        sys.modules["transformers"] = transformers


def install_ui_stubs_if_needed() -> None:
    class DummyComponent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.title = kwargs.get("title") or (args[0] if args else "")
            self.blocks = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def click(self, *_args, **_kwargs):
            return self

        def tick(self, *_args, **_kwargs):
            return self

        def load(self, *_args, **_kwargs):
            return self

        def queue(self, *_args, **_kwargs):
            return self

        def launch(self, *_args, **_kwargs):
            return self

    if importlib.util.find_spec("gradio") is None:
        gradio = types.ModuleType("gradio")
        for name in [
            "Accordion",
            "Blocks",
            "Button",
            "Checkbox",
            "Code",
            "Column",
            "Dataframe",
            "Dropdown",
            "HTML",
            "JSON",
            "Markdown",
            "Number",
            "Radio",
            "Row",
            "Slider",
            "State",
            "Textbox",
            "Timer",
        ]:
            setattr(gradio, name, DummyComponent)
        gradio.themes = types.SimpleNamespace(Soft=lambda *_args, **_kwargs: DummyComponent())
        sys.modules["gradio"] = gradio

    if importlib.util.find_spec("folium") is None:
        class DummyRoot:
            def render(self) -> str:
                return "<html></html>"

        class DummyMap(DummyComponent):
            def get_root(self):
                return DummyRoot()

            def get_name(self) -> str:
                return "dummy_map"

        class DummyFoliumComponent(DummyComponent):
            def add_to(self, _map):
                return self

        folium = types.ModuleType("folium")
        folium.Map = DummyMap
        folium.Circle = DummyFoliumComponent
        folium.CircleMarker = DummyFoliumComponent
        folium.Marker = DummyFoliumComponent
        folium.DivIcon = DummyFoliumComponent
        folium.Popup = DummyFoliumComponent
        sys.modules["folium"] = folium


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the Gradio Space without downloading model weights.")
    parser.add_argument("--live-scan", action="store_true", help="Also hit the public ADS-B provider used by the Live Snapshot tab.")
    args = parser.parse_args()

    install_ml_stubs_if_needed()
    install_ui_stubs_if_needed()

    from scripts import gradio_rarity_tester as tester

    app = tester.build_app("google/gemma-3-27b-it", "", True, 2048)
    payload = tester.payload_from_form(
        "ae4ece",
        "RCH123",
        "",
        "C17",
        "BOEING C-17 Globemaster",
        "USAF",
        22000,
        410,
        18,
        "",
        "central Orange County",
        "SNA",
        "Joint Forces Training Base Los Alamitos",
        "away_from_base_pattern",
    )
    rendered = tester.result_markdown('{"is_rare":true,"confidence":0.91,"reason":"C-17 away from base pattern."}', {"latency_seconds": 1.23})
    if "C17" not in payload or "Rare" not in rendered:
        raise RuntimeError("Gradio helper smoke check failed.")

    if args.live_scan:
        def fake_fetch_json(_url: str) -> dict:
            return {
                "aircraft": [
                    {
                        "hex": "a00001",
                        "flight": "GTI456",
                        "r": "N249BA",
                        "t": "BLCF",
                        "desc": "BOEING 747-400 Dreamlifter",
                        "ownOp": "Atlas Air",
                        "lat": 33.72,
                        "lon": -117.84,
                        "alt_baro": 22000,
                        "gs": 410,
                        "track": 80,
                        "dst": 8.2,
                        "squawk": "7700",
                    }
                ]
            }

        tester.SCAN_CACHE.clear()
        tester.DEFAULT_MODEL_ID = ""
        tester.fetch_json = fake_fetch_json
        rows, top_payload, status, map_html, photo_html = tester.scan_aircraft("adsb.fi", 33.7175, -117.8311, 10, 5)
        if not status.get("ok") or not rows or not top_payload or "Live aircraft map" not in map_html or "Aircraft photos" not in photo_html:
            raise RuntimeError(f"Live scan smoke check failed: {status}")
        tester.fetch_json = lambda _url: (_ for _ in ()).throw(HTTPError(_url, 429, "Too Many Requests", None, None))
        cached_rows, _cached_payload, cached_status, _cached_map, _cached_photos = tester.scan_aircraft("adsb.fi", 33.7175, -117.8311, 10, 5)
        if not cached_status.get("cached") or not cached_rows:
            raise RuntimeError(f"Cached 429 fallback smoke check failed: {cached_status}")

    if app.title != "rarebirds":
        raise RuntimeError(f"Unexpected app title: {app.title}")
    print({"blocks": len(app.blocks), "title": app.title, "live_scan": args.live_scan})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
