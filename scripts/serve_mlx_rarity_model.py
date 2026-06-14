#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_socal_aircraft_dataset import make_prompt  # noqa: E402


SYSTEM_PROMPT = (
    "You are rarebirds, a strict aircraft rarity classifier for plane spotters. "
    "You must output exactly one JSON object with keys is_rare, confidence, reason. "
    "No markdown, no metadata, no extra keys."
)


def format_training_text(prompt: str) -> str:
    return f"### System\n{SYSTEM_PROMPT}\n\n### Input JSON\n{prompt}\n\n### Output JSON\n"


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def aircraft_from_request(payload: dict[str, Any]) -> dict[str, Any]:
    aircraft = payload.get("aircraft") or {}
    if not isinstance(aircraft, dict):
        aircraft = {}
    return {
        "provider": "ios-dev-bridge",
        "icao_hex": aircraft.get("icaoHex") or aircraft.get("icao_hex"),
        "callsign": aircraft.get("callsign"),
        "registration": aircraft.get("registration"),
        "type_designator": aircraft.get("typeDesignator") or aircraft.get("type_designator"),
        "description": aircraft.get("description"),
        "operator": aircraft.get("operatorName") or aircraft.get("operator"),
        "lat": aircraft.get("latitude") or aircraft.get("lat"),
        "lon": aircraft.get("longitude") or aircraft.get("lon"),
        "altitude_ft": aircraft.get("altitudeFeet") or aircraft.get("altitude_ft"),
        "ground_speed_kt": aircraft.get("groundSpeedKnots") or aircraft.get("ground_speed_kt"),
        "heading_deg": aircraft.get("headingDegrees") or aircraft.get("heading_deg"),
        "distance_nm": aircraft.get("distanceNauticalMiles") or aircraft.get("distance_nm"),
        "squawk": aircraft.get("squawk"),
        "emergency": aircraft.get("emergency"),
    }


def observer_context_from_request(payload: dict[str, Any]) -> dict[str, Any]:
    observer = payload.get("observer_context") or payload.get("observerContext") or {}
    if not isinstance(observer, dict):
        observer = {}
    return {
        "region": "orange_county_los_angeles_southern_california",
        "observer_lat": 33.7175,
        "observer_lon": -117.8311,
        "current_local_area": payload.get("observerArea") or observer.get("current_local_area"),
        "nearest_airport": observer.get("nearest_airport"),
        "nearest_military_area": observer.get("nearest_military_area"),
        "distance_to_nearest_military_nm": (
            payload.get("distanceToNearestMilitaryNauticalMiles")
            or observer.get("distance_to_nearest_military_nm")
        ),
        "military_pattern": payload.get("militaryPattern") or observer.get("military_pattern"),
    }


class RarityServer:
    def __init__(self, model_path: Path, max_tokens: int):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.model, self.tokenizer = load(str(model_path))

    def classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        aircraft = aircraft_from_request(payload)
        observer_context = observer_context_from_request(payload)
        prompt = format_training_text(make_prompt(aircraft, observer_context))
        started = time.perf_counter()
        raw = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        ).strip()
        parsed = extract_json(raw)
        if not parsed or not isinstance(parsed.get("is_rare"), bool):
            return {
                "is_rare": False,
                "confidence": 0,
                "reason": "Model did not return valid rarity JSON.",
                "raw": raw,
                "latency_seconds": time.perf_counter() - started,
            }
        return {
            "is_rare": parsed["is_rare"],
            "confidence": float(parsed.get("confidence") or 0),
            "reason": str(parsed.get("reason") or ""),
            "raw": raw,
            "latency_seconds": time.perf_counter() - started,
        }


def make_handler(classifier: RarityServer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self.write_json({"ok": True, "model": str(classifier.model_path)})

        def do_POST(self) -> None:
            if self.path != "/classify":
                self.send_error(404)
                return
            length = int(self.headers.get("content-length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                self.send_error(400, "invalid JSON")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "JSON body must be an object")
                return
            self.write_json(classifier.classify(payload))

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

        def write_json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("model/output/rarity-gemma4-oc-la-hard-v3-mlx-4bit"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    classifier = RarityServer(args.model, args.max_tokens)
    server = HTTPServer((args.host, args.port), make_handler(classifier))
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "model": str(args.model)}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
