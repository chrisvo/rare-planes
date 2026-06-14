#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from build_quick_1000_dataset import SOCAL_LOCATIONS, observer_context_for, training_row


def aircraft(
    case_id: str,
    type_designator: str,
    description: str,
    operator: str,
    callsign: str,
    squawk: str | None,
    emergency: str,
    location: dict,
) -> dict:
    return {
        "provider": "ga_eval",
        "collected_at": None,
        "icao_hex": case_id,
        "callsign": callsign,
        "registration": callsign if callsign.startswith("N") else f"N{case_id[-3:]}RB",
        "type_designator": type_designator,
        "description": description,
        "operator": operator,
        "origin_country": None,
        "lat": location["lat"],
        "lon": location["lon"],
        "local_area": location["local_area"],
        "nearest_airport": location["nearest_airport"],
        "nearest_military_area": location["nearest_military_area"],
        "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
        "military_pattern": location["military_pattern"],
        "altitude_ft": 2500,
        "ground_speed_kt": 105,
        "heading_deg": 180,
        "vertical_rate_fpm": 0,
        "squawk": squawk,
        "emergency": emergency,
        "distance_nm": 8,
        "category": "A1",
        "seen_seconds": 0.3,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/eval/ga_hard_cases.csv"))
    args = parser.parse_args()

    locations = {item["id"]: item for item in SOCAL_LOCATIONS}
    central_oc = locations["central_orange_county"]
    valley = locations["san_fernando_valley"]
    long_beach = locations["long_beach_south_bay"]

    cases = [
        (
            "ga_eval001",
            aircraft("ga_eval001", "C172", "CESSNA 172 Skyhawk", "local flight school", "N1967S", "1200", "none", central_oc),
            {
                "is_rare": False,
                "confidence": 0.94,
                "reason": "Cessna 172 flight-school traffic is common local GA around SNA and is not rare without emergency, special-mission, vintage/warbird, or watchlist evidence.",
            },
            central_oc,
        ),
        (
            "ga_eval002",
            aircraft("ga_eval002", "C152", "CESSNA 152", "local flight school", "N761YE", "1200", "none", long_beach),
            {
                "is_rare": False,
                "confidence": 0.94,
                "reason": "Cessna 152 training traffic is common in Southern California and should be routine with no special signal.",
            },
            long_beach,
        ),
        (
            "ga_eval003",
            aircraft("ga_eval003", "P28A", "PIPER PA-28 Cherokee", "local flight school", "TRAIN24", "1200", "none", valley),
            {
                "is_rare": False,
                "confidence": 0.93,
                "reason": "PA-28 trainer traffic near VNY/BUR is ordinary local GA and is not rare without a stronger signal.",
            },
            valley,
        ),
        (
            "ga_eval004",
            aircraft("ga_eval004", "SR22", "CIRRUS SR22", "private owner", "N728CB", "1200", "none", central_oc),
            {
                "is_rare": False,
                "confidence": 0.91,
                "reason": "Cirrus SR22 private GA traffic is common enough locally and has no rare-aircraft signal.",
            },
            central_oc,
        ),
        (
            "ga_eval005",
            aircraft("ga_eval005", "C172", "CESSNA 172 Skyhawk", "County Sheriff", "SHERIFF1", "4371", "none", central_oc),
            {
                "is_rare": False,
                "confidence": 0.88,
                "reason": "Cessna 172 sheriff patrol traffic is recognizable, but not rare without emergency, rescue/evacuation, notable registration, unusual type, or watchlist evidence.",
            },
            central_oc,
        ),
        (
            "ga_eval006",
            aircraft("ga_eval006", "C172", "CESSNA 172 Skyhawk", "County Sheriff", "RESCUE12", "4371", "none", central_oc),
            {
                "is_rare": True,
                "confidence": 0.9,
                "reason": "Cessna 172 is usually common, but explicit rescue callsign RESCUE12 makes this a special-incident exception.",
            },
            central_oc,
        ),
        (
            "ga_eval007",
            aircraft("ga_eval007", "C152", "CESSNA 152", "local flight school", "N152WA", "7700", "general", long_beach),
            {
                "is_rare": True,
                "confidence": 0.95,
                "reason": "Cessna 152 is normally routine, but emergency squawk 7700 makes it alert-worthy.",
            },
            long_beach,
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        for _, aircraft_payload, label, location in cases:
            writer.writerow(training_row(aircraft_payload, label, observer_context_for(location)))

    print(json.dumps({"output": str(args.output), "examples": len(cases)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
