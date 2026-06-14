#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from build_quick_1000_dataset import SOCAL_LOCATIONS, observer_context_for, training_row


def location(location_id: str) -> dict:
    return next(item for item in SOCAL_LOCATIONS if item["id"] == location_id)


def aircraft(
    case_id: str,
    type_designator: str,
    description: str,
    operator: str | None,
    callsign: str,
    location_id: str,
    *,
    registration: str | None = None,
    squawk: str | None = "1200",
    emergency: str = "none",
    altitude_ft: int = 2500,
    ground_speed_kt: int = 140,
) -> tuple[dict, dict]:
    loc = location(location_id)
    payload = {
        "provider": "policy_regression",
        "collected_at": None,
        "icao_hex": case_id,
        "callsign": callsign,
        "registration": registration or (callsign if callsign.startswith("N") else f"N{case_id[-3:]}RB"),
        "type_designator": type_designator,
        "description": description,
        "operator": operator,
        "origin_country": None,
        "lat": loc["lat"],
        "lon": loc["lon"],
        "local_area": loc["local_area"],
        "nearest_airport": loc["nearest_airport"],
        "nearest_military_area": loc["nearest_military_area"],
        "distance_to_nearest_military_nm": loc["distance_to_nearest_military_nm"],
        "military_pattern": loc["military_pattern"],
        "altitude_ft": altitude_ft,
        "ground_speed_kt": ground_speed_kt,
        "heading_deg": 180,
        "vertical_rate_fpm": 0,
        "squawk": squawk,
        "emergency": emergency,
        "distance_nm": 8,
        "category": None,
        "seen_seconds": 0.3,
    }
    return payload, loc


def case(
    case_id: str,
    type_designator: str,
    description: str,
    operator: str | None,
    callsign: str,
    location_id: str,
    is_rare: bool,
    reason: str,
    *,
    confidence: float = 0.9,
    registration: str | None = None,
    squawk: str | None = "1200",
    emergency: str = "none",
    altitude_ft: int = 2500,
    ground_speed_kt: int = 140,
) -> dict[str, str]:
    payload, loc = aircraft(
        case_id,
        type_designator,
        description,
        operator,
        callsign,
        location_id,
        registration=registration,
        squawk=squawk,
        emergency=emergency,
        altitude_ft=altitude_ft,
        ground_speed_kt=ground_speed_kt,
    )
    return training_row(
        payload,
        {
            "is_rare": is_rare,
            "confidence": confidence,
            "reason": reason,
        },
        observer_context_for(loc),
    )


def build_cases() -> list[dict[str, str]]:
    rows = [
        case("policy001", "A320", "AIRBUS A320", "United Airlines", "UAL123", "lax_corridor", False, "Routine A320-family airline traffic around LAX is not rare without another signal."),
        case("policy002", "B738", "BOEING 737-800", "Southwest Airlines", "SWA220", "central_orange_county", False, "Routine 737-family traffic near SNA is not rare without another signal."),
        case("policy003", "C172", "CESSNA 172 Skyhawk", "local flight school", "TRAIN24", "central_orange_county", False, "Common trainer traffic near SNA is routine."),
        case("policy004", "SR22", "CIRRUS SR22", "private owner", "N728CB", "san_fernando_valley", False, "Common private GA traffic near VNY/BUR is not rare without a hard signal."),
        case("policy005", "C152", "CESSNA 152", "local flight school", "N152WA", "long_beach_south_bay", True, "Emergency squawk 7700 makes otherwise routine trainer traffic alert-worthy.", squawk="7700", emergency="general", confidence=0.96),
        case("policy006", "C172", "CESSNA 172 Skyhawk", "County Sheriff", "SHERIFF1", "central_orange_county", False, "Plain sheriff patrol traffic is public-safety relevant but not rare without emergency, rescue/evacuation, unusual type, notable registration, or watchlist evidence."),
        case("policy007", "AS50", "AEROSPATIALE AS-350 Ecureuil", "SAN BERNARDINO COUNTY SHERIFFS DEPT", "N835SB", "central_orange_county", False, "Local sheriff helicopter activity is routine/near-miss by default, not claimable rare."),
        case("policy008", "C172", "CESSNA 172 Skyhawk", "County Sheriff", "RESCUE12", "central_orange_county", True, "Explicit rescue callsign makes this a special-incident exception."),
        case("policy009", "SR22", "CIRRUS SR22", "CAL FIRE", "FIRE12", "central_orange_county", True, "Emergency squawk makes this public-safety support flight alert-worthy.", squawk="7700", emergency="general"),
        case("policy010", "C130", "LOCKHEED C-130 Hercules", "CALIFORNIA DEPT OF FORESTRY AND FIRE PROTECTION", "CFR605", "central_orange_county", True, "CAL FIRE C-130 combines an unusual aircraft type with a special mission."),
        case("policy011", "H60", "SIKORSKY UH-60 Black Hawk", "United States Army", "KNIFE07", "los_alamitos_pattern", False, "H60 traffic in the Los Alamitos base pattern is locally routine without a hard signal."),
        case("policy012", "H60", "SIKORSKY UH-60 Black Hawk", "United States Army", "KNIFE07", "downtown_los_angeles", True, "H60 away from a known local base pattern over downtown LA is locally alert-worthy."),
        case("policy013", "C17", "BOEING C-17 Globemaster III", "United States Air Force", "RCH321", "march_arb_pattern", False, "C-17 activity in the March ARB pattern is locally expected."),
        case("policy014", "C17", "BOEING C-17 Globemaster III", "United States Air Force", "RCH321", "downtown_los_angeles", True, "C-17 away from a base pattern is alert-worthy for the observer."),
        case("policy015", "T38", "NORTHROP T-38 Talon", "United States Air Force", "TALON68", "edwards_pattern", False, "T-38 traffic in the Edwards pattern is routine training/test traffic."),
        case("policy016", "T38", "NORTHROP T-38 Talon", "United States Air Force", "TALON68", "lax_corridor", True, "T-38 traffic away from Edwards/Plant 42 is locally unusual."),
        case("policy017", "B744", "BOEING 747-400", "Kalitta Air", "CKS369", "lax_corridor", False, "Routine B744 cargo flow is contextual, not automatically rare."),
        case("policy018", "B748", "BOEING 747-8", "Lufthansa", "DLH456", "lax_corridor", False, "Routine B748 airline traffic is contextual without a special livery, notable registration, emergency, or unusual route."),
        case("policy019", "B748", "BOEING 747-8", "Lufthansa", "DLH456", "lax_corridor", True, "D-ABYN is a known special-livery 747-8.", registration="D-ABYN"),
        case("policy020", "BLCF", "BOEING 747-400 Dreamlifter", "Atlas Air", "GTI456", "lax_corridor", True, "Dreamlifter is a very limited specialized cargo aircraft."),
        case("policy021", "MD11", "MCDONNELL DOUGLAS MD-11F", "FedEx", "FDX901", "lax_corridor", True, "MD-11 freighter is a disappearing trijet with spotter interest."),
        case("policy022", "B789", "BOEING 787-9 Dreamliner", "United Airlines", "UAL900", "lax_corridor", False, "Modern 787 hub traffic is interesting but not rare without another signal."),
    ]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/eval/policy_regression_cases.csv"))
    args = parser.parse_args()

    rows = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "examples": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
