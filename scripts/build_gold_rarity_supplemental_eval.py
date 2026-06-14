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
    altitude_ft: int = 5000,
    ground_speed_kt: int = 210,
) -> tuple[dict, dict]:
    loc = location(location_id)
    payload = {
        "provider": "gold_supplemental",
        "collected_at": None,
        "icao_hex": case_id,
        "callsign": callsign,
        "registration": registration or (callsign if callsign.startswith("N") else f"N{case_id[-4:]}RB"),
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
        "distance_nm": 11,
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
    reason: str,
    *,
    registration: str | None = None,
    squawk: str | None = "1200",
    emergency: str = "none",
    altitude_ft: int = 5000,
    ground_speed_kt: int = 210,
    confidence: float = 0.92,
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
            "is_rare": True,
            "confidence": confidence,
            "reason": reason,
        },
        observer_context_for(loc),
    )


def build_cases() -> list[dict[str, str]]:
    rare_type_cases = [
        ("goldrt001", "A124", "ANTONOV AN-124 Ruslan", "Antonov Airlines", "ADB350F", "lax_corridor", "An-124 is a scarce heavy-lift aircraft and alert-worthy when seen locally."),
        ("goldrt002", "A3ST", "AIRBUS A300-600ST Beluga", "Airbus Transport International", "BGA123", "lax_corridor", "Beluga is a specialized oversize transporter with high spotter interest."),
        ("goldrt003", "A343", "AIRBUS A340-300", "Edelweiss Air", "EDW33", "lax_corridor", "A340-300 is a disappearing four-engine passenger type and locally unusual."),
        ("goldrt004", "A346", "AIRBUS A340-600", "Lufthansa", "DLH456", "lax_corridor", "A340-600 is a rare long quadjet with strong spotter interest."),
        ("goldrt005", "B712", "BOEING 717-200", "Delta Air Lines", "DAL717", "lax_corridor", "Boeing 717 is uncommon in Southern California compared with routine narrowbodies."),
        ("goldrt006", "B722", "BOEING 727-200", "private operator", "N727NK", "long_beach_south_bay", "727 is a classic tri-jet and a rare local sighting."),
        ("goldrt007", "B742", "BOEING 747-200", "charter operator", "N742RB", "lax_corridor", "747-200 is a classic jumbo and rare enough to alert even without livery context."),
        ("goldrt008", "C5", "LOCKHEED C-5M Super Galaxy", "United States Air Force", "SAMSON21", "lax_corridor", "C-5M is an uncommon strategic airlifter away from a local base pattern."),
        ("goldrt009", "DC10", "MCDONNELL DOUGLAS DC-10", "cargo operator", "N10RB", "lax_corridor", "DC-10 is a disappearing classic tri-jet and locally alert-worthy."),
        ("goldrt010", "DC8", "DOUGLAS DC-8", "cargo operator", "N805RB", "long_beach_south_bay", "DC-8 is a classic jet rarely seen in current local traffic."),
        ("goldrt011", "MD11", "MCDONNELL DOUGLAS MD-11F", "FedEx", "FDX901", "lax_corridor", "MD-11 freighter is a disappearing tri-jet with clear spotter interest."),
        ("goldrt012", "L101", "LOCKHEED L-1011 TriStar", "museum ferry", "N101RB", "lax_corridor", "L-1011 is a classic tri-jet and an exceptional local sighting."),
        ("goldrt013", "IL76", "ILYUSHIN IL-76", "Silk Way Airlines", "AZQ412", "lax_corridor", "Il-76 is an uncommon Soviet-designed heavy transport in Southern California."),
        ("goldrt014", "A306", "AIRBUS A300-600", "DHL", "DHL306", "lax_corridor", "A300-600 is an older widebody freighter now uncommon compared with routine cargo traffic."),
        ("goldrt015", "CONC", "AEROSPATIALE BAC Concorde", "museum ferry", "N101CV", "lax_corridor", "Concorde is an exceptional historic aircraft sighting."),
        ("goldrt016", "F100", "FOKKER 100", "charter operator", "N100FK", "lax_corridor", "Fokker 100 is an uncommon regional jet in current local traffic."),
        ("goldrt017", "F28", "FOKKER F28 Fellowship", "charter operator", "N280FK", "lax_corridor", "Fokker F28 is a rare classic regional jet sighting."),
        ("goldrt018", "B461", "BRITISH AEROSPACE BAe 146-100", "charter operator", "N146BA", "long_beach_south_bay", "BAe 146 is an uncommon four-engine regional jet."),
        ("goldrt019", "B462", "BRITISH AEROSPACE BAe 146-200", "charter operator", "N462BA", "long_beach_south_bay", "BAe 146-200 is a rare local type compared with routine regional traffic."),
        ("goldrt020", "AN12", "ANTONOV AN-12", "cargo operator", "N12AN", "lax_corridor", "An-12 is an uncommon turboprop cargo aircraft with strong spotter interest."),
        ("goldrt021", "AN22", "ANTONOV AN-22 Antei", "cargo operator", "UR09307", "lax_corridor", "An-22 is an extremely rare heavy turboprop transport."),
        ("goldrt022", "YK42", "YAKOVLEV Yak-42", "charter operator", "RA4242", "lax_corridor", "Yak-42 is a rare Soviet-era tri-jet for Southern California."),
    ]
    emergency_cases = [
        ("goldem001", "A320", "AIRBUS A320", "United Airlines", "UAL7700", "lax_corridor", "Emergency squawk 7700 makes otherwise routine A320 traffic alert-worthy.", "7700", "general"),
        ("goldem002", "B738", "BOEING 737-800", "Southwest Airlines", "SWA7600", "central_orange_county", "Radio-failure squawk 7600 is an operational exception worth alerting.", "7600", "radio"),
        ("goldem003", "C172", "CESSNA 172 Skyhawk", "local flight school", "N172EM", "central_orange_county", "Emergency status makes common trainer traffic alert-worthy.", "7700", "general"),
        ("goldem004", "B789", "BOEING 787-9 Dreamliner", "United Airlines", "UAL7500", "lax_corridor", "Security squawk 7500 is a hard alert regardless of aircraft type.", "7500", "security"),
    ]
    rows = [
        case(case_id, type_designator, description, operator, callsign, location_id, reason)
        for case_id, type_designator, description, operator, callsign, location_id, reason in rare_type_cases
    ]
    rows.extend(
        case(
            case_id,
            type_designator,
            description,
            operator,
            callsign,
            location_id,
            reason,
            squawk=squawk,
            emergency=emergency,
            confidence=0.96,
        )
        for case_id, type_designator, description, operator, callsign, location_id, reason, squawk, emergency in emergency_cases
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/eval/gold_rarity_supplemental_cases.csv"))
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
