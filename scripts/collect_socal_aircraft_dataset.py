#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_LAT = 33.7175
DEFAULT_LON = -117.8311
DEFAULT_DIST_NM = 90
DEFAULT_OUT_DIR = Path("data/datasets/socal-aircraft")

ORANGE_COUNTY_OBSERVER_CONTEXT = {
    "region": "orange_county_los_angeles_southern_california",
    "observer_lat": DEFAULT_LAT,
    "observer_lon": DEFAULT_LON,
    "home_airports": [
        {"code": "SNA", "name": "John Wayne Airport", "kind": "civilian", "distance_nm": 5},
        {"code": "LGB", "name": "Long Beach Airport", "kind": "civilian", "distance_nm": 18},
        {"code": "LAX", "name": "Los Angeles International Airport", "kind": "civilian", "distance_nm": 35},
        {"code": "ONT", "name": "Ontario International Airport", "kind": "civilian", "distance_nm": 28},
        {"code": "BUR", "name": "Hollywood Burbank Airport", "kind": "civilian", "distance_nm": 46},
        {"code": "VNY", "name": "Van Nuys Airport", "kind": "civilian", "distance_nm": 52},
    ],
    "nearby_military_areas": [
        {"name": "Joint Forces Training Base Los Alamitos", "distance_nm": 18, "routine_types": ["H60", "H47", "C130"]},
        {"name": "Camp Pendleton / MCAS Camp Pendleton", "distance_nm": 36, "routine_types": ["H60", "V22", "H53", "C130"]},
        {"name": "March Air Reserve Base", "distance_nm": 38, "routine_types": ["C17", "KC135", "KC46", "C130"]},
        {"name": "MCAS Miramar", "distance_nm": 63, "routine_types": ["F18", "F35", "V22", "H53", "C130"]},
        {"name": "NAS Point Mugu", "distance_nm": 82, "routine_types": ["E2", "C130", "P3", "P8"]},
        {"name": "Plant 42 / Palmdale flight test corridor", "distance_nm": 68, "routine_types": ["T38", "F16", "F35", "C130"]},
        {"name": "Edwards Air Force Base", "distance_nm": 92, "routine_types": ["T38", "F16", "F35", "C17", "KC135", "B52"]},
    ],
    "local_rarity_guidance": [
        "Common SNA/LAX/LGB/BUR/ONT airline traffic such as A320, A321, 737, E175, and routine business jets should usually be not rare.",
        "Military aircraft are less locally rare when close to Los Alamitos, Camp Pendleton, March ARB, Miramar, or Point Mugu.",
        "Flight-test and military aircraft are less locally rare near Palmdale, Plant 42, or Edwards corridors.",
        "The same military aircraft can be locally notable over central Orange County, coastal Los Angeles, downtown Los Angeles, or dense civilian-airport corridors away from a base pattern.",
        "Ultra-rare aircraft, special liveries, classic types, and specialized cargo remain alert-worthy even in Southern California.",
    ],
}

RARE_TYPE_DESIGNATORS = {
    "A10",
    "A124",
    "A225",
    "A306",
    "A3ST",
    "A337",
    "A342",
    "A343",
    "A345",
    "A346",
    "B1",
    "B2",
    "B703",
    "B712",
    "B721",
    "B722",
    "B741",
    "B742",
    "B743",
    "BLCF",
    "B52",
    "C5",
    "C17",
    "C30J",
    "C130",
    "CONI",
    "CVLT",
    "CVLP",
    "DC10",
    "DC3",
    "DC6",
    "DC8",
    "DC85",
    "DC86",
    "DC87",
    "DC91",
    "DC93",
    "DC95",
    "E2",
    "E3",
    "E6",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "H47",
    "H53",
    "H60",
    "IL18",
    "IL62",
    "IL76",
    "L101",
    "KC10",
    "KC135",
    "KC46",
    "MD11",
    "MD80",
    "MD81",
    "MD82",
    "MD83",
    "MD87",
    "MD88",
    "MD90",
    "P3",
    "P8",
    "T38",
    "U2",
    "V22",
}

CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS = {
    "H47",
    "H53",
    "H60",
}

LAST_OF_TYPE_DESIGNATORS = {
    "B753",
}

CONTEXTUAL_TYPE_DESIGNATORS = {
    "B744",
    "B748",
}

SPECIAL_REGISTRATIONS = {
    "A7-BEG",
    "B-LRJ",
    "D-ABYN",
}

RARE_CALLSIGN_PREFIXES = {
    "AF",
    "ASY",
    "BOLT",
    "DEATH",
    "DOOM",
    "EVAC",
    "FORGE",
    "GRIM",
    "GUARD",
    "JOSA",
    "NACHO",
    "NASA",
    "NAVY",
    "PATON",
    "RCH",
    "REACH",
    "RESCUE",
    "SHADY",
    "SPUR",
    "TITAN",
    "VENUS",
}

RARE_ANY_TEXT_PATTERNS = [
    r"\bAIR FORCE\b",
    r"\bUSAF\b",
    r"\bU S AIR FORCE\b",
    r"\bNAVY\b",
    r"\bMARINES?\b",
    r"\bARMY\b",
    r"\bCOAST GUARD\b",
    r"\bNASA\b",
    r"\bBORDER\b",
    r"\bCUSTOMS\b",
    r"\bHOMELAND\b",
    r"\bVINTAGE\b",
    r"\bWARBIRD\b",
    r"\bPRESIDENT\b",
    r"\bVIP\b",
]

RARE_OPERATOR_PATTERNS = [
    r"\bAIR KORYO\b",
    r"\bLOCKHEED\b",
    r"\bNORTHROP\b",
    r"\bBOEING\b",
    r"\bRAYTHEON\b",
    r"\bGENERAL ATOMICS\b",
    r"\bSCLED\b",
    r"\bORBITAL\b",
    r"\bORBIS\b",
]

RARE_DESCRIPTION_PATTERNS = [
    r"\bANTONOV\b",
    r"\bAN-?124\b",
    r"\bAN-?225\b",
    r"\bBELUGA\b",
    r"\bDREAMLIFTER\b",
    r"\bDOOMSDAY\b",
    r"\bTRIJET\b",
    r"\bTRI-?JET\b",
    r"\bDC-?10\b",
    r"\bDC-?8\b",
    r"\bMD-?11\b",
    r"\bIL-?62\b",
    r"\bCONVAIR\b",
    r"\bP-?51\b",
    r"\bT-?6\b",
    r"\bMUSTANG\b",
    r"\bCORSAIR\b",
    r"\bSPITFIRE\b",
    r"\bTEXAN\b",
]

COMMON_AIRLINE_PREFIXES = {
    "AAL",
    "ASA",
    "DAL",
    "FFT",
    "JBU",
    "SKW",
    "SWA",
    "UAL",
    "UPS",
    "FDX",
}

COMMON_GA_TRAINER_TYPE_DESIGNATORS = {
    "BE33",
    "BE36",
    "C150",
    "C152",
    "C172",
    "C182",
    "PA28",
    "P28A",
    "P28R",
    "SR20",
    "SR22",
}

MILITARY_OR_SPECIAL_TYPE_DESIGNATORS = {
    "A10",
    "B1",
    "B2",
    "B52",
    "C5",
    "C17",
    "C30J",
    "C130",
    "E2",
    "E3",
    "E6",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "H47",
    "H53",
    "H60",
    "KC10",
    "KC135",
    "KC46",
    "P3",
    "P8",
    "T38",
    "U2",
    "V22",
}


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "rare-bird-dataset/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def adsbfi_url(lat: float, lon: float, dist_nm: int) -> str:
    return f"https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{dist_nm}"


def adsblol_url(lat: float, lon: float, dist_nm: int) -> str:
    return f"https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{dist_nm}"


def normalize_adsb_aircraft(row: dict[str, Any], collected_at: int) -> dict[str, Any]:
    callsign = clean_str(row.get("flight"))
    altitude = row.get("alt_baro")
    if altitude == "ground":
        altitude = 0
    return {
        "provider": "adsbfi",
        "collected_at": collected_at,
        "icao_hex": clean_str(row.get("hex")),
        "callsign": callsign,
        "registration": clean_str(row.get("r")),
        "type_designator": clean_str(row.get("t")),
        "description": clean_str(row.get("desc")),
        "operator": clean_str(row.get("ownOp")),
        "origin_country": None,
        "lat": as_float(row.get("lat")),
        "lon": as_float(row.get("lon")),
        "altitude_ft": as_float(altitude),
        "ground_speed_kt": as_float(row.get("gs")),
        "heading_deg": as_float(row.get("track")),
        "vertical_rate_fpm": as_float(row.get("baro_rate")),
        "squawk": clean_str(row.get("squawk")),
        "emergency": clean_str(row.get("emergency")),
        "distance_nm": as_float(row.get("dst")),
        "category": clean_str(row.get("category")),
        "seen_seconds": as_float(row.get("seen")),
    }


def clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def callsign_prefix(callsign: str | None) -> str | None:
    if not callsign:
        return None
    match = re.match(r"([A-Z]+)", callsign.upper())
    return match.group(1) if match else None


def weak_label(aircraft: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    type_designator = (aircraft.get("type_designator") or "").upper()
    callsign = (aircraft.get("callsign") or "").upper()
    prefix = callsign_prefix(callsign)
    squawk = aircraft.get("squawk")
    registration = (aircraft.get("registration") or "").upper()
    any_text = " ".join(
        value.upper()
        for value in [
            aircraft.get("operator"),
            aircraft.get("callsign"),
            aircraft.get("registration"),
        ]
        if value
    )
    operator_text = (aircraft.get("operator") or "").upper()
    description_text = (aircraft.get("description") or "").upper()

    if type_designator in RARE_TYPE_DESIGNATORS and type_designator not in CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS:
        reasons.append(f"{type_designator} is on the rare aircraft type list")
    if type_designator in LAST_OF_TYPE_DESIGNATORS:
        reasons.append(f"{type_designator} is a last-of-type or disappearing commercial variant")
    if registration in SPECIAL_REGISTRATIONS:
        reasons.append(f"{registration} is a known special livery or notable individual aircraft")
    if prefix in RARE_CALLSIGN_PREFIXES:
        reasons.append(f"{prefix} callsign prefix is commonly special-use or military")
    if squawk in {"7500", "7600", "7700"}:
        reasons.append(f"{squawk} is an emergency squawk")
    for pattern in RARE_ANY_TEXT_PATTERNS:
        if re.search(pattern, any_text):
            reasons.append(f"matched text signal {pattern.replace(chr(92) + 'b', '')}")
    for pattern in RARE_OPERATOR_PATTERNS:
        if re.search(pattern, operator_text):
            reasons.append(f"matched operator signal {pattern.replace(chr(92) + 'b', '')}")
    for pattern in RARE_DESCRIPTION_PATTERNS:
        if re.search(pattern, description_text):
            reasons.append(f"matched description signal {pattern.replace(chr(92) + 'b', '')}")

    is_common_airline = prefix in COMMON_AIRLINE_PREFIXES
    is_rare = bool(reasons)
    if is_rare:
        confidence = 0.9 if len(reasons) > 1 else 0.78
    elif is_common_airline:
        confidence = 0.88
    else:
        confidence = 0.62

    reason = "; ".join(reasons)
    if not reason:
        if type_designator in CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS:
            reason = f"{type_designator} helicopter is contextual; no military, emergency, special-mission, callsign, operator, or registration signal found"
        elif is_common_airline:
            reason = "ordinary airline or cargo traffic with no rare-aircraft signals"
        else:
            reason = "no rare-aircraft type, callsign, operator, squawk, or description signal found"

    return {
        "is_rare": is_rare,
        "confidence": confidence,
        "reason": reason,
        "label_source": "weak_rules_v1",
    }


def local_frequency_context(aircraft: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    type_designator = (aircraft.get("type_designator") or "").upper()
    callsign = aircraft.get("callsign")
    prefix = callsign_prefix(callsign)
    military_type = type_designator in MILITARY_OR_SPECIAL_TYPE_DESIGNATORS
    base_pattern = context.get("military_pattern") == "base_pattern"
    common_airline = prefix in COMMON_AIRLINE_PREFIXES or type_designator in {"A320", "A321", "A20N", "A21N", "B737", "B738", "B739", "B38M", "B39M", "E75L"}
    common_ga_trainer = type_designator in COMMON_GA_TRAINER_TYPE_DESIGNATORS

    if military_type and base_pattern:
        context_label = {
            "class": "routine_near_military_base_or_test_range",
            "alert_policy": "suppress_alert_unless_special_livery_emergency_or_unique_airframe",
        }
        if type_designator == "T38":
            context_label["rationale"] = "T-38 Talon traffic in the Edwards or Palmdale/Plant 42 pattern is routine local training/test traffic, not an alert."
        return context_label
    if military_type:
        return {
            "class": "uncommon_military_away_from_base_pattern",
            "alert_policy": "alert",
        }
    if common_airline:
        return {
            "class": "common_local_civil_traffic",
            "alert_policy": "suppress_alert_unless_special_livery_emergency_or_unique_airframe",
        }
    if common_ga_trainer:
        return {
            "class": "common_local_ga_training_or_private_traffic",
            "alert_policy": "suppress_alert_unless_special_mission_emergency_vintage_or_watchlist",
        }
    return {
        "class": "unknown_or_contextual",
        "alert_policy": "use_aircraft_rarity_signals",
    }


def compact_observer_context(aircraft: dict[str, Any], observer_context: dict[str, Any] | None) -> dict[str, Any]:
    context = observer_context or ORANGE_COUNTY_OBSERVER_CONTEXT
    return {
        "region": context.get("region", ORANGE_COUNTY_OBSERVER_CONTEXT["region"]),
        "observer_lat": context.get("observer_lat", ORANGE_COUNTY_OBSERVER_CONTEXT["observer_lat"]),
        "observer_lon": context.get("observer_lon", ORANGE_COUNTY_OBSERVER_CONTEXT["observer_lon"]),
        "current_local_area": context.get("current_local_area"),
        "nearest_airport": context.get("nearest_airport"),
        "nearest_military_area": context.get("nearest_military_area"),
        "distance_to_nearest_military_nm": context.get("distance_to_nearest_military_nm"),
        "military_pattern": context.get("military_pattern"),
        "local_frequency_context": local_frequency_context(aircraft, context),
        "local_rules": [
            "routine A320/737/E175/business-jet traffic near SNA/LAX/LGB/BUR/ONT is not rare without another signal",
            "military aircraft near its base or flight-test pattern can be locally routine",
            "the same military aircraft away from a base pattern can be locally alert-worthy",
            "ultra-rare/special-livery/classic/specialized-cargo aircraft remain alert-worthy",
        ],
    }


def make_prompt(aircraft: dict[str, Any], observer_context: dict[str, Any] | None = None) -> str:
    payload = {
        "task": "Return exactly one JSON object with keys is_rare, confidence, reason. Do not include markdown, commentary, metadata, or extra keys. Classify whether this Southern California aircraft sighting is rare enough to alert an aviation-curious iPhone user.",
        "aircraft": aircraft,
        "observer_context": compact_observer_context(aircraft, observer_context),
        "reference": {
            "decision_policy": [
                "First apply observer_context.local_frequency_context.alert_policy.",
            "If alert_policy is suppress_alert_unless_special_livery_emergency_or_unique_airframe, return is_rare=false unless the aircraft has an emergency squawk, known special registration/livery, or ultra-rare unique airframe signal.",
            "If alert_policy is suppress_alert_unless_special_mission_emergency_vintage_or_watchlist, return is_rare=false unless the aircraft has emergency, explicit rescue/evacuation or special-incident evidence, unusual aircraft type for the mission, vintage/warbird, or explicit watchlist evidence.",
            "Treat B744 / Boeing 747-400 cargo traffic, including routine Kalitta-style flow, as contextual: not automatically rare without special livery, notable registration, emergency, unusual operator, or clearly unusual route.",
            "Treat B748 / Boeing 747-8 as contextual: not automatically rare for routine airline or cargo traffic, but alert-worthy for special livery, unusual operator, notable individual airframe, emergency, or clearly unusual route.",
            "If alert_policy is alert, return is_rare=true for military or special-mission aircraft away from a base pattern.",
            "Use the broader rarity factors only after applying the local frequency policy.",
        ],
            "rarity_factors": [
                "low production count or few remaining active examples",
                "near end-of-life or disappearing type",
                "limited operator, government, VIP, military, test, research, or special mission use",
                "unique shape, role, livery, or notable individual airframe",
                "geographic infrequency for Orange County, Los Angeles County, and Southern California",
                "historical or enthusiast value",
                "ordinary local A320 and 737 family airline traffic near SNA, LAX, LGB, BUR, ONT, or SAN is not rare without another signal",
                "ordinary Boeing 747-400 / B744 cargo traffic, including routine Kalitta-style flow, is contextual rather than automatically rare; require special livery, notable registration, emergency, unusual operator, or unusual route evidence",
                "ordinary Boeing 747-8 / B748 airline or cargo traffic is contextual rather than automatically rare; require special livery, unusual operator, notable airframe, emergency, or unusual route evidence",
                "ordinary C150, C152, C172, C182, PA-28/P28A/P28R, SR20, SR22, BE33, and BE36 training or private GA traffic is common in Southern California and is not rare without emergency, special mission, vintage/warbird, or watchlist evidence",
                "military traffic near a military base can be locally routine even if globally noteworthy",
            ],
            "rare_type_designators": sorted(RARE_TYPE_DESIGNATORS),
            "contextual_type_designators": sorted(CONTEXTUAL_TYPE_DESIGNATORS),
            "last_of_type_designators": sorted(LAST_OF_TYPE_DESIGNATORS),
            "common_ga_trainer_type_designators": sorted(COMMON_GA_TRAINER_TYPE_DESIGNATORS),
            "rare_callsign_prefixes": sorted(RARE_CALLSIGN_PREFIXES),
            "special_registrations": sorted(SPECIAL_REGISTRATIONS),
            "emergency_squawks": ["7500", "7600", "7700"],
        },
        "output_schema": {
            "is_rare": "boolean",
            "confidence": "number from 0 to 1",
            "reason": "short human-readable explanation",
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["adsbfi", "adsblol"], default="adsbfi")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--dist-nm", type=int, default=DEFAULT_DIST_NM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-file", type=Path, help="Rebuild normalized/train files from an existing raw snapshot.")
    args = parser.parse_args()

    collected_at = int(time.time())
    url = adsbfi_url(args.lat, args.lon, args.dist_nm) if args.provider == "adsbfi" else adsblol_url(args.lat, args.lon, args.dist_nm)
    if args.raw_file:
        raw = json.loads(args.raw_file.read_text(encoding="utf-8"))
        snapshot_name = args.raw_file.name
    else:
        raw = fetch_json(url)
        snapshot_name = f"{args.provider}-{collected_at}.json"
    aircraft_rows = raw.get("aircraft") or raw.get("ac") or []
    if not isinstance(aircraft_rows, list):
        raise ValueError("API response did not include an aircraft list")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    normalized_dir = args.out_dir / "normalized"
    raw_dir.mkdir(exist_ok=True)
    normalized_dir.mkdir(exist_ok=True)

    if not args.raw_file:
        (raw_dir / snapshot_name).write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    normalized_by_hex: dict[str, dict[str, Any]] = {}
    for row in aircraft_rows:
        if not isinstance(row, dict):
            continue
        normalized = normalize_adsb_aircraft(row, collected_at)
        if not normalized["icao_hex"]:
            continue
        normalized_by_hex[normalized["icao_hex"]] = normalized

    normalized_rows = sorted(normalized_by_hex.values(), key=lambda item: (item.get("distance_nm") is None, item.get("distance_nm") or 9999, item["icao_hex"]))
    examples: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    rare_count = 0
    for aircraft in normalized_rows:
        label = weak_label(aircraft)
        rare_count += int(label["is_rare"])
        response = {
            "is_rare": label["is_rare"],
            "confidence": label["confidence"],
            "reason": label["reason"],
        }
        example = {
            "prompt": make_prompt(aircraft),
            "response": json.dumps(response, sort_keys=True, separators=(",", ":")),
            "aircraft": aircraft,
            "label": label,
        }
        examples.append(example)
        csv_rows.append({"prompt": example["prompt"], "response": example["response"]})

    write_jsonl(normalized_dir / snapshot_name.replace(".json", ".jsonl"), normalized_rows)
    write_jsonl(args.out_dir / "train.jsonl", examples)
    with (args.out_dir / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(csv_rows)

    summary = {
        "provider": args.provider,
        "url": url,
        "collected_at": collected_at,
        "center": {"lat": args.lat, "lon": args.lon, "dist_nm": args.dist_nm},
        "aircraft_count": len(normalized_rows),
        "rare_weak_label_count": rare_count,
        "not_rare_weak_label_count": len(normalized_rows) - rare_count,
        "raw_snapshot": str(raw_dir / snapshot_name),
        "normalized_snapshot": str(normalized_dir / snapshot_name.replace(".json", ".jsonl")),
        "train_jsonl": str(args.out_dir / "train.jsonl"),
        "train_csv": str(args.out_dir / "train.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
