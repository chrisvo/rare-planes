#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from collect_socal_aircraft_dataset import ORANGE_COUNTY_OBSERVER_CONTEXT, RARE_TYPE_DESIGNATORS, make_prompt


DEFAULT_LIVE = Path("data/datasets/socal-aircraft/train.jsonl")
DEFAULT_SEED = Path("data/datasets/rarity-seed/examples.jsonl")
DEFAULT_OUT_DIR = Path("data/datasets/rarity-quick-1000")

SOCAL_LOCATIONS = [
    {
        "id": "central_orange_county",
        "lat": 33.7175,
        "lon": -117.8311,
        "local_area": "central Orange County near SNA",
        "nearest_airport": "SNA",
        "nearest_military_area": "Joint Forces Training Base Los Alamitos",
        "distance_to_nearest_military_nm": 18,
        "military_pattern": "not_base_pattern",
    },
    {
        "id": "coastal_orange_county",
        "lat": 33.6189,
        "lon": -117.9298,
        "local_area": "coastal Orange County",
        "nearest_airport": "SNA",
        "nearest_military_area": "Camp Pendleton / MCAS Camp Pendleton",
        "distance_to_nearest_military_nm": 34,
        "military_pattern": "not_base_pattern",
    },
    {
        "id": "los_alamitos_pattern",
        "lat": 33.7900,
        "lon": -118.0510,
        "local_area": "Los Alamitos base pattern",
        "nearest_airport": "SLI",
        "nearest_military_area": "Joint Forces Training Base Los Alamitos",
        "distance_to_nearest_military_nm": 2,
        "military_pattern": "base_pattern",
    },
    {
        "id": "camp_pendleton_pattern",
        "lat": 33.3013,
        "lon": -117.3559,
        "local_area": "Camp Pendleton / north San Diego County base pattern",
        "nearest_airport": "NFG",
        "nearest_military_area": "Camp Pendleton / MCAS Camp Pendleton",
        "distance_to_nearest_military_nm": 4,
        "military_pattern": "base_pattern",
    },
    {
        "id": "march_arb_pattern",
        "lat": 33.8807,
        "lon": -117.2590,
        "local_area": "March ARB / Inland Empire base pattern",
        "nearest_airport": "RIV",
        "nearest_military_area": "March Air Reserve Base",
        "distance_to_nearest_military_nm": 3,
        "military_pattern": "base_pattern",
    },
    {
        "id": "lax_corridor",
        "lat": 33.9425,
        "lon": -118.4081,
        "local_area": "LAX corridor",
        "nearest_airport": "LAX",
        "nearest_military_area": "Joint Forces Training Base Los Alamitos",
        "distance_to_nearest_military_nm": 20,
        "military_pattern": "not_base_pattern",
    },
    {
        "id": "downtown_los_angeles",
        "lat": 34.0522,
        "lon": -118.2437,
        "local_area": "downtown Los Angeles",
        "nearest_airport": "LAX",
        "nearest_military_area": "Joint Forces Training Base Los Alamitos",
        "distance_to_nearest_military_nm": 22,
        "military_pattern": "not_base_pattern",
    },
    {
        "id": "san_fernando_valley",
        "lat": 34.1899,
        "lon": -118.4514,
        "local_area": "San Fernando Valley near VNY/BUR",
        "nearest_airport": "VNY",
        "nearest_military_area": "Plant 42 / Palmdale flight test corridor",
        "distance_to_nearest_military_nm": 30,
        "military_pattern": "not_base_pattern",
    },
    {
        "id": "long_beach_south_bay",
        "lat": 33.7701,
        "lon": -118.1937,
        "local_area": "Long Beach and South Bay",
        "nearest_airport": "LGB",
        "nearest_military_area": "Joint Forces Training Base Los Alamitos",
        "distance_to_nearest_military_nm": 8,
        "military_pattern": "not_base_pattern",
    },
    {
        "id": "palmdale_plant_42_pattern",
        "lat": 34.6294,
        "lon": -118.0846,
        "local_area": "Palmdale / Plant 42 flight test pattern",
        "nearest_airport": "PMD",
        "nearest_military_area": "Plant 42 / Palmdale flight test corridor",
        "distance_to_nearest_military_nm": 2,
        "military_pattern": "base_pattern",
    },
    {
        "id": "edwards_pattern",
        "lat": 34.9054,
        "lon": -117.8837,
        "local_area": "Edwards Air Force Base pattern",
        "nearest_airport": "EDW",
        "nearest_military_area": "Edwards Air Force Base",
        "distance_to_nearest_military_nm": 3,
        "military_pattern": "base_pattern",
    },
]

RARE_TEMPLATES = [
    ("A337", "AIRBUS Beluga XL", "Airbus Transport International", "low-production, visually unique Beluga aircraft"),
    ("A3ST", "AIRBUS A300-600ST Beluga", "Airbus Transport International", "low-production, visually unique Beluga aircraft"),
    ("BLCF", "BOEING 747-400 Dreamlifter", "Atlas Air", "very limited modified 747 Dreamlifter fleet"),
    ("A124", "ANTONOV An-124 Ruslan", "Antonov Airlines", "massive specialized cargo aircraft and infrequent visitor"),
    ("IL62", "ILYUSHIN Il-62", "Air Koryo", "rare Soviet-era classic with limited operator access"),
    ("IL76", "ILYUSHIN Il-76", "Volga-Dnepr", "uncommon Soviet-designed heavy transport"),
    ("MD11", "MCDONNELL DOUGLAS MD-11F", "FedEx", "disappearing trijet freighter"),
    ("DC10", "MCDONNELL DOUGLAS DC-10", "Orbis", "near end-of-life classic widebody"),
    ("DC8", "DOUGLAS DC-8", "Samaritan's Purse", "historic classic jet with few active examples"),
    ("B753", "BOEING 757-300", "United Airlines", "last-of-type disappearing 757-300 variant"),
    ("C17", "BOEING C-17 Globemaster III", "United States Air Force", "military heavy transport"),
    ("C5", "LOCKHEED C-5M Super Galaxy", "United States Air Force", "very large military transport"),
    ("E3", "BOEING E-3 Sentry", "United States Air Force", "special mission AWACS aircraft"),
    ("E6", "BOEING E-6 Mercury", "United States Navy", "special mission command aircraft"),
    ("V22", "BELL BOEING V-22 Osprey", "United States Marine Corps", "unusual tiltrotor military aircraft"),
    ("H60", "SIKORSKY UH-60 Black Hawk", "United States Army", "military helicopter with special-mission interest"),
]

SPECIAL_REGISTRATION_TEMPLATES = [
    ("D-ABYN", "B748", "BOEING 747-8", "Lufthansa", "known special-livery 747-8"),
    ("A7-BEG", "B77W", "BOEING 777-300ER", "Qatar Airways", "known special-livery individual airframe"),
    ("B-LRJ", "A359", "AIRBUS A350-900", "Cathay Pacific", "known special-livery individual airframe"),
]

NEGATIVE_TEMPLATES = [
    ("A320", "AIRBUS A320", "United Airlines"),
    ("A321", "AIRBUS A321", "American Airlines"),
    ("A20N", "AIRBUS A320neo", "Delta Air Lines"),
    ("A21N", "AIRBUS A321neo", "JetBlue Airways"),
    ("B737", "BOEING 737-700", "Southwest Airlines"),
    ("B738", "BOEING 737-800", "American Airlines"),
    ("B739", "BOEING 737-900", "United Airlines"),
    ("B38M", "BOEING 737 MAX 8", "Southwest Airlines"),
    ("B39M", "BOEING 737 MAX 9", "Alaska Airlines"),
    ("E75L", "EMBRAER ERJ-175", "SkyWest Airlines"),
    ("C172", "CESSNA 172", "flight school"),
    ("P28A", "PIPER PA-28 Cherokee", "flight school"),
    ("B789", "BOEING 787-9 Dreamliner", "United Airlines"),
    ("A359", "AIRBUS A350-900", "Delta Air Lines"),
    ("B744", "BOEING 747-400", "Kalitta Air"),
    ("B748", "BOEING 747-8", "Lufthansa"),
]

COMMON_GA_TRAINER_TEMPLATES = [
    ("C150", "CESSNA 150", "local flight school"),
    ("C152", "CESSNA 152", "local flight school"),
    ("C172", "CESSNA 172 Skyhawk", "local flight school"),
    ("C182", "CESSNA 182 Skylane", "private owner"),
    ("PA28", "PIPER PA-28 Cherokee", "local flight school"),
    ("P28A", "PIPER PA-28 Cherokee", "local flight school"),
    ("P28R", "PIPER PA-28R Arrow", "private owner"),
    ("SR20", "CIRRUS SR20", "private owner"),
    ("SR22", "CIRRUS SR22", "private owner"),
    ("BE33", "BEECH 33 Bonanza", "private owner"),
    ("BE36", "BEECH 36 Bonanza", "private owner"),
]

COMMON_GA_LOCATIONS = [
    item
    for item in SOCAL_LOCATIONS
    if item["id"] in {
        "central_orange_county",
        "coastal_orange_county",
        "long_beach_south_bay",
        "san_fernando_valley",
        "downtown_los_angeles",
    }
]

PUBLIC_SAFETY_ROUTINE_TEMPLATES = [
    ("C182", "CESSNA 182 Skylane", "California Highway Patrol", "routine patrol aircraft"),
    ("C172", "CESSNA 172 Skyhawk", "County Sheriff", "routine law-enforcement patrol"),
    ("PA28", "PIPER PA-28 Cherokee", "Civil Air Patrol", "routine patrol or training support"),
    ("SR22", "CIRRUS SR22", "CAL FIRE", "routine agency support flight"),
    ("AS50", "AEROSPATIALE AS-350 Ecureuil", "San Bernardino County Sheriff Dept", "routine public-safety helicopter"),
]

PUBLIC_SAFETY_HARD_ALERT_TEMPLATES = [
    ("C172", "CESSNA 172 Skyhawk", "County Sheriff", "RESCUE12", "explicit rescue callsign"),
    ("PA28", "PIPER PA-28 Cherokee", "Civil Air Patrol", "EVAC24", "explicit evacuation callsign"),
    ("SR22", "CIRRUS SR22", "CAL FIRE", "FIRE12", "emergency squawk"),
    ("C182", "CESSNA 182 Skylane", "California Highway Patrol", "CHP51", "watchlist registration"),
]

GA_VINTAGE_EXCEPTIONS = [
    ("C150", "CESSNA 150", "private owner", "vintage restoration watchlist aircraft"),
    ("C172", "CESSNA 172 Skyhawk", "private owner", "explicit watchlist registration"),
]

MILITARY_LOCAL_CONTRAST_TEMPLATES = [
    ("H60", "SIKORSKY UH-60 Black Hawk", "United States Army", "military helicopter"),
    ("H47", "BOEING CH-47 Chinook", "United States Army", "heavy-lift military helicopter"),
    ("H53", "SIKORSKY CH-53E Super Stallion", "United States Marine Corps", "heavy military helicopter"),
    ("V22", "BELL BOEING V-22 Osprey", "United States Marine Corps", "tiltrotor military aircraft"),
    ("C130", "LOCKHEED C-130 Hercules", "United States Air Force", "military transport"),
    ("C17", "BOEING C-17 Globemaster III", "United States Air Force", "military heavy transport"),
    ("KC135", "BOEING KC-135 Stratotanker", "United States Air Force", "military tanker"),
    ("T38", "NORTHROP T-38 Talon", "United States Air Force", "military trainer"),
]

MILITARY_CALLSIGN_PREFIXES = ["KNIFE", "SHWK", "VVNH", "RCH", "REACH", "NAVY", "SPUR", "TITAN", "PATON", "GUARD"]

T38_HARD_BASE_LOCATIONS = [
    item for item in SOCAL_LOCATIONS if item["id"] in {"edwards_pattern", "palmdale_plant_42_pattern"}
]

T38_HARD_AWAY_LOCATIONS = [
    item
    for item in SOCAL_LOCATIONS
    if item["id"] in {"downtown_los_angeles", "san_fernando_valley", "lax_corridor", "central_orange_county"}
]

PUBLIC_SAFETY_TEXT = (
    "SHERIFF",
    "POLICE",
    "PATROL",
    "HIGHWAY PATROL",
    "CAL FIRE",
    "FIRE",
    "CIVIL AIR PATROL",
    "COUNTY OF LOS ANGELES",
    "SAN BERNARDINO COUNTY",
    "RIVERSIDE COUNTY",
)

INCIDENT_TEXT = (
    "RESCUE",
    "EVAC",
    "MEDEVAC",
    "LIFEGUARD",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def observer_context_for(location: dict) -> dict:
    context = json.loads(json.dumps(ORANGE_COUNTY_OBSERVER_CONTEXT))
    context.update(
        {
            "current_local_area": location["local_area"],
            "nearest_airport": location["nearest_airport"],
            "nearest_military_area": location["nearest_military_area"],
            "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
            "military_pattern": location["military_pattern"],
        }
    )
    return context


def compact_reason(reason: str, limit: int = 140) -> str:
    reason = " ".join(reason.split())
    if len(reason) <= limit:
        return reason
    truncated = reason[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return truncated or reason[:limit].rstrip(" ,;:-")


def training_row(aircraft: dict, label: dict, observer_context: dict | None = None) -> dict[str, str]:
    response = {
        "is_rare": label["is_rare"],
        "confidence": label["confidence"],
        "reason": compact_reason(label["reason"]),
    }
    return {
        "prompt": make_prompt(aircraft, observer_context=observer_context),
        "response": json.dumps(response, separators=(",", ":")),
    }


def has_text(aircraft: dict, terms: tuple[str, ...]) -> bool:
    text = " ".join(
        str(aircraft.get(key) or "").upper()
        for key in ("callsign", "registration", "description", "operator")
    )
    return any(term in text for term in terms)


def normalize_existing_label(aircraft: dict, label: dict) -> dict:
    type_designator = (aircraft.get("type_designator") or "").upper()
    registration = (aircraft.get("registration") or "").upper()
    emergency = (aircraft.get("emergency") or "").lower()
    squawk = str(aircraft.get("squawk") or "")
    description = aircraft.get("description") or type_designator or "Aircraft"

    hard_alert = (
        registration in {"D-ABYN", "A7-BEG", "B-LRJ"}
        or emergency not in {"", "none", "null"}
        or squawk in {"7500", "7600", "7700"}
        or has_text(aircraft, INCIDENT_TEXT)
        or has_text(aircraft, ("VINTAGE", "WARBIRD", "WATCHLIST", "SPECIAL LIVERY"))
    )
    if (
        type_designator in {"B744", "B748"}
        and registration not in {"D-ABYN", "A7-BEG", "B-LRJ"}
        and not hard_alert
    ):
        return {
            "is_rare": False,
            "confidence": 0.86,
            "reason": f"Ordinary {type_designator} airline or cargo traffic is contextual in Southern California and is not rare without special livery, notable registration, emergency, unusual operator, or unusual route evidence.",
        }
    if has_text(aircraft, PUBLIC_SAFETY_TEXT) and type_designator not in RARE_TYPE_DESIGNATORS and not hard_alert:
        return {
            "is_rare": False,
            "confidence": 0.86,
            "reason": f"{description} public-safety or agency traffic is recognizable, but not rare without emergency, rescue/evacuation, notable registration, unusual type, or watchlist evidence.",
        }
    return label


def from_existing(example: dict) -> dict[str, str]:
    if "prompt" in example and "response" in example:
        prompt = json.loads(example["prompt"])
        aircraft = prompt.get("aircraft") or {}
        observer_context = prompt.get("observer_context")
        label = json.loads(example["response"])
        label = normalize_existing_label(aircraft, label)
        return training_row(aircraft, label, observer_context=observer_context)
    return training_row(example["aircraft"], normalize_existing_label(example["aircraft"], example["label"]))


def synthetic_aircraft(index: int, type_designator: str, description: str, operator: str, rare: bool) -> dict:
    rare_prefixes = ["RCH", "REACH", "NASA", "NAVY", "SPUR", "TITAN"]
    common_prefixes = ["SWA", "UAL", "AAL", "DAL", "SKW", "ASA"]
    prefix = random.choice(rare_prefixes if rare else common_prefixes)
    location = random.choice(SOCAL_LOCATIONS)
    return {
        "provider": "synthetic",
        "collected_at": None,
        "icao_hex": f"synthetic{index:05d}",
        "callsign": f"{prefix}{100 + index % 900}",
        "registration": f"N{10000 + index % 89999}",
        "type_designator": type_designator,
        "description": description,
        "operator": operator,
        "origin_country": None,
        "lat": round(location["lat"] + random.uniform(-0.08, 0.08), 6),
        "lon": round(location["lon"] + random.uniform(-0.08, 0.08), 6),
        "local_area": location["local_area"],
        "nearest_airport": location["nearest_airport"],
        "nearest_military_area": location["nearest_military_area"],
        "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
        "military_pattern": location["military_pattern"],
        "altitude_ft": random.choice([0, 1200, 3500, 12000, 28000, 35000]),
        "ground_speed_kt": round(random.uniform(70, 510), 1),
        "heading_deg": round(random.uniform(0, 359), 1),
        "vertical_rate_fpm": random.choice([None, -512, 0, 640, 1600]),
        "squawk": random.choice([None, "1200", "4212", "7377"]),
        "emergency": "none",
        "distance_nm": round(random.uniform(2, 140), 3),
        "category": None,
        "seen_seconds": round(random.uniform(0, 5), 1),
    }


def local_context_label(type_designator: str, description: str, factor: str, location: dict) -> dict:
    base_pattern = location["military_pattern"] == "base_pattern"
    if type_designator in {"H60", "H47", "H53", "V22", "C130", "C17", "KC135", "KC46", "T38", "F18", "F35"} and base_pattern:
        return {
            "is_rare": False,
            "confidence": round(random.uniform(0.78, 0.9), 2),
            "reason": f"{description} is globally noteworthy, but this sighting is in the {location['local_area']} near {location['nearest_military_area']}; suppress the alert because it matches routine local military/base-pattern traffic.",
        }
    return {
        "is_rare": True,
        "confidence": round(random.uniform(0.86, 0.97), 2),
        "reason": f"{description} is rare for {location['local_area']} because it is away from a routine base pattern and represents {factor}.",
    }


def military_contrast_rows(start_index: int, repeats: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = start_index
    altitude_options = [900, 1500, 2800, 6500, 14000, 26000, 34000]
    speed_options = [95, 135, 180, 240, 310, 420, 455]
    for _ in range(repeats):
        for type_designator, description, operator, factor in MILITARY_LOCAL_CONTRAST_TEMPLATES:
            for location in SOCAL_LOCATIONS:
                aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=True)
                aircraft["lat"] = round(location["lat"] + random.uniform(-0.055, 0.055), 6)
                aircraft["lon"] = round(location["lon"] + random.uniform(-0.055, 0.055), 6)
                aircraft["local_area"] = location["local_area"]
                aircraft["nearest_airport"] = location["nearest_airport"]
                aircraft["nearest_military_area"] = location["nearest_military_area"]
                aircraft["distance_to_nearest_military_nm"] = location["distance_to_nearest_military_nm"]
                aircraft["military_pattern"] = location["military_pattern"]
                aircraft["callsign"] = f"{random.choice(MILITARY_CALLSIGN_PREFIXES)}{100 + index % 900}"
                aircraft["registration"] = f"{random.choice(['AF', 'ARMY', 'NAVY', 'MC'])}{index % 100000:05d}"
                aircraft["altitude_ft"] = random.choice(altitude_options)
                aircraft["ground_speed_kt"] = random.choice(speed_options)
                aircraft["distance_nm"] = round(random.uniform(1.5, 85), 3)
                label = local_context_label(type_designator, description, factor, location)
                rows.append(training_row(aircraft, label, observer_context_for(location)))
                index += 1
    return rows


def t38_edwards_curriculum_rows(start_index: int, repeats: int = 48) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = start_index
    for _ in range(repeats):
        for location in T38_HARD_BASE_LOCATIONS:
            aircraft = synthetic_aircraft(index, "T38", "NORTHROP T-38 Talon", "United States Air Force", rare=True)
            aircraft.update(
                {
                    "lat": round(location["lat"] + random.uniform(-0.035, 0.035), 6),
                    "lon": round(location["lon"] + random.uniform(-0.035, 0.035), 6),
                    "local_area": location["local_area"],
                    "nearest_airport": location["nearest_airport"],
                    "nearest_military_area": location["nearest_military_area"],
                    "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                    "military_pattern": location["military_pattern"],
                    "callsign": f"{random.choice(['EDWARDS', 'VIPER', 'TEST', 'TALON'])}{10 + index % 80}",
                    "registration": f"AF{index % 100000:05d}",
                    "altitude_ft": random.choice([4500, 9000, 14000, 18000, 22000]),
                    "ground_speed_kt": random.choice([260, 310, 360, 410]),
                    "distance_nm": round(random.uniform(1.5, 8), 3),
                }
            )
            rows.append(training_row(aircraft, {
                "is_rare": False,
                "confidence": round(random.uniform(0.86, 0.94), 2),
                "reason": f"NORTHROP T-38 Talon traffic in the {location['local_area']} near {location['nearest_military_area']} is routine local training/test traffic, so suppress the Southern California alert.",
            }, observer_context_for(location)))
            index += 1

        for location in T38_HARD_AWAY_LOCATIONS:
            aircraft = synthetic_aircraft(index, "T38", "NORTHROP T-38 Talon", "United States Air Force", rare=True)
            aircraft.update(
                {
                    "lat": round(location["lat"] + random.uniform(-0.04, 0.04), 6),
                    "lon": round(location["lon"] + random.uniform(-0.04, 0.04), 6),
                    "local_area": location["local_area"],
                    "nearest_airport": location["nearest_airport"],
                    "nearest_military_area": location["nearest_military_area"],
                    "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                    "military_pattern": location["military_pattern"],
                    "callsign": f"{random.choice(['EDWARDS', 'VIPER', 'TEST', 'TALON'])}{10 + index % 80}",
                    "registration": f"AF{index % 100000:05d}",
                    "altitude_ft": random.choice([7000, 12000, 18000, 24000]),
                    "ground_speed_kt": random.choice([280, 330, 360, 420]),
                    "distance_nm": round(random.uniform(12, 55), 3),
                }
            )
            rows.append(training_row(aircraft, {
                "is_rare": True,
                "confidence": round(random.uniform(0.88, 0.96), 2),
                "reason": f"NORTHROP T-38 Talon is alert-worthy for {location['local_area']} because it is away from an Edwards or Plant 42 base pattern.",
            }, observer_context_for(location)))
            index += 1
    return rows


def ga_hard_negative_rows(start_index: int, repeats: int = 18) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = start_index
    for _ in range(repeats):
        for type_designator, description, operator in COMMON_GA_TRAINER_TEMPLATES:
            location = random.choice(COMMON_GA_LOCATIONS)
            aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=False)
            aircraft.update(
                {
                    "lat": round(location["lat"] + random.uniform(-0.045, 0.045), 6),
                    "lon": round(location["lon"] + random.uniform(-0.045, 0.045), 6),
                    "local_area": location["local_area"],
                    "nearest_airport": location["nearest_airport"],
                    "nearest_military_area": location["nearest_military_area"],
                    "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                    "military_pattern": location["military_pattern"],
                    "callsign": random.choice([f"N{100 + index % 900}{random.choice(['CB', 'LA', 'OC', 'WA'])}", f"TRAIN{10 + index % 80}", f"SKY{10 + index % 80}"]),
                    "registration": f"N{10000 + index % 89999}",
                    "operator": operator,
                    "altitude_ft": random.choice([900, 1200, 2500, 3500, 5500, 7500]),
                    "ground_speed_kt": round(random.uniform(70, 145), 1),
                    "distance_nm": round(random.uniform(1.5, 28), 3),
                    "squawk": random.choice(["1200", "1201", "4371", "4721", None]),
                    "emergency": "none",
                }
            )
            rows.append(training_row(aircraft, {
                "is_rare": False,
                "confidence": round(random.uniform(0.88, 0.96), 2),
                "reason": f"{description} is common local GA training or private traffic around {location['nearest_airport']}; suppress the alert because there is no emergency, special-mission, vintage/warbird, or watchlist evidence.",
            }, observer_context_for(location)))
            index += 1
    return rows


def ga_exception_rows(start_index: int, repeats: int = 8) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = start_index
    for _ in range(repeats):
        for type_designator, description, operator, factor in PUBLIC_SAFETY_ROUTINE_TEMPLATES:
            location = random.choice(COMMON_GA_LOCATIONS)
            aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=False)
            aircraft.update(
                {
                    "lat": round(location["lat"] + random.uniform(-0.045, 0.045), 6),
                    "lon": round(location["lon"] + random.uniform(-0.045, 0.045), 6),
                    "local_area": location["local_area"],
                    "nearest_airport": location["nearest_airport"],
                    "nearest_military_area": location["nearest_military_area"],
                    "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                    "military_pattern": location["military_pattern"],
                    "callsign": random.choice(["SHERIFF1", "CAP452", "FIRE12", "PATROL7", "CHP51"]),
                    "registration": f"N{10000 + index % 89999}",
                    "operator": operator,
                    "altitude_ft": random.choice([900, 1800, 3200, 6500]),
                    "ground_speed_kt": round(random.uniform(75, 155), 1),
                    "distance_nm": round(random.uniform(1.5, 35), 3),
                    "squawk": random.choice(["1200", "4371", None]),
                    "emergency": "none",
                }
            )
            rows.append(training_row(aircraft, {
                "is_rare": False,
                "confidence": round(random.uniform(0.82, 0.92), 2),
                "reason": f"{description} is recognizable {factor}, but it is common enough locally and has no emergency, rescue/evacuation, notable registration, unusual type, or watchlist signal.",
            }, observer_context_for(location)))
            index += 1

        for type_designator, description, operator, callsign, factor in PUBLIC_SAFETY_HARD_ALERT_TEMPLATES:
            location = random.choice(COMMON_GA_LOCATIONS)
            aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=True)
            aircraft.update(
                {
                    "lat": round(location["lat"] + random.uniform(-0.045, 0.045), 6),
                    "lon": round(location["lon"] + random.uniform(-0.045, 0.045), 6),
                    "local_area": location["local_area"],
                    "nearest_airport": location["nearest_airport"],
                    "nearest_military_area": location["nearest_military_area"],
                    "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                    "military_pattern": location["military_pattern"],
                    "callsign": callsign,
                    "registration": "N777RB" if factor == "watchlist registration" else f"N{10000 + index % 89999}",
                    "operator": operator,
                    "description": f"{description} watchlist aircraft" if factor == "watchlist registration" else description,
                    "altitude_ft": random.choice([900, 1800, 3200, 6500]),
                    "ground_speed_kt": round(random.uniform(75, 155), 1),
                    "distance_nm": round(random.uniform(1.5, 35), 3),
                    "squawk": "7700" if factor == "emergency squawk" else random.choice(["1200", "4371", None]),
                    "emergency": "general" if factor == "emergency squawk" else "none",
                }
            )
            rows.append(training_row(aircraft, {
                "is_rare": True,
                "confidence": round(random.uniform(0.86, 0.94), 2),
                "reason": f"{description} is normally common, but this sighting is alert-worthy because it has {factor}.",
            }, observer_context_for(location)))
            index += 1

        location = random.choice(COMMON_GA_LOCATIONS)
        type_designator, description, operator = random.choice(COMMON_GA_TRAINER_TEMPLATES)
        aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=True)
        aircraft.update(
            {
                "lat": round(location["lat"] + random.uniform(-0.045, 0.045), 6),
                "lon": round(location["lon"] + random.uniform(-0.045, 0.045), 6),
                "local_area": location["local_area"],
                "nearest_airport": location["nearest_airport"],
                "nearest_military_area": location["nearest_military_area"],
                "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                "military_pattern": location["military_pattern"],
                "callsign": f"N{10000 + index % 89999}",
                "registration": f"N{10000 + index % 89999}",
                "operator": operator,
                "altitude_ft": random.choice([1200, 2500, 4500]),
                "ground_speed_kt": round(random.uniform(70, 145), 1),
                "distance_nm": round(random.uniform(1.5, 25), 3),
                "squawk": "7700",
                "emergency": "general",
            }
        )
        rows.append(training_row(aircraft, {
            "is_rare": True,
            "confidence": round(random.uniform(0.9, 0.97), 2),
            "reason": f"{description} is normally routine, but emergency squawk 7700 makes this an alert-worthy exception.",
        }, observer_context_for(location)))
        index += 1

        for type_designator, description, operator, factor in GA_VINTAGE_EXCEPTIONS:
            location = random.choice(COMMON_GA_LOCATIONS)
            aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=True)
            aircraft.update(
                {
                    "lat": round(location["lat"] + random.uniform(-0.045, 0.045), 6),
                    "lon": round(location["lon"] + random.uniform(-0.045, 0.045), 6),
                    "local_area": location["local_area"],
                    "nearest_airport": location["nearest_airport"],
                    "nearest_military_area": location["nearest_military_area"],
                    "distance_to_nearest_military_nm": location["distance_to_nearest_military_nm"],
                    "military_pattern": location["military_pattern"],
                    "callsign": f"N{100 + index % 900}RB",
                    "registration": f"N{100 + index % 900}RB",
                    "operator": operator,
                    "description": f"{description} vintage restoration",
                    "altitude_ft": random.choice([1200, 2500, 4500]),
                    "ground_speed_kt": round(random.uniform(70, 145), 1),
                    "distance_nm": round(random.uniform(1.5, 25), 3),
                    "squawk": "1200",
                    "emergency": "none",
                }
            )
            rows.append(training_row(aircraft, {
                "is_rare": True,
                "confidence": round(random.uniform(0.84, 0.92), 2),
                "reason": f"{description} is normally routine, but this example includes {factor}, so it is an exception to the common-GA suppression rule.",
            }, observer_context_for(location)))
            index += 1
    return rows


def build_synthetic_rows(start_index: int, target_count: int) -> list[dict[str, str]]:
    contrast_repeats = 10 if target_count >= 2200 else 6
    rows: list[dict[str, str]] = military_contrast_rows(start_index, repeats=contrast_repeats)
    rows.extend(t38_edwards_curriculum_rows(start_index + len(rows), repeats=48 if target_count >= 2200 else 20))
    rows.extend(ga_hard_negative_rows(start_index + len(rows), repeats=24 if target_count >= 2200 else 12))
    rows.extend(ga_exception_rows(start_index + len(rows), repeats=8 if target_count >= 2200 else 4))
    index = start_index + len(rows)
    while len(rows) < target_count:
        type_designator, description, operator, factor = random.choice(RARE_TEMPLATES)
        aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=True)
        location = next(item for item in SOCAL_LOCATIONS if item["local_area"] == aircraft["local_area"])
        rows.append(training_row(aircraft, local_context_label(type_designator, description, factor, location), observer_context_for(location)))
        index += 1

        if len(rows) >= target_count:
            break
        registration, type_designator, description, operator, factor = random.choice(SPECIAL_REGISTRATION_TEMPLATES)
        aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=True)
        aircraft["registration"] = registration
        location = next(item for item in SOCAL_LOCATIONS if item["local_area"] == aircraft["local_area"])
        rows.append(training_row(aircraft, {
            "is_rare": True,
            "confidence": round(random.uniform(0.88, 0.96), 2),
            "reason": f"{registration} is rare for Orange County or Los Angeles County because it is a {factor}.",
        }, observer_context_for(location)))
        index += 1

        if len(rows) >= target_count:
            break
        type_designator, description, operator = random.choice(NEGATIVE_TEMPLATES)
        aircraft = synthetic_aircraft(index, type_designator, description, operator, rare=False)
        location = next(item for item in SOCAL_LOCATIONS if item["local_area"] == aircraft["local_area"])
        rows.append(training_row(aircraft, {
            "is_rare": False,
            "confidence": round(random.uniform(0.78, 0.94), 2),
            "reason": f"Ordinary airline, training, or hub traffic around {location['nearest_airport']} is not rare without special livery, limited operator, geographic anomaly, or another noteworthy signal.",
        }, observer_context_for(location)))
        index += 1
    return rows[:target_count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--live", type=Path, action="append", help="Live/weak-labeled JSONL source. Can be passed multiple times.")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed-random", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed_random)
    live_paths = args.live or [DEFAULT_LIVE]
    live_examples = [example for path in live_paths for example in read_jsonl(path)]
    seed_examples = read_jsonl(args.seed)
    existing_examples = live_examples + seed_examples
    rows = [from_existing(example) for example in existing_examples]
    if len(rows) < args.target:
        rows.extend(build_synthetic_rows(len(rows), args.target - len(rows)))
    rows = rows[: args.target]
    random.shuffle(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "train.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    rare_count = sum(1 for row in rows if json.loads(row["response"])["is_rare"])
    summary = {
        "target": args.target,
        "region": "orange_county_los_angeles_southern_california",
        "source_mix": {
            "live_adsb_examples": len(live_examples),
            "live_adsb_sources": [str(path) for path in live_paths],
            "curated_seed_examples": len(seed_examples),
            "synthetic_examples": max(0, args.target - len(existing_examples)),
            "regional_context": "static repo profile for Orange County and Los Angeles County airports, nearby military areas, and flight-test corridors",
        },
        "live_examples": len(live_examples),
        "seed_examples": len(seed_examples),
        "synthetic_generated_examples": max(0, args.target - len(existing_examples)),
        "training_examples": len(rows),
        "rare_examples": rare_count,
        "not_rare_examples": len(rows) - rare_count,
        "train_csv": str(args.out_dir / "train.csv"),
        "train_jsonl": str(args.out_dir / "train.jsonl"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
