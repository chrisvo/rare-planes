#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any

from collect_socal_aircraft_dataset import (
    COMMON_AIRLINE_PREFIXES,
    COMMON_GA_TRAINER_TYPE_DESIGNATORS,
    CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS,
    CONTEXTUAL_TYPE_DESIGNATORS,
    LAST_OF_TYPE_DESIGNATORS,
    MILITARY_OR_SPECIAL_TYPE_DESIGNATORS,
    RARE_CALLSIGN_PREFIXES,
    RARE_DESCRIPTION_PATTERNS,
    RARE_OPERATOR_PATTERNS,
    RARE_TYPE_DESIGNATORS,
    SPECIAL_REGISTRATIONS,
    callsign_prefix,
)

COMMON_AIRLINE_TYPES = {
    "A318", "A319", "A320", "A321", "A20N", "A21N",
    "B737", "B738", "B739", "B38M", "B39M", "B752", "B763", "E170", "E175", "E75L",
}
EMERGENCY_SQUAWKS = {"7500", "7600", "7700"}
COAST_GUARD_PATTERNS = [r"\bCOAST\s+GUARD\b", r"\bUSCG\b"]
GOVERNMENT_PATTERNS = [r"\bNASA\b", r"\bAIR\s+FORCE\b", r"\bUSAF\b", r"\bNAVY\b", r"\bARMY\b", r"\bMARINES?\b"]


def _text(*values: Any) -> str:
    return " ".join(str(value).upper() for value in values if value is not None and str(value).strip())


def _add_factor(factors: list[dict[str, Any]], code: str, label: str, points: int) -> None:
    if any(factor["code"] == code for factor in factors):
        return
    factors.append({"code": code, "label": label, "points": points})


def aircraft_label(aircraft: dict[str, Any]) -> str:
    parts = [
        aircraft.get("operator"),
        aircraft.get("description"),
        aircraft.get("type_designator"),
        aircraft.get("callsign"),
    ]
    compact = [str(part).strip() for part in parts if part]
    if not compact:
        return str(aircraft.get("icao_hex") or "Unknown aircraft")
    label = " ".join(compact)
    label = re.sub(r"\s+", " ", label)
    return label[:120]


def score_aircraft(aircraft: dict[str, Any], observer_context: dict[str, Any] | None = None) -> dict[str, Any]:
    observer_context = observer_context or {}
    type_designator = (aircraft.get("type_designator") or "").upper()
    callsign = (aircraft.get("callsign") or "").upper()
    prefix = callsign_prefix(callsign)
    registration = (aircraft.get("registration") or "").upper()
    squawk = str(aircraft.get("squawk") or "")
    operator_text = _text(aircraft.get("operator"))
    description_text = _text(aircraft.get("description"))
    all_text = _text(aircraft.get("operator"), aircraft.get("description"), aircraft.get("callsign"), aircraft.get("registration"))
    military_pattern = str(observer_context.get("military_pattern") or "").lower()
    near_base = military_pattern == "base_pattern"
    factors: list[dict[str, Any]] = []
    suppressors: list[dict[str, Any]] = []

    if squawk in EMERGENCY_SQUAWKS:
        _add_factor(factors, "emergency_squawk", f"Emergency squawk {squawk}", 100)
    if registration in SPECIAL_REGISTRATIONS:
        _add_factor(factors, "special_registration", f"Known special registration {registration}", 45)
    if prefix in RARE_CALLSIGN_PREFIXES:
        _add_factor(factors, "rare_callsign_prefix", f"Special-use callsign prefix {prefix}", 25)
    if any(re.search(pattern, all_text) for pattern in COAST_GUARD_PATTERNS):
        _add_factor(factors, "coast_guard", "Coast Guard operator", 35)
    if any(re.search(pattern, all_text) for pattern in GOVERNMENT_PATTERNS):
        _add_factor(factors, "government_or_military", "Government or military operator", 25)
    if type_designator in LAST_OF_TYPE_DESIGNATORS:
        _add_factor(factors, "last_of_type", f"{type_designator} is a disappearing or last-of-type aircraft", 35)
    if type_designator in RARE_TYPE_DESIGNATORS and type_designator not in CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS and type_designator not in CONTEXTUAL_TYPE_DESIGNATORS:
        _add_factor(factors, "rare_type", f"{type_designator} is on the rare aircraft type list", 35)
    if type_designator in CONTEXTUAL_TYPE_DESIGNATORS:
        _add_factor(factors, "contextual_heavy", f"{type_designator} is context-dependent and worth review", 18)
    if type_designator in CONTEXTUAL_HELICOPTER_TYPE_DESIGNATORS:
        _add_factor(factors, "contextual_helicopter", f"{type_designator} helicopter is context-dependent", 20)
    for pattern in RARE_OPERATOR_PATTERNS:
        if re.search(pattern, operator_text):
            _add_factor(factors, "rare_operator", "Unusual operator signal", 20)
            break
    for pattern in RARE_DESCRIPTION_PATTERNS:
        if re.search(pattern, description_text):
            _add_factor(factors, "rare_description", "Rare or enthusiast-notable aircraft description", 25)
            break

    is_military_or_special = type_designator in MILITARY_OR_SPECIAL_TYPE_DESIGNATORS or any(
        code in {"government_or_military", "coast_guard", "rare_callsign_prefix"} for code in [factor["code"] for factor in factors]
    )
    if is_military_or_special and near_base and squawk not in EMERGENCY_SQUAWKS and registration not in SPECIAL_REGISTRATIONS:
        suppressors.append({
            "code": "military_near_base_context",
            "label": f"Globally interesting aircraft appears to be in a routine local base/test-range pattern near {observer_context.get('nearest_military_area') or 'a military area'}",
            "points": -35,
        })
    elif is_military_or_special and not near_base:
        _add_factor(factors, "military_or_special_away_from_base", "Special-use aircraft away from a base pattern", 25)

    if prefix in COMMON_AIRLINE_PREFIXES or type_designator in COMMON_AIRLINE_TYPES:
        suppressors.append({"code": "common_airline", "label": "Routine airline or cargo traffic", "points": -30})
    if type_designator in COMMON_GA_TRAINER_TYPE_DESIGNATORS:
        suppressors.append({"code": "common_ga", "label": "Common local GA training/private aircraft", "points": -25})

    positive_points = sum(int(factor["points"]) for factor in factors)
    negative_points = sum(int(factor["points"]) for factor in suppressors)
    score = max(0, min(100, positive_points + negative_points))
    if squawk in EMERGENCY_SQUAWKS:
        score = 100
    if near_base and is_military_or_special and squawk not in EMERGENCY_SQUAWKS and registration not in SPECIAL_REGISTRATIONS:
        score = max(35, min(score, 65))

    reason_codes = [factor["code"] for factor in factors] + [factor["code"] for factor in suppressors]
    recommendation = "show" if score >= 70 else "review" if score >= 35 else "suppress"
    is_rare = recommendation == "show"
    summary = _summary(aircraft, score, recommendation, factors, suppressors)
    return {
        "is_rare": is_rare,
        "rarity_score": score,
        "recommendation": recommendation,
        "confidence": round(0.5 + min(score, 90) / 200, 2),
        "reason_codes": reason_codes,
        "factors": factors,
        "suppressors": suppressors,
        "summary": summary,
        "reason": summary,
        "aircraft_label": aircraft_label(aircraft),
        "label_source": "deterministic_rules_v2",
    }


def _summary(
    aircraft: dict[str, Any],
    score: int,
    recommendation: str,
    factors: list[dict[str, Any]],
    suppressors: list[dict[str, Any]],
) -> str:
    label = aircraft_label(aircraft)
    if recommendation == "show":
        lead = f"Rare aircraft candidate: {label} scores {score}/100."
    elif recommendation == "review":
        lead = f"Contextual aircraft: {label} scores {score}/100 and should be reviewed before alerting."
    else:
        lead = f"Routine/unknown aircraft: {label} scores {score}/100."
    evidence = factors[:3] if factors else suppressors[:2]
    if not evidence:
        return lead + " No rare-aircraft signals were found."
    joined = "; ".join(str(item["label"]) for item in evidence)
    if recommendation == "suppress" and any(item["code"] == "common_airline" for item in suppressors):
        return lead + " Suppressed as routine airline/cargo traffic with no overriding rare signal."
    if any(item["code"] == "military_near_base_context" for item in suppressors):
        return lead + " Globally interesting, but local base-pattern context prevents automatic alerting."
    return lead + f" Evidence: {joined}."


def explanation_from_score(score: dict[str, Any]) -> str:
    label = score.get("aircraft_label") or "Aircraft"
    rarity_score = score.get("rarity_score", "?")
    recommendation = score.get("recommendation") or "review"
    factors = score.get("factors") or []
    if factors:
        evidence = "; ".join(str(factor.get("label")) for factor in factors[:3])
    else:
        evidence = "no strong rare-aircraft factors"
    if recommendation == "show":
        return f"Rare aircraft detected: {label} ({rarity_score}/100). Evidence: {evidence}."
    if recommendation == "review":
        return f"Aircraft worth reviewing: {label} ({rarity_score}/100). Evidence: {evidence}."
    return f"No rare aircraft alert: {label} ({rarity_score}/100). Evidence: {evidence}."
