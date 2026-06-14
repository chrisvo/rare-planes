#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMMUNITY_PROVIDERS = {"twitter_planespotting", "reddit_planespotting"}
COMMON_LOCAL_TYPES = {
    "A319",
    "A320",
    "A321",
    "B737",
    "B738",
    "B739",
    "B38M",
    "B39M",
    "C150",
    "C152",
    "C172",
    "C182",
    "PA28",
    "P28A",
    "P28R",
    "SR20",
    "SR22",
    "BE33",
    "BE36",
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
COMMUNITY_SIGNAL_PATTERNS = {
    "special_livery": r"\b(special|retro|heritage|oneworld|star alliance|flag|tribute|livery|colors?|colours?)\b",
    "first_visit": r"\b(first(?:-| )?(?:time|visit|arrival|flight)|inaugural|debut)\b",
    "rare_word": r"\brare\b",
    "incident": r"\b(divert|go-?around|emergency|collapse|incident|winds?hield|gear)\b",
    "classic_or_vintage": r"\b(classic|vintage|warbird|last|final)\b",
    "route_return": r"\b(finally\s+returns?|returns?\s+to|back\s+at|back\s+to)\b",
    "stored_or_retired": r"\b(stored|storage|retired|withdrawn|fleet\s+stored)\b",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json(value: str, *, row_index: int, field: str, errors: list[str]) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        errors.append(f"row {row_index}: {field} is invalid JSON: {exc}")
        return {}
    if not isinstance(parsed, dict):
        errors.append(f"row {row_index}: {field} is not a JSON object")
        return {}
    return parsed


def source_text(aircraft: dict[str, Any]) -> str:
    return " ".join(
        str(aircraft.get(key) or "")
        for key in ["source_title", "description", "operator", "callsign", "type_designator"]
    ).strip()


def community_signals(text: str) -> list[str]:
    lowered = text.lower()
    return [name for name, pattern in COMMUNITY_SIGNAL_PATTERNS.items() if re.search(pattern, lowered)]


def operational_signals(aircraft: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    squawk = str(aircraft.get("squawk") or "")
    emergency = str(aircraft.get("emergency") or "").lower()
    callsign = str(aircraft.get("callsign") or "").upper()
    if squawk in {"7500", "7600", "7700"} or emergency not in {"", "none", "null"}:
        signals.append("emergency")
    if any(callsign.startswith(prefix) for prefix in RARE_CALLSIGN_PREFIXES):
        signals.append("special_mission_callsign")
    return signals


def category_for(prompt: dict[str, Any], response: dict[str, Any]) -> str:
    aircraft = prompt.get("aircraft") or {}
    context = prompt.get("observer_context") or {}
    provider = aircraft.get("provider") or "unknown"
    type_designator = str(aircraft.get("type_designator") or "").upper()
    policy = ((context.get("local_frequency_context") or {}).get("alert_policy") or "")
    military_pattern = context.get("military_pattern") or aircraft.get("military_pattern")
    text = source_text(aircraft)
    signals = [*community_signals(text), *operational_signals(aircraft)]

    if "emergency" in signals:
        return "emergency"
    if "special_mission_callsign" in signals:
        return "special_mission"
    if provider in COMMUNITY_PROVIDERS:
        if signals:
            return "community_signal"
        return "community_no_signal"
    if type_designator in {"B744", "B748", "BLCF"}:
        return "747_contextual"
    if type_designator in COMMON_LOCAL_TYPES:
        return "common_local"
    if policy == "alert" and military_pattern == "not_base_pattern":
        return "military_away_from_base"
    if military_pattern == "base_pattern" or "routine_near_military" in json.dumps(context):
        return "military_near_base"
    if response.get("is_rare") is True:
        return "rare_type_or_special"
    return "routine_other"


def audit_row(index: int, row: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    flags: list[dict[str, Any]] = []
    prompt = parse_json(row.get("prompt", ""), row_index=index, field="prompt", errors=errors)
    response = parse_json(row.get("response", ""), row_index=index, field="response", errors=errors)
    aircraft = prompt.get("aircraft") if isinstance(prompt.get("aircraft"), dict) else {}
    context = prompt.get("observer_context") if isinstance(prompt.get("observer_context"), dict) else {}
    provider = aircraft.get("provider") or "unknown"
    type_designator = str(aircraft.get("type_designator") or "").upper()
    label = response.get("is_rare")
    confidence = response.get("confidence")
    text = source_text(aircraft)
    signals = [*community_signals(text), *operational_signals(aircraft)]
    policy = ((context.get("local_frequency_context") or {}).get("alert_policy") or "")

    if set(response) != {"is_rare", "confidence", "reason"}:
        errors.append(f"row {index}: response must contain exactly is_rare, confidence, reason")
    if not isinstance(label, bool):
        errors.append(f"row {index}: is_rare must be boolean")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append(f"row {index}: confidence must be 0..1")

    def flag(code: str, severity: str, message: str) -> None:
        flags.append(
            {
                "row": index,
                "code": code,
                "severity": severity,
                "provider": provider,
                "type_designator": type_designator,
                "is_rare": label,
                "confidence": confidence,
                "category": category_for(prompt, response),
                "source_title": aircraft.get("source_title"),
                "source_url": aircraft.get("source_url"),
                "message": message,
            }
        )

    if provider in COMMUNITY_PROVIDERS and label is True and not signals:
        flag(
            "community_positive_without_signal",
            "high",
            "Community-sourced positive has no explicit special-livery, first-visit, incident, rare, or vintage signal.",
        )
    if provider in COMMUNITY_PROVIDERS and label is True and type_designator in COMMON_LOCAL_TYPES and not signals:
        flag(
            "common_type_community_positive_without_signal",
            "critical",
            "Common aircraft type is labeled rare only because it came from a community post.",
        )
    if policy.startswith("suppress_alert") and label is True and not signals and type_designator in COMMON_LOCAL_TYPES:
        flag(
            "suppression_policy_overridden_without_signal",
            "critical",
            "Suppression policy is overridden by a positive label without a concrete rarity signal.",
        )
    if label is True and isinstance(confidence, (int, float)) and confidence < 0.7:
        flag("low_confidence_positive", "medium", "Positive label has low confidence.")
    if label is False and signals:
        flag(
            "negative_with_community_signal",
            "medium",
            "Negative label contains a community-interest signal; verify whether this is a hard negative or a mislabeled positive.",
        )

    summary = {
        "provider": provider,
        "type_designator": type_designator,
        "is_rare": label,
        "category": category_for(prompt, response),
        "community_signals": signals,
        "source_url": aircraft.get("source_url"),
    }
    return summary, flags, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/audit/rarity-dataset"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    provider_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    provider_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    signal_counts: Counter[str] = Counter()
    all_flags: list[dict[str, Any]] = []
    errors: list[str] = []

    for index, row in enumerate(rows):
        summary, flags, row_errors = audit_row(index, row)
        provider = str(summary["provider"])
        label = str(summary["is_rare"])
        provider_counts[provider] += 1
        label_counts[label] += 1
        provider_label_counts[provider][label] += 1
        category_counts[str(summary["category"])] += 1
        signal_counts.update(summary["community_signals"])
        all_flags.extend(flags)
        errors.extend(row_errors)

    report = {
        "input": str(args.input),
        "examples": len(rows),
        "labels": dict(label_counts),
        "providers": dict(provider_counts),
        "provider_labels": {provider: dict(counts) for provider, counts in provider_label_counts.items()},
        "categories": dict(category_counts),
        "community_signal_counts": dict(signal_counts),
        "flag_counts": dict(Counter(flag["code"] for flag in all_flags)),
        "severity_counts": dict(Counter(flag["severity"] for flag in all_flags)),
        "errors": errors[:100],
        "flagged_rows_csv": str(args.out_dir / "flagged_rows.csv"),
        "report_json": str(args.out_dir / "report.json"),
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with (args.out_dir / "flagged_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "row",
            "severity",
            "code",
            "provider",
            "type_designator",
            "is_rare",
            "confidence",
            "category",
            "source_title",
            "source_url",
            "message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_flags)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
