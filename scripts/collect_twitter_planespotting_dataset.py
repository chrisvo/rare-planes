#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from collect_reddit_planespotting_dataset import AIRCRAFT_PATTERNS, COMMON_PATTERNS, registration_from_title, write_jsonl
from collect_socal_aircraft_dataset import make_prompt


DEFAULT_INPUT = Path("~/Downloads/twitter-UserTweets-1780810505144.json").expanduser()
DEFAULT_OUT_DIR = Path("data/datasets/twitter-planespotting")
DEFAULT_BALANCED_LIMIT = 160

COMMUNITY_SIGNAL_PATTERNS = [
    r"\bSPECIAL\s+LIVERY\b",
    r"\bLIVERY\b",
    r"\bNEW\s+(?:LIVERY|COLORS?)\b",
    r"\bRETRO\b",
    r"\bHERITAGE\b",
    r"\bONEWORLD\b",
    r"\bSTAR\s+ALLIANCE\b",
    r"\bFLAG\s+LIVERY\b",
    r"\bFIRST(?:-|\s+)(?:EVER\s+)?(?:VISIT|TIME|ARRIVAL|FLIGHT)\b",
    r"\bINAUGURAL\b",
    r"\bDREAM\s+PLANE\b",
    r"\bRARE\b",
    r"\bDIVERT(?:S|ED|ING)?\b",
    r"\bGO-?AROUND\b",
    r"\bWINDSHIELD\b",
    r"\bCOLOURFUL\b|\bCOLORFUL\b",
]

ACTION_CONTEXT_PATTERNS = [
    r"\bARRIV(?:ES|ED|ING|AL)\b",
    r"\bDEPART(?:S|ED|ING|URE)\b",
    r"\bTOUCH(?:ES|ED)?\s+DOWN\b",
    r"\bTAK(?:ES|ING)\s+OFF\b",
    r"\bLAND(?:S|ED|ING)\b",
    r"\bTAXI(?:S|ED|ING)?\b",
    r"\bSEEN\s+(?:LIVE|ON)\b",
    r"\bCAPTURED\s+LIVE\b",
    r"\bPOWERS?\s+(?:DOWN|OFF)\b",
    r"\bHEADS?\s+OUT\b",
    r"\bCOMPLETES\s+ITS\s+JOURNEY\b",
    r"\bCLIMBS?\s+INTO\b",
    r"\bCAUGHT\s+LIVE\b",
    r"\bCAUGHT\s+IN\b",
]

PROMO_ONLY_PATTERNS = [
    r"\bLIVE\s+(?:FROM|LOS\s+ANGELES|LAX)\b",
    r"\bTUNE\s+IN\b",
    r"\bJOIN\s+(?:US|PLANE\s+JOCKEY)\b",
    r"\bAIRPORT\s+ACTION\b",
    r"\bPOP-?UP\s+LIVE\s+STREAM\b",
]

EXTRA_AIRCRAFT_PATTERNS: list[dict[str, Any]] = [
    {
        "patterns": [r"\bA350-?900\b", r"\bA359\b", r"\bAIRBUS\s+A350\b"],
        "type_designator": "A359",
        "description": "AIRBUS A350-900",
        "operator": None,
        "is_rare": False,
        "confidence": 0.74,
        "reason": "A350 traffic is high-interest for spotters, but not alert-worthy without special livery, first-visit, unusual operator, or other contextual evidence.",
    },
    {
        "patterns": [r"\b777-?300ER\b", r"\bB77W\b", r"\bBOEING\s+777\b"],
        "type_designator": "B77W",
        "description": "BOEING 777-300ER",
        "operator": None,
        "is_rare": False,
        "confidence": 0.72,
        "reason": "777 long-haul traffic is spotter-interesting but usually routine at major international airports without a special signal.",
    },
    {
        "patterns": [r"\bA330-?200\b", r"\bA332\b", r"\bA330\b"],
        "type_designator": "A332",
        "description": "AIRBUS A330",
        "operator": None,
        "is_rare": False,
        "confidence": 0.72,
        "reason": "A330 traffic is spotter-interesting but not rare by itself at major international airports.",
    },
    {
        "patterns": [r"\b737\s+MAX\s+9\b", r"\b737-?9\b", r"\bB39M\b"],
        "type_designator": "B39M",
        "description": "BOEING 737 MAX 9",
        "operator": None,
        "is_rare": False,
        "confidence": 0.72,
        "reason": "737 MAX traffic is not alert-worthy without special livery, unusual operator, emergency, or another noteworthy signal.",
    },
]


def clean_tweet_text(text: str) -> str:
    text = re.sub(r"^RT\s+@\w+:\s*", "", text.strip())
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tweet_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


def media_urls(tweet: dict[str, Any]) -> list[str]:
    urls = []
    for item in tweet.get("media") or []:
        if not isinstance(item, dict):
            continue
        for key in ["original", "thumbnail", "url"]:
            if item.get(key):
                urls.append(str(item[key]))
    return sorted(set(urls))


def community_signals(text: str) -> list[str]:
    upper = text.upper()
    return [pattern for pattern in COMMUNITY_SIGNAL_PATTERNS if re.search(pattern, upper)]


def has_action_context(text: str) -> bool:
    upper = text.upper()
    return any(re.search(pattern, upper) for pattern in ACTION_CONTEXT_PATTERNS)


def looks_promo_only(text: str) -> bool:
    upper = text.upper()
    return any(re.search(pattern, upper) for pattern in PROMO_ONLY_PATTERNS) and not has_action_context(text)


def infer_aircraft_from_tweet(text: str) -> dict[str, Any] | None:
    upper = text.upper()
    for item in [*AIRCRAFT_PATTERNS, *EXTRA_AIRCRAFT_PATTERNS, *COMMON_PATTERNS]:
        if any(re.search(pattern, upper) for pattern in item["patterns"]):
            return {
                **item,
                "is_rare": bool(item.get("is_rare", False)),
                "confidence": float(item.get("confidence", 0.74)),
            }
    return None


def curation_decision(tweet: dict[str, Any], inferred: dict[str, Any] | None, duplicate: bool) -> dict[str, Any]:
    text = clean_tweet_text(tweet.get("full_text") or "")
    signals = community_signals(text)
    decision = {
        "tweet_id": tweet.get("id"),
        "screen_name": tweet.get("screen_name"),
        "created_at": tweet.get("created_at"),
        "url": tweet.get("url") or None,
        "text": text,
        "include_in_dataset": False,
        "reason_code": "",
        "reason": "",
        "inferred_type_designator": inferred.get("type_designator") if inferred else None,
        "inferred_description": inferred.get("description") if inferred else None,
        "community_signals": signals,
    }
    if not text:
        return {
            **decision,
            "reason_code": "empty_text",
            "reason": "Excluded because the tweet has no usable text.",
        }
    if duplicate:
        return {
            **decision,
            "reason_code": "duplicate_or_retweet",
            "reason": "Excluded because this normalized tweet text was already processed.",
        }
    if inferred is None:
        return {
            **decision,
            "reason_code": "no_aircraft_match",
            "reason": "Excluded because Codex could not extract a concrete aircraft type or model from the tweet.",
        }
    if looks_promo_only(text) and not signals:
        return {
            **decision,
            "reason_code": "generic_livestream_promo",
            "reason": "Excluded because this is a generic livestream or airport-action promo, not an aircraft rarity example.",
        }
    if inferred.get("is_rare") or signals:
        return {
            **decision,
            "include_in_dataset": True,
            "reason_code": "spotter_interest_positive",
            "reason": "Included because it names an aircraft and contains a rare-type, special-livery, first-visit, or chase-worthy spotter signal.",
        }
    if has_action_context(text):
        return {
            **decision,
            "include_in_dataset": True,
            "reason_code": "useful_routine_negative",
            "reason": "Included as a not-rare counterexample because it names an aircraft in a concrete arrival/departure/taxi/landing context without a rarity signal.",
        }
    return {
        **decision,
        "reason_code": "insufficient_sighting_context",
        "reason": "Excluded because it names an aircraft but lacks enough sighting or rarity context for training.",
    }


def upgrade_for_community_interest(inferred: dict[str, Any], text: str) -> dict[str, Any]:
    signals = community_signals(text)
    if not signals:
        return inferred
    description = inferred["description"]
    return {
        **inferred,
        "is_rare": True,
        "confidence": max(float(inferred.get("confidence", 0.0)), 0.88),
        "reason": (
            f"{description} is alert-worthy because the planespotting community post highlights "
            "special livery, first-visit, rare, or otherwise chase-worthy context."
        ),
        "community_signals": signals,
    }


def make_example(tweet: dict[str, Any], inferred: dict[str, Any]) -> dict[str, Any]:
    text = clean_tweet_text(tweet.get("full_text") or "")
    aircraft = {
        "provider": "twitter_planespotting",
        "collected_at": tweet_timestamp(tweet.get("created_at")),
        "icao_hex": None,
        "callsign": callsign_from_text(text),
        "registration": registration_from_title(text),
        "type_designator": inferred["type_designator"],
        "description": inferred["description"],
        "operator": inferred.get("operator") or operator_from_text(text),
        "origin_country": None,
        "lat": None,
        "lon": None,
        "altitude_ft": None,
        "ground_speed_kt": None,
        "heading_deg": None,
        "vertical_rate_fpm": None,
        "squawk": None,
        "emergency": None,
        "distance_nm": None,
        "category": None,
        "seen_seconds": None,
        "source_title": text,
        "source_url": tweet.get("url") or None,
        "source_media_urls": media_urls(tweet),
        "community_interest_source": "x_planespotting_account",
    }
    label = {
        "is_rare": inferred["is_rare"],
        "confidence": inferred["confidence"],
        "reason": inferred["reason"],
        "label_source": "twitter_planespotting_text_rules_v1",
        "community_signals": inferred.get("community_signals", []),
    }
    response = {
        "is_rare": label["is_rare"],
        "confidence": label["confidence"],
        "reason": label["reason"],
    }
    return {
        "prompt": make_prompt(aircraft),
        "response": json.dumps(response, sort_keys=True, separators=(",", ":")),
        "aircraft": aircraft,
        "label": label,
        "source": {
            "platform": "x",
            "tweet_id": tweet.get("id"),
            "screen_name": tweet.get("screen_name"),
            "name": tweet.get("name"),
            "text": text,
            "url": tweet.get("url") or None,
            "created_at": tweet.get("created_at") or None,
            "favorite_count": tweet.get("favorite_count"),
            "retweet_count": tweet.get("retweet_count"),
            "reply_count": tweet.get("reply_count"),
            "quote_count": tweet.get("quote_count"),
            "views_count": tweet.get("views_count"),
            "retweeted_status": tweet.get("retweeted_status"),
            "media_urls": media_urls(tweet),
        },
    }


def callsign_from_text(text: str) -> str | None:
    match = re.search(r"\b[A-Z]{2,4}\d{1,4}[A-Z]?\b", text.upper())
    return match.group(0) if match else None


def operator_from_text(text: str) -> str | None:
    operators = [
        "Emirates",
        "Lufthansa",
        "EVA Air",
        "Air Tahiti Nui",
        "ZipAir",
        "Western Global",
        "Southwest Airlines",
        "Alaska Airlines",
        "Iberia",
        "Vietnam Airlines",
        "Saudia",
        "SWISS",
        "Qantas",
    ]
    upper = text.upper()
    for operator in operators:
        if operator.upper() in upper:
            return operator
    return None


def review_row(tweet: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "tweet_id": tweet.get("id"),
        "screen_name": tweet.get("screen_name"),
        "text": clean_tweet_text(tweet.get("full_text") or ""),
        "url": tweet.get("url") or None,
        "created_at": tweet.get("created_at") or None,
        "media_urls": media_urls(tweet),
        "review_reason": reason,
    }


def review_row_from_decision(decision: dict[str, Any], tweet: dict[str, Any]) -> dict[str, Any]:
    return {
        "tweet_id": decision["tweet_id"],
        "screen_name": decision["screen_name"],
        "text": decision["text"],
        "url": decision["url"],
        "created_at": decision["created_at"],
        "media_urls": media_urls(tweet),
        "review_reason": decision["reason"],
        "reason_code": decision["reason_code"],
        "inferred_type_designator": decision["inferred_type_designator"],
        "inferred_description": decision["inferred_description"],
        "community_signals": decision["community_signals"],
    }


def write_train_csv(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt", "response"])
        writer.writeheader()
        for example in examples:
            writer.writerow({"prompt": example["prompt"], "response": example["response"]})


def balanced_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(examples) <= limit:
        return list(examples)
    rare = [item for item in examples if item["label"]["is_rare"]]
    not_rare = [item for item in examples if not item["label"]["is_rare"]]
    target_rare = min(len(rare), limit // 2)
    target_not_rare = min(len(not_rare), limit - target_rare)
    if target_rare + target_not_rare < limit:
        target_rare = min(len(rare), limit - target_not_rare)

    def key(item: dict[str, Any]) -> tuple[str, str]:
        return (str(item["aircraft"].get("type_designator") or ""), str(item["source"].get("text") or ""))

    def stratified_take(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for item in sorted(items, key=key):
            buckets.setdefault(str(item["aircraft"].get("type_designator") or "unknown"), []).append(item)
        selected = []
        while len(selected) < count and buckets:
            for type_designator in sorted(list(buckets)):
                bucket = buckets[type_designator]
                if bucket:
                    selected.append(bucket.pop(0))
                    if len(selected) >= count:
                        break
                if not bucket:
                    buckets.pop(type_designator, None)
        return selected

    return sorted(
        [
            *stratified_take(rare, target_rare),
            *stratified_take(not_rare, target_not_rare),
        ],
        key=lambda item: (not item["label"]["is_rare"], item["aircraft"]["type_designator"], item["source"]["text"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--keep-retweets", action="store_true", help="Keep retweets instead of deduping by normalized text.")
    parser.add_argument("--balanced-limit", type=int, default=DEFAULT_BALANCED_LIMIT)
    args = parser.parse_args()

    tweets = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(tweets, list):
        raise ValueError("--input must be a JSON array of tweets")

    seen_texts: set[str] = set()
    examples: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    curation_rows: list[dict[str, Any]] = []
    duplicate_tweets = 0
    for tweet in tweets:
        if not isinstance(tweet, dict):
            continue
        text = clean_tweet_text(tweet.get("full_text") or "")
        dedupe_key = text.upper()
        duplicate = bool(text and not args.keep_retweets and dedupe_key in seen_texts)
        if duplicate:
            duplicate_tweets += 1
        elif text:
            seen_texts.add(dedupe_key)

        inferred = infer_aircraft_from_tweet(text) if text else None
        decision = curation_decision(tweet, inferred, duplicate)
        curation_rows.append(decision)
        if not decision["include_in_dataset"]:
            review_rows.append(review_row_from_decision(decision, tweet))
            continue
        examples.append(make_example(tweet, upgrade_for_community_interest(inferred, text)))  # type: ignore[arg-type]

    examples.sort(key=lambda item: (not item["label"]["is_rare"], item["aircraft"]["type_designator"], item["source"]["text"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "examples.jsonl", examples)
    write_jsonl(args.out_dir / "review.jsonl", review_rows)
    write_jsonl(args.out_dir / "curation.jsonl", curation_rows)
    write_train_csv(args.out_dir / "train.csv", examples)
    balanced = balanced_examples(examples, args.balanced_limit)
    write_jsonl(args.out_dir / "examples-balanced.jsonl", balanced)
    write_train_csv(args.out_dir / "train-balanced.csv", balanced)
    summary = {
        "source": "twitter/x planespotting export",
        "input": str(args.input),
        "tweets_seen": len(tweets),
        "unique_texts_seen": len(seen_texts),
        "duplicate_tweets": duplicate_tweets,
        "examples": len(examples),
        "balanced_examples": len(balanced),
        "rare_examples": sum(1 for item in examples if item["label"]["is_rare"]),
        "not_rare_examples": sum(1 for item in examples if not item["label"]["is_rare"]),
        "balanced_rare_examples": sum(1 for item in balanced if item["label"]["is_rare"]),
        "balanced_not_rare_examples": sum(1 for item in balanced if not item["label"]["is_rare"]),
        "review_posts": len(review_rows),
        "curation_rows": len(curation_rows),
        "curation_reason_counts": {
            reason: sum(1 for item in curation_rows if item["reason_code"] == reason)
            for reason in sorted({item["reason_code"] for item in curation_rows})
        },
        "examples_jsonl": str(args.out_dir / "examples.jsonl"),
        "examples_balanced_jsonl": str(args.out_dir / "examples-balanced.jsonl"),
        "curation_jsonl": str(args.out_dir / "curation.jsonl"),
        "review_jsonl": str(args.out_dir / "review.jsonl"),
        "train_csv": str(args.out_dir / "train.csv"),
        "train_balanced_csv": str(args.out_dir / "train-balanced.csv"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
