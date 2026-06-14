#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


AIRCRAFT_FIELDS = (
    "type_designator",
    "description",
    "operator",
    "callsign",
    "registration",
    "squawk",
    "emergency",
    "category",
    "local_area",
    "nearest_airport",
    "nearest_military_area",
    "military_pattern",
)

CONTEXT_FIELDS = (
    "current_local_area",
    "nearest_airport",
    "nearest_military_area",
    "military_pattern",
    "region",
)

SPECIAL_MISSION_TERMS = (
    "rescue",
    "evac",
    "medevac",
    "lifeguard",
    "search-and-rescue",
    "special incident",
)

WATCHLIST_TERMS = (
    "vintage",
    "warbird",
    "restoration",
    "special livery",
    "special-livery",
    "watchlist",
)

SPECIAL_REGISTRATIONS = {"D-ABYN", "A7-BEG", "B-LRJ"}
CONTEXTUAL_LONGHAUL_TYPES = {
    "A332",
    "A333",
    "A339",
    "A359",
    "B744",
    "B748",
    "B788",
    "B789",
    "B78X",
    "B77W",
}
RARE_TYPE_DESIGNATORS = {
    "A124",
    "A225",
    "A3ST",
    "A306",
    "A342",
    "A343",
    "A345",
    "A346",
    "B1",
    "B2",
    "B52",
    "B703",
    "B712",
    "B721",
    "B722",
    "B741",
    "B742",
    "B743",
    "B753",
    "BLCF",
    "CONI",
    "CVLP",
    "CVLT",
    "DC10",
    "DC3",
    "DC6",
    "DC8",
    "IL62",
    "IL76",
    "L101",
    "MD11",
    "MD80",
    "MD81",
    "MD82",
    "MD83",
    "MD87",
    "MD88",
    "MD90",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_prompt(row: dict[str, str]) -> dict:
    return json.loads(row["prompt"])


def parse_response(row: dict[str, str]) -> dict:
    return json.loads(row["response"])


def bool_label(row: dict[str, str]) -> bool:
    return bool(parse_response(row)["is_rare"])


def add_text_features(features: Counter[str], prefix: str, value: object) -> None:
    if value is None:
        return
    text = str(value).strip().lower()
    if not text:
        return
    features[f"{prefix}={text}"] += 3
    tokens = TOKEN_RE.findall(text)
    for token in tokens:
        features[f"{prefix}:{token}"] += 1
    for left, right in zip(tokens, tokens[1:]):
        features[f"{prefix}:{left}_{right}"] += 1


def featurize(prompt: dict) -> Counter[str]:
    features: Counter[str] = Counter()
    aircraft = prompt.get("aircraft") or {}
    context = prompt.get("observer_context") or {}
    frequency = context.get("local_frequency_context") or {}

    for field in AIRCRAFT_FIELDS:
        add_text_features(features, f"aircraft.{field}", aircraft.get(field))

    for field in CONTEXT_FIELDS:
        add_text_features(features, f"context.{field}", context.get(field))

    add_text_features(features, "context.frequency_class", frequency.get("class"))
    add_text_features(features, "context.alert_policy", frequency.get("alert_policy"))

    for numeric in ("distance_nm", "distance_to_nearest_military_nm", "altitude_ft", "ground_speed_kt"):
        value = aircraft.get(numeric)
        if isinstance(value, (int, float)):
            bucket = int(value // 10 * 10)
            features[f"aircraft.{numeric}_bucket={bucket}"] += 1

    callsign = str(aircraft.get("callsign") or "").upper()
    if callsign:
        match = re.match(r"[A-Z]+", callsign)
        if match:
            features[f"callsign_prefix={match.group(0)}"] += 2

    squawk = str(aircraft.get("squawk") or "")
    emergency = str(aircraft.get("emergency") or "").lower()
    hard_emergency = squawk in {"7500", "7600", "7700"} or emergency not in {"", "none", "null"}
    if hard_emergency:
        features["signal.emergency"] += 10
        features[f"signal.emergency_squawk={squawk}"] += 10

    searchable = " ".join(
        str(aircraft.get(field) or "").lower()
        for field in ("callsign", "registration", "description", "operator")
    )
    registration = str(aircraft.get("registration") or "").upper()
    special_registration = registration in SPECIAL_REGISTRATIONS
    if special_registration:
        features["signal.special_registration"] += 10
        features[f"signal.special_registration={registration}"] += 10

    for term in SPECIAL_MISSION_TERMS:
        if term in searchable:
            features["signal.special_mission"] += 8
            features[f"signal.special_mission={term}"] += 4
    for term in WATCHLIST_TERMS:
        if term in searchable:
            features["signal.watchlist"] += 8
            features[f"signal.watchlist={term}"] += 4

    type_designator = str(aircraft.get("type_designator") or "").upper()
    if type_designator in {"B744", "B748"} and not hard_emergency and not special_registration:
        features["signal.contextual_widebody_suppressed"] += 12
        features[f"signal.contextual_widebody_suppressed={type_designator}"] += 8

    return features


def has_hard_alert_signal(aircraft: dict) -> bool:
    squawk = str(aircraft.get("squawk") or "")
    emergency = str(aircraft.get("emergency") or "").lower()
    if squawk in {"7500", "7600", "7700"} or emergency not in {"", "none", "null"}:
        return True
    if str(aircraft.get("registration") or "").upper() in SPECIAL_REGISTRATIONS:
        return True
    searchable = " ".join(
        str(aircraft.get(field) or "").lower()
        for field in ("callsign", "registration", "description", "operator")
    )
    return any(term in searchable for term in SPECIAL_MISSION_TERMS + WATCHLIST_TERMS)


def adjusted_probability(prompt: dict, probability: float) -> float:
    aircraft = prompt.get("aircraft") or {}
    if has_hard_alert_signal(aircraft):
        return max(probability, 0.95)
    type_designator = str(aircraft.get("type_designator") or "").upper()
    if type_designator in RARE_TYPE_DESIGNATORS:
        return max(probability, 0.95)
    if type_designator in CONTEXTUAL_LONGHAUL_TYPES:
        return min(probability, 0.05)
    return probability


class MultinomialNB:
    def __init__(self, alpha: float = 0.5, min_count: int = 1) -> None:
        self.alpha = alpha
        self.min_count = min_count
        self.vocabulary: set[str] = set()
        self.class_doc_counts: Counter[bool] = Counter()
        self.class_feature_counts: dict[bool, Counter[str]] = {False: Counter(), True: Counter()}
        self.class_total_features: Counter[bool] = Counter()

    def fit(self, rows: list[dict[str, str]]) -> None:
        document_counts: Counter[str] = Counter()
        prepared: list[tuple[bool, Counter[str]]] = []
        for row in rows:
            label = bool_label(row)
            features = featurize(parse_prompt(row))
            prepared.append((label, features))
            document_counts.update(features)

        self.vocabulary = {feature for feature, count in document_counts.items() if count >= self.min_count}
        for label, features in prepared:
            self.class_doc_counts[label] += 1
            for feature, count in features.items():
                if feature in self.vocabulary:
                    self.class_feature_counts[label][feature] += count
                    self.class_total_features[label] += count

    def predict_proba(self, row: dict[str, str]) -> float:
        features = featurize(parse_prompt(row))
        total_docs = sum(self.class_doc_counts.values())
        scores: dict[bool, float] = {}
        vocab_size = len(self.vocabulary)
        for label in (False, True):
            prior = (self.class_doc_counts[label] + self.alpha) / (total_docs + 2 * self.alpha)
            score = math.log(prior)
            denominator = self.class_total_features[label] + self.alpha * max(vocab_size, 1)
            counts = self.class_feature_counts[label]
            for feature, count in features.items():
                if feature not in self.vocabulary:
                    continue
                score += count * math.log((counts[feature] + self.alpha) / denominator)
            scores[label] = score
        max_score = max(scores.values())
        rare_score = math.exp(scores[True] - max_score)
        common_score = math.exp(scores[False] - max_score)
        return rare_score / (rare_score + common_score)

    def model_size_bytes(self) -> int:
        payload = {
            "alpha": self.alpha,
            "min_count": self.min_count,
            "vocabulary": sorted(self.vocabulary),
            "class_doc_counts": dict(self.class_doc_counts),
            "class_total_features": dict(self.class_total_features),
            "class_feature_counts": {
                str(label): dict(counts) for label, counts in self.class_feature_counts.items()
            },
        }
        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def evaluate(model: MultinomialNB, rows: list[dict[str, str]], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    samples = []
    for row in rows:
        prompt = parse_prompt(row)
        probability = adjusted_probability(prompt, model.predict_proba(row))
        prediction = probability >= threshold
        truth = bool_label(row)
        if prediction and truth:
            tp += 1
        elif prediction and not truth:
            fp += 1
        elif not prediction and truth:
            fn += 1
        else:
            tn += 1
        if prediction != truth and len(samples) < 12:
            aircraft = prompt.get("aircraft") or {}
            response = parse_response(row)
            samples.append(
                {
                    "truth": truth,
                    "prediction": prediction,
                    "rare_probability": round(probability, 4),
                    "type": aircraft.get("type_designator"),
                    "callsign": aircraft.get("callsign"),
                    "operator": aircraft.get("operator"),
                    "description": aircraft.get("description"),
                    "reason": response.get("reason"),
                }
            )
    total = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {
        "examples": total,
        "accuracy": (tp + tn) / total if total else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "mistakes": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=Path("data/datasets/rarity-oc-la-socal-hard-v5-3600-split/train.csv"))
    parser.add_argument("--eval-csv", type=Path, default=Path("data/datasets/rarity-oc-la-socal-hard-v5-3600-split/eval.csv"))
    parser.add_argument("--extra-eval-csv", type=Path, action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("model/output/rarity-text-classifier-eval.json"))
    args = parser.parse_args()

    train_rows = read_csv(args.train_csv)
    model = MultinomialNB(alpha=args.alpha, min_count=args.min_count)
    model.fit(train_rows)

    evals = {str(args.eval_csv): evaluate(model, read_csv(args.eval_csv), args.threshold)}
    for path in args.extra_eval_csv:
        evals[str(path)] = evaluate(model, read_csv(path), args.threshold)

    metrics = {
        "model": "multinomial_nb_text_features",
        "train_csv": str(args.train_csv),
        "train_examples": len(train_rows),
        "threshold": args.threshold,
        "alpha": args.alpha,
        "min_count": args.min_count,
        "vocabulary_size": len(model.vocabulary),
        "estimated_serialized_model_bytes": model.model_size_bytes(),
        "estimated_serialized_model_mb": round(model.model_size_bytes() / 1_000_000, 3),
        "evaluations": evals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
