# ADR 0001: Use Gemma For Rarity Classification, Keep Delivery Deterministic

## Status

Accepted

## Context

The project includes Gemma 4 and a mobile app for rare-aircraft alerts. The model is intended to quickly decide whether a plane is rare without requiring expensive per-sighting database searches. The notification path still has reliability requirements: delivery, dedupe, location matching, and user preferences must be deterministic, observable, and testable.

## Decision

Use `google/gemma-4-E2B-it` to classify candidate aircraft as rare or not rare after the backend has already filtered by user area. Keep model work in `model/` and app/backend work in `ios/` and `backend/`.

Use deterministic rules as guardrails: known watchlists, squawk codes, cooldowns, blocklists, and minimum-confidence thresholds.

## Consequences

- The app can avoid repeated database searches for every nearby aircraft.
- Alert delivery remains testable even when the model changes.
- The model needs compact context for facts not present in live ADS-B fields.
- Classifications should be logged so false positives and misses can become training data.
