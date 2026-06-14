# Context

Rare Bird helps aviation-curious iPhone users notice unusual aircraft near them.

## Domain Terms

- **Rare aircraft**: An aircraft worth alerting on because of type, operator, role, route, altitude, squawk, registration, or user-specific interest.
- **Sighting**: A live aircraft state that matched one or more rare-aircraft rules.
- **Alert**: A notification sent to a user for a sighting within their configured area.
- **User area**: A user's current location plus alert radius, or a saved fixed location.
- **Rarity score**: The model's fast judgment of whether a live aircraft is worth alerting on.
- **Rule**: A machine-readable condition used to decide whether an aircraft is rare or to constrain model behavior.
- **Cooldown**: A suppression window that prevents repeated alerts for the same aircraft near the same user.

## Initial Product Decisions

- Build a native iPhone app with SwiftUI.
- Put live aircraft polling and APNs delivery in a backend service.
- Use `google/gemma-4-E2B-it` as the fast rarity classifier for candidate aircraft.
- Keep app-critical delivery logic deterministic: location matching, cooldowns, dedupe, and APNs sending.
- Use explicit rules and compact reference lists as guardrails around model classification.

## Candidate Rare-Aircraft Signals

- ICAO type designator, such as military, vintage, tanker, cargo, or special mission aircraft.
- Registration or ICAO hex allowlist.
- Operator or callsign pattern.
- Squawk codes of interest.
- Unusual low-altitude route through the user's radius.
- User-defined watchlist aircraft.
- Model confidence and explanation from the current aircraft state.
