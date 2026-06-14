# Architecture

## Tracks

Rare Bird has two independent tracks.

### 1. Gemma 4 Workspace

Use `model/` for local experiments with `google/gemma-4-E2B-it`. The intended product role for the model is fast rarity classification: given a normalized live aircraft state, return whether this aircraft is rare enough to alert on, a confidence score, and a short reason.

Good early model tasks:

- Classify a normalized aircraft state as rare or not rare.
- Summarize an aircraft sighting in plain language.
- Explain why a matched aircraft is unusual.
- Convert natural-language user preferences into deterministic rules.

The model should not need a full database lookup for every sighting. It should learn the concept of spotter rarity: uncommon, noteworthy, or chase-worthy aircraft in context. It can classify quickly from fields like aircraft type, callsign, registration, operator, altitude, route behavior, and squawk. For facts not present in the live feed, give the model compact reference context such as rarity factors, notable example types, special registrations, local baseline traffic, and user watchlists.

See [model-training-strategy.md](model-training-strategy.md).

### 2. iPhone App And Alert Backend

Use `ios/` for the native app and `backend/` for live aircraft ingestion and notifications.

Recommended runtime flow:

```text
Aircraft data source
  -> backend ingest job or stream consumer
  -> normalize aircraft states
  -> geospatial match active user areas
  -> Gemma rarity classification
  -> deterministic guardrails, cooldown, and dedupe
  -> APNs push notification
  -> iPhone app detail screen
```

## Backend Responsibilities

- Fetch or stream live aircraft states.
- Normalize provider-specific fields into one aircraft-state shape.
- Maintain user alert regions and preferences.
- Ask Gemma to classify candidate aircraft after geospatial filtering.
- Apply deterministic guardrails and blocklists.
- Deduplicate alerts by user, aircraft, and time window.
- Send APNs notifications.
- Store alert history for app display.

## iPhone Responsibilities

- Request location and notification permissions.
- Let users configure radius, categories, quiet hours, and watchlists.
- Show nearby rare aircraft and recent alerts.
- Open an aircraft detail screen from a push notification.
- Avoid continuous background polling. Use server-side monitoring plus APNs.

## Data Sources

OpenSky is suitable for a prototype or research-oriented usage. ADSB Exchange is a strong candidate for live aircraft coverage when commercial use and terms are settled. FlightAware AeroAPI is useful if the app needs flight status, schedules, historical flight objects, or alert endpoints.

The backend should hide provider-specific details behind a `FlightDataProvider` boundary so the app is not coupled to one data vendor.

## Minimal Backend Data Model

```text
users
  id
  apns_token
  home_lat
  home_lon
  alert_radius_nm
  quiet_hours

aircraft_states
  provider
  icao_hex
  callsign
  registration
  type_designator
  operator
  lat
  lon
  altitude_ft
  ground_speed_kt
  heading_deg
  seen_at

rare_rules
  id
  label
  enabled
  match_json

rarity_classifications
  id
  aircraft_state_id
  model_id
  is_rare
  confidence
  reason
  classified_at

alerts
  id
  user_id
  icao_hex
  rule_id
  title
  body
  lat
  lon
  sent_at
```

## First Milestone

Build a backend prototype that can:

- Query aircraft within a fixed radius.
- Classify candidates with a local or hosted Gemma runner.
- Apply guardrails from `data/rare-aircraft-rules.example.json`.
- Print would-send alerts to logs.

Then build the iOS app shell with:

- Map/list view.
- Settings for radius and categories.
- Local mock alert history.
