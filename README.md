---
title: Rare Planes
emoji: 🛩️
sdk: gradio
sdk_version: 6.16.0
python_version: 3.12
app_file: app.py
suggested_hardware: a10g-large
short_description: Aircraft rarity classifier with live ADS-B map
models:
  - google/gemma-3-27b-it
tags:
  - gradio
  - aircraft
  - ads-b
  - gemma
---

# Rare Planes

Dedicated to Benjamin Vo, plane spotter extraordinaire.

<img src="assets/ben.jpg" alt="Benjamin Vo" width="320">

Rare Planes is a two-track project:

1. A Gradio/Gemma workspace for experimenting with sub-32B rarity classifiers.
2. An iPhone app that tells a user when rare aircraft are flying near them and sends push notifications.

The product should treat the model and the mobile app as separate concerns, but Gemma is part of the alert pipeline: it should quickly classify whether a live aircraft looks rare from the aircraft state, instead of forcing every sighting through slow database searches. The backend still owns geospatial matching, cooldowns, and APNs delivery.

## Repository Layout

```text
backend/    Server-side aircraft polling, rare-aircraft matching, and APNs fanout.
data/       Rule lists and seed data used by the backend.
docs/       Architecture notes and decisions.
ios/        Native SwiftUI iPhone app workspace notes.
model/      Gemma 4 tuning and inference workspace.
```

## Demo and Model Artifacts

- Hugging Face demo Space: [build-small-hackathon/rarebirds](https://huggingface.co/spaces/build-small-hackathon/rarebirds)
- Fine-tuned MLX model: [vochris/rare-bird-gemma4-e2b-mlx-4bit](https://huggingface.co/vochris/rare-bird-gemma4-e2b-mlx-4bit)
- Model artifacts are kept out of this GitHub repo because they are multi-GB generated outputs. The app-sized model card lives in [model/huggingface/rare-bird-gemma4-e2b-mlx-4bit](model/huggingface/rare-bird-gemma4-e2b-mlx-4bit).

## Product Shape

The first version should answer one question quickly: "Is something rare near me right now?"

Core flows:

- User grants location permission and notification permission.
- User sets an alert radius, aircraft categories, and quiet hours.
- Backend polls or streams aircraft positions for active user regions.
- Backend asks Gemma to score whether candidate aircraft are rare, with deterministic rules as guardrails.
- Backend sends an APNs notification when a new match crosses the user's threshold.
- App shows a current nearby list, aircraft details, and recent alert history.

## Current Data Source Assumptions

- OpenSky is useful for research and non-commercial prototypes.
- ADSB Exchange has strong live ADS-B coverage and commercial API options.
- FlightAware AeroAPI is better when flight status, schedules, or richer commercial metadata matter.

See [docs/architecture.md](docs/architecture.md) for the initial design.

## Public Repo Hygiene

Keep API keys, App Store Connect credentials, provisioning profiles, private keys, generated model outputs, local virtual environments, and scratch collection artifacts out of git. Use GitHub Actions secrets or local environment variables for tokens such as `HF_TOKEN`.
