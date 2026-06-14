---
license: apache-2.0
base_model: google/gemma-4-E2B-it
library_name: mlx
pipeline_tag: text-generation
tags:
  - gemma
  - mlx
  - aircraft
  - aviation
  - ios
  - local-model
  - rare-bird
---

# Rare Bird Gemma 4 E2B MLX 4-bit

Rare Bird is a local aircraft-rarity classifier for an iPhone app. Given a
normalized aircraft sighting and regional observer context, it predicts whether
the aircraft is uncommon, noteworthy, or chase-worthy enough to surface to an
aviation-curious user.

This repository contains the app-sized MLX 4-bit conversion of the fine-tuned
Rare Bird model.

## Model Details

- Base model: `google/gemma-4-E2B-it`
- Fine-tuning method: LoRA/QLoRA adapter
- Deployment export: LoRA merged into the base model, then converted to MLX
- Quantization: MLX 4-bit, 4.501 bits per weight
- Artifact size: about 2.5 GB
- Target use: local iPhone app prototype and Apple Silicon development

## Task

The model classifies Southern California aircraft sightings. It is trained to
consider:

- aircraft type and description
- callsign, registration, and operator
- altitude, speed, heading, and distance
- Orange County and Los Angeles regional context
- whether military traffic is near a routine base/test pattern
- rare type, rare callsign, special registration, and emergency-squawk signals

Expected output is one JSON object:

```json
{
  "is_rare": true,
  "confidence": 0.9,
  "reason": "Boeing 747-400 Dreamlifter is rare for Orange County or Los Angeles County because it represents very limited modified freighter examples."
}
```

## Prompt Format

Use the full Rare Bird training-style prompt. Short prompts without the reference
policy are not reliable.

```text
### System
You are Rare Bird, a strict aircraft rarity classifier for plane spotters. You must output exactly one JSON object with keys is_rare, confidence, reason. No markdown, no metadata, no extra keys.

### Input JSON
{...full Rare Bird payload with aircraft, observer_context, reference, output_schema...}

### Output JSON
```

The repository script `scripts/collect_socal_aircraft_dataset.py` contains the
canonical `make_prompt()` function used to build the full payload.

## Evaluation

Merged Hugging Face checkpoint before MLX conversion:

- Eval examples: 150
- Strict accuracy: 0.9867
- F1: 0.9875
- Precision: 1.0
- Recall: 0.9753
- Invalid JSON: 0

MLX 4-bit regional contrast eval:

- Eval examples: 8
- Accuracy: 1.0
- Invalid JSON: 0

The regional contrast eval focuses on the important local-context behavior:
military aircraft can be alert-worthy away from a base pattern but routine near
Los Alamitos, March ARB, Edwards, Palmdale, or similar local training/test
patterns.

## Usage With MLX

```bash
mlx_lm.generate \
  --model rare-bird-gemma4-e2b-mlx-4bit \
  --prompt - \
  --ignore-chat-template \
  --max-tokens 120 \
  --temp 0
```

For local iOS simulator testing, the Rare Bird repo includes a development
bridge:

```bash
python scripts/serve_mlx_rarity_model.py \
  --model model/output/rarity-gemma4-oc-la-hard-v2-mlx-4bit \
  --host 127.0.0.1 \
  --port 8765
```

The simulator app calls `http://127.0.0.1:8765/classify`. This is a development
bridge only; the shipping app should use an on-device MLX or LiteRT runtime.

## Intended Use

This model is intended for:

- local/offline aircraft-rarity classification prototypes
- Rare Bird iPhone app development
- evaluating regional rarity logic for plane spotting

It is not intended for:

- aviation safety or operational air traffic decisions
- real-time navigation
- regulatory, law-enforcement, or emergency use

## Limitations

- Rarity is contextual and changes as aircraft retire, move operators, or change
  routes.
- The model depends on normalized aircraft fields and regional observer context.
- The 4-bit model is optimized for size. The BF16 MLX conversion produced
  cleaner explanations in some cases but is too large for the current app target.
- The model should be paired with deterministic product guardrails for
  notification cooldowns, user preferences, and claimability.

## Provenance

Built by the Rare Bird project from a fine-tuned `google/gemma-4-E2B-it`
checkpoint trained on synthetic and real Southern California aircraft rarity
examples.
