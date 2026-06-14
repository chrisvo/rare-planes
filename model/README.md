# Gemma 4 Workspace

This directory is for experiments with `google/gemma-4-E2B-it`. The main product goal is fast rare-aircraft classification from normalized live aircraft states.

The repo includes a local `gemma-tuner-multimodal` skill with setup instructions for Apple Silicon. Use a separate virtual environment for model work.

## Starting Point

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio
pip install -U transformers accelerate
huggingface-cli login
```

## Candidate App-Facing Uses

- Decide if a candidate aircraft is rare enough to alert on.
- Generate short explanations for why an aircraft matched.
- Summarize a user's daily rare-aircraft sightings.
- Turn a natural language request like "tell me about tankers and old warbirds within 50 miles" into structured rules.

## Classifier Contract

Input should be a compact JSON aircraft state:

```json
{
  "icao_hex": "ae4ece",
  "callsign": "RCH123",
  "registration": null,
  "type_designator": "C17",
  "operator": "USAF",
  "altitude_ft": 22000,
  "ground_speed_kt": 410,
  "distance_nm": 18,
  "squawk": null
}
```

Output should be structured:

```json
{
  "is_rare": true,
  "confidence": 0.91,
  "reason": "C-17 military transport using a REACH callsign within the alert radius."
}
```

Use compact reference context for facts that are not in the aircraft state, such as rarity factors, notable type examples, callsign prefixes, special registrations, local baseline traffic, and user watchlists. Do not train this as an exhaustive list memorization task; train the model to apply spotter rarity factors to the current aircraft state.
