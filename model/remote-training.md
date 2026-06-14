# Remote Training Host

Use `cvo@192.168.1.159` for CUDA training.

Verified host facts:

```text
host: cvo@192.168.1.159
os: Ubuntu 24.04
python: 3.12.3
gpu: NVIDIA GeForce RTX 5090, 32607 MiB
driver: 580.159.03
disk: about 2 TB free on /
```

PyTorch is not installed globally on the host, so use a project virtual environment.

## Sync The Repo

From this machine:

```bash
scripts/sync-remote-training.sh
```

This copies the repo to:

```text
~/rare-bird
```

## Create The Remote Environment

SSH into the host:

```bash
ssh cvo@192.168.1.159
cd ~/rare-bird
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r model/requirements.cuda.txt
```

Then log in to Hugging Face so the host can access Gemma:

```bash
huggingface-cli login
```

## Smoke Test

After the environment is installed:

```bash
scripts/remote-gemma-smoke.sh
```

The smoke test checks that CUDA is visible and loads the Gemma tokenizer/processor. Loading full model weights can take much longer and requires Hugging Face access to `google/gemma-4-E2B-it`.

Current verified smoke result:

```text
torch=2.12.0+cu130
cuda_available=True
device=NVIDIA GeForce RTX 5090
vram_gb=31.3
processor=Gemma4Processor
```

## Training Direction

The first training target is an instruction classifier that maps normalized aircraft JSON to structured rarity output. Build the combined live + seed training set with:

```bash
python3 scripts/build_rarity_training_dataset.py
```

The older live + seed starter dataset is then in:

```text
data/datasets/rarity-training/train.csv
```

For a fast 1,000-example pipeline-validation set, build:

```bash
python3 scripts/build_quick_1000_dataset.py
python3 scripts/validate_rarity_dataset.py --input data/datasets/rarity-quick-1000/train.csv
python3 scripts/split_rarity_dataset.py
```

The configured profile currently points at:

```text
data/datasets/rarity-quick-1000-split/train.csv
```

Run a dry check before training:

```bash
python3 scripts/train_rarity_lora.py --dry-run
```

Then train the LoRA adapter:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/train_rarity_lora.py \
  --qlora \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-seq-length 1024 \
  --output-dir model/output/rarity-gemma4-qlora
```

Evaluate the base model or an adapter:

```bash
python3 scripts/evaluate_rarity_model.py --max-examples 50
python3 scripts/evaluate_rarity_model.py --adapter-dir model/output/rarity-gemma4-lora/adapter --max-examples 150
```

Run the Gradio tester:

```bash
python3 scripts/gradio_rarity_tester.py --host 0.0.0.0 --port 7860
```

Then open:

```text
http://192.168.1.159:7860
```

## Hugging Face Space Demo

The repo has a root `app.py` entry point for Gradio Spaces. Use a GPU Space and set these variables as needed:

```text
RAREBIRD_MODEL_ID=google/gemma-3-27b-it
RAREBIRD_ADAPTER_DIR=
RAREBIRD_LOAD_IN_4BIT=1
RAREBIRD_MAX_SEQ_LENGTH=2048
RAREBIRD_MAX_NEW_TOKENS=160
RAREBIRD_WATCH_RADIUS_NM=15
RAREBIRD_SCAN_REFRESH_SECONDS=30
RAREBIRD_SCAN_CACHE_SECONDS=25
RAREBIRD_SCAN_MIN_INTERVAL_SECONDS=30
```

`RAREBIRD_MODEL_ID` can be any Hugging Face text/chat model that fits the Space GPU. The default is a larger under-32B model for the hackathon demo; the Gradio UI also includes presets for the current local Gemma model and other sub-32B candidates. Use `RAREBIRD_ADAPTER_DIR` for either a local adapter path committed to the Space or a private/public Hugging Face adapter repo. For gated base models, configure the standard `HF_TOKEN` Space secret.

The README metadata suggests `a10g-large`, but Hugging Face does not automatically assign suggested hardware. Select the GPU in the Space settings before the demo.

The app lazily loads the selected model on the first classification request, then reuses it until the model, adapter, quantization, or sequence-length settings change.

Live ADS-B scans are cached in memory to avoid provider throttling. If a provider returns `429 Too Many Requests` and a previous snapshot exists, the Space keeps showing the cached map and rare-aircraft list. If `adsb.fi` rate-limits a fresh request, the app tries `adsb.lol` before falling back to cached data.

The Space opens on a non-technical plane-watching landing page:

- A hero panel branded as `RareBirds`.
- A live Leaflet/Folium aircraft map around the configured latitude/longitude.
- A compact list of nearby aircraft that lightweight rules consider rare.

Technical classifier surfaces are kept under a collapsed `Developer tools` accordion. This keeps the first page focused on the plane-watching experience while preserving manual sighting and JSON inspector workflows for testing.

Run a fast UI smoke check without downloading model weights:

```bash
python3 scripts/smoke_gradio_space.py
python3 scripts/smoke_gradio_space.py --live-scan
```

Input:

```json
{
  "aircraft": {
    "icao_hex": "ae4ece",
    "callsign": "RCH123",
    "registration": null,
    "type_designator": "C17",
    "operator": "USAF",
    "altitude_ft": 22000,
    "ground_speed_kt": 410,
    "distance_nm": 18,
    "squawk": null
  },
  "reference": {
    "rare_type_designators": ["C17", "F35", "B52", "KC135"],
    "rare_callsign_prefixes": ["RCH", "REACH", "SPUR"]
  }
}
```

Output:

```json
{
  "is_rare": true,
  "confidence": 0.91,
  "reason": "C-17 military transport using a REACH/RCH callsign within the alert radius."
}
```

Keep the training examples compact. The model should learn fast classification behavior, not perform open-ended aviation research at inference time.
