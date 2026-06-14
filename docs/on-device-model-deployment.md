# On-Device Model Deployment

rarebirds runs rarity classification locally on iPhone. The backend can provide
candidate aircraft sightings and regional context, but the final rarity decision
belongs on device.

## Current App Runtime

The primary iPhone classifier is now a bundled multinomial text classifier:

```text
ios/RareBird/Sources/RareBirdApp/Resources/RarityTextClassifier.json
```

The exported artifact is about 0.90 MB and is copied into the app bundle by
XcodeGen. App startup uses this chain:

```text
TextRarityClassifier -> deterministic guardrails
```

`TextRarityClassifier` performs the on-device decision with no model download,
no bridge, and no generated JSON parsing. It applies a deterministic policy
layer for hard alert signals, rare/classic types, and routine contextual
long-haul traffic before returning the final decision.

To regenerate the bundled classifier:

```bash
python3 scripts/export_rarity_text_classifier.py \
  --train-csv data/datasets/rarity-training-v7-combined-split/train.csv \
  --threshold 0.70 \
  --output ios/RareBird/Sources/RareBirdApp/Resources/RarityTextClassifier.json
```

Current gate results:

```text
v7 held-out eval:          96.7% accuracy, precision 0.992, recall 0.935, F1 0.963
gold eval:                 92.6% accuracy, precision 1.000, recall 0.849, F1 0.919
gold invalid JSON rate:    0.0%
serialized model size:     ~0.90 MB
```

These numbers are a gate, not a claim of real-world accuracy. The gold set
should continue to grow with manually reviewed live sightings.

## MLX Experiment

The Swift source keeps `LocalModelRarityClassifier` behind `canImport(...)`
guards, but the App Store project no longer links `MLXLLM`, `MLXLMCommon`, or
`Tokenizers`. This keeps the shipping app small and makes the bundled text
classifier the only active model path.

If MLX is re-enabled for experiments, `LocalModelRarityClassifier` looks for a
directory named:

```text
RareBirdsRarityModel
```

It checks the app bundle first, then the app's Application Support directory.
For simulator testing:

```bash
scripts/install_ios_mlx_model.sh
```

That currently defaults to:

```text
model/output/rarity-gemma4-oc-la-hard-v3-mlx-4bit
```

The MLX path remains experimental because the artifact is about 2.5 GB and
local LLM generation can still produce malformed JSON.

## Latest Small LLM Runs

The Gemma 4 E2B v4/v5/v7 experiments are not better phone candidates yet.

```text
model/output/rarity-gemma4-e2b-hard-v5-merged
model/output/rarity-gemma4-e2b-hard-v5-mlx-4bit
model/output/rarity-gemma4-e2b-v7-combined-qlora
```

Measured results:

```text
v7 QLoRA gold:             19.6% accuracy, recall 0.041, invalid JSON 69.6%
audited QLoRA gold:        9.5% accuracy, recall 0.014, invalid JSON 85.8%
current MLX gold:          46.6% accuracy, recall 0.014, invalid JSON 4.7%
v3 MLX regional contrast: 7/8 strict, 1 invalid JSON
v3 MLX GA hard:           3/6 strict, 1 invalid JSON
v5 merged 150 strict:     0.86 accuracy, 15/150 invalid JSON
v5 merged regional:       7/8 strict, 1 invalid JSON, 8/8 lenient
v5 merged GA hard:        3/6 strict, 2 invalid JSON
v5 MLX regional:          5/8 strict, 3 invalid JSON
v5 MLX GA hard:           4/6 strict, 0 invalid JSON
```

The generative models still produce malformed JSON and miss most gold rare
cases. Do not promote a Gemma/MLX model as the default iPhone classifier until
it passes the same gold gate as the bundled text classifier.

## Convert To MLX

Setup on Apple Silicon:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-mlx
. .venv-mlx/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install mlx-lm
```

Convert a merged Hugging Face checkpoint:

```bash
scripts/convert_rarity_model_mlx.sh \
  model/output/rarity-gemma4-e2b-hard-v5-merged \
  model/output/rarity-gemma4-e2b-hard-v5-mlx-4bit
```

Smoke eval:

```bash
. .venv-mlx/bin/activate
python scripts/evaluate_mlx_rarity_model.py \
  --eval-csv data/eval/regional_contrast_cases.csv
```

## Development Bridge

The HTTP bridge remains useful for development and comparing Mac-side MLX
behavior against the app, but it is not enabled by the App Store plist:

```bash
. .venv-mlx/bin/activate
python scripts/serve_mlx_rarity_model.py \
  --host 127.0.0.1 \
  --port 8765
```

The App Store path is local-model-first. `NetworkRarityClassifier` is
development plumbing, not the preferred iPhone classifier.
