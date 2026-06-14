# Candidate Model Evaluation Plan

## Goal

Evaluate whether smaller instruction models can serve as the RareBirds rarity classifier for the hackathon Gradio app and future iOS/on-device deployment.

Candidate models:

- `Qwen/Qwen3-4B` — realistic mobile/on-device target.
- `Qwen/Qwen3-8B` — likely quality/size sweet spot for MLX or strong on-device NPU scenarios.
- `google/gemma-4-E4B-it` — stronger small Gemma-family reasoning baseline.

## Product Constraint

The hackathon deliverable is the Gradio app. The future product target includes iOS. Therefore:

1. Prioritize a robust Gradio path now.
2. Keep model choices compatible with later MLX / quantized / on-device experiments.
3. Do not let model invalid JSON break the app; deterministic rules remain the fallback.

## Recommendation

Use a two-track experiment:

### Track A — zero/few-shot baseline

Before fine-tuning, run the same held-out eval examples against each base instruct model with the exact classifier contract:

```json
{
  "is_rare": true,
  "confidence": 0.91,
  "reason": "short explanation"
}
```

Measure:

- Strict JSON validity.
- Accuracy, precision, recall, F1.
- Latency.
- Failure modes by category.

This establishes whether fine-tuning is actually needed and exposes prompt/schema issues.

### Track B — Unsloth LoRA/QLoRA fine-tunes

Use Unsloth for Qwen and Gemma-family LoRA/QLoRA where supported. Unsloth should make training faster and less memory-hungry than the current vanilla PEFT path, especially for Qwen 4B/8B.

Train each candidate with the same dataset split and output schema. Save model-specific outputs under:

```text
model/output/rarity-qwen3-4b-unsloth-qlora/
model/output/rarity-qwen3-8b-unsloth-qlora/
model/output/rarity-gemma4-e4b-unsloth-qlora/
```

## Suggested Evaluation Gates

A model is hackathon-demo acceptable only if it meets all of:

- Strict JSON validity: >= 98% after one retry, ideally 100%.
- Precision: >= 90% so alerts are not spammy.
- Recall: >= 80% so it catches meaningful rare sightings.
- Mean latency acceptable for the Space hardware.
- Failure fallback is safe: invalid/timeout means deterministic fallback, not silent suppression.

A model is future-iOS promising if it also has:

- Viable quantization path: MLX, Core ML, or llama.cpp-compatible export.
- Reasonable memory footprint after 4-bit quantization.
- Good behavior with short prompts and compact `reason_code` outputs.

## Implementation Steps

1. Add a shared model candidate config listing model IDs, output dirs, and architecture notes.
2. Add or adapt a baseline eval script that can run base models without adapters.
3. Add Unsloth training script for text-only rarity classification.
4. Add strict structured-output parsing with retry support.
5. Run a small smoke eval first: 10 examples per model.
6. Run a 150-example eval for the models that pass smoke.
7. Compare metrics in one JSON/markdown summary.
8. Wire the Gradio app to prefer deterministic prefilter + best validated model adjudicator.

## Notes

- Qwen models may need a different LoRA target module pattern than the current Gemma 4 regex.
- Gemma 4 currently uses `AutoModelForImageTextToText` fallback behavior in the existing script because of multimodal model structure.
- The previous fresh eval collapse must be debugged in parallel; otherwise a fine-tuned model can look good in one environment and fail in another.
- For production and iOS, prefer compact structured output with `reason_code` plus optional prose, rather than long free-form reasons.
