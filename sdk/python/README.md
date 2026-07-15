# AutoVision Python SDK

SDK version 0.2.0 — supports the v5.13 car-recognition model (897 classes:
896 car model-generations + 1 background class, 76 makes, EfficientNet-V2-S).

## Installation

```bash
pip install autovision

# With TFLite backend (recommended for edge/mobile)
pip install autovision[tflite]

# With ONNX backend (recommended for server)
pip install autovision[onnx]
```

## Usage

```python
from autovision import AutoVision

# Point at a model release directory containing model_manifest.json,
# class_mapping.json, confusion_pairs.json and the model weights
# (car_classifier.tflite / model.tflite / *.onnx are auto-detected).
engine = AutoVision("models/v5.13.0")

result = engine.classify("photo.jpg", top_k=5)

if result.rejected:
    print(f"Rejected: {result.rejection_reason}")
    # Candidates are still available in result.predictions
else:
    top = result.top1
    print(f"{top.make} {top.model} ({top.year_start}-{top.year_end})")
    print(f"  Confidence: {top.confidence:.1%}  |  Rarity: {top.rarity}")

for pred in result.predictions:
    print(f"{pred.rank}. {pred.class_name}  {pred.confidence:.1%}")

# Batch classification — returns list[ClassificationResult]
all_results = engine.classify_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### Legacy constructor (backwards compatible)

```python
engine = AutoVision(
    model_path="models/car_classifier.tflite",
    class_mapping_path="models/class_mapping.json",  # optional if next to model
    manifest_path="models/model_manifest.json",      # optional if next to model
)
```

If no `model_manifest.json` is found next to the model, the SDK falls back to
built-in defaults matching the v5.13.0 release and emits a warning.

## Result contract

`classify()` returns a `ClassificationResult`:

| Field | Type | Description |
|---|---|---|
| `predictions` | `list[Prediction]` | Top-k predictions sorted by confidence descending, **excluding the background class**. Populated even when rejected. |
| `top1` | `Optional[Prediction]` | Accepted top prediction; `None` when rejected. |
| `rejected` | `bool` | True when a gating check failed. |
| `rejection_reason` | `Optional[str]` | `"not_a_car"`, `"low_confidence"`, or `"ambiguous"`. |
| `inference_ms` | `float` | Model inference latency in milliseconds. |
| `engine_version` | `str` | SDK version (`"0.2.0"`). |
| `model_version` | `str` | Manifest version (e.g. `"5.13.0"`). |
| `taxonomy_version` | `str` | `"{model_version}-{num_classes}"` (e.g. `"5.13.0-897"`). |

Each `Prediction` has: `rank` (1-based), `index`, `class_name`, `make`,
`model`, `generation`, `year_start`, `year_end`, `rarity`, `confidence`,
`logit`. Rarity tiers: `COMMON`, `UNCOMMON`, `RARE`, `ULTRA_RARE`, `EPIC`,
`LEGENDARY` (plus `BACKGROUND` for the internal background class).

## Gating logic

Gates are applied in order to the argmax over all 897 temperature-scaled
softmax probabilities:

1. **not_a_car** — the argmax is the background class, or the top probability
   is below the manifest `ood_threshold` (0.05).
2. **low_confidence** — the top probability is below the per-rarity
   confidence threshold from the manifest (COMMON 0.4 ... LEGENDARY 0.7;
   0.5 for unknown rarities).
3. **ambiguous** — the top-2 car classes form a known confusion pair (from
   `confusion_pairs.json`) and their probability gap is below that pair's
   `margin_threshold` (fallback: manifest `confusion_pair_margin`, 0.08).

Even when rejected, `predictions` is populated so callers can display
candidates; only `top1` is `None`.

## Errors

All exceptions derive from `autovision.AutoVisionError`:

- `ModelLoadError` — weights missing or backend failed to load
- `MappingMismatchError` — class mapping size != manifest `num_classes`
- `InvalidImageError` — PIL cannot decode the image
- `UnsupportedFormatError` — model file is not `.tflite` or `.onnx`

## Model Files

A release directory (e.g. `models/v5.13.0/`) contains:

- `car_classifier.tflite` (44 MB, float16) — or `.onnx` equivalent
- `model_manifest.json` — input size, normalization, temperature, thresholds
  (source of truth for preprocessing and gating)
- `class_mapping.json` — class index to make/model/year/rarity metadata
- `confusion_pairs.json` — visually similar pairs with margin thresholds

## Preprocessing

The SDK handles all preprocessing automatically (parameters come from the
manifest):

1. EXIF rotation correction
2. Center crop to square
3. Resize to 384x384
4. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
5. NHWC float32, pixel range 0-1 before normalization

## Requirements

- Python 3.9+
- NumPy, Pillow
- One of: `tflite-runtime`, `tensorflow`, or `onnxruntime`
