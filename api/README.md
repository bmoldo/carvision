# AutoVision API

Self-hosted REST API for car recognition (server version 0.2.0, model v5.13).

## Quick Start

```bash
# Build
docker build -t autovision-api -f api/Dockerfile .

# Run (mount your model release directory)
docker run -p 8000:8000 -v /path/to/models:/app/models autovision-api
```

## Endpoints

### `GET /health`

Health check.

```json
{"status": "ok", "model_loaded": true}
```

`status` is `"no_model"` when the model failed to load.

### `GET /version`

```json
{
  "engine_version": "0.2.0",
  "model_version": "5.13.0",
  "taxonomy_version": "5.13.0-897"
}
```

`model_version` / `taxonomy_version` are `null` when no model is loaded.

### `GET /metadata`

```json
{
  "model_loaded": true,
  "model_version": "5.13.0",
  "taxonomy_version": "5.13.0-897",
  "backend": "tflite",
  "input_size": 384,
  "num_classes": 897,
  "num_makes": 76,
  "quantization": "float16"
}
```

### `POST /classify`

Classify a car image.

**Request (multipart/form-data):**
- `image` (file): Image upload. Accepted content types: `image/jpeg`,
  `image/png`, `image/webp`. Maximum size: 10 MB.
- `top_k` (form, optional): Number of predictions (default: 5, clamped to 1-20)

**Response** (mirrors the SDK `ClassificationResult`):
```json
{
  "predictions": [
    {
      "rank": 1,
      "class_name": "bmw_3_series_f30",
      "make": "BMW",
      "model": "3 Series",
      "generation": "F30",
      "year_start": 2012,
      "year_end": 2018,
      "rarity": "COMMON",
      "confidence": 0.9234
    }
  ],
  "top1": {
    "rank": 1,
    "class_name": "bmw_3_series_f30",
    "make": "BMW",
    "model": "3 Series",
    "generation": "F30",
    "year_start": 2012,
    "year_end": 2018,
    "rarity": "COMMON",
    "confidence": 0.9234
  },
  "rejected": false,
  "rejection_reason": null,
  "inference_ms": 45.2,
  "engine_version": "0.2.0",
  "model_version": "5.13.0",
  "taxonomy_version": "5.13.0-897"
}
```

When the image is rejected by gating, `rejected` is `true`,
`rejection_reason` is one of `"not_a_car"`, `"low_confidence"`,
`"ambiguous"`, `top1` is `null` — and `predictions` still contains the
top-k candidates.

## Errors

All errors use a structured shape (no raw exception text):

```json
{"error": {"code": "invalid_image", "message": "Image could not be decoded"}}
```

| HTTP status | code | When |
|---|---|---|
| 503 | `model_not_loaded` | Model is not loaded |
| 422 | `invalid_image` | Upload could not be decoded as an image |
| 400 | `bad_request` | Unsupported content type, empty upload, invalid parameters |
| 413 | `bad_request` | Upload larger than 10 MB |
| 500 | `inference_failed` | Unexpected inference error |

## Configuration

Environment variables:

- `AUTOVISION_MODEL_DIR` (preferred) — model release directory containing
  `model_manifest.json`, `class_mapping.json`, `confusion_pairs.json` and
  weights (default: `/app/models/v5.13.0`)
- `AUTOVISION_MODEL` — legacy: direct path to a `.tflite`/`.onnx` model file
  (takes precedence over `AUTOVISION_MODEL_DIR` when set)
- `AUTOVISION_CLASSES` — legacy: path to `class_mapping.json`

## curl examples

```bash
# Health / version / metadata
curl http://localhost:8000/health
curl http://localhost:8000/version
curl http://localhost:8000/metadata

# Classify an image
curl -X POST http://localhost:8000/classify \
  -F "image=@photo.jpg;type=image/jpeg" \
  -F "top_k=5"
```
