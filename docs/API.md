# AutoVision REST API Reference

The self-hosted API server (see `api/`) exposes the same logical inference contract as the SDKs. All responses are JSON.

Base URL in the examples below: `http://localhost:8000`.

## Endpoints

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| POST | `/classify` | Classify a car image | required |
| GET | `/health` | Liveness/readiness check | none |
| GET | `/version` | Engine, model, and taxonomy versions | required |
| GET | `/metadata` | Loaded model details | required |

---

## Authentication

The server authenticates requests with an API key in the **`X-API-Key`** header. Auth is enabled by configuring accepted keys at startup:

```bash
# Comma-separated keys
docker run -e AUTOVISION_API_KEYS="av_yourkey1,av_yourkey2" ...

# Or a file with one key per line (wins over AUTOVISION_API_KEYS; use with
# Docker/Kubernetes secrets so keys don't appear in `docker inspect`)
docker run -e AUTOVISION_API_KEYS_FILE=/run/secrets/autovision_keys ...
```

```bash
curl -H "X-API-Key: av_yourkey1" http://localhost:8000/version
```

- Generate strong keys: `python3 -c "import secrets; print('av_' + secrets.token_urlsafe(32))"`
- **If no keys are configured, auth is disabled** and the server logs a startup warning. This is intended for local development only — always set keys in production.
- Multiple keys enable zero-downtime rotation: add the new key, migrate clients, remove the old key.
- `GET /health` never requires a key, so container health checks and load-balancer probes work unmodified.
- Keys are only accepted in the header — never in the query string.
- The Swagger UI at `/docs` has an **Authorize** button for the key.
- Requests without a valid key receive `401` with code `unauthorized` and a `WWW-Authenticate: ApiKey` header.

---

## POST /classify

Classify a single image. Multipart form upload.

### Request

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `image` | file | yes | JPEG, PNG, or WebP; max 10 MB |
| `top_k` | form field (int) | no | 1–20 (values outside the range are clamped), default 5 |

```bash
curl -X POST http://localhost:8000/classify \
  -H "X-API-Key: av_yourkey1" \
  -F "image=@photo.jpg" \
  -F "top_k=5"
```

### Response fields

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array | Top-k candidates, always returned (even when rejected) |
| `top1` | object or null | Best prediction; `null` when `rejected` is true |
| `rejected` | bool | Whether the result was rejected by gating |
| `rejection_reason` | string or null | `"not_a_car"`, `"low_confidence"`, or `"ambiguous"` |
| `inference_ms` | number | Model inference time in milliseconds |
| `engine_version` | string | API/SDK code version (currently `0.2.0`) |
| `model_version` | string | Model artifact version (currently `5.13.0`) |
| `taxonomy_version` | string | Label-space version (currently `5.13.0-897`) |

Each entry in `predictions` (and `top1`):

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | 1-based rank |
| `class_name` | string | Canonical class id, e.g. `bmw_3_series_f30` |
| `make` | string | e.g. `BMW` |
| `model` | string | e.g. `3 Series` |
| `generation` | string or null | e.g. `F30` |
| `year_start` | int | Production start year (class metadata) |
| `year_end` | int | Production end year (class metadata) |
| `rarity` | string | `COMMON`, `UNCOMMON`, `RARE`, `ULTRA_RARE`, `EPIC`, or `LEGENDARY` |
| `confidence` | number | Calibrated softmax score, 0–1 |

### Example: accepted prediction

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
      "confidence": 0.91
    },
    {
      "rank": 2,
      "class_name": "bmw_3_series_g20",
      "make": "BMW",
      "model": "3 Series",
      "generation": "G20",
      "year_start": 2019,
      "year_end": 2025,
      "rarity": "COMMON",
      "confidence": 0.05
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
    "confidence": 0.91
  },
  "rejected": false,
  "rejection_reason": null,
  "inference_ms": 142.7,
  "engine_version": "0.2.0",
  "model_version": "5.13.0",
  "taxonomy_version": "5.13.0-897"
}
```

### Example: rejected image

Rejection is a first-class outcome, not an error. HTTP status is still 200; `top1` is `null`, and the candidate predictions are returned for inspection.

```json
{
  "predictions": [
    {
      "rank": 1,
      "class_name": "ferrari_f40",
      "make": "Ferrari",
      "model": "F40",
      "generation": null,
      "year_start": 1987,
      "year_end": 1992,
      "rarity": "LEGENDARY",
      "confidence": 0.44
    },
    {
      "rank": 2,
      "class_name": "ferrari_288_gto",
      "make": "Ferrari",
      "model": "288 GTO",
      "generation": null,
      "year_start": 1984,
      "year_end": 1987,
      "rarity": "LEGENDARY",
      "confidence": 0.31
    }
  ],
  "top1": null,
  "rejected": true,
  "rejection_reason": "low_confidence",
  "inference_ms": 138.2,
  "engine_version": "0.2.0",
  "model_version": "5.13.0",
  "taxonomy_version": "5.13.0-897"
}
```

Rejection reasons:

| Reason | Trigger |
|--------|---------|
| `not_a_car` | Background class wins, or top softmax < 0.05 |
| `low_confidence` | Top confidence below the per-rarity threshold (COMMON 0.4 … LEGENDARY 0.7) |
| `ambiguous` | Top-2 are a known confusion pair (51 pairs) with margin below the pair threshold (default 0.08) |

---

## GET /health

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok", "model_loaded": true }
```

`status` is `"no_model"` and `model_loaded` is `false` if the server started without a loaded model.

---

## GET /version

```bash
curl http://localhost:8000/version
```

```json
{
  "engine_version": "0.2.0",
  "model_version": "5.13.0",
  "taxonomy_version": "5.13.0-897"
}
```

---

## GET /metadata

Details of the loaded model release.

```bash
curl http://localhost:8000/metadata
```

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

`num_classes` counts all output classes including background (897); `num_makes` counts real makes only (76). If no model is loaded, `model_loaded` is `false` and the other fields are `null`.

---

## Errors

All errors return a structured payload:

```json
{
  "error": {
    "code": "invalid_image",
    "message": "Image could not be decoded"
  }
}
```

| Code | HTTP status | Meaning |
|------|-------------|---------|
| `unauthorized` | 401 | Missing or invalid `X-API-Key` header (when auth is enabled) |
| `model_not_loaded` | 503 | Server started but model is not loaded |
| `invalid_image` | 422 | Upload accepted but the image could not be decoded |
| `bad_request` | 400 | Missing/empty `image` field, unsupported content type, or malformed request |
| `bad_request` | 413 | Upload exceeds the 10 MB limit |
| `inference_failed` | 500 | Internal inference error |

Out-of-range `top_k` values are clamped to 1–20 rather than rejected.

A rejected image is **not** an error — see the rejected-image example above.

---

## SDK Contract Parity

The Python and Android SDKs return the same logical shape:

```python
# Python
engine = AutoVision("models/v5.13.0")
result = engine.classify("photo.jpg", top_k=5)
# result.predictions, result.top1 (None if rejected), result.rejected,
# result.rejection_reason, result.inference_ms,
# result.engine_version, result.model_version, result.taxonomy_version
```

```kotlin
// Kotlin — same fields, camelCase
val autoVision = AutoVision(context, "car_classifier.tflite")
val result: ClassificationResult = autoVision.classify(bitmap, topK = 5)
// result.predictions, result.top1 (null if rejected), result.rejected,
// result.rejectionReason, result.inferenceMs,
// result.engineVersion, result.modelVersion, result.taxonomyVersion
```
