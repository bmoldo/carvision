# Getting Started with AutoVision

This guide takes a new integrator from zero to a first classification, on any of
the three supported surfaces. For worked, end-to-end business scenarios see
[`usecases/`](../usecases/README.md).

Contents:

1. [Choose your integration](#1-choose-your-integration)
2. [Get the model weights](#2-get-the-model-weights)
3. [Path A — Python SDK](#3-path-a--python-sdk)
4. [Path B — Self-hosted Docker API](#4-path-b--self-hosted-docker-api)
5. [Path C — Android SDK](#5-path-c--android-sdk)
6. [Production checklist](#6-production-checklist)
7. [Troubleshooting](#7-troubleshooting)

> **Licensing:** AutoVision is source-available — free for personal, research,
> and noncommercial use ([PolyForm Noncommercial](../LICENSE)); commercial use
> requires a [paid license](../LICENSE-COMMERCIAL.md). The model weights are
> covered by the same dual terms.

---

## 1. Choose your integration

| | On-device SDK (Python / Android / iOS) | Self-hosted REST API |
|---|---|---|
| **Deployment** | Embedded in your app/process | Docker container you run |
| **Latency** | < 50 ms on-device (Android GPU) | ~100–200 ms (network + inference) |
| **Privacy** | Images never leave the device | Images sent to your server |
| **Scaling** | Per-device / per-process | Horizontal behind a load balancer |
| **Best for** | Mobile apps, edge devices, batch scripts colocated with images | Web backends, multi-language stacks, centralized batch processing |

Rules of thumb:

- **Mobile app** (car spotting, consumer camera features) → **Path C** (Android)
  or Core ML (iOS). Offline, private, fastest.
- **Backend service called from several languages/services** → **Path B** (API).
  One container, one contract, HTTP from anywhere.
- **Python pipeline that already has the images on disk** (inventory tagging,
  nightly audits) → **Path A** (Python SDK). No server hop, `classify_batch`
  built in.

All three surfaces return the same logical result shape (see
[docs/API.md](API.md), "SDK Contract Parity"), so you can start with one and
add another later without changing your domain logic.

## 2. Get the model weights

The repository ships the release **metadata** in `models/v5.13.0/`
(`model_manifest.json`, `class_mapping.json`, `confusion_pairs.json`,
`SHA256SUMS`). The **weight binaries** are distributed via the GitHub Release:

**https://github.com/bmoldo/carvision/releases/tag/v5.13.0**

| Asset | Size | Needed for |
|---|---|---|
| `car_classifier.tflite` | 44 MB (FP16) | Python SDK (TFLite backend), API server, Android |
| `car_classifier_coreml_fp16.mlpackage.zip` | — | iOS (Core ML) only |
| `SHA256SUMS` | — | Integrity verification |

Download into the release directory and verify checksums:

```bash
cd AutoVision  # repo root

curl -L -o models/v5.13.0/car_classifier.tflite \
  https://github.com/bmoldo/carvision/releases/download/v5.13.0/car_classifier.tflite

# iOS only:
# curl -L -o models/v5.13.0/car_classifier_coreml_fp16.mlpackage.zip \
#   https://github.com/bmoldo/carvision/releases/download/v5.13.0/car_classifier_coreml_fp16.mlpackage.zip

cd models/v5.13.0
sha256sum -c SHA256SUMS --ignore-missing
# car_classifier.tflite: OK
# class_mapping.json: OK
# model_manifest.json: OK
# confusion_pairs.json: OK
```

Do not skip verification, and never mix files across releases: the SDKs
validate that `class_mapping.json`, `model_manifest.json`, and the model output
size agree, and refuse to start otherwise (see
[MappingMismatchError](#7-troubleshooting)).

## 3. Path A — Python SDK

### 3.1 Install

Install the SDK from the repo (or vendor the `autovision` package directly
into your codebase — it is a plain Python package):

```bash
pip install -e sdk/python
```

Then install **one** inference backend:

| Backend | Install | When to use |
|---|---|---|
| `tflite-runtime` | `pip install -e "sdk/python[tflite]"` | **Recommended.** Small, fast CPU inference on Linux x86_64/ARM. Runs the released `.tflite` weights. |
| `tensorflow` | `pip install -e "sdk/python[tf]"` | Fallback where `tflite-runtime` wheels are unavailable (e.g. Windows, some macOS setups). Heavyweight (~GBs) but runs the same `.tflite` file. |
| `onnxruntime` | `pip install -e "sdk/python[onnx]"` | Server deployments wanting GPU (`CUDAExecutionProvider`) or maximum CPU throughput. Requires the `.onnx` export of the weights (server format — contact for access if not in your release bundle). |

The SDK picks the backend by weight-file extension: `.tflite` tries
`tflite-runtime` first, then `tensorflow`; `.onnx` uses `onnxruntime`.

### 3.2 First classification

```python
from autovision import AutoVision

# Point at the release directory — weights, manifest, class mapping and
# confusion pairs are auto-discovered and cross-validated.
engine = AutoVision("models/v5.13.0")

result = engine.classify("photo.jpg", top_k=5)
```

### 3.3 Reading the result

`classify()` returns a `ClassificationResult`:

- `result.top1` — the accepted best prediction, or **`None` when rejected**.
  Always check `rejected` before dereferencing.
- `result.rejected` / `result.rejection_reason` — rejection is a first-class
  outcome, **not an error**. Reasons: `"not_a_car"`, `"low_confidence"`,
  `"ambiguous"`.
- `result.predictions` — top-k candidates (background class excluded),
  **populated even when rejected**, so you can build "did you mean?" UIs.
- `result.model_version`, `result.taxonomy_version`, `result.engine_version` —
  log these with every stored result (see the
  [production checklist](#6-production-checklist)).

```python
if result.rejected:
    print(f"Rejected: {result.rejection_reason}")
    for p in result.predictions:       # candidates are still available
        print(f"  candidate #{p.rank}: {p.class_name} {p.confidence:.1%}")
else:
    top = result.top1
    print(f"{top.make} {top.model} {top.generation or ''} "
          f"({top.year_start}-{top.year_end})  {top.confidence:.1%}  [{top.rarity}]")
```

### 3.4 Handling each rejection reason

Map each reason to a distinct user experience — never a generic "error":

| `rejection_reason` | What it means | What to show your end user |
|---|---|---|
| `not_a_car` | The background class won, or the top score was near-zero. The photo most likely doesn't show a car (or shows an interior, a toy, a truck/motorcycle — out of scope). | "We couldn't find a car in this photo. Point the camera at the vehicle's exterior and try again." Do **not** show candidates — they are noise. |
| `low_confidence` | A car is probably there, but the top score is below the per-rarity threshold (COMMON 0.4 … LEGENDARY 0.7). Often: distance, occlusion, night shots, extreme angles. | "We couldn't identify this car confidently. Get closer, improve lighting, or shoot from a front-three-quarter angle." Optionally show `predictions` as clearly-labeled *unverified* guesses. |
| `ambiguous` | The top-2 candidates are a known confusion pair (51 documented pairs, e.g. Golf Mk7 vs Golf GTI Mk7) with too small a margin. The car is one of them. | "This could be one of the following — please pick." Show the top 2–3 `predictions` as a choice list. This converts a model limitation into a one-tap user action. |

## 4. Path B — Self-hosted Docker API

### 4.1 Build

Build from the **repo root** (the image copies in `sdk/python`):

```bash
docker build -t autovision-api -f api/Dockerfile .
```

### 4.2 Generate an API key

```bash
python3 -c "import secrets; print('av_' + secrets.token_urlsafe(32))"
# e.g. av_k3XErc9dR0y...
```

### 4.3 Run

Mount your models directory (with the downloaded weights) and set the accepted
keys:

```bash
docker run -d --name autovision \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -e AUTOVISION_API_KEYS="av_yourgeneratedkey" \
  autovision-api
```

- The server loads `/app/models/v5.13.0` by default
  (override with `AUTOVISION_MODEL_DIR`).
- **If `AUTOVISION_API_KEYS` is unset, auth is disabled** and a startup warning
  is logged — acceptable for local development only.
- For secrets hygiene, prefer `AUTOVISION_API_KEYS_FILE=/run/secrets/...` with
  Docker/Kubernetes secrets.

### 4.4 First request

```bash
export AV_KEY="av_yourgeneratedkey"

# Liveness — never requires a key
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}

# Pinnable versions
curl -H "X-API-Key: $AV_KEY" http://localhost:8000/version
# {"engine_version": "0.2.0", "model_version": "5.13.0", "taxonomy_version": "5.13.0-897"}

# Loaded-model details
curl -H "X-API-Key: $AV_KEY" http://localhost:8000/metadata

# Classify
curl -X POST http://localhost:8000/classify \
  -H "X-API-Key: $AV_KEY" \
  -F "image=@photo.jpg;type=image/jpeg" \
  -F "top_k=5"
```

The JSON response mirrors the SDK result (`predictions`, `top1`, `rejected`,
`rejection_reason`, `inference_ms`, plus the three version fields). A rejected
image still returns **HTTP 200** — see
[section 3.4](#34-handling-each-rejection-reason) for how to handle each
reason. Full reference: [docs/API.md](API.md).

## 5. Path C — Android SDK

The Android SDK is a single drop-in file — no library dependency on AutoVision
itself.

1. Copy [`sdk/android/AutoVision.kt`](../sdk/android/AutoVision.kt) into your
   project (package `dev.autovision`).

2. Add TFLite dependencies to `build.gradle`:

   ```gradle
   dependencies {
       implementation 'org.tensorflow:tensorflow-lite:2.14.0'
       implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
       implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
       implementation 'androidx.exifinterface:exifinterface:1.3.7'
   }
   ```

3. Bundle the model assets in `assets/` — **all from the same release**:

   | Asset | Required |
   |---|---|
   | `car_classifier.tflite` (44 MB) | Yes |
   | `class_mapping.json` | Yes |
   | `model_manifest.json` | Recommended (built-in v5.13.0 defaults otherwise) |
   | `confusion_pairs.json` | Recommended (no ambiguity gating otherwise) |

4. Classify:

   ```kotlin
   val engine = AutoVision(context, "car_classifier.tflite")
   val result = engine.classify(bitmap, topK = 5)

   if (result.rejected) {
       // "not_a_car" | "low_confidence" | "ambiguous" — see section 3.4
       showRejectionUi(result.rejectionReason, result.predictions)
   } else {
       val top = result.top1!!
       showCar("${top.make} ${top.model}", top.confidence)
   }

   engine.close() // when done — releases interpreter + GPU delegate
   ```

GPU delegate is used automatically with CPU fallback. Full API and gating
details: [`sdk/android/README.md`](../sdk/android/README.md).

## 6. Production checklist

Before going live:

- [ ] **API keys set.** `AUTOVISION_API_KEYS` (or `_FILE`) configured; strong
  keys (`av_` + 32 url-safe bytes); at least two keys so you can rotate with
  zero downtime. Never ship keys in client-side code.
- [ ] **TLS via a reverse proxy.** The container speaks plain HTTP; terminate
  TLS at nginx/Caddy/Traefik or your cloud load balancer. Don't expose port
  8000 directly to the internet.
- [ ] **Pin and log versions.** Record `model_version` and `taxonomy_version`
  with every stored classification. Class indices and names are only
  meaningful within a taxonomy version — treat a version change as a schema
  migration, not a hot swap.
- [ ] **Handle `rejected` as UX, not error.** HTTP 200 + `rejected: true` is a
  correct, expected outcome. Wire all three `rejection_reason` values to
  distinct user flows (section 3.4). Alert on *error codes* (5xx), not on
  rejections.
- [ ] **Monitor confidence distributions.** Log `top1.confidence` (and the
  rejection rate per reason) over time. A drifting distribution — falling
  confidence, rising `low_confidence` rate — is your early warning that your
  image mix has moved away from what the model was validated on. Re-evaluate
  on your own distribution before trusting accuracy targets
  (see [MODEL_CARD.md](MODEL_CARD.md)).
- [ ] **Verify checksums in your deploy pipeline**, not just once on a laptop
  (`sha256sum -c SHA256SUMS --ignore-missing`).
- [ ] **Health checks** on `GET /health` (no key needed), so orchestrators and
  load balancers work unmodified.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModelLoadError: No model weights found in models/v5.13.0` | Weights not downloaded — the repo only ships metadata JSONs. | Download `car_classifier.tflite` from the [release](https://github.com/bmoldo/carvision/releases/tag/v5.13.0) into `models/v5.13.0/` and verify checksums (section 2). |
| `ModelLoadError: TFLite backend requires tflite-runtime or tensorflow` | No inference backend installed. | `pip install -e "sdk/python[tflite]"` (Linux) or `[tf]` / `[onnx]` per the matrix in section 3.1. |
| `MappingMismatchError: class_mapping has N entries but the manifest declares num_classes=897` | Mixed files from different releases, or a truncated download. | Re-download **all** files from the same release tag; verify with `SHA256SUMS`. Never hand-edit `class_mapping.json`. |
| API returns `401` `unauthorized` | Missing/wrong `X-API-Key` header, or key not in `AUTOVISION_API_KEYS`. | Send the header exactly as configured (`-H "X-API-Key: $AV_KEY"`); keys are never accepted in the query string. Check container env: `docker exec autovision env \| grep AUTOVISION`. |
| API returns `422` `invalid_image` | Upload accepted but not decodable — corrupt file, HTML error page saved as `.jpg`, HEIC/AVIF, etc. | Send real JPEG/PNG/WebP under 10 MB; set the part's content type (`-F "image=@photo.jpg;type=image/jpeg"`). Convert HEIC before upload. |
| API returns `503` `model_not_loaded` | Server started without weights (check startup logs). | Mount the models volume correctly (`-v "$(pwd)/models:/app/models"`) and confirm `models/v5.13.0/car_classifier.tflite` exists on the host. |
| Slow CPU inference (seconds per image) | Full TensorFlow fallback, cold-start per image, or an undersized container. | Prefer `tflite-runtime` over `tensorflow`; create **one** `AutoVision` instance and reuse it (model load is the expensive part); give the container real CPU quota; for server throughput or GPU, use the ONNX weights with `onnxruntime` / `onnxruntime-gpu`. |
| High memory usage | Full TensorFlow import (~GBs) or one engine instance per request. | Switch to `tflite-runtime` (tiny footprint); construct the engine once per process, not per request; the model itself needs well under 1 GB. |

Still stuck? Commercial licensing and evaluation support:
**bogdanmoldovan29@gmail.com**.
