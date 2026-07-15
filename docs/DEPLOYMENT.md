# Deployment Guide

## Option 1: On-Device (Mobile SDK)

Best for: mobile apps requiring offline inference and low latency.

### Android

1. Add `AutoVision.kt` to your project
2. Add TFLite dependencies (see `sdk/android/README.md`)
3. Bundle or download `car_classifier.tflite` + `class_mapping.json` (+ `confusion_pairs.json` for ambiguity gating)
4. Initialize `AutoVision(context, "car_classifier.tflite")` and call `classify(bitmap, topK = 5)`

**Requirements**: Android API 24+, ~44 MB model storage (FP16 TFLite)

### iOS

Available. The model ships as a Core ML `mlpackage` (FP16), same 384x384 input and preprocessing contract as the TFLite model.

## Option 2: Self-Hosted API (Docker)

Best for: web applications, batch processing, centralized inference.

```bash
# Build the container
docker build -t autovision-api -f api/Dockerfile .

# Run with model volume (mount the release directory)
docker run -d \
  --name autovision \
  -p 8000:8000 \
  -v /path/to/models:/app/models \
  autovision-api
```

The server loads the release directory (default `/app/models/v5.13.0`, override with `AUTOVISION_MODEL_DIR`) and selects the backend from the weight file present — TFLite if available, otherwise ONNX (use ONNX for GPU inference). Endpoint reference: [API.md](API.md).

### Scaling

- **Horizontal**: Run multiple containers behind a load balancer
- **GPU**: Mount NVIDIA runtime for ONNX GPU inference
- **Caching**: Add Redis for duplicate image detection

### Resource Requirements

| | CPU Only | GPU |
|---|---|---|
| RAM | 512 MB | 512 MB + GPU VRAM |
| Model Size | 44 MB (FP16) | 44 MB (FP16) |
| Latency | ~200-400ms | ~50-100ms |
| Throughput | ~3-5 req/s | ~10-20 req/s |

## Option 3: Python SDK (Embedded)

Best for: data pipelines, Jupyter notebooks, batch analysis.

```bash
pip install autovision[tflite]  # or autovision[onnx] for GPU
```

```python
from autovision import AutoVision

engine = AutoVision("models/v5.13.0")
result = engine.classify("car.jpg", top_k=5)
```

The SDK loads the release directory via its manifest — model file, class mapping, thresholds, and preprocessing settings all come from `model_manifest.json`.

## Option 4: Edge Devices

The TFLite model runs on:
- Raspberry Pi 4+ (ARM, ~500ms inference)
- Jetson Nano/Xavier (GPU delegate, ~50ms)
- Coral Edge TPU (requires INT8 quantized model — not yet available)

## Model Distribution

Model weights are not included in this repo (44 MB FP16 TFLite). Release **metadata** is tracked in the repo using a one-directory-per-release layout:

```
models/
  v5.13.0/
    model_manifest.json     # input size, normalization, thresholds, versions
    class_mapping.json      # 897 classes with make/model/generation/years/rarity
    confusion_pairs.json    # 51 known confusion pairs with margin thresholds
    car_classifier.tflite   # (weights, not tracked)
    model.onnx              # (weights, not tracked)
```

One release directory = one deployable engine version. SDKs and the API load from the manifest, not from hardcoded filenames.

Weight distribution options:

1. **GitHub Releases**: model files attached to release tags, dropped into `models/<release>/`
2. **S3/GCS bucket**: host privately with signed URLs
3. **App bundling**: include in Android APK assets / iOS app bundle, or download on first launch
4. **Commercial licensees**: direct distribution — see [LICENSE-COMMERCIAL.md](../LICENSE-COMMERCIAL.md)
