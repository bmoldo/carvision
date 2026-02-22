# Deployment Guide

## Option 1: On-Device (Mobile SDK)

Best for: mobile apps requiring offline inference and low latency.

### Android

1. Add `AutoVision.kt` to your project
2. Add TFLite dependencies (see `sdk/android/README.md`)
3. Bundle or download `car_classifier.tflite` + `class_mapping.json`
4. Initialize `AutoVision(context, "model.tflite")` and call `classify(bitmap)`

**Requirements**: Android API 24+, ~83 MB model storage

### iOS

Coming soon. The model is compatible with Core ML via conversion.

## Option 2: Self-Hosted API (Docker)

Best for: web applications, batch processing, centralized inference.

```bash
# Build the container
docker build -t autovision-api -f api/Dockerfile .

# Run with model volume
docker run -d \
  --name autovision \
  -p 8000:8000 \
  -v /path/to/models:/app/models \
  autovision-api
```

### Scaling

- **Horizontal**: Run multiple containers behind a load balancer
- **GPU**: Mount NVIDIA runtime for ONNX GPU inference
- **Caching**: Add Redis for duplicate image detection

### Resource Requirements

| | CPU Only | GPU |
|---|---|---|
| RAM | 512 MB | 512 MB + GPU VRAM |
| Model Size | 83 MB | 83 MB |
| Latency | ~200-400ms | ~50-100ms |
| Throughput | ~3-5 req/s | ~10-20 req/s |

## Option 3: Python SDK (Embedded)

Best for: data pipelines, Jupyter notebooks, batch analysis.

```bash
pip install autovision[tflite]  # or autovision[onnx] for GPU
```

```python
from autovision import AutoVision

engine = AutoVision("path/to/model.tflite")
results = engine.classify("car.jpg")
```

## Option 4: Edge Devices

The TFLite model runs on:
- Raspberry Pi 4+ (ARM, ~500ms inference)
- Jetson Nano/Xavier (GPU delegate, ~50ms)
- Coral Edge TPU (requires INT8 quantized model — not yet available)

## Model Distribution

Models are not included in this repo (83 MB). Distribution options:

1. **GitHub Releases**: Attach model files to release tags
2. **S3/GCS bucket**: Host privately with signed URLs
3. **App bundling**: Include in Android APK assets or download on first launch
