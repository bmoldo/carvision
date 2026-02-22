# AutoVision

**Identify 667 car makes, models & generations from a single photo.**

AutoVision is a production-ready car recognition engine powered by EfficientNet-V2-S, trained on a curated dataset spanning 60 makes and 666 model-generation classes. It delivers generation-level granularity — distinguishing not just "BMW 3 Series" but "BMW 3 Series E90 (2005-2011)" from "BMW 3 Series F30 (2012-2018)".

## Features

- **667 classes** — 666 car model-generations + background rejection
- **60 makes** — all major manufacturers covered
- **Generation-level ID** — tells apart facelifts and redesigns
- **On-device inference** — TFLite for mobile/edge, ONNX for server
- **< 50ms latency** — on modern Android devices (GPU delegate)
- **Year range output** — each prediction includes production years
- **Rarity classification** — Common, Uncommon, Rare, Ultra Rare, Legendary
- **Background rejection** — OOD detection for non-car images

## Quick Start

### Python SDK

```bash
pip install autovision
```

```python
from autovision import AutoVision

engine = AutoVision("models/car_classifier.tflite")
results = engine.classify("photo.jpg", top_k=5)

for pred in results:
    print(f"{pred.make} {pred.model} ({pred.year_start}-{pred.year_end}): {pred.confidence:.1%}")
```

### Android SDK

Drop `AutoVision.kt` + your `.tflite` model into your project:

```kotlin
val autoVision = AutoVision(context, "car_classifier.tflite")
val results = autoVision.classify(bitmap, topK = 5)

results.forEach { pred ->
    Log.d("AutoVision", "${pred.make} ${pred.model}: ${pred.confidence}")
}
```

### REST API

```bash
docker build -t autovision-api api/
docker run -p 8000:8000 -v ./models:/app/models autovision-api

curl -X POST http://localhost:8000/classify \
  -F "image=@photo.jpg" \
  -F "top_k=5"
```

## SDK vs API

| | SDK | API |
|---|---|---|
| **Deployment** | Embedded in your app | Self-hosted server |
| **Latency** | < 50ms (on-device) | ~100-200ms (network + inference) |
| **Privacy** | Images never leave device | Images sent to server |
| **Platforms** | Android, Python | Any (HTTP) |
| **Scaling** | Per-device | Horizontal |
| **Best for** | Mobile apps, edge devices | Web apps, batch processing |

## Model

- **Architecture**: EfficientNet-V2-S
- **Input**: 384 x 384 RGB, ImageNet normalized
- **Output**: 666 class logits + background
- **Format**: TFLite (float32, 83 MB) or ONNX
- **Accuracy**: ~79% top-1, ~91% top-3 on validation set

> Model files are not included in this repo. Download from releases or contact for access.

## Project Structure

```
AutoVision/
├── sdk/            # Drop-in SDKs (Android, Python, JS)
├── api/            # Self-hosted FastAPI server
├── examples/       # Usage examples
├── models/         # Model files (gitignored)
└── docs/           # Documentation
```

## License

Proprietary. See [LICENSE](LICENSE) for details.
