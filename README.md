# AutoVision

**Identify car make, model & generation from a single photo — 896 model-generations across 76 makes.**

AutoVision is a production-ready car recognition engine powered by EfficientNet-V2-S. It delivers generation-level granularity — distinguishing not just "BMW 3 Series" but "BMW 3 Series E90 (2005-2011)" from "BMW 3 Series F30 (2012-2018)" — and ships as an embeddable SDK for on-device inference or a self-hosted REST API.

## Features

- **897 classes** — 896 car model-generations + 1 background/non-car class
- **76 makes** — from Acura to Volvo, including BYD, Cupra, Dacia, Koenigsegg, Lada, Lucid, Polestar, and Rivian
- **Generation-level ID** — tells apart facelifts and redesigns
- **On-device inference** — TFLite FP16 (44 MB) for mobile/edge, Core ML for iOS, ONNX for server
- **< 50 ms latency** — on modern Android devices (GPU delegate)
- **Year range output** — each prediction includes production years
- **Rarity classification** — Common, Uncommon, Rare, Ultra Rare, Epic, Legendary
- **Rejection gating** — non-car, low-confidence, and ambiguous inputs are rejected explicitly, not guessed
- **Calibrated confidence** — ECE 0.049; scores are usable as probabilities
- **Versioned responses** — every result carries `engine_version`, `model_version`, and `taxonomy_version`

**Accuracy: 93.85% top-1 / 97.88% top-5** on a clean held-out validation set (21,381 images, 897 classes). See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for full details and limitations.

> **New integrator?** The step-by-step zero-to-first-classification guide (weights download + verification, Python SDK, Docker API, Android, production checklist, troubleshooting) is [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Quick Start

### Python SDK

```bash
pip install autovision
```

```python
from autovision import AutoVision

engine = AutoVision("models/v5.13.0")
result = engine.classify("photo.jpg", top_k=5)

if result.rejected:
    print(f"Rejected: {result.rejection_reason}")  # not_a_car | low_confidence | ambiguous
else:
    top = result.top1
    print(f"{top.make} {top.model} ({top.year_start}-{top.year_end}): {top.confidence:.1%}")

for pred in result.predictions:
    print(f"#{pred.rank} {pred.class_name} [{pred.rarity}] {pred.confidence:.1%}")
```

### Android SDK

Drop `AutoVision.kt` + the `.tflite` model into your project:

```kotlin
val autoVision = AutoVision(context, "car_classifier.tflite")
val result = autoVision.classify(bitmap, topK = 5)

if (result.rejected) {
    Log.d("AutoVision", "Rejected: ${result.rejectionReason}")
} else {
    val top = result.top1!!
    Log.d("AutoVision", "${top.make} ${top.model}: ${top.confidence}")
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

Full endpoint reference: [docs/API.md](docs/API.md).

## SDK vs API

| | SDK | API |
|---|---|---|
| **Deployment** | Embedded in your app | Self-hosted server |
| **Latency** | < 50 ms (on-device, GPU) | ~100-200 ms (network + inference) |
| **Privacy** | Images never leave device | Images sent to server |
| **Platforms** | Android, iOS (Core ML), Python | Any (HTTP) |
| **Scaling** | Per-device | Horizontal |
| **Best for** | Mobile apps, edge devices | Web apps, batch processing |

## Model

- **Version**: 5.13.0 (taxonomy `5.13.0-897`)
- **Architecture**: EfficientNet-V2-S
- **Input**: 384 x 384 RGB, ImageNet normalized
- **Output**: 897 class logits (896 model-generations + background)
- **Formats**: TFLite FP16 (44 MB, primary — verified 32/32 top-1 parity vs FP32), Core ML mlpackage FP16 (iOS), ONNX (server)
- **Accuracy**: 93.85% top-1, 97.88% top-5 on the clean held-out validation set
- **Training data**: ~156k curated, deduplicated images

Full details, intended use, and honest limitations: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

> **Download the weights:** [Release v5.13.0](https://github.com/bmoldo/carvision/releases/tag/v5.13.0) — FP16 TFLite (44 MB), Core ML package, metadata, and SHA-256 checksums. Drop them into `models/v5.13.0/` and you're ready. The weights are dual-licensed like the code: free for non-commercial use, commercial use requires a [license](LICENSE-COMMERCIAL.md).

## Project Structure

```
AutoVision/
├── sdk/                # Drop-in SDKs (Android, Python, JS)
├── api/                # Self-hosted FastAPI server
├── examples/           # Usage examples
├── usecases/           # Worked business scenarios (5 runnable scripts + Android pattern)
├── models/
│   └── v5.13.0/        # Release metadata: manifest, class mapping, confusion pairs
└── docs/
    ├── GETTING_STARTED.md  # Zero-to-first-classification guide
    ├── API.md          # REST API reference
    ├── MODEL_CARD.md   # Model card: capabilities, limitations, evaluation
    ├── FEATURES.md     # Feature overview
    └── DEPLOYMENT.md   # Deployment guide
```

## Use Cases

Six worked scenarios live in [`usecases/`](usecases/README.md): marketplace listing autofill, dealer inventory batch tagging, insurance claims photo intake, ANPR/parking enrichment, and fleet yard audits — each with a runnable Python script against the real contract — plus a mobile car-spotting integration pattern in Kotlin. Every scenario's README covers which integration surface to use and how to handle rejections in that context.

## Pricing

Personal, research, and noncommercial use is **free**. Commercial use is licensed with published, self-serve pricing — no "contact sales" wall for standard tiers. Annual billing gets 2 months free.

| Tier | Price | Covers |
|---|---|---|
| **Evaluation** | Free (30 days) | Full commercial evaluation, non-production |
| **Self-Hosted API — Starter** | **$99 / month** | 1 production instance, up to 100k images/month |
| **Self-Hosted API — Business** | **$299 / month** | Up to 3 instances, 500k images/month, priority support |
| **On-Device SDK — Indie** | **$1,990 / year** | 1 app, unlimited devices, companies under $1M revenue |
| **On-Device SDK — Standard** | **$4,990 / year** | 1 app, unlimited devices, no revenue cap, day-one model releases |
| **Enterprise / OEM** | Contact us | Air-gapped, OEM/redistribution, custom classes, SLAs |

Every paid license includes a **license-continuity clause**: if AutoVision ceases operations, your license converts to a perpetual license for the model version you deployed. Full terms: [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) · Website: [bmoldo.github.io/carvision](https://bmoldo.github.io/carvision/) · Contact: **bogdanmoldovan29@gmail.com**

## License

AutoVision is **source-available** under a dual license:

- **Personal, research, and noncommercial use** — free under the [PolyForm Noncommercial License 1.0.0](LICENSE)
- **Commercial use** (revenue-generating products, or internal use at a for-profit company) — requires a paid license, see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) and the [pricing table](#pricing) above

This is not an OSI-approved open-source license. Model weights are covered by the same dual terms as the code. Commercial inquiries: **bogdanmoldovan29@gmail.com**.
