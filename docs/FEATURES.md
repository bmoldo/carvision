# AutoVision Features

## Classification Capabilities

- **897 classes**: 896 car model-generations + 1 background/non-car class
- **76 makes**: Acura, Alfa Romeo, Alpine, Aston Martin, Audi, Bentley, BMW, Bugatti, Buick, BYD, Cadillac, Chevrolet, Chrysler, Citroen, Cupra, Dacia, Dodge, DS, Ferrari, Fiat, Fisker, Ford, Geely, Genesis, GMC, Great Wall, Haval, Honda, Hyundai, Infiniti, Jaguar, Jeep, Kia, Koenigsegg, KTM, Lada, Lamborghini, Lancia, Land Rover, Lexus, Lincoln, London Taxi, Lotus, Lucid, Maserati, Maxus, Mazda, McLaren, Mercedes-AMG, Mercedes-Benz, MG, Mini, Mitsubishi, Mitsuoka, Nissan, Opel, Pagani, Peugeot, Polestar, Porsche, Ram, Renault, Rivian, Rolls-Royce, Seat, Skoda, Smart, SsangYong, Subaru, Suzuki, Tesla, Toyota, TVR, Vauxhall, Volkswagen, Volvo
- **Generation-level granularity**: Distinguishes facelifts and redesigns (e.g., BMW 3 Series E46 vs F30 vs G20)
- **Year ranges**: Each class includes production year range (class metadata, not per-image estimation)

## Model Architecture

- **Backbone**: EfficientNet-V2-S
- **Input**: 384 x 384 RGB, ImageNet normalization
- **Output**: 897 class logits (896 model-generations + background)
- **Formats**:
  - TFLite FP16, 44 MB (primary; verified 32/32 top-1 parity vs FP32 on sampled validation images)
  - Core ML mlpackage FP16 (iOS)
  - ONNX (server)

## Accuracy

Model v5.13.0, clean held-out validation set (21,381 images, 897 classes):

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 93.85% |
| Top-5 Accuracy | 97.88% |
| Calibration (ECE) | 0.049 |

> Real-world accuracy on uncurated street photos is lower than clean validation accuracy. Rejection thresholds are tuned to favor precision over recall: the engine would rather reject an image than return a wrong answer. See [MODEL_CARD.md](MODEL_CARD.md).

## Rarity Classification

Each car class is tagged with a rarity tier:

| Rarity | Classes | Description |
|--------|---------|-------------|
| COMMON | 150 | Frequently seen on roads |
| UNCOMMON | 415 | Regular but less frequent |
| RARE | 182 | Uncommon to encounter |
| ULTRA_RARE | 66 | Very rare, limited production |
| EPIC | 39 | Exceptional finds (new tier in v5.13) |
| LEGENDARY | 44 | Exotic supercars, classics |

## Rejection Gating

Rejection is a first-class outcome. Every result includes `rejected` and `rejection_reason`; when a result is rejected, `top1` is null but the raw candidate predictions are still returned. Three gates run in order:

1. **Non-car** (`not_a_car`): the background class wins, or the top softmax score is below 0.05.
2. **Low confidence** (`low_confidence`): the top prediction's confidence is below its rarity tier's threshold — COMMON 0.4, UNCOMMON 0.5, RARE 0.6, ULTRA_RARE 0.65, EPIC 0.67, LEGENDARY 0.7. Rarer classes require more confidence.
3. **Ambiguous** (`ambiguous`): the top-2 predictions form one of 51 known confusion pairs (shipped in `models/v5.13.0/confusion_pairs.json`) and the confidence margin between them is below the pair's threshold (default 0.08).

## Preprocessing Pipeline

All SDKs implement identical preprocessing:

1. **EXIF rotation** — correct camera sensor orientation
2. **Center crop** — square crop using shorter dimension
3. **Resize** — bilinear interpolation to 384x384
4. **Normalize** — ImageNet mean/std normalization

## Versioning

Every response carries three independent versions:

| Field | Current | Meaning |
|-------|---------|---------|
| `engine_version` | 0.2.0 | SDK/API code |
| `model_version` | 5.13.0 | Model artifact |
| `taxonomy_version` | 5.13.0-897 | Class mapping / label space |

## Performance

| Platform | Latency | Notes |
|----------|---------|-------|
| Android (GPU) | ~30-50ms | With TFLite GPU delegate |
| Android (CPU) | ~150-300ms | 4-thread inference |
| Python (CPU) | ~200-400ms | TFLite runtime |
| Python (GPU) | ~50-100ms | ONNX with CUDA |
