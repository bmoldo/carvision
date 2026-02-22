# AutoVision Features

## Classification Capabilities

- **667 classes**: 666 car model-generations + 1 background/non-car class
- **60 makes**: Acura, Alfa Romeo, Aston Martin, Audi, Bentley, BMW, Bugatti, Buick, Cadillac, Chevrolet, Chrysler, Dodge, Ferrari, Fiat, Ford, Genesis, GMC, Honda, Hyundai, Infiniti, Jaguar, Jeep, Kia, Lamborghini, Land Rover, Lexus, Lincoln, Lotus, Maserati, Mazda, McLaren, Mercedes-Benz, Mini, Mitsubishi, Nissan, Pagani, Peugeot, Pontiac, Porsche, Ram, Renault, Rolls-Royce, Saab, Scion, Seat, Skoda, Smart, Subaru, Suzuki, Tesla, Toyota, Volkswagen, Volvo, and more
- **Generation-level granularity**: Distinguishes facelifts and redesigns (e.g., BMW 3 Series E46 vs F30 vs G20)
- **Year ranges**: Each class includes production year range

## Model Architecture

- **Backbone**: EfficientNet-V2-S
- **Input**: 384 x 384 RGB, ImageNet normalization
- **Output**: 666 class logits (+ background)
- **Format**: TFLite float32 (83 MB) or ONNX
- **Quantization**: Float32 only (dynamic range quantization degrades accuracy significantly)

## Accuracy

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | ~79% (validation) |
| Top-3 Accuracy | ~91% (validation) |
| Top-5 Accuracy | ~94% (validation) |

> Accuracy numbers from v4 model. Will be updated after v5 training.

## Rarity Classification

Each car class is tagged with a rarity level:

| Rarity | Description |
|--------|-------------|
| COMMON | Frequently seen on roads |
| UNCOMMON | Regular but less frequent |
| RARE | Uncommon to encounter |
| ULTRA_RARE | Very rare, limited production |
| LEGENDARY | Exotic supercars, classics |

## Preprocessing Pipeline

All SDKs implement identical preprocessing:

1. **EXIF rotation** — correct camera sensor orientation
2. **Center crop** — square crop using shorter dimension
3. **Resize** — bilinear interpolation to 384x384
4. **Normalize** — ImageNet mean/std normalization

## Background Rejection

The model includes a dedicated background class for non-car images. Combined with OOD (out-of-distribution) detection thresholds, the engine can reliably reject:

- Non-vehicle images
- Partial/obscured vehicles
- Artwork or illustrations
- Screenshots of cars (via separate screen detection)

## Performance

| Platform | Latency | Notes |
|----------|---------|-------|
| Android (GPU) | ~30-50ms | With TFLite GPU delegate |
| Android (CPU) | ~150-300ms | 4-thread inference |
| Python (CPU) | ~200-400ms | TFLite runtime |
| Python (GPU) | ~50-100ms | ONNX with CUDA |
