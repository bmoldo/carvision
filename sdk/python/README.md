# AutoVision Python SDK

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

# Initialize with model path
engine = AutoVision(
    model_path="models/car_classifier.tflite",
    class_mapping_path="models/class_mapping.json",  # optional if next to model
    temperature=1.0,
)

# Classify a single image
results = engine.classify("photo.jpg", top_k=5)

for pred in results:
    print(f"{pred.make} {pred.model} ({pred.year_start}-{pred.year_end})")
    print(f"  Confidence: {pred.confidence:.1%}")
    print(f"  Generation: {pred.generation}")
    print(f"  Rarity: {pred.rarity}")

# Batch classification
all_results = engine.classify_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

## Model Files

Place your model and class mapping in the `models/` directory:

- `car_classifier.tflite` (83 MB, float32) — or `.onnx` equivalent
- `class_mapping.json` — class index to make/model/year metadata

## Preprocessing

The SDK handles all preprocessing automatically:

1. EXIF rotation correction
2. Center crop to square
3. Resize to 384x384
4. ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Requirements

- Python 3.9+
- NumPy, Pillow
- One of: `tflite-runtime`, `tensorflow`, or `onnxruntime`
