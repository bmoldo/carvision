"""Basic image classification example."""

import sys
from pathlib import Path

# Add SDK to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from autovision import AutoVision


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_image.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/car_classifier.tflite"

    engine = AutoVision(model_path)
    results = engine.classify(image_path, top_k=5)

    print(f"\nResults for: {image_path}\n")
    for i, pred in enumerate(results, 1):
        years = ""
        if pred.year_start and pred.year_end:
            years = f" ({pred.year_start}-{pred.year_end})"
        gen = f" {pred.generation}" if pred.generation else ""
        print(f"  {i}. {pred.make} {pred.model}{gen}{years}")
        print(f"     Confidence: {pred.confidence:.1%}  |  Rarity: {pred.rarity}")


if __name__ == "__main__":
    main()
