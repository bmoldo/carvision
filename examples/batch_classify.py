"""Batch classification example — process a directory of images."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from autovision import AutoVision

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_classify.py <image_dir> [model_path]")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/car_classifier.tflite"

    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory")
        sys.exit(1)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    engine = AutoVision(model_path)

    print(f"Processing {len(images)} images...\n")
    start = time.time()

    for img_path in images:
        results = engine.classify(str(img_path), top_k=3)
        top = results[0]
        years = f" ({top.year_start}-{top.year_end})" if top.year_start else ""
        print(f"  {img_path.name:<40} → {top.make} {top.model}{years}  ({top.confidence:.1%})")

    elapsed = time.time() - start
    avg = elapsed / len(images) * 1000
    print(f"\nDone. {len(images)} images in {elapsed:.1f}s ({avg:.0f}ms avg)")


if __name__ == "__main__":
    main()
