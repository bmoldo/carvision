"""Batch classification example — process a directory of images."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from autovision import AutoVision

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_classify.py <image_dir> [model_dir]")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "models/v5.13.0"

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

    print(f"Loading model from {model_dir}...")
    engine = AutoVision(model_dir)

    print(f"Processing {len(images)} images...\n")
    start = time.time()

    results = engine.classify_batch([str(p) for p in images], top_k=3)

    accepted = 0
    for img_path, result in zip(images, results):
        if result.rejected:
            candidate = ""
            if result.predictions:
                p = result.predictions[0]
                candidate = f"  (best guess: {p.make} {p.model}, {p.confidence:.1%})"
            print(f"  {img_path.name:<40} → REJECTED [{result.rejection_reason}]{candidate}")
            continue

        accepted += 1
        top = result.top1
        years = f" ({top.year_start}-{top.year_end})" if top.year_start else ""
        print(f"  {img_path.name:<40} → {top.make} {top.model}{years}  ({top.confidence:.1%})")

    elapsed = time.time() - start
    avg = elapsed / len(images) * 1000
    print(f"\nDone. {len(images)} images ({accepted} accepted, "
          f"{len(images) - accepted} rejected) in {elapsed:.1f}s ({avg:.0f}ms avg)")


if __name__ == "__main__":
    main()
