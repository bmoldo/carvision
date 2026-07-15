"""Basic image classification example."""

import sys
from pathlib import Path

# Add SDK to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))

from autovision import AutoVision


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_image.py <image_path> [model_dir]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "models/v5.13.0"

    engine = AutoVision(model_dir)
    result = engine.classify(image_path, top_k=5)

    print(f"\nResults for: {image_path}")
    print(f"Engine {result.engine_version} | model {result.model_version} | "
          f"taxonomy {result.taxonomy_version} | {result.inference_ms:.0f}ms\n")

    if result.rejected:
        print(f"  REJECTED ({result.rejection_reason})")
        if result.rejection_reason == "not_a_car":
            print("  The image does not appear to contain a car.")
        elif result.rejection_reason == "low_confidence":
            print("  The model was not confident enough in its top prediction.")
        elif result.rejection_reason == "ambiguous":
            print("  Two visually similar models are too close to call.")
        print("\n  Candidates:")
    else:
        top = result.top1
        print(f"  Top match: {top.make} {top.model} ({top.confidence:.1%})\n")

    for pred in result.predictions:
        years = ""
        if pred.year_start and pred.year_end:
            years = f" ({pred.year_start}-{pred.year_end})"
        gen = f" {pred.generation}" if pred.generation else ""
        print(f"  {pred.rank}. {pred.make} {pred.model}{gen}{years}")
        print(f"     Confidence: {pred.confidence:.1%}  |  Rarity: {pred.rarity}")


if __name__ == "__main__":
    main()
