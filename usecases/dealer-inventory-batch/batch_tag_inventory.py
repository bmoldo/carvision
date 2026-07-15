#!/usr/bin/env python3
"""Dealer / auction inventory batch tagging with the AutoVision Python SDK.

Classifies every photo in a directory, writes:
  * an inventory CSV  (file, class, make, model, generation, years,
    confidence, rejected, rejection_reason)
  * a manual-review CSV for rejected photos (low_confidence / ambiguous /
    not_a_car), with the top candidate pre-filled so a human tagger confirms
    rather than starts from scratch
and prints a per-class inventory rollup at the end.

Integration surface: Python SDK, on the machine that holds the photos
(no server, images never leave the box).

Usage:
    python3 batch_tag_inventory.py /path/to/photos --model-dir models/v5.13.0

See ../../docs/GETTING_STARTED.md (Path A) for SDK + weights setup.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# The SDK is the only non-stdlib dependency. Fail with actionable guidance
# rather than a bare ImportError traceback.
try:
    from autovision import AutoVision
    from autovision.errors import (
        AutoVisionError,
        InvalidImageError,
        MappingMismatchError,
        ModelLoadError,
    )
except ImportError:
    sys.exit(
        "The 'autovision' SDK is not installed.\n"
        "From the repo root:  pip install -e \"sdk/python[tflite]\"\n"
        "See docs/GETTING_STARTED.md, Path A."
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

CSV_COLUMNS = [
    "file",
    "class_name",
    "make",
    "model",
    "generation",
    "year_start",
    "year_end",
    "confidence",
    "rejected",
    "rejection_reason",
    "model_version",
    "taxonomy_version",
]


def load_engine(model_dir):
    """Construct the engine with clear, non-traceback failure messages."""
    try:
        return AutoVision(model_dir)
    except ModelLoadError as exc:
        sys.exit(
            f"Could not load the model from '{model_dir}': {exc}\n"
            "Did you download car_classifier.tflite from the v5.13.0 release "
            "and verify SHA256SUMS? See docs/GETTING_STARTED.md, section 2."
        )
    except MappingMismatchError as exc:
        sys.exit(
            f"Model release files are inconsistent: {exc}\n"
            "Re-download ALL files from the same release tag."
        )


def find_photos(photo_dir):
    root = Path(photo_dir)
    if not root.is_dir():
        sys.exit(f"Not a directory: {photo_dir}")
    photos = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not photos:
        sys.exit(f"No images (jpg/jpeg/png/webp) found under {photo_dir}")
    return photos


def result_row(photo, result):
    """Flatten one ClassificationResult into a CSV row.

    Contract reminder: when result.rejected is True, result.top1 is None but
    result.predictions still holds the top-k candidates -- we surface the
    best candidate's fields so review is a confirm/correct task.
    """
    best = result.top1 if result.top1 is not None else (
        result.predictions[0] if result.predictions else None
    )
    return {
        "file": str(photo),
        "class_name": best.class_name if best else "",
        "make": best.make if best else "",
        "model": best.model if best else "",
        "generation": (best.generation or "") if best else "",
        "year_start": best.year_start if best else "",
        "year_end": best.year_end if best else "",
        "confidence": f"{best.confidence:.4f}" if best else "",
        "rejected": result.rejected,
        "rejection_reason": result.rejection_reason or "",
        # Pin versions per row: tags stay auditable across model upgrades.
        "model_version": result.model_version,
        "taxonomy_version": result.taxonomy_version,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch-tag a directory of vehicle intake photos to CSV "
        "using the AutoVision Python SDK."
    )
    parser.add_argument("photo_dir", help="Directory of intake photos (searched recursively)")
    parser.add_argument("--model-dir", default="models/v5.13.0",
                        help="Model release directory (default: models/v5.13.0)")
    parser.add_argument("--out", default="inventory.csv",
                        help="Inventory CSV path (default: inventory.csv)")
    parser.add_argument("--review-out", default="review_queue.csv",
                        help="Manual-review CSV for rejected photos "
                             "(default: review_queue.csv)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Candidates kept per photo (default 3)")
    args = parser.parse_args(argv)

    engine = load_engine(args.model_dir)
    photos = find_photos(args.photo_dir)
    print(f"Classifying {len(photos)} photos with model "
          f"{engine.model_version} (taxonomy {engine.taxonomy_version})...")

    # classify_batch returns one ClassificationResult per input path, in
    # order. Undecodable files raise InvalidImageError, so run in chunks of
    # one file when robustness to corrupt files matters -- here we pre-filter
    # per file and keep going.
    rows, review_rows = [], []
    rollup = Counter()

    for photo in photos:
        try:
            # classify_batch exists for lists; per-file classify keeps one
            # corrupt image from aborting the nightly run.
            [result] = engine.classify_batch([str(photo)], top_k=args.top_k)
        except InvalidImageError:
            print(f"  [skip] undecodable image: {photo}", file=sys.stderr)
            continue
        except AutoVisionError as exc:
            print(f"  [skip] {photo}: {exc}", file=sys.stderr)
            continue

        row = result_row(photo, result)
        rows.append(row)

        if result.rejected:
            # Rejection is routing, not failure: low_confidence and ambiguous
            # photos become the morning review worklist; not_a_car photos are
            # flagged so they aren't counted as vehicles.
            review_rows.append(row)
        else:
            top = result.top1
            gen = f" {top.generation}" if top.generation else ""
            rollup[f"{top.make} {top.model}{gen} "
                   f"({top.year_start}-{top.year_end})"] += 1

    for path, data in ((args.out, rows), (args.review_out, review_rows)):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(data)

    accepted = len(rows) - len(review_rows)
    print(f"\nTagged {len(rows)} photos: {accepted} accepted, "
          f"{len(review_rows)} routed to {args.review_out}")

    if rollup:
        print("\nInventory rollup (accepted photos):")
        for vehicle, count in rollup.most_common():
            print(f"  {count:5d}  {vehicle}")
    print(f"\nInventory CSV: {args.out}")


if __name__ == "__main__":
    main()
