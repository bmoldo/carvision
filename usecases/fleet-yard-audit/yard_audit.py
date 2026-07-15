#!/usr/bin/env python3
"""Nightly fleet / rental yard audit with the AutoVision Python SDK.

Classifies every overnight camera still in a directory, tallies accepted
identifications per model-generation class, and reconciles the tally against
the expected fleet database:

  * UNEXPECTED -- models seen in the yard but not in the fleet definition
  * MISSING / SHORT -- expected models seen fewer times than expected
  * UNVERIFIED -- stills the engine rejected (never guessed into the tally)

Exits non-zero when discrepancies are found so a cron wrapper can page.

Throughput note: for large multi-branch runs, use the ONNX export with
onnxruntime-gpu -- the SDK auto-selects the ONNX backend for `.onnx` weights
and prefers CUDAExecutionProvider. Just point --model-dir at a release
directory containing the .onnx file; no code change needed.

Integration surface: Python SDK (on-prem batch; images stay on your network).

Usage:
    python3 yard_audit.py /path/to/yard-stills --model-dir models/v5.13.0
    python3 yard_audit.py /path/to/yard-stills --fleet-json branch_042.json

See ../../docs/GETTING_STARTED.md (Path A) for SDK + weights setup.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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
        "(or [onnx] + onnxruntime-gpu on a GPU server)\n"
        "See docs/GETTING_STARTED.md, Path A."
    )

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Demo fleet definition: expected count per class id (class ids come from
# models/v5.13.0/class_mapping.json). In production, export this from your
# fleet DB and pass --fleet-json.
DEMO_FLEET = {
    "volkswagen_golf_mk8": 12,
    "skoda_octavia_mk4": 8,
    "tesla_model_3": 4,
}


def load_engine(model_dir):
    """Construct the engine with actionable failure messages."""
    try:
        return AutoVision(model_dir)
    except ModelLoadError as exc:
        sys.exit(
            f"Could not load the model from '{model_dir}': {exc}\n"
            "Download the weights per docs/GETTING_STARTED.md, section 2, and "
            "verify SHA256SUMS."
        )
    except MappingMismatchError as exc:
        sys.exit(
            f"Model release files are inconsistent: {exc}\n"
            "Re-download ALL files from the same release tag."
        )


def audit(engine, stills, fleet):
    """Classify all stills and tally per-class sightings.

    Returns (tally Counter, unverified count, skipped count).

    Rejection handling keeps the count honest:
      * not_a_car        -> excluded silently (empty bay, pedestrian, gear)
      * low_confidence   -> counted as unverified, never guessed
      * ambiguous        -> counted toward an EXPECTED fleet model if either
                            confusion-pair candidate is one (pair-mates are
                            trim/facelift siblings; fleet counts don't care),
                            otherwise unverified
    """
    tally = Counter()
    unverified = 0
    skipped = 0

    for still in stills:
        try:
            # Per-file call so one corrupt frame can't abort the nightly run;
            # classify_batch is the same loop without that isolation.
            [result] = engine.classify_batch([str(still)], top_k=2)
        except InvalidImageError:
            skipped += 1
            continue
        except AutoVisionError as exc:
            print(f"  [skip] {still}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if not result.rejected:
            tally[result.top1.class_name] += 1
        elif result.rejection_reason == "ambiguous":
            # Candidates are still returned on rejection (contract).
            match = next(
                (p.class_name for p in result.predictions[:2]
                 if p.class_name in fleet),
                None,
            )
            if match is not None:
                tally[match] += 1
            else:
                unverified += 1
        elif result.rejection_reason == "low_confidence":
            unverified += 1
        # not_a_car: excluded silently -- the gate doing its job.

    return tally, unverified, skipped


def reconcile(tally, fleet):
    """Diff seen counts against the expected fleet definition."""
    unexpected = {cls: n for cls, n in tally.items() if cls not in fleet}
    short = {
        cls: (expected, tally.get(cls, 0))
        for cls, expected in sorted(fleet.items())
        if tally.get(cls, 0) < expected
    }
    return unexpected, short


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reconcile yard camera stills against the expected fleet "
        "(nightly batch audit)."
    )
    parser.add_argument("stills_dir",
                        help="Directory of yard camera stills "
                             "(one dominant vehicle per still)")
    parser.add_argument("--model-dir", default="models/v5.13.0",
                        help="Model release directory; point at an .onnx "
                             "release for GPU throughput (default: models/v5.13.0)")
    parser.add_argument("--fleet-json",
                        help="JSON file {class_id: expected_count}; "
                             "defaults to a built-in demo fleet")
    args = parser.parse_args(argv)

    if args.fleet_json:
        try:
            with open(args.fleet_json) as fh:
                fleet = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not read fleet file {args.fleet_json}: {exc}")
    else:
        fleet = DEMO_FLEET
        print("(using built-in demo fleet -- supply --fleet-json in production)")

    root = Path(args.stills_dir)
    if not root.is_dir():
        sys.exit(f"Not a directory: {args.stills_dir}")
    stills = sorted(p for p in root.rglob("*")
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not stills:
        sys.exit(f"No images (jpg/jpeg/png/webp) found under {args.stills_dir}")

    engine = load_engine(args.model_dir)
    print(f"Auditing {len(stills)} stills with model {engine.model_version} "
          f"(taxonomy {engine.taxonomy_version}, backend {engine.backend})...")

    tally, unverified, skipped = audit(engine, stills, fleet)
    unexpected, short = reconcile(tally, fleet)

    # --- Report -------------------------------------------------------------
    print("\nSeen (accepted identifications, per class):")
    for cls, n in tally.most_common():
        marker = "" if cls in fleet else "   <-- NOT IN FLEET"
        print(f"  {n:4d}  {cls}{marker}")

    if unexpected:
        print("\nUNEXPECTED models in the yard:")
        for cls, n in sorted(unexpected.items()):
            print(f"  {cls}: seen {n}x -- not in the fleet definition "
                  "(wrong-branch return? un-logged transfer?)")

    if short:
        print("\nMISSING / SHORT counts:")
        for cls, (expected, seen) in short.items():
            print(f"  {cls}: expected {expected}, saw {seen}")

    print(f"\nUnverified stills (rejected, excluded from tally): {unverified}")
    if skipped:
        print(f"Skipped (undecodable) stills: {skipped}")
    print("Note: the tally counts stills, not vehicles -- dedupe by "
          "bay/camera in your pipeline. A high unverified rate is usually a "
          "camera/lighting work order, not a model issue.")

    if unexpected or short:
        print("\nRESULT: DISCREPANCIES FOUND -- route to branch manager.")
        sys.exit(1)
    print("\nRESULT: yard matches the expected fleet.")


if __name__ == "__main__":
    main()
