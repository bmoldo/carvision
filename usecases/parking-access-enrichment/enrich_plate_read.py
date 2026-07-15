#!/usr/bin/env python3
"""ANPR enrichment: cross-check a plate read against what the lane camera
actually sees, using the AutoVision REST API.

The plate decision comes from your existing ANPR system -- AutoVision does
NOT read plates (no OCR). This script adds a vehicle-attribute check on top:
if the plate is registered to a VW Golf but the camera sees an SUV, raise a
mismatch alert for the attendant. Complements plate recognition; never
replaces it.

Verdicts:
    PASS             prediction agrees with the registered vehicle
    MISMATCH_ALERT   confident prediction contradicts the registration
                     -> attendant review (never automatic enforcement)
    UNVERIFIED       rejected frame / low confidence / plate not in registry
                     -> plate decision stands, event is logged

Also prints a rarity hint for VIP/valet flows when the predicted vehicle is
EPIC or LEGENDARY.

Integration surface: REST API (POST /classify with X-API-Key auth).

Environment:
    AUTOVISION_API_URL   Base URL (default http://localhost:8000)
    AUTOVISION_API_KEY   API key, required if the server has auth enabled

Usage:
    python3 enrich_plate_read.py lane_still.jpg --plate "B-123-XYZ"
    python3 enrich_plate_read.py lane_still.jpg --plate "B-123-XYZ" \
        --registry-json registry.json

See ../../docs/GETTING_STARTED.md for server setup.
"""

import argparse
import json
import mimetypes
import os
import sys

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("This script needs the 'requests' package: pip install requests")

DEFAULT_API_URL = "http://localhost:8000"

# Only alert on CONFIDENT contradictions -- a high-throughput lane cannot
# afford false alarms, and the enrichment must never make the lane less
# reliable than plain ANPR. Confidence is calibrated (ECE 0.049).
ALERT_CONFIDENCE = 0.80

# Rarity tiers that suggest the VIP/valet flow at sites that have one.
VIP_RARITIES = {"EPIC", "LEGENDARY"}

# Demo registration records (plate -> vehicle on file). In production this
# is your registration DB; supply an export with --registry-json.
DEMO_REGISTRY = {
    "B-123-XYZ": {"make": "Volkswagen", "model": "Golf", "generation": "Mk7"},
    "CJ-45-ABC": {"make": "BMW", "model": "3 Series", "generation": "F30"},
    "TM-99-GTO": {"make": "Ferrari", "model": "288 GTO", "generation": None},
}


def norm(s):
    """Loose text normalization: '3 Series' == '3-series' == '3series'."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def classify_frame(api_url, api_key, image_path, top_k=3):
    """POST the lane still to /classify; exit with guidance on hard failures."""
    if not os.path.isfile(image_path):
        sys.exit(f"Lane still not found: {image_path}")

    headers = {"X-API-Key": api_key} if api_key else {}
    content_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    try:
        with open(image_path, "rb") as fh:
            resp = requests.post(
                f"{api_url}/classify",
                headers=headers,
                files={"image": (os.path.basename(image_path), fh, content_type)},
                data={"top_k": str(top_k)},
                timeout=15,  # lane decisions are time-sensitive
            )
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Cannot reach the AutoVision API at {api_url}. Is the server "
            "running? See docs/GETTING_STARTED.md, Path B."
        )

    if resp.status_code == 401:
        sys.exit("401 unauthorized: set AUTOVISION_API_KEY (see docs/API.md).")
    if resp.status_code == 503:
        sys.exit("503 model_not_loaded: server has no weights loaded "
                 "(docs/GETTING_STARTED.md, section 2).")
    if resp.status_code == 422:
        return None  # frame undecodable -> UNVERIFIED
    if resp.status_code != 200:
        sys.exit(f"API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def vehicle_matches(pred, registered, check_generation):
    """Does a prediction dict agree with the registered vehicle record?"""
    if norm(pred["make"]) != norm(registered["make"]):
        return False
    if norm(pred["model"]) != norm(registered["model"]):
        return False
    if check_generation and registered.get("generation"):
        return norm(pred["generation"]) == norm(registered["generation"])
    return True


def enrich(result, registered, check_generation):
    """Map the classify response + registration record to a lane verdict."""
    if result is None:
        return "UNVERIFIED", "frame could not be decoded"

    if result["rejected"]:
        reason = result["rejection_reason"]
        if reason == "ambiguous":
            # Confusion pairs are trim/facelift siblings -- if either
            # candidate matches the registration, that's a PASS.
            candidates = result["predictions"][:2]
            if registered and any(
                vehicle_matches(p, registered, check_generation)
                for p in candidates
            ):
                return "PASS", "ambiguous pair, but a candidate matches registration"
            return "UNVERIFIED", "ambiguous between confusion-pair candidates"
        # not_a_car: usually a bad frame (barrier arm, glare, pedestrian).
        # low_confidence: night/rain/motion blur. Either way: don't alert.
        return "UNVERIFIED", f"engine rejected the frame ({reason})"

    top1 = result["top1"]  # contract: non-null when not rejected
    seen = "{} {} {}".format(
        top1["make"], top1["model"], top1["generation"] or ""
    ).strip()

    if registered is None:
        return "UNVERIFIED", (f"camera sees {seen} "
                              f"({top1['confidence']:.1%}); plate not in registry")

    if vehicle_matches(top1, registered, check_generation):
        return "PASS", f"camera sees {seen} ({top1['confidence']:.1%}) -- matches"

    if top1["confidence"] >= ALERT_CONFIDENCE:
        return "MISMATCH_ALERT", (
            f"plate registered to {registered['make']} {registered['model']}, "
            f"camera sees {seen} ({top1['confidence']:.1%})"
        )
    # Disagreement below the alert bar: log it, don't page anyone.
    return "UNVERIFIED", (f"possible mismatch (camera sees {seen} at only "
                          f"{top1['confidence']:.1%}) -- below alert threshold")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Enrich an ANPR plate read with vehicle attributes and "
        "raise plate/vehicle mismatch alerts (complements, never replaces, "
        "plate recognition)."
    )
    parser.add_argument("image", help="Lane camera still (single dominant vehicle)")
    parser.add_argument("--plate", required=True, help="Plate text from your ANPR system")
    parser.add_argument("--registry-json",
                        help="JSON file {plate: {make, model, generation}}; "
                             "defaults to a small built-in demo registry")
    parser.add_argument("--check-generation", action="store_true",
                        help="Also require the generation to match "
                             "(only if your records store it)")
    args = parser.parse_args(argv)

    api_url = os.environ.get("AUTOVISION_API_URL", DEFAULT_API_URL).rstrip("/")
    api_key = os.environ.get("AUTOVISION_API_KEY", "")

    if args.registry_json:
        try:
            with open(args.registry_json) as fh:
                registry = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"Could not read registry file {args.registry_json}: {exc}")
    else:
        registry = DEMO_REGISTRY
        print("(using built-in demo registry -- supply --registry-json in production)")

    registered = registry.get(args.plate)
    result = classify_frame(api_url, api_key, args.image)
    verdict, detail = enrich(result, registered, args.check_generation)

    reg_desc = (
        f"{registered['make']} {registered['model']}"
        + (f" {registered['generation']}" if registered and registered.get("generation") else "")
        if registered else "NOT FOUND in registry"
    )
    print(f"Plate {args.plate}: registered vehicle: {reg_desc}")
    print(f"Verdict: {verdict} -- {detail}")

    actions = {
        "PASS": "Normal entry.",
        "MISMATCH_ALERT": "Route to attendant screen / hold barrier per site "
                          "policy. Human decides -- body kits, re-registered "
                          "vehicles and stale records are common benign causes.",
        "UNVERIFIED": "Plate decision stands; event logged. The enrichment is "
                      "additive and must never make the lane less reliable "
                      "than plain ANPR.",
    }
    print(f"Action: {actions[verdict]}")

    # VIP/valet hint: rarity travels with every prediction.
    if result is not None and result["predictions"]:
        top = result["predictions"][0]
        if top["rarity"] in VIP_RARITIES:
            print(f"VIP hint: predicted vehicle is {top['rarity']} "
                  f"({top['make']} {top['model']}) -- consider the "
                  "valet/covered-parking flow.")

    if result is not None:
        print(f"(model {result['model_version']}, "
              f"taxonomy {result['taxonomy_version']}, "
              f"inference {result['inference_ms']:.0f} ms)")


if __name__ == "__main__":
    main()
