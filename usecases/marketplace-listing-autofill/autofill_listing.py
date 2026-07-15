#!/usr/bin/env python3
"""Marketplace listing autofill using the AutoVision REST API.

Given one or more seller-uploaded photos, suggest the listing's make / model /
generation / year-range fields. When the engine rejects the top prediction as
`ambiguous` (a known confusion pair), the top-3 candidates are offered as a
pick list instead. When the seller's *claimed* vehicle disagrees with a
confident prediction, the listing is flagged for manual review (fraud /
mislabel signal) -- flagged, never auto-blocked.

Integration surface: REST API (POST /classify with X-API-Key auth).

Environment:
    AUTOVISION_API_URL   Base URL of the API server (default http://localhost:8000)
    AUTOVISION_API_KEY   API key, required if the server has auth enabled

Usage:
    python3 autofill_listing.py front.jpg rear.jpg
    python3 autofill_listing.py front.jpg --claimed-make BMW --claimed-model "5 Series"
    python3 autofill_listing.py front.jpg --json

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
    sys.exit(
        "This script needs the 'requests' package: pip install requests"
    )

DEFAULT_API_URL = "http://localhost:8000"

# Product rule: only autofill silently when the accepted top-1 confidence is
# at least this high. Between the engine's own gating threshold and this value
# we still *suggest*, but mark the suggestion as needing confirmation.
# (Engine confidence is calibrated -- ECE 0.049 -- so this reads as a
# probability.)
AUTOFILL_CONFIDENCE = 0.75

# Only raise a mislabel/fraud flag when the prediction contradicting the
# seller's claim is itself confident. A low-confidence disagreement is noise.
MISMATCH_FLAG_CONFIDENCE = 0.80


def classify_photo(session, api_url, api_key, photo_path, top_k=5):
    """POST one photo to /classify and return the parsed JSON response.

    Raises SystemExit with a clear message on connection / auth / server
    problems so callers in a listing pipeline fail loudly, not silently.
    """
    if not os.path.isfile(photo_path):
        sys.exit(f"Photo not found: {photo_path}")

    content_type = mimetypes.guess_type(photo_path)[0] or "image/jpeg"
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        with open(photo_path, "rb") as fh:
            resp = session.post(
                f"{api_url}/classify",
                headers=headers,
                files={"image": (os.path.basename(photo_path), fh, content_type)},
                data={"top_k": str(top_k)},
                timeout=30,
            )
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"Cannot reach the AutoVision API at {api_url}.\n"
            "Is the server running? See docs/GETTING_STARTED.md (Path B):\n"
            "  docker run -p 8000:8000 -v $(pwd)/models:/app/models "
            "-e AUTOVISION_API_KEYS=... autovision-api"
        )

    if resp.status_code == 401:
        sys.exit(
            "401 unauthorized: set AUTOVISION_API_KEY to a key configured in "
            "the server's AUTOVISION_API_KEYS."
        )
    if resp.status_code == 503:
        sys.exit(
            "503 model_not_loaded: the server started without model weights. "
            "Download them per docs/GETTING_STARTED.md section 2."
        )
    if resp.status_code == 422:
        # Not fatal for the batch -- this one photo is undecodable.
        print(f"  [skip] {photo_path}: not a decodable image (422)", file=sys.stderr)
        return None
    if resp.status_code != 200:
        sys.exit(f"API error {resp.status_code}: {resp.text[:300]}")

    return resp.json()


def suggest_listing_fields(results):
    """Aggregate per-photo classify results into a single form suggestion.

    Strategy: prefer the highest-confidence ACCEPTED top1 across all photos.
    If no photo was accepted but at least one was rejected as `ambiguous`,
    fall back to a pick list built from that photo's top-3 candidates.

    Returns a dict:
        {"status": "autofill" | "suggest" | "pick_list" | "no_car",
         "fields": {...} | None,
         "choices": [ ... up to 3 candidate dicts ... ],
         "confidence": float | None}
    """
    best = None            # (confidence, top1 dict)
    best_ambiguous = None  # predictions list from an ambiguous rejection

    for res in results:
        if res is None:
            continue
        if not res["rejected"]:
            top1 = res["top1"]  # contract: non-null whenever rejected is False
            if best is None or top1["confidence"] > best[0]:
                best = (top1["confidence"], top1)
        elif res["rejection_reason"] == "ambiguous" and res["predictions"]:
            # Candidates are still returned on rejection -- ideal pick list.
            if best_ambiguous is None:
                best_ambiguous = res["predictions"][:3]

    if best is not None:
        conf, top1 = best
        fields = {
            "make": top1["make"],
            "model": top1["model"],
            "generation": top1["generation"],
            "year_start": top1["year_start"],
            "year_end": top1["year_end"],
            "class_name": top1["class_name"],
        }
        status = "autofill" if conf >= AUTOFILL_CONFIDENCE else "suggest"
        return {"status": status, "fields": fields, "choices": [], "confidence": conf}

    if best_ambiguous is not None:
        return {
            "status": "pick_list",
            "fields": None,
            "choices": [
                {
                    "make": p["make"],
                    "model": p["model"],
                    "generation": p["generation"],
                    "year_start": p["year_start"],
                    "year_end": p["year_end"],
                    "confidence": p["confidence"],
                }
                for p in best_ambiguous
            ],
            "confidence": None,
        }

    # Everything rejected as not_a_car / low_confidence (or undecodable).
    return {"status": "no_car", "fields": None, "choices": [], "confidence": None}


def check_claim(results, claimed_make, claimed_model):
    """Compare the seller's claimed vehicle with confident predictions.

    Returns "FLAG_FOR_REVIEW" when a photo yields an accepted prediction with
    confidence >= MISMATCH_FLAG_CONFIDENCE whose make/model contradicts the
    claim; "CONSISTENT" when a confident prediction agrees; "UNVERIFIED" when
    no prediction was confident enough to judge either way.
    """
    verdict = "UNVERIFIED"
    for res in results:
        if res is None or res["rejected"]:
            continue
        top1 = res["top1"]
        if top1["confidence"] < MISMATCH_FLAG_CONFIDENCE:
            continue
        same_make = top1["make"].lower() == claimed_make.lower()
        # Model comparison is loose on purpose ("3 Series" vs "3-series").
        norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
        same_model = claimed_model is None or norm(top1["model"]) == norm(claimed_model)
        if same_make and same_model:
            verdict = "CONSISTENT"
        else:
            return "FLAG_FOR_REVIEW"  # any confident contradiction wins
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Suggest used-car listing fields from seller photos "
        "via the AutoVision REST API."
    )
    parser.add_argument("photos", nargs="+", help="Seller photo paths (JPEG/PNG/WebP)")
    parser.add_argument("--claimed-make", help="Make the seller entered, if any")
    parser.add_argument("--claimed-model", help="Model the seller entered, if any")
    parser.add_argument("--top-k", type=int, default=5, help="Candidates per photo (default 5)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    api_url = os.environ.get("AUTOVISION_API_URL", DEFAULT_API_URL).rstrip("/")
    api_key = os.environ.get("AUTOVISION_API_KEY", "")

    session = requests.Session()
    results = [
        classify_photo(session, api_url, api_key, photo, top_k=args.top_k)
        for photo in args.photos
    ]

    suggestion = suggest_listing_fields(results)

    # Pin versions with the stored decision -- auditability (see checklist in
    # docs/GETTING_STARTED.md section 6).
    versions = next(
        (
            {"model_version": r["model_version"], "taxonomy_version": r["taxonomy_version"]}
            for r in results
            if r is not None
        ),
        {},
    )
    suggestion["versions"] = versions

    if args.claimed_make:
        suggestion["claim_check"] = check_claim(
            results, args.claimed_make, args.claimed_model
        )

    if args.json:
        print(json.dumps(suggestion, indent=2))
        return

    # --- Human-readable output ---------------------------------------------
    status = suggestion["status"]
    if status in ("autofill", "suggest"):
        f = suggestion["fields"]
        years = f"{f['year_start']}-{f['year_end']}"
        gen = f" {f['generation']}" if f["generation"] else ""
        print(f"Suggested listing: {f['make']} {f['model']}{gen} ({years})")
        print(f"Confidence: {suggestion['confidence']:.1%}")
        if status == "suggest":
            print("Below the silent-autofill bar -- show as a suggestion the "
                  "seller must confirm.")
    elif status == "pick_list":
        print("This could be one of the following -- ask the seller to pick:")
        for i, c in enumerate(suggestion["choices"], start=1):
            gen = f" {c['generation']}" if c["generation"] else ""
            print(f"  {i}. {c['make']} {c['model']}{gen} "
                  f"({c['year_start']}-{c['year_end']})  {c['confidence']:.1%}")
    else:
        print("No car identified in the uploaded photos.")
        print("Prompt the seller: 'Add a clear daylight exterior photo so we "
              "can fill in the details for you.'")

    if "claim_check" in suggestion:
        claimed = " ".join(x for x in (args.claimed_make, args.claimed_model) if x)
        print(f"Claim check vs '{claimed}': {suggestion['claim_check']}")
        if suggestion["claim_check"] == "FLAG_FOR_REVIEW":
            print("-> Route this listing to the trust-and-safety review queue "
                  "(possible mislabel or fraud). Do NOT auto-block.")

    if versions:
        print(f"(model {versions['model_version']}, "
              f"taxonomy {versions['taxonomy_version']})")


if __name__ == "__main__":
    main()
