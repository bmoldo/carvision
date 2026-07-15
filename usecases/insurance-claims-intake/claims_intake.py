#!/usr/bin/env python3
"""FNOL (first notice of loss) photo intake verification via the AutoVision
REST API.

For each claimant photo, checks:
  1. that the photo actually shows a vehicle (OOD rejection -> retake prompt),
  2. that the photographed vehicle is CONSISTENT with the policy vehicle's
     make/model/generation (optionally generation-tolerant).

Deliberately conservative: a client-side confidence floor (default 0.85) is
stacked ON TOP of the engine's own gating -- precision over recall. Anything
uncertain is UNVERIFIED, never a MISMATCH.

IMPORTANT -- responsible use (see docs/MODEL_CARD.md):
  * Verdicts are TRIAGE LABELS for adjuster worklists. A human reviews every
    claim; automated denial based on this output is explicitly NOT a
    recommended use of the model.
  * The engine assumes a single dominant vehicle per photo. Multi-car scenes
    can produce a confident label for the WRONG car.

Integration surface: REST API (POST /classify with X-API-Key auth).

Environment:
    AUTOVISION_API_URL   Base URL (default http://localhost:8000)
    AUTOVISION_API_KEY   API key, required if the server has auth enabled

Usage:
    python3 claims_intake.py photo1.jpg photo2.jpg \
        --policy-make Volkswagen --policy-model Golf --policy-generation Mk7

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

# Client-side floor applied on top of engine gating. The engine's per-rarity
# thresholds (0.4-0.7) are tuned for general use; claims triage wants a much
# stricter bar. Confidence is calibrated (ECE 0.049), so 0.85 reads as
# roughly "85% probability this is the right class".
DEFAULT_MIN_CONFIDENCE = 0.85


def norm(s):
    """Loose text normalization for make/model comparison
    ('3 Series' == '3-series' == '3series')."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def classify_photo(session, api_url, api_key, photo_path, top_k=3):
    """POST one photo to /classify; return parsed JSON or None if undecodable."""
    if not os.path.isfile(photo_path):
        sys.exit(f"Photo not found: {photo_path}")

    headers = {"X-API-Key": api_key} if api_key else {}
    content_type = mimetypes.guess_type(photo_path)[0] or "image/jpeg"

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
            f"Cannot reach the AutoVision API at {api_url}. Is the server "
            "running? See docs/GETTING_STARTED.md, Path B."
        )

    if resp.status_code == 401:
        sys.exit("401 unauthorized: set AUTOVISION_API_KEY (see docs/API.md).")
    if resp.status_code == 503:
        sys.exit("503 model_not_loaded: server has no weights loaded "
                 "(docs/GETTING_STARTED.md, section 2).")
    if resp.status_code == 422:
        return None  # undecodable upload -- caller marks it UNVERIFIED
    if resp.status_code != 200:
        sys.exit(f"API error {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def matches_policy(pred, policy, generation_tolerant):
    """Does one prediction dict agree with the policy vehicle?"""
    if norm(pred["make"]) != norm(policy["make"]):
        return False
    if norm(pred["model"]) != norm(policy["model"]):
        return False
    if generation_tolerant or not policy.get("generation"):
        return True
    return norm(pred["generation"]) == norm(policy["generation"])


def assess_photo(result, policy, min_confidence, generation_tolerant):
    """Map one /classify response to (verdict, detail).

    Verdicts: CONSISTENT | MISMATCH | UNVERIFIED | RETAKE
    Conservative by construction:
      * MISMATCH only from an ACCEPTED prediction above the client floor.
      * `ambiguous` rejections count as CONSISTENT if either confusion-pair
        candidate matches the policy vehicle (the pair-mate is usually a
        trim/facelift sibling) -- and never escalate to MISMATCH.
    """
    if result is None:
        return "UNVERIFIED", "photo could not be decoded -- request re-upload"

    if result["rejected"]:
        reason = result["rejection_reason"]
        if reason == "not_a_car":
            # Cheapest fix: prompt while the claimant is still on site.
            return "RETAKE", ("no vehicle detected -- ask the claimant to "
                              "photograph the car's exterior")
        if reason == "ambiguous":
            candidates = result["predictions"][:2]
            if any(matches_policy(p, policy, generation_tolerant)
                   for p in candidates):
                names = " / ".join(p["class_name"] for p in candidates)
                return "CONSISTENT", (f"ambiguous between {names}, but a "
                                      "candidate matches the policy vehicle")
            return "UNVERIFIED", "ambiguous result; no candidate matches"
        # low_confidence: never block a claim on model confidence.
        return "UNVERIFIED", ("low confidence -- suggest a retake (whole car "
                              "in frame, daylight), accept photo regardless")

    top1 = result["top1"]  # contract: non-null when not rejected
    label = "{} {} {} ({}-{})".format(
        top1["make"], top1["model"], top1["generation"] or "",
        top1["year_start"], top1["year_end"],
    )

    # Client-side conservative floor on top of engine gating.
    if top1["confidence"] < min_confidence:
        return "UNVERIFIED", (f"prediction {label} at {top1['confidence']:.1%} "
                              f"is below the {min_confidence:.0%} floor")

    if matches_policy(top1, policy, generation_tolerant):
        return "CONSISTENT", f"{label} at {top1['confidence']:.1%}"
    return "MISMATCH", (f"photo shows {label} at {top1['confidence']:.1%}; "
                        "policy vehicle differs")


def combine(verdicts):
    """Roll per-photo verdicts into one claim-level triage label.

    Any confident MISMATCH outranks everything (priority human review);
    otherwise any CONSISTENT photo suffices; otherwise unverified.
    """
    if "MISMATCH" in verdicts:
        return "MISMATCH"
    if "CONSISTENT" in verdicts:
        return "CONSISTENT"
    return "UNVERIFIED"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify FNOL claim photos against the policy vehicle "
        "(triage labels only -- a human reviews every claim)."
    )
    parser.add_argument("photos", nargs="+", help="Claimant photo paths")
    parser.add_argument("--policy-make", required=True, help="Policy vehicle make")
    parser.add_argument("--policy-model", required=True, help="Policy vehicle model")
    parser.add_argument("--policy-generation",
                        help="Policy vehicle generation (e.g. Mk7, F30)")
    parser.add_argument("--generation-tolerant", action="store_true",
                        help="Accept any generation of the policy make/model")
    parser.add_argument("--min-confidence", type=float,
                        default=DEFAULT_MIN_CONFIDENCE,
                        help="Client-side confidence floor on top of engine "
                             f"gating (default {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    api_url = os.environ.get("AUTOVISION_API_URL", DEFAULT_API_URL).rstrip("/")
    api_key = os.environ.get("AUTOVISION_API_KEY", "")
    policy = {
        "make": args.policy_make,
        "model": args.policy_model,
        "generation": args.policy_generation,
    }

    session = requests.Session()
    report = {"policy_vehicle": policy, "photos": [], "versions": {}}

    for photo in args.photos:
        result = classify_photo(session, api_url, api_key, photo)
        verdict, detail = assess_photo(
            result, policy, args.min_confidence, args.generation_tolerant
        )
        report["photos"].append({"file": photo, "verdict": verdict, "detail": detail})
        if result is not None and not report["versions"]:
            # Pin versions with the stored triage decision (auditability).
            report["versions"] = {
                "model_version": result["model_version"],
                "taxonomy_version": result["taxonomy_version"],
            }

    report["claim_verdict"] = combine([p["verdict"] for p in report["photos"]])

    if args.json:
        print(json.dumps(report, indent=2))
        return

    gen = f" {policy['generation']}" if policy["generation"] else ""
    print(f"Policy vehicle: {policy['make']} {policy['model']}{gen}"
          + (" (generation-tolerant)" if args.generation_tolerant else ""))
    for p in report["photos"]:
        print(f"  [{p['verdict']:<10}] {p['file']}: {p['detail']}")
    print(f"\nClaim triage verdict: {report['claim_verdict']}")
    routing = {
        "CONSISTENT": "Normal adjuster queue (standard human review).",
        "MISMATCH": "PRIORITY human review -- never an automatic denial; "
                    "legitimate explanations are common.",
        "UNVERIFIED": "Adjuster reviews photos manually / app requests retake.",
    }
    print(f"Routing: {routing[report['claim_verdict']]}")
    if report["versions"]:
        print(f"(model {report['versions']['model_version']}, "
              f"taxonomy {report['versions']['taxonomy_version']})")


if __name__ == "__main__":
    main()
