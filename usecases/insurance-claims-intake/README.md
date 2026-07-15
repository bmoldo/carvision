# Use Case: Insurance Claims Photo Intake (FNOL)

**Integration surface:** REST API (`POST /classify`) — see
[Getting Started, Path B](../../docs/GETTING_STARTED.md#4-path-b--self-hosted-docker-api).

## The business scenario

At first notice of loss (FNOL), a claimant photographs their vehicle in the
insurer's app or web portal. Two intake problems cost adjusters time every
day:

1. **Wrong-subject photos** — screenshots, documents, pets, the garage wall —
   that bounce back and forth before an adjuster even sees the claim.
2. **Vehicle mismatch** — the photographed car is not the vehicle on the
   policy (wrong car in a multi-car household, stale policy data, or —
   rarely — deliberate misrepresentation).

This script verifies at upload time that (a) each photo actually shows a
vehicle, and (b) the photographed vehicle is *consistent with* the policy
vehicle's make/model/generation — and routes everything else to a human.

## Why AutoVision fits

- **OOD rejection at the door.** The dedicated background class plus softmax
  gating returns `not_a_car` for non-vehicle photos, so the app can ask for a
  retake *while the claimant is still standing next to the car*.
- **Generation-level comparison** is the right granularity for consistency
  checks: "policy says Golf Mk7, photo shows a Mk7-era Golf" — without
  pretending to read a VIN.
- **Calibrated confidence** supports the deliberately conservative thresholds
  this domain requires.

## Conservative thresholds: precision over recall

Insurance is exactly the setting where a wrong automated answer costs more
than no answer. This script therefore stacks a **stricter client-side
confidence floor** (default 0.85, `--min-confidence`) on top of the engine's
own gating. Anything below the floor — even if the engine accepted it — is
reported as `UNVERIFIED`, not as a match or mismatch. Expect more
`UNVERIFIED` outcomes and fewer wrong ones; that is the point.

**A human reviews every outcome.** Per the
[model card's responsible-use guidance](../../docs/MODEL_CARD.md#fairness-geography-and-responsible-use),
automated insurance denial or claims decisions are **not recommended** uses.
This script's verdicts are triage labels for adjuster worklists:

| Verdict | Meaning | Routing |
|---|---|---|
| `CONSISTENT` | Confident prediction matches the policy vehicle | Normal queue — adjuster confirms as part of standard review |
| `MISMATCH` | Confident prediction contradicts the policy vehicle | Priority human review — *never* an automatic denial. Legitimate explanations abound (household's other car, updated vehicle not yet on policy, data-entry error). |
| `UNVERIFIED` | Rejected photo or below the confidence floor | Adjuster reviews photos manually, or app requests a retake |

## What to do with rejections in this context

| Reason | Intake behavior |
|---|---|
| `not_a_car` | Immediate in-app feedback: "This photo doesn't appear to show a vehicle. Please photograph the car's exterior." Cheapest possible fix — the claimant is still on site. |
| `low_confidence` | Ask for a better shot ("step back to fit the whole car, daylight if possible"), but **accept the photo anyway** after a retry — never block a claim on model confidence. Mark `UNVERIFIED`. |
| `ambiguous` | Treat as consistent *if either* confusion-pair candidate matches the policy vehicle (the pair-mate is usually a trim/facelift sibling); otherwise `UNVERIFIED`. Never escalate to `MISMATCH` from an ambiguous result. |

## Run it

```bash
export AUTOVISION_API_URL="http://localhost:8000"
export AUTOVISION_API_KEY="av_yourkey"

# Strict: make + model + generation must match
python3 claims_intake.py photo1.jpg photo2.jpg \
  --policy-make Volkswagen --policy-model Golf --policy-generation Mk7

# Generation-tolerant: accept any generation of the policy make/model
# (useful when policy records don't store generation)
python3 claims_intake.py photo1.jpg \
  --policy-make Volkswagen --policy-model Golf --generation-tolerant

# Even more conservative
python3 claims_intake.py photo1.jpg --policy-make BMW --policy-model "3 Series" \
  --min-confidence 0.92
```

## Honest limits (read before deploying)

- **Single-vehicle assumption.** The engine assumes one dominant vehicle per
  frame — it is not a detector and does not handle multi-car accident scenes.
  Photos with several vehicles can yield the *wrong car's* label with high
  confidence. Instruct claimants to photograph their own vehicle, filling the
  frame.
- The model **cannot** verify trim, color, VIN, plates, or damage — this check
  is "consistent with the policy vehicle's model-generation", not identity
  verification.
- A car outside the 896-class taxonomy is mapped to its nearest covered
  lookalike or rejected — check
  `models/v5.13.0/class_mapping.json` covers your book of business.
- Real-world accuracy on claimant photos (night, rain, damage, odd angles) is
  **lower** than the 93.85% clean-validation figure. Evaluate on your own
  intake distribution first, and keep a human in the loop **always**.
