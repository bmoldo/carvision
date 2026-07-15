# Use Case: Parking / Toll ANPR Enrichment

**Integration surface:** REST API (`POST /classify`) — see
[Getting Started, Path B](../../docs/GETTING_STARTED.md#4-path-b--self-hosted-docker-api).

## The business scenario

A parking or toll operator already runs ANPR (automatic number-plate
recognition): the camera reads the plate, the barrier opens for registered
plates. The blind spot is **cloned or swapped plates** — a plate registered to
a VW Golf arriving on an SUV should not sail through on the plate alone.

This script enriches each plate read with what the *vehicle itself* looks
like: the operator's lane camera still goes to AutoVision, and the predicted
make/model is compared with the vehicle registered to that plate. A confident
disagreement raises a mismatch alert for the attendant.

## Why AutoVision fits — as a complement, not a replacement

AutoVision does **not** read plates and is not an ANPR system (no OCR — see
[MODEL_CARD.md](../../docs/MODEL_CARD.md)). It answers the question ANPR
can't: *does the car attached to this plate look like the car in the
registration record?*

- **Make/model/generation output** compares directly against registration
  records (which usually store make + model; generation comparison is
  optional and off by default here, since records rarely store it).
- **Calibrated confidence** means the alert threshold is a real probability —
  the script only alerts on *confident* contradictions, keeping false alarms
  low in a high-throughput lane.
- **Rejection gating** handles the realities of lane cameras (glare, rain,
  partial frames) by saying "couldn't verify" instead of guessing.

## Integration pattern

1. ANPR fires: plate text + a full-frame lane still.
2. Backend looks up the plate in the registration DB (mocked as a small dict
   in the script; supply your own with `--registry-json`).
3. Lane still → `POST /classify`.
4. Compare: registered make/model vs predicted.
   - Agree → `PASS` (normal entry).
   - Confident disagree → `MISMATCH_ALERT` → attendant screen / hold barrier
     per site policy. **Human decides** — never auto-enforcement; body kits,
     re-registered vehicles, and stale records are common benign causes.
   - Rejected / low confidence / plate not in DB → `UNVERIFIED` (default:
     allow and log; tighten per site policy).
5. Log everything with `model_version`/`taxonomy_version` for later analysis.

### Bonus: rarity cross-check for VIP / valet flows

Every prediction carries a rarity tier (`COMMON` … `LEGENDARY`). Sites with
valet or VIP programs can cross-check it: an `EPIC`/`LEGENDARY` prediction at
the entry lane can auto-suggest the valet/covered-parking flow, or flag a
high-value vehicle for camera-recorded handling. The script prints this hint
when it applies.

## What to do with rejections in this context

| Reason | Lane behavior |
|---|---|
| `not_a_car` | Almost always a bad frame (barrier arm, pedestrian, glare) — log `UNVERIFIED`, don't alert, optionally trigger a second frame grab. |
| `low_confidence` | Night/rain/motion blur. Log `UNVERIFIED` and let the plate decision stand — the enrichment is additive, it must never make the lane *less* reliable than plain ANPR. |
| `ambiguous` | Confusion-pair sibling (often same model, different trim/facelift). If **either** candidate matches the registered vehicle, treat as `PASS` — the pair-mate almost never crosses make/model lines. |

## Run it

```bash
export AUTOVISION_API_URL="http://localhost:8000"
export AUTOVISION_API_KEY="av_yourkey"

# Uses the built-in demo registry (see the dict in the script)
python3 enrich_plate_read.py lane_still.jpg --plate "B-123-XYZ"

# Real deployment: supply your registry export
python3 enrich_plate_read.py lane_still.jpg --plate "B-123-XYZ" \
  --registry-json registry.json

# Compare generation too, if your records store it
python3 enrich_plate_read.py lane_still.jpg --plate "B-123-XYZ" --check-generation
```

`registry.json` format: `{"<PLATE>": {"make": "...", "model": "...",
"generation": null}}`.

## Honest limits

Single dominant vehicle per frame — tailgating scenes can classify the wrong
car; crop to the lane's vehicle region if your camera sees multiple lanes. The
model can't tell trims apart or verify identity; a same-model swap (Golf for
Golf) passes this check by design. Treat mismatch alerts as attendant
worklist items, not enforcement actions.
