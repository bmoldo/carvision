# Use Case: Fleet / Rental Yard Audit

**Integration surface:** Python SDK (`classify_batch`) — see
[Getting Started, Path A](../../docs/GETTING_STARTED.md#3-path-a--python-sdk).

## The business scenario

A rental branch or corporate fleet yard is supposed to contain a known mix of
vehicles — say 12 Golf Mk8s, 8 Octavias, 4 Model 3s. Reality drifts: returns
parked at the wrong branch, un-logged transfers, vehicles that quietly never
came back. A physical clipboard audit is monthly at best.

This script turns the yard's existing overnight camera stills into a nightly
reconciliation: classify every still, tally accepted identifications per
model-generation, and diff against the expected fleet database — reporting
**unexpected** models (shouldn't be here) and **missing/short** counts
(fewer seen than expected).

## Why AutoVision fits

- **`classify_batch`** over a directory of stills is the whole inference
  loop; the reconciliation is a dictionary diff.
- **Generation-level classes** match how fleets are actually purchased
  (a batch of Mk8 Golfs), so the tally keys line up with fleet records.
- **Rejection gating keeps the count honest.** Night shots and partial frames
  are the norm for yard cameras; rejected stills are *excluded from the
  tally* and reported separately, rather than polluting the reconciliation
  with guesses.
- Runs on the yard box or a central server — images never need to leave your
  network.

### ONNX on a server GPU for throughput

Large operations (hundreds of branches × dozens of stills) should run the
**ONNX export with `onnxruntime-gpu`**: the SDK auto-selects the ONNX backend
for `.onnx` weights and prefers `CUDAExecutionProvider` when available —
no code change, just point `--model-dir` at a release directory containing
the `.onnx` file. TFLite (`tflite-runtime`) remains the right choice for a
small on-site box. See the install matrix in
[Getting Started, section 3.1](../../docs/GETTING_STARTED.md#31-install).

## Integration pattern

1. Overnight, yard cameras drop stills into a per-branch directory
   (one dominant vehicle per still — see limits below).
2. Nightly cron: `yard_audit.py <stills-dir> --fleet-json branch_042.json`.
3. The report goes to the branch manager: unexpected models, short counts,
   and the unverified-still count.
4. Non-zero exit code when discrepancies are found, so the cron wrapper can
   page or open a ticket.

## What to do with rejections in this context

| Reason | Audit behavior |
|---|---|
| `low_confidence` | Expected at night. The still is counted as `unverified`, never guessed into the tally. A consistently high unverified rate is a camera/lighting work order, not a model bug. |
| `ambiguous` | If either confusion-pair candidate is an expected fleet model, the script counts it toward that model (pair-mates are trim/facelift siblings — fleet counts don't care). Otherwise unverified. |
| `not_a_car` | Empty bay, pedestrian, equipment. Excluded from the tally silently — this is the gate doing its job. |

## Run it

```bash
pip install -e "sdk/python[tflite]"   # or [onnx] + onnxruntime-gpu on a GPU server

# Built-in demo fleet definition:
python3 yard_audit.py /path/to/yard-stills --model-dir models/v5.13.0

# Real fleet DB export:
python3 yard_audit.py /path/to/yard-stills --fleet-json branch_042.json
```

`branch_042.json` format — expected count per class id (class ids come from
`models/v5.13.0/class_mapping.json`):

```json
{
  "volkswagen_golf_mk8": 12,
  "skoda_octavia_mk4": 8,
  "tesla_model_3": 4
}
```

## Honest limits

- **One still ≈ one vehicle.** The engine is a classifier, not a detector or
  counter — it assumes a single dominant vehicle per frame
  ([MODEL_CARD.md](../../docs/MODEL_CARD.md)). Wide-angle whole-yard shots
  will not work; use per-bay/per-lane crops or camera presets.
- The tally counts *stills*, so the same vehicle seen by two cameras counts
  twice — dedupe by bay/camera position in your pipeline if that matters.
- Night/low-light performance is degraded by design of the domain; expect and
  monitor the unverified rate rather than forcing it to zero.
- The audit says "a Golf Mk8 was seen", not *which* Golf — reconciling
  individual VINs/plates needs your ANPR or asset tags alongside this.
