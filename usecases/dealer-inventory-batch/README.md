# Use Case: Dealer Inventory Batch Tagging

**Integration surface:** Python SDK (`AutoVision.classify_batch`) — see
[Getting Started, Path A](../../docs/GETTING_STARTED.md#3-path-a--python-sdk).

## The business scenario

A dealership group (or auction house) photographs every arriving vehicle —
trade-ins, lease returns, auction lots. Photos land in a directory per intake
day; someone then has to tag each car so it can be matched to stock records,
priced, and published. Manual tagging is slow, and taggers routinely mislabel
*generations*, which matters for pricing.

This script batch-tags an entire photo directory in one pass: one CSV with
make/model/generation/years/confidence per photo, a second CSV with the photos
that need a human look, and a per-class inventory rollup at the end.

## Why AutoVision fits

- **Runs where the photos are.** The SDK loads the 44 MB TFLite model
  in-process — no server to stand up, no images leaving the intake machine.
- **`classify_batch`** does the loop for you and returns one
  `ClassificationResult` per file.
- **Generation-level labels + year ranges** map straight onto stock-record
  fields.
- **Rejection gating is the review queue.** Instead of silently mis-tagging
  the blurry end-of-day shots, the engine marks them `low_confidence` and the
  script routes them to `review_queue.csv` for a human tagger — precision
  where it's cheap, people where it's needed.

## Integration pattern

1. Nightly cron (or a watch on the intake folder) runs the script over the
   day's directory.
2. `inventory.csv` → imported into the DMS/stock system.
3. `review_queue.csv` → the morning worklist for a human tagger (every
   rejected photo, with the reason and the top candidate as a starting point).
4. The rollup printed at the end doubles as the intake-day summary email.
5. `model_version`/`taxonomy_version` are embedded in the CSV so tags remain
   auditable across model upgrades.

## What to do with rejections in this context

| Reason | Batch behavior |
|---|---|
| `low_confidence` | Route to `review_queue.csv` with the top candidate pre-filled — the tagger confirms or corrects rather than starting from scratch. Usually fixable with a re-shoot (distance/lighting). |
| `ambiguous` | Also routed to review, with the confusion-pair candidates listed. These are one-glance decisions for a human (e.g. GTI badge vs base Golf). |
| `not_a_car` | Routed to review flagged as "no vehicle" — typically interior shots, paperwork photos, or the odd forklift. Don't count them in inventory. |

## Run it

```bash
pip install -e "sdk/python[tflite]"   # once — see Getting Started, Path A

python3 batch_tag_inventory.py /path/to/intake-photos \
  --model-dir models/v5.13.0 \
  --out inventory.csv \
  --review-out review_queue.csv
```

Sample output:

```
Tagged 143 photos: 128 accepted, 15 routed to review_queue.csv

Inventory rollup (accepted photos):
  count  vehicle
     12  Volkswagen Golf Mk7 (2013-2019)
      9  BMW 3 Series F30 (2012-2018)
      ...
```

## Honest limits

One photo = one vehicle: the engine assumes a **single dominant vehicle** per
frame and does not detect or count multiple cars
([MODEL_CARD.md](../../docs/MODEL_CARD.md)). It also cannot distinguish trims
(GTI vs base Golf is a known confusion pair, gated as `ambiguous`), and the
year range is the class's production range, not the specific model year —
confirm the exact year from the VIN/paperwork, not the photo.
