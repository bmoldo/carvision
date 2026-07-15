# Use Case: Marketplace Listing Autofill

**Integration surface:** REST API (`POST /classify`) — see
[Getting Started, Path B](../../docs/GETTING_STARTED.md#4-path-b--self-hosted-docker-api).

## The business scenario

A used-car marketplace wants sellers to create listings faster and with fewer
errors. Today the seller uploads photos and then manually picks make, model,
generation, and year range from cascading dropdowns — the single biggest
drop-off point in the listing funnel, and the biggest source of mislabeled
inventory (wrong generation is the classic one: an F30 3 Series listed as a
G20 commands a different price).

With AutoVision, the moment the seller uploads their first photos, the backend
classifies them and **pre-fills the form**: make, model, generation, and the
production year range. The seller confirms instead of typing.

## Why AutoVision fits

- **Generation-level output** maps directly onto listing taxonomy fields —
  including the year range (`year_start`–`year_end`) to constrain the "year"
  dropdown to plausible values.
- **Calibrated confidence** (ECE 0.049) lets you set a simple product rule:
  auto-fill silently above a threshold, suggest below it.
- **Rejection gating** keeps garbage out of the form: interior shots, document
  photos, and hand-on-hood selfies come back `not_a_car` instead of a wrong
  autofill.
- A second signal for free: comparing the **seller's claimed** vehicle with the
  prediction is a cheap fraud/mislabel detector.

## Integration pattern

1. Seller uploads photos → backend sends each to `POST /classify`
   (multipart, `X-API-Key` header).
2. Aggregate across photos: take the highest-confidence **accepted** `top1`.
3. Pre-fill the form; keep every field editable — the model assists, the
   seller decides.
4. If the seller had already typed a make/model that disagrees with a
   high-confidence prediction, flag the listing for the trust-and-safety
   review queue (do **not** auto-block: exterior swaps, replicas, and simple
   model errors all look the same to a classifier).
5. Store `model_version` / `taxonomy_version` with the listing so autofill
   decisions are auditable.

## What to do with rejections in this context

| Reason | Product behavior |
|---|---|
| `not_a_car` | Skip the photo for autofill purposes (it may be a legitimate interior/detail shot for the gallery). If **no** photo yields a car, prompt: "Add a clear exterior photo so we can fill in the details for you." |
| `low_confidence` | Don't autofill from this photo. If it's the only one, ask for "a daylight, front-three-quarter exterior shot". |
| `ambiguous` | The best UX in the folder: the top candidates are a known confusion pair (e.g. Golf vs Golf GTI). Show the **top-3 as a one-tap pick list** — "Is this one of these?" — instead of an empty form. The script demonstrates exactly this. |

## Run it

```bash
export AUTOVISION_API_URL="http://localhost:8000"
export AUTOVISION_API_KEY="av_yourkey"

# Basic: suggest listing fields from photos
python3 autofill_listing.py front.jpg rear.jpg side.jpg

# With the seller's claimed vehicle — enables the mislabel/fraud check
python3 autofill_listing.py front.jpg --claimed-make BMW --claimed-model "5 Series"

# JSON output for backend integration
python3 autofill_listing.py front.jpg --json
```

The script prints the suggested form fields, a pick list when the result is
ambiguous, and a `FLAG_FOR_REVIEW` verdict when the claim disagrees with a
confident prediction.

## Honest limits

The model reads the *body style era*, not the paperwork: it cannot detect trim
level, engine, mileage, or a rebadged/replica body (see
[MODEL_CARD.md](../../docs/MODEL_CARD.md)). Autofill is a convenience and the
claim-mismatch flag is a *signal for human review* — never an automated
delisting decision.
