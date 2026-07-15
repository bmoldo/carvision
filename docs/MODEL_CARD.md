# AutoVision Model Card — v5.13.0

This card describes what the AutoVision car recognition model does, what it does not do, and how it was evaluated. It is written for engineering and procurement evaluation; claims below are limited to what we have measured.

## Model Details

| | |
|---|---|
| Model version | 5.13.0 |
| Taxonomy version | 5.13.0-897 |
| Architecture | EfficientNet-V2-S, single classification head |
| Input | 384 x 384 RGB, center-cropped, ImageNet mean/std normalization |
| Output | 897 class logits: 896 car model-generation classes + 1 background/non-car class |
| Formats | TFLite FP16 (44 MB, primary), Core ML mlpackage FP16 (iOS), ONNX (server) |
| FP16 parity | Verified 32/32 top-1 agreement vs FP32 on sampled validation images |
| Training data | ~156k images, curated and deduplicated (byte-level and decode-verified) |
| License | Dual: PolyForm Noncommercial 1.0.0 / commercial license ([LICENSE](../LICENSE), [LICENSE-COMMERCIAL.md](../LICENSE-COMMERCIAL.md)) |

## Intended Use

Classification of **exterior photos of cars** with a **single dominant vehicle** in frame. Designed for on-device (Android/iOS) and server deployment. Typical uses: car-spotting and enthusiast apps, inventory tagging, photo organization, content enrichment.

## What It Can Do

- **Make / model / generation identification** across 896 model-generation classes and 76 makes, distinguishing facelifts and redesigns (e.g., BMW 3 Series E90 vs F30).
- **Production year ranges** attached to each predicted class.
- **Rarity tags** per class across six tiers: COMMON (150 classes), UNCOMMON (415), RARE (182), ULTRA_RARE (66), EPIC (39), LEGENDARY (44).
- **Non-car rejection** via a dedicated background class plus softmax gating.
- **Calibrated confidence scores** (ECE 0.049) — scores can be treated as approximate probabilities.
- **On-device inference under 50 ms** on modern Android phones with GPU delegate.

## What It Cannot Do

Be explicit with your users about these limits:

- **Closed taxonomy.** The model only knows the 896 covered model-generations. A car outside the taxonomy **will be mapped to its nearest covered lookalike or rejected** — it cannot say "unknown car model X". If coverage of a specific class matters to you, check `models/v5.13.0/class_mapping.json` before integrating.
- **No trim, engine, or drivetrain detection.** It cannot distinguish a 320i from a 330d, or AWD from RWD, unless they are separate classes in the taxonomy.
- **No color, damage, license plate, or VIN reading.** It is a classifier, not an inspection or OCR system.
- **Not a vehicle detector or counter.** It assumes a single dominant vehicle; it does not localize vehicles or handle multi-car scenes.
- **Degraded performance** on interiors, heavy occlusion, extreme angles, night/low-light shots, and renders/toys/artwork.
- **Not covered:** motorcycles, trucks/HGVs, and buses.
- **Geographic skew.** The taxonomy skews toward US/EU market vehicles; coverage of Asian domestic-market and other regional vehicles is thinner.
- **Year ranges are class metadata, not estimation.** The returned years are the production range of the predicted class — the model does not estimate a specific model year from the image.

## Rejection Gating

Rejection is a first-class outcome, built into the SDKs and API. Responses include `rejected` and `rejection_reason`; `top1` is null when rejected, and candidate predictions are still returned for inspection.

1. `not_a_car` — background class wins or top softmax < 0.05
2. `low_confidence` — top confidence below the per-rarity threshold (COMMON 0.4 rising to LEGENDARY 0.7)
3. `ambiguous` — top-2 form a known confusion pair and the margin is below the pair threshold (default 0.08)

## Known Issues

- **BMW 2 Series F22** predictions can be attracted to **BMW 5 Series E60** — documented; fix scheduled for v5.13.1.
- **51 known confusion pairs** (e.g., VW Golf Mk7 vs Golf GTI Mk7, Tesla Model S pre/post-refresh) ship in `models/v5.13.0/confusion_pairs.json` and are gated at inference via the `ambiguous` rejection path.

## Evaluation

| Metric | Value | Dataset |
|--------|-------|---------|
| Top-1 accuracy | 93.85% | Clean held-out validation set, 21,381 images, 897 classes |
| Top-5 accuracy | 97.88% | Same |
| Calibration (ECE) | 0.049 | Same |

**Important:** real-world accuracy on uncurated street photos is lower than clean validation accuracy. Rejection thresholds are deliberately tuned precision-over-recall — the engine prefers rejecting an image over returning a wrong answer. Evaluate on your own image distribution before committing to accuracy targets.

## Fairness, Geography, and Responsible Use

The training data and taxonomy skew toward US/EU market vehicles, so error rates are not uniform across regions or vehicle populations.

The following uses are **not recommended** without a human in the loop, and in some cases not at all:

- Law enforcement identification of individuals or their vehicles
- Automated insurance denial or claims decisions
- Surveillance applications without human review

These are recommendations, not technical enforcement — the [license](../LICENSE) governs permitted use.

## Contact

Commercial licensing, weight access, and evaluation support: **bogdanmoldovan29@gmail.com**. See [LICENSE-COMMERCIAL.md](../LICENSE-COMMERCIAL.md).
