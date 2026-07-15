# AutoVision Use Cases

Possible use cases: situations where AutoVision's car make/model/generation
recognition can be integrated. Each folder contains a README describing the
business scenario and integration pattern. Five scenarios include a runnable,
well-commented example Python script written against the real v5.13.0 contract;
[mobile-carspotting-app](mobile-carspotting-app/) is an Android integration
pattern in Kotlin (code in its README, no standalone script).

**These are illustrations, not case studies.** The engine's production
deployment today is [Boby's Garage](mobile-carspotting-app/), the car-spotting
app it was built for — the other scenarios show how the same contract applies
elsewhere and have been exercised against the real model, but not in a
production deployment of that kind.

New here? Do the zero-to-first-classification setup in
[docs/GETTING_STARTED.md](../docs/GETTING_STARTED.md) first.

## Index

| Scenario | What it demonstrates | Integration surface |
|---|---|---|
| [marketplace-listing-autofill](marketplace-listing-autofill/) | Auto-filling a used-car listing form from seller photos; top-3 pick lists on `ambiguous`; claimed-vs-predicted fraud/mislabel flagging | REST API |
| [dealer-inventory-batch](dealer-inventory-batch/) | Batch-tagging a photo directory to CSV; routing `low_confidence` rows to a manual-review queue; per-class inventory rollup | Python SDK |
| [insurance-claims-intake](insurance-claims-intake/) | FNOL photo verification against the policy vehicle; OOD rejection at upload time; conservative thresholds with a human always in the loop | REST API |
| [parking-access-enrichment](parking-access-enrichment/) | Enriching ANPR plate reads with vehicle attributes for plate/vehicle mismatch alerts; rarity cross-checks for VIP/valet flows | REST API |
| [mobile-carspotting-app](mobile-carspotting-app/) | On-device camera capture → classify → rarity-based gamification; rejection-reason-driven UX; offline-first privacy | Android SDK (Kotlin) |
| [fleet-yard-audit](fleet-yard-audit/) | Nightly batch over yard camera stills reconciled against an expected fleet DB; unexpected/missing model reporting; ONNX-on-GPU throughput note | Python SDK |

## Shared setup

All scenarios assume the v5.13.0 weights are in place
(`models/v5.13.0/car_classifier.tflite`, verified against `SHA256SUMS` — see
[Getting Started, section 2](../docs/GETTING_STARTED.md#2-get-the-model-weights)).

**REST-API-based scripts** need a running server
([Getting Started, Path B](../docs/GETTING_STARTED.md#4-path-b--self-hosted-docker-api))
and read two environment variables:

```bash
export AUTOVISION_API_URL="http://localhost:8000"   # default if unset
export AUTOVISION_API_KEY="av_yourkey"              # required if the server has auth enabled
```

**SDK-based scripts** need the `autovision` package plus a backend
(`pip install -e "sdk/python[tflite]"` —
[Getting Started, Path A](../docs/GETTING_STARTED.md#3-path-a--python-sdk)).

All scripts use only the Python standard library plus `requests` (REST
scenarios) or the `autovision` SDK (on-device scenarios), degrade gracefully
with clear messages when the server is down or weights are missing, and expose
an `argparse` CLI — run any of them with `--help`.

## A note on honesty

These scripts inherit the limits documented in
[docs/MODEL_CARD.md](../docs/MODEL_CARD.md): closed 896-class taxonomy, single
dominant vehicle per photo, no trim/color/damage/plate reading, degraded
performance on night shots and occlusion, US/EU coverage skew. Every scenario
below treats rejection as a designed outcome and keeps a human in the loop
wherever a decision affects a person.
