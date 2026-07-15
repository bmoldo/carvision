# Use Case: Mobile Car-Spotting App

**Integration surface:** Android SDK (drop-in `AutoVision.kt`, on-device
TFLite) — see
[Getting Started, Path C](../../docs/GETTING_STARTED.md#5-path-c--android-sdk).
This folder is a README-with-code pattern, not a Gradle project — copy the
snippet into your app.

## The business scenario

A consumer app for car enthusiasts: point the camera at a car on the street,
the app identifies it, and rare finds earn points, badges, and a spot in the
user's collection — Pokédex mechanics for cars. Sessions are driven by the
thrill of the `EPIC` and `LEGENDARY` find.

## Why AutoVision fits

- **Rarity is built in.** Every prediction carries one of six tiers —
  `COMMON` (150 classes), `UNCOMMON` (415), `RARE` (182), `ULTRA_RARE` (66),
  `EPIC` (39), `LEGENDARY` (44) — a ready-made gamification ladder.
- **Generation-level ID** is exactly the enthusiast's language: spotting an
  E90 is not the same as spotting an F30.
- **On-device and offline-first.** The 44 MB FP16 model runs under 50 ms on
  modern phones with the GPU delegate. No network needed at the trackside car
  meet with zero bars — and a genuine privacy pitch: **photos never leave the
  phone**. No upload consent flows, no image retention policy to write, no
  server bill that scales with your DAU.
- **Rejection reasons map 1:1 to camera UX** (see below) — the difference
  between an app that feels smart and one that guesses.

## Integration pattern

Camera frame (CameraX) → `Bitmap` → `AutoVision.classify()` → route on
`rejected`/`rejectionReason` → rarity-based celebration + collection update.

Bundle `car_classifier.tflite`, `class_mapping.json`, `model_manifest.json`,
and `confusion_pairs.json` in `assets/` (all from the same release), and add
the TFLite Gradle dependencies — both listed in
[Getting Started, Path C](../../docs/GETTING_STARTED.md#5-path-c--android-sdk).

## What to do with rejections in this context

| Reason | Camera UX |
|---|---|
| `not_a_car` | "Point the camera at a car" — passive hint overlay, no shutter sound, no penalty. Happens constantly while the user pans; treat it as idle state, not failure. |
| `low_confidence` | "Almost! Get closer or find better light." Keep the session alive — this is a retry prompt, not a dead end. |
| `ambiguous` | "Could be one of these — you were there, you pick:" show the top candidates as tappable cards (they're still in `result.predictions`). Enthusiasts *love* making the call between a Golf and a GTI — it's a feature, not an apology. |

## Kotlin snippet

```kotlin
package com.example.carspotter

import android.content.Context
import android.graphics.Bitmap
import dev.autovision.AutoVision

/**
 * Car-spotting game logic on top of the drop-in AutoVision engine.
 *
 * Construction is the expensive part (model load + GPU delegate init):
 * create ONE CarSpotter per screen/session and close() it when done.
 */
class CarSpotter(context: Context) : AutoCloseable {

    private val engine = AutoVision(context, "car_classifier.tflite")
    // class_mapping.json / model_manifest.json / confusion_pairs.json are
    // loaded from assets/ by default -- all four files from the same release.

    /** Points per rarity tier -- the gamification ladder is built into every prediction. */
    private val rarityPoints = mapOf(
        "COMMON" to 10,
        "UNCOMMON" to 25,
        "RARE" to 100,
        "ULTRA_RARE" to 400,
        "EPIC" to 1500,        // full-screen confetti territory
        "LEGENDARY" to 5000    // the reason people open the app
    )

    sealed class SpotOutcome {
        /** Accepted identification -> add to collection, award points. */
        data class Spotted(
            val title: String,        // "BMW 3 Series F30 (2012-2018)"
            val rarity: String,       // COMMON .. LEGENDARY
            val points: Int,
            val confidence: Float,
            val classId: String       // stable collection key, e.g. "bmw_3_series_f30"
        ) : SpotOutcome()

        /** Ambiguous confusion pair -> let the spotter make the call. */
        data class PickOne(val choices: List<String>) : SpotOutcome()

        /** Retryable states -> hint overlay, keep the camera session alive. */
        data class Hint(val message: String) : SpotOutcome()
    }

    /** Classify one camera frame (e.g. from CameraX ImageCapture). */
    fun spot(frame: Bitmap): SpotOutcome {
        val result = engine.classify(frame, topK = 3)

        // Rejection is a designed outcome, not an error: each reason gets
        // its own UX instead of a generic failure toast.
        if (result.rejected) {
            return when (result.rejectionReason) {
                AutoVision.REJECT_NOT_A_CAR ->
                    SpotOutcome.Hint("Point the camera at a car")

                AutoVision.REJECT_LOW_CONFIDENCE ->
                    SpotOutcome.Hint("Almost! Get closer or find better light")

                AutoVision.REJECT_AMBIGUOUS ->
                    // Candidates are still populated when rejected --
                    // perfect for a "you pick" card row.
                    SpotOutcome.PickOne(
                        result.predictions.take(3).map { p ->
                            "${p.make} ${p.model} ${p.generation ?: ""}".trim()
                        }
                    )

                else -> SpotOutcome.Hint("Try another angle")
            }
        }

        // Contract: top1 is non-null whenever rejected == false.
        val top = result.top1!!
        val gen = top.generation?.let { " $it" } ?: ""
        return SpotOutcome.Spotted(
            title = "${top.make} ${top.model}$gen (${top.yearStart}-${top.yearEnd})",
            rarity = top.rarity,
            points = rarityPoints[top.rarity] ?: 10,
            confidence = top.confidence,
            classId = top.className
        )
    }

    /** Big-find check for the celebration animation. */
    fun isJackpot(outcome: SpotOutcome): Boolean =
        outcome is SpotOutcome.Spotted &&
            (outcome.rarity == "EPIC" || outcome.rarity == "LEGENDARY")

    override fun close() = engine.close()  // releases interpreter + GPU delegate
}
```

Wiring it into a ViewModel:

```kotlin
val spotter = CarSpotter(applicationContext)  // once per session, not per frame

when (val outcome = spotter.spot(bitmap)) {
    is CarSpotter.SpotOutcome.Spotted -> {
        collection.add(outcome.classId)   // dedupe by classId, not title
        score += outcome.points
        if (spotter.isJackpot(outcome)) playLegendaryAnimation(outcome.title)
    }
    is CarSpotter.SpotOutcome.PickOne -> showPickerCards(outcome.choices)
    is CarSpotter.SpotOutcome.Hint -> showCameraHint(outcome.message)
}
```

## Honest limits

Renders, toys, posters, and game screenshots degrade accuracy — decide
whether "spotting" a poster counts in your game rules before players decide
for you. The taxonomy is 896 model-generations with a US/EU skew: a
JDM-import meet will produce more `low_confidence` results than a suburban
parking lot. And rarity is a *class-level* tag, not a market-value estimate —
see [MODEL_CARD.md](../../docs/MODEL_CARD.md) for the full limits list.
