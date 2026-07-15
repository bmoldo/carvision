# AutoVision Android SDK

Engine version: **0.2.0** — built for model **v5.13.0**.

## Model facts (v5.13.0)

- EfficientNet-V2-S, TFLite FP16 (44 MB)
- 897 classes = 896 car model-generations + 1 background class (`zzz_background`, index 896)
- 76 makes
- 93.85% top-1 validation accuracy
- Input: 384×384 RGB, center crop, ImageNet mean/std normalization
- Rarity tiers: `COMMON`, `UNCOMMON`, `RARE`, `ULTRA_RARE`, `EPIC`, `LEGENDARY` (plus `BACKGROUND` for the background class)

Note: the FP16-quantized model still produces a **float32** output tensor via TFLite. The engine reads the output tensor's dtype and shape from the interpreter instead of assuming, and fails fast with a clear error if they do not match.

## Integration

1. Copy `AutoVision.kt` into your project
2. Add TFLite dependencies to `build.gradle`:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'androidx.exifinterface:exifinterface:1.3.7'
}
```

3. Bundle the model assets in `assets/` (or download at runtime):

| Asset | Required | Description |
|---|---|---|
| `car_classifier.tflite` | Yes | FP16 TFLite model (44 MB) |
| `class_mapping.json` | Yes | 897 class entries (must match the model's output size) |
| `model_manifest.json` | Recommended | Input size, temperature, OOD threshold, per-rarity confidence thresholds, confusion-pair margin. Falls back to built-in v5.13.0 defaults if missing. |
| `confusion_pairs.json` | Recommended | 51 known confusion pairs with per-pair margins. Empty list if missing. |

All four files must come from the same model release. The engine throws `IllegalStateException` at construction if `class_mapping.json` size, the manifest's `num_classes`, and the model's output tensor size disagree.

4. Initialize and classify:

```kotlin
val autoVision = AutoVision(context, "car_classifier.tflite")

val result = autoVision.classify(bitmap, topK = 5)

if (result.rejected) {
    // "not_a_car" | "low_confidence" | "ambiguous"
    Log.d("AutoVision", "Rejected: ${result.rejectionReason}")
} else {
    val top1 = result.top1!!
    Log.d("AutoVision", "${top1.make} ${top1.model} (${top1.rarity})")
    Log.d("AutoVision", "  Years: ${top1.yearStart}-${top1.yearEnd}")
    Log.d("AutoVision", "  Confidence: ${"%.1f%%".format(top1.confidence * 100)}")
}

// Top-K candidates are populated even when rejected (background excluded)
result.predictions.forEach { p ->
    Log.d("AutoVision", "#${p.rank} ${p.className} ${"%.3f".format(p.confidence)}")
}

Log.d("AutoVision", "model=${result.modelVersion} taxonomy=${result.taxonomyVersion} " +
    "engine=${result.engineVersion} ${result.inferenceTimeMs}ms")

// Clean up when done
autoVision.close()
```

`classifyFile(path, topK)` works the same way and additionally handles EXIF rotation.

## Gating / rejection behavior

Gating runs on the argmax over all 897 temperature-scaled softmax probabilities (background class included), checked in this order:

| # | Condition | `rejectionReason` |
|---|---|---|
| a | Argmax is the background class (rarity `BACKGROUND`) | `not_a_car` |
| b | Top probability < `ood_threshold` (0.05) | `not_a_car` |
| c | Top probability < confidence threshold for its rarity tier (fallback 0.5) | `low_confidence` |
| d | Top-2 car classes are a known confusion pair (either order) and p1 − p2 < that pair's `margin_threshold` (fallback: manifest `confusion_pair_margin`, 0.08) | `ambiguous` |

When rejected, `top1` is `null` and `rejected` is `true`, but `predictions` is still populated with the top-K car candidates (background class always excluded) so callers can show "did you mean" UIs.

Per-rarity confidence thresholds (from `model_manifest.json`):

| Rarity | Threshold |
|---|---|
| COMMON | 0.40 |
| UNCOMMON | 0.50 |
| RARE | 0.60 |
| ULTRA_RARE | 0.65 |
| EPIC | 0.67 |
| LEGENDARY | 0.70 |

## API

```kotlin
class AutoVision(
    context: Context,
    modelFileName: String,
    classMappingFileName: String = "class_mapping.json",
    manifestFileName: String = "model_manifest.json",
    confusionPairsFileName: String = "confusion_pairs.json"
) : Closeable

fun classify(bitmap: Bitmap, topK: Int = 5): ClassificationResult
fun classifyFile(imagePath: String, topK: Int = 5): ClassificationResult

data class ClassificationResult(
    val predictions: List<Prediction>,   // top-K, background class excluded
    val top1: Prediction?,               // null when rejected
    val rejected: Boolean,
    val rejectionReason: String?,        // "not_a_car" | "low_confidence" | "ambiguous"
    val inferenceTimeMs: Long,
    val engineVersion: String,           // "0.2.0"
    val modelVersion: String,            // manifest version, e.g. "5.13.0"
    val taxonomyVersion: String          // "<modelVersion>-<numClasses>", e.g. "5.13.0-897"
)

data class Prediction(
    val index: Int,
    val className: String,
    val make: String,
    val model: String,
    val generation: String?,
    val yearStart: Int?,
    val yearEnd: Int?,
    val rarity: String,                  // COMMON | UNCOMMON | RARE | ULTRA_RARE | EPIC | LEGENDARY
    val confidence: Float,
    val logit: Float,
    val rank: Int                        // 1-based rank within predictions
)
```

## Features

- GPU delegate with automatic CPU fallback
- EXIF rotation handling
- ImageNet normalization
- Temperature-scaled softmax (temperature from manifest)
- Manifest-driven thresholds (OOD, per-rarity confidence, confusion-pair margins)
- Background / not-a-car, low-confidence, and ambiguity rejection
- Defensive output-tensor handling (dtype/shape read from the interpreter)
- Asset consistency validation (mapping vs manifest vs model output size)
- Thread-safe inference

## Requirements

- Android API 24+
- TFLite model in FP16 format (44 MB)
- `class_mapping.json` in assets (897 entries)
- `model_manifest.json` and `confusion_pairs.json` in assets (recommended; built-in fallbacks otherwise)
