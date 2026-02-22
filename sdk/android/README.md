# AutoVision Android SDK

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

3. Place model files in `assets/` or download at runtime
4. Initialize and classify:

```kotlin
val autoVision = AutoVision(context, "car_classifier.tflite")
val results = autoVision.classify(bitmap, topK = 5)

results.forEach { prediction ->
    Log.d("AutoVision", "${prediction.make} ${prediction.model}")
    Log.d("AutoVision", "  Years: ${prediction.yearStart}-${prediction.yearEnd}")
    Log.d("AutoVision", "  Confidence: ${"%.1f%%".format(prediction.confidence * 100)}")
}

// Clean up when done
autoVision.close()
```

## Features

- GPU delegate with automatic CPU fallback
- EXIF rotation handling
- ImageNet normalization
- Temperature-scaled softmax
- Thread-safe inference

## Requirements

- Android API 24+
- TFLite model in float32 format (83 MB)
- `class_mapping.json` in assets
