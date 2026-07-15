package dev.autovision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.min

/**
 * AutoVision — standalone car make/model/generation classifier.
 *
 * Runs EfficientNet-V2-S via TFLite with GPU delegate fallback to CPU.
 * Handles EXIF rotation, center crop, ImageNet normalization.
 *
 * v5.13 models ship with a manifest (input size, temperature, OOD threshold,
 * per-rarity confidence thresholds, confusion-pair margin) and an explicit
 * background class. Predictions are gated: background / out-of-distribution,
 * low-confidence, and ambiguous (confusion-pair) results are rejected.
 *
 * Usage:
 *   val engine = AutoVision(context, "car_classifier.tflite")
 *   val result = engine.classify(bitmap, topK = 5)
 *   if (!result.rejected) Log.d("AutoVision", result.top1!!.className)
 *   engine.close()
 */
class AutoVision(
    private val context: Context,
    modelFileName: String,
    classMappingFileName: String = "class_mapping.json",
    manifestFileName: String = "model_manifest.json",
    confusionPairsFileName: String = "confusion_pairs.json"
) : Closeable {

    data class Prediction(
        val index: Int,
        val className: String,
        val make: String,
        val model: String,
        val generation: String?,
        val yearStart: Int?,
        val yearEnd: Int?,
        val rarity: String,
        val confidence: Float,
        val logit: Float,
        val rank: Int
    )

    data class ClassificationResult(
        val predictions: List<Prediction>,   // top-K, background class excluded
        val top1: Prediction?,               // null when rejected
        val rejected: Boolean,
        val rejectionReason: String?,        // "not_a_car" | "low_confidence" | "ambiguous"
        val inferenceTimeMs: Long,
        val engineVersion: String,
        val modelVersion: String,
        val taxonomyVersion: String
    )

    data class ModelManifest(
        val version: String,
        val numClasses: Int,
        val inputSize: Int,
        val temperature: Float,
        val oodThreshold: Float,
        val confidenceThresholds: Map<String, Float>,
        val confusionPairMargin: Float
    )

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?
    private val manifest: ModelManifest
    private val classMapping: List<Map<String, Any?>>
    private val confusionPairs: Map<Pair<String, String>, Float>
    private val numClasses: Int
    private val outputNumBytes: Int
    private val inputSize: Int

    companion object {
        const val ENGINE_VERSION = "0.2.0"

        const val REJECT_NOT_A_CAR = "not_a_car"
        const val REJECT_LOW_CONFIDENCE = "low_confidence"
        const val REJECT_AMBIGUOUS = "ambiguous"

        private const val RARITY_BACKGROUND = "BACKGROUND"
        private const val FALLBACK_CONFIDENCE_THRESHOLD = 0.5f

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        private val DEFAULT_MANIFEST = ModelManifest(
            version = "5.13.0",
            numClasses = 897,
            inputSize = 384,
            temperature = 1.0f,
            oodThreshold = 0.05f,
            confidenceThresholds = mapOf(
                "COMMON" to 0.4f,
                "UNCOMMON" to 0.5f,
                "RARE" to 0.6f,
                "ULTRA_RARE" to 0.65f,
                "EPIC" to 0.67f,
                "LEGENDARY" to 0.7f
            ),
            confusionPairMargin = 0.08f
        )

        private fun pairKey(a: String, b: String): Pair<String, String> =
            if (a <= b) a to b else b to a
    }

    init {
        // Load manifest, class mapping, and confusion pairs from assets
        manifest = loadManifest(manifestFileName)
        classMapping = loadClassMapping(classMappingFileName)
        confusionPairs = loadConfusionPairs(confusionPairsFileName)
        inputSize = manifest.inputSize

        if (classMapping.size != manifest.numClasses) {
            throw IllegalStateException(
                "Class mapping / manifest mismatch: class_mapping has ${classMapping.size} " +
                    "entries but manifest declares ${manifest.numClasses} classes. " +
                    "Make sure all model assets come from the same model release."
            )
        }

        // Load model with GPU fallback
        val modelBuffer = loadModelFile(modelFileName)
        var delegate: GpuDelegate? = null
        var interp: Interpreter

        try {
            delegate = GpuDelegate()
            val options = Interpreter.Options()
                .addDelegate(delegate)
                .setNumThreads(4)
            interp = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            delegate?.close()
            delegate = null
            val options = Interpreter.Options().setNumThreads(4)
            interp = Interpreter(modelBuffer, options)
        }

        gpuDelegate = delegate
        interpreter = interp

        // FP16 models still expose a float32 output tensor via TFLite, but be
        // defensive: read dtype/shape from the interpreter instead of assuming.
        val outputTensor = interpreter.getOutputTensor(0)
        if (outputTensor.dataType() != DataType.FLOAT32) {
            throw IllegalStateException(
                "Unexpected output tensor dtype ${outputTensor.dataType()} " +
                    "(expected FLOAT32). FP16-quantized TFLite models still produce " +
                    "float32 outputs; this model is not compatible with this engine."
            )
        }
        val outputShape = outputTensor.shape()
        numClasses = outputShape[outputShape.size - 1]
        outputNumBytes = outputTensor.numBytes()

        if (classMapping.size != numClasses) {
            throw IllegalStateException(
                "Class mapping / model mismatch: class_mapping has ${classMapping.size} " +
                    "entries but the model output tensor has $numClasses classes. " +
                    "Make sure all model assets come from the same model release."
            )
        }
    }

    /**
     * Classify a Bitmap image.
     *
     * @param bitmap Input image (any size, will be center-cropped and resized)
     * @param topK Number of top predictions to return (background class excluded)
     * @return ClassificationResult with gated top-1 and top-K predictions
     */
    fun classify(bitmap: Bitmap, topK: Int = 5): ClassificationResult {
        val inputBuffer = preprocessBitmap(bitmap)

        val outputBuffer = ByteBuffer.allocateDirect(outputNumBytes)
        outputBuffer.order(ByteOrder.nativeOrder())

        val startTime = System.currentTimeMillis()
        interpreter.run(inputBuffer, outputBuffer)
        val inferenceTime = System.currentTimeMillis() - startTime

        outputBuffer.rewind()
        val logits = FloatArray(numClasses) { outputBuffer.float }

        return buildResult(logits, topK, inferenceTime)
    }

    /**
     * Classify an image from file path. Handles EXIF rotation.
     *
     * @param imagePath Absolute path to image file
     * @param topK Number of top predictions to return (background class excluded)
     * @return ClassificationResult with gated top-1 and top-K predictions
     */
    fun classifyFile(imagePath: String, topK: Int = 5): ClassificationResult {
        val bitmap = loadAndRotate(imagePath)
        val result = classify(bitmap, topK)
        bitmap.recycle()
        return result
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val cropped = centerCrop(bitmap)
        val resized = Bitmap.createScaledBitmap(cropped, inputSize, inputSize, true)
        if (cropped !== bitmap) cropped.recycle()

        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        resized.recycle()

        for (pixel in pixels) {
            val r = ((pixel shr 16 and 0xFF) / 255.0f - MEAN[0]) / STD[0]
            val g = ((pixel shr 8 and 0xFF) / 255.0f - MEAN[1]) / STD[1]
            val b = ((pixel and 0xFF) / 255.0f - MEAN[2]) / STD[2]
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        buffer.rewind()
        return buffer
    }

    private fun centerCrop(bitmap: Bitmap): Bitmap {
        val size = min(bitmap.width, bitmap.height)
        val x = (bitmap.width - size) / 2
        val y = (bitmap.height - size) / 2
        return Bitmap.createBitmap(bitmap, x, y, size, size)
    }

    private fun loadAndRotate(path: String): Bitmap {
        val bitmap = BitmapFactory.decodeFile(path)
            ?: throw IllegalArgumentException("Cannot decode image: $path")

        val exif = ExifInterface(path)
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
            else -> return bitmap
        }

        val rotated = Bitmap.createBitmap(
            bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
        )
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    private fun buildResult(
        logits: FloatArray,
        topK: Int,
        inferenceTimeMs: Long
    ): ClassificationResult {
        // Temperature-scaled softmax over all classes (background included)
        val scaled = DoubleArray(numClasses) { logits[it].toDouble() / manifest.temperature }
        val maxScaled = scaled.max()
        val exps = DoubleArray(numClasses) { exp(scaled[it] - maxScaled) }
        val sumExp = exps.sum()
        val probs = DoubleArray(numClasses) { exps[it] / sumExp }

        val ranked = (0 until numClasses).sortedByDescending { probs[it] }
        val carRanked = ranked.filter { rarityOf(it) != RARITY_BACKGROUND }

        // Top-K predictions always exclude the background class
        val predictions = carRanked.take(topK).mapIndexed { i, idx ->
            val cls = classMapping[idx]
            Prediction(
                index = idx,
                className = cls["class_name"] as String,
                make = cls["make"] as String,
                model = cls["model"] as String,
                generation = cls["generation"] as? String,
                yearStart = (cls["year_start"] as? Number)?.toInt(),
                yearEnd = (cls["year_end"] as? Number)?.toInt(),
                rarity = (cls["rarity"] as? String) ?: "UNKNOWN",
                confidence = probs[idx].toFloat(),
                logit = logits[idx],
                rank = i + 1
            )
        }

        val rejectionReason = gate(ranked, carRanked, probs)

        return ClassificationResult(
            predictions = predictions,
            top1 = if (rejectionReason == null) predictions.firstOrNull() else null,
            rejected = rejectionReason != null,
            rejectionReason = rejectionReason,
            inferenceTimeMs = inferenceTimeMs,
            engineVersion = ENGINE_VERSION,
            modelVersion = manifest.version,
            taxonomyVersion = "${manifest.version}-$numClasses"
        )
    }

    /** Returns a rejection reason, or null if the prediction is accepted. */
    private fun gate(
        ranked: List<Int>,
        carRanked: List<Int>,
        probs: DoubleArray
    ): String? {
        val topIdx = ranked[0]
        val topProb = probs[topIdx].toFloat()
        val topRarity = rarityOf(topIdx)

        // (a) argmax is the background class -> not a car
        if (topRarity == RARITY_BACKGROUND) return REJECT_NOT_A_CAR

        // (b) out-of-distribution: top prob below OOD threshold
        if (topProb < manifest.oodThreshold) return REJECT_NOT_A_CAR

        // (c) below the rarity-specific confidence threshold
        val threshold = manifest.confidenceThresholds[topRarity]
            ?: FALLBACK_CONFIDENCE_THRESHOLD
        if (topProb < threshold) return REJECT_LOW_CONFIDENCE

        // (d) top-2 car classes form a known confusion pair with too-small margin
        if (carRanked.size >= 2) {
            val name1 = classMapping[carRanked[0]]["class_name"] as String
            val name2 = classMapping[carRanked[1]]["class_name"] as String
            val margin = confusionPairs[pairKey(name1, name2)]
            if (margin != null) {
                val p1 = probs[carRanked[0]]
                val p2 = probs[carRanked[1]]
                if (p1 - p2 < margin) return REJECT_AMBIGUOUS
            }
        }

        return null
    }

    private fun rarityOf(index: Int): String =
        (classMapping[index]["rarity"] as? String) ?: "UNKNOWN"

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fd = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    private fun loadManifest(fileName: String): ModelManifest {
        val json = try {
            context.assets.open(fileName).bufferedReader().readText()
        } catch (e: Exception) {
            return DEFAULT_MANIFEST
        }

        val obj = JSONObject(json)

        val thresholds = mutableMapOf<String, Float>()
        val thresholdsObj = obj.optJSONObject("confidence_thresholds")
        if (thresholdsObj != null) {
            for (key in thresholdsObj.keys()) {
                thresholds[key] = thresholdsObj.getDouble(key).toFloat()
            }
        } else {
            thresholds.putAll(DEFAULT_MANIFEST.confidenceThresholds)
        }

        return ModelManifest(
            version = obj.optString("version", DEFAULT_MANIFEST.version),
            numClasses = obj.optInt("num_classes", DEFAULT_MANIFEST.numClasses),
            inputSize = obj.optInt("input_size", DEFAULT_MANIFEST.inputSize),
            temperature = obj.optDouble(
                "temperature", DEFAULT_MANIFEST.temperature.toDouble()
            ).toFloat(),
            oodThreshold = obj.optDouble(
                "ood_threshold", DEFAULT_MANIFEST.oodThreshold.toDouble()
            ).toFloat(),
            confidenceThresholds = thresholds,
            confusionPairMargin = obj.optDouble(
                "confusion_pair_margin", DEFAULT_MANIFEST.confusionPairMargin.toDouble()
            ).toFloat()
        )
    }

    private fun loadConfusionPairs(fileName: String): Map<Pair<String, String>, Float> {
        val json = try {
            context.assets.open(fileName).bufferedReader().readText()
        } catch (e: Exception) {
            return emptyMap()
        }

        val array = JSONArray(json)
        val result = mutableMapOf<Pair<String, String>, Float>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            val classA = obj.getString("class_a")
            val classB = obj.getString("class_b")
            val margin = if (obj.has("margin_threshold")) {
                obj.getDouble("margin_threshold").toFloat()
            } else {
                manifest.confusionPairMargin
            }
            result[pairKey(classA, classB)] = margin
        }
        return result
    }

    private fun loadClassMapping(fileName: String): List<Map<String, Any?>> {
        val json = context.assets.open(fileName).bufferedReader().readText()
        val array = JSONArray(json)
        val result = mutableListOf<Map<String, Any?>>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            result.add(
                mapOf(
                    "class_name" to obj.getString("class_name"),
                    "make" to obj.getString("make"),
                    "model" to obj.getString("model"),
                    "generation" to obj.optString("generation", null),
                    "year_start" to if (obj.has("year_start")) obj.getInt("year_start") else null,
                    "year_end" to if (obj.has("year_end")) obj.getInt("year_end") else null,
                    "rarity" to obj.optString("rarity", "UNKNOWN")
                )
            )
        }
        return result
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}
