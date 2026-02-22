package dev.autovision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.exifinterface.media.ExifInterface
import org.json.JSONArray
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * AutoVision — standalone car make/model/generation classifier.
 *
 * Runs EfficientNet-V2-S via TFLite with GPU delegate fallback to CPU.
 * Handles EXIF rotation, center crop, ImageNet normalization.
 *
 * Usage:
 *   val engine = AutoVision(context, "car_classifier.tflite")
 *   val results = engine.classify(bitmap, topK = 5)
 *   engine.close()
 */
class AutoVision(
    private val context: Context,
    modelFileName: String,
    classMappingFileName: String = "class_mapping.json",
    private val temperature: Float = 1.0f
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
        val inferenceTimeMs: Long = 0
    )

    private val interpreter: Interpreter
    private val gpuDelegate: GpuDelegate?
    private val classMapping: List<Map<String, Any?>>
    private val numClasses: Int

    companion object {
        private const val INPUT_SIZE = 384
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    init {
        // Load class mapping from assets
        classMapping = loadClassMapping(classMappingFileName)

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
        numClasses = interpreter.getOutputTensor(0).shape()[1]
    }

    /**
     * Classify a Bitmap image.
     *
     * @param bitmap Input image (any size, will be center-cropped and resized)
     * @param topK Number of top predictions to return
     * @return List of predictions sorted by confidence (descending)
     */
    fun classify(bitmap: Bitmap, topK: Int = 5): List<Prediction> {
        val inputBuffer = preprocessBitmap(bitmap)

        val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        val startTime = System.currentTimeMillis()
        interpreter.run(inputBuffer, outputBuffer)
        val inferenceTime = System.currentTimeMillis() - startTime

        outputBuffer.rewind()
        val logits = FloatArray(numClasses) { outputBuffer.float }

        return topKPredictions(logits, topK, inferenceTime)
    }

    /**
     * Classify an image from file path. Handles EXIF rotation.
     *
     * @param imagePath Absolute path to image file
     * @param topK Number of top predictions to return
     * @return List of predictions sorted by confidence (descending)
     */
    fun classifyFile(imagePath: String, topK: Int = 5): List<Prediction> {
        val bitmap = loadAndRotate(imagePath)
        val result = classify(bitmap, topK)
        bitmap.recycle()
        return result
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val cropped = centerCrop(bitmap)
        val resized = Bitmap.createScaledBitmap(cropped, INPUT_SIZE, INPUT_SIZE, true)
        if (cropped !== bitmap) cropped.recycle()

        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
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

    private fun topKPredictions(
        logits: FloatArray,
        topK: Int,
        inferenceTimeMs: Long
    ): List<Prediction> {
        // Temperature-scaled softmax
        val scaled = DoubleArray(numClasses) { logits[it].toDouble() / temperature }
        val maxScaled = scaled.max()
        val exps = DoubleArray(numClasses) { exp(scaled[it] - maxScaled) }
        val sumExp = exps.sum()
        val probs = DoubleArray(numClasses) { exps[it] / sumExp }

        // Get top-K indices
        val indices = (0 until numClasses).sortedByDescending { probs[it] }.take(topK)

        return indices.map { idx ->
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
                inferenceTimeMs = inferenceTimeMs
            )
        }
    }

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fd = context.assets.openFd(fileName)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    @Suppress("UNCHECKED_CAST")
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
