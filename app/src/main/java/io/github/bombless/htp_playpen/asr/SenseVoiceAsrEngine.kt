package io.github.bombless.htp_playpen.asr

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineSenseVoiceModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Offline SenseVoiceSmall INT8 engine backed by the project's sherpa-onnx 1.13.8 AAR. */
class SenseVoiceAsrEngine(
    context: Context,
    private val language: String = LANGUAGE_AUTO,
    private val numThreads: Int = DEFAULT_NUM_THREADS,
    private val useInverseTextNormalization: Boolean = true,
) : AsrEngine {
    private val appContext = context.applicationContext
    private val lock = Any()
    @Volatile private var recognizer: OfflineRecognizer? = null
    @Volatile private var closed = false

    init {
        require(language in SUPPORTED_LANGUAGES) { "Unsupported SenseVoice language: $language" }
        require(numThreads > 0) { "numThreads must be positive" }
    }

    /** Loads native model state off the main thread. Safe to call repeatedly. */
    suspend fun prepare() = withContext(Dispatchers.IO) {
        synchronized(lock) { ensureRecognizerLocked() }
    }

    override suspend fun transcribe(pcm16: ShortArray, sampleRate: Int): AsrResult =
        withContext(Dispatchers.Default) {
            require(sampleRate > 0) { "sampleRate must be positive" }
            require(pcm16.isNotEmpty()) { "PCM input is empty" }
            val started = System.nanoTime()
            val resampled = Pcm16Resampler.resample(pcm16, sampleRate, TARGET_SAMPLE_RATE)
            val samples = FloatArray(resampled.size) { index ->
                (resampled[index].toInt() / 32768.0f).coerceIn(-1.0f, 1.0f)
            }
            val text = synchronized(lock) {
                check(!closed) { "SenseVoice engine is closed" }
                val activeRecognizer = ensureRecognizerLocked()
                val stream = activeRecognizer.createStream()
                try {
                    stream.acceptWaveform(samples, TARGET_SAMPLE_RATE)
                    activeRecognizer.decode(stream)
                    // SenseVoice's language/emotion/event markers are separate result fields;
                    // the UI consumes only the recognized text.
                    activeRecognizer.getResult(stream).text
                        .replace(ANGLE_TAG_PATTERN, "")
                        .trim()
                } finally {
                    stream.release()
                }
            }
            AsrResult(text, (System.nanoTime() - started) / 1_000_000L)
        }

    private fun ensureRecognizerLocked(): OfflineRecognizer {
        check(!closed) { "SenseVoice engine is closed" }
        recognizer?.let { return it }
        val modelDir = MODEL_ASSET_DIR
        val senseVoice = OfflineSenseVoiceModelConfig(
            model = "$modelDir/model.int8.onnx",
            language = language,
            useInverseTextNormalization = useInverseTextNormalization,
        )
        val modelConfig = OfflineModelConfig(
            senseVoice = senseVoice,
            tokens = "$modelDir/tokens.txt",
            numThreads = numThreads,
            provider = "cpu",
        )
        val config = OfflineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = TARGET_SAMPLE_RATE, featureDim = 80, dither = 0.0f),
            modelConfig = modelConfig,
        )
        Log.i(TAG, "Loading SenseVoice INT8 language=$language threads=$numThreads provider=cpu")
        return OfflineRecognizer(appContext.assets, config).also { recognizer = it }
    }

    override fun close() {
        synchronized(lock) {
            if (closed) return
            closed = true
            recognizer?.release()
            recognizer = null
        }
    }

    companion object {
        const val TARGET_SAMPLE_RATE = 16_000
        const val LANGUAGE_AUTO = "auto"
        const val DEFAULT_NUM_THREADS = 4
        const val MODEL_ASSET_DIR = "models/asr/sense_voice_small_int8"
        private const val TAG = "SENSEVOICE_ASR"
        private val SUPPORTED_LANGUAGES = setOf("auto", "zh", "en", "ja", "ko", "yue")
        private val ANGLE_TAG_PATTERN = Regex("<\\|[^|>]+\\|>")
    }
}

/** Linear-interpolation PCM16 resampler; never relabels samples with a different rate. */
object Pcm16Resampler {
    fun resample(input: ShortArray, sourceRate: Int, targetRate: Int): ShortArray {
        require(sourceRate > 0 && targetRate > 0) { "Sample rates must be positive" }
        if (input.isEmpty() || sourceRate == targetRate) return input.copyOf()
        val outputSize = ((input.size.toLong() * targetRate + sourceRate / 2) / sourceRate)
            .coerceAtLeast(1L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        val output = ShortArray(outputSize)
        val ratio = sourceRate.toDouble() / targetRate.toDouble()
        for (i in output.indices) {
            val position = i * ratio
            val left = position.toInt().coerceAtMost(input.lastIndex)
            val right = (left + 1).coerceAtMost(input.lastIndex)
            val fraction = (position - left).coerceIn(0.0, 1.0)
            val value = input[left] * (1.0 - fraction) + input[right] * fraction
            output[i] = value.toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return output
    }
}