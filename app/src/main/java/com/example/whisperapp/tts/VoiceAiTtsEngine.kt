package com.example.whisperapp.tts

import android.content.Context
import com.qualcomm.qti.voice.assist.tts.sdk.TTS
import com.qualcomm.qti.voice.assist.tts.sdk.TTSResultCallback
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CancellationException
import java.io.ByteArrayOutputStream
import java.io.File

/** Qualcomm VoiceAI adapter for the packaged MeloTTS-ZH model. */
class VoiceAiTtsEngine(
    private val context: Context,
    private val skelLibPath: String? = null,
) : TtsEngine {
    companion object {
        private const val ASSET_DIR = "models/tts/melotts_zh"
        private const val MODEL_DIR_NAME = "melotts_zh"
        private const val LANGUAGE = "Chinese"
        private const val AUDIO_ENCODING_LINEAR16 = "0"
        private const val SPEECH_RATE = "1.0"
        private const val PITCH = "0.0"
        private const val VOLUME_GAIN = "0.0"
        private const val SAMPLE_RATE = "44100"

        init {
            System.loadLibrary("tts")
            System.loadLibrary("tts_jni")
        }
    }

    private val lock = Any()
    private var tts: TTS? = null
    private var modelDir: File? = null
    private var active: Request? = null

    private data class Request(
        val deferred: CompletableDeferred<TtsResult>,
        val pcm: ByteArrayOutputStream = ByteArrayOutputStream(),
        var sampleRate: Int = 44_100,
        val startedAt: Long = android.os.SystemClock.elapsedRealtime(),
    )

    private val callback = object : TTSResultCallback {
        override fun onStart(sampleRateInHz: Int, audioFormat: Int, channelCount: Int) {
            synchronized(lock) { active?.sampleRate = sampleRateInHz }
        }

        override fun onAudioAvailable(buff: ByteArray, offset: Int, size: Int) {
            synchronized(lock) { active?.let { if (size > 0) it.pcm.write(buff, offset, size) } }
        }

        override fun onDone() {
            synchronized(lock) {
                val request = active ?: return
                active = null
                val bytes = request.pcm.toByteArray()
                val samples = ShortArray(bytes.size / 2)
                for (i in samples.indices) {
                    val lo = bytes[i * 2].toInt() and 0xff
                    val hi = bytes[i * 2 + 1].toInt()
                    samples[i] = ((hi shl 8) or lo).toShort()
                }
                request.deferred.complete(
                    TtsResult(
                        pcm16 = samples,
                        sampleRate = request.sampleRate,
                        latencyMs = android.os.SystemClock.elapsedRealtime() - request.startedAt,
                    )
                )
            }
        }

        override fun onError(errorCode: Int) {
            synchronized(lock) {
                val request = active ?: return
                active = null
                request.deferred.completeExceptionally(
                    IllegalStateException("VoiceAI TTS error code=$errorCode")
                )
            }
        }

        override fun onPerformanceResult(processingTimeMs: Int, audioLengthInMs: Int, realTimeFactor: Float) = Unit
    }

    override suspend fun synthesize(text: String): TtsResult {
        val instance = ensureInitialized()
        val request = Request(CompletableDeferred())
        synchronized(lock) {
            check(active == null) { "VoiceAI TTS already has an active request" }
            active = request
            instance.start(text)
        }
        return try {
            request.deferred.await()
        } catch (cancelled: CancellationException) {
            synchronized(lock) {
                if (active === request) {
                    active = null
                    runCatching { instance.stop() }
                }
            }
            throw cancelled
        }
    }

    private fun ensureInitialized(): TTS {
        synchronized(lock) {
            tts?.let { return it }
            val dir = prepareModelDir()
            val instance = TTS.newInstance(callback)
            instance.setLanguage(LANGUAGE)
            instance.setModelPath(dir.absolutePath)
            instance.setAudioEncoding(AUDIO_ENCODING_LINEAR16)
            instance.setSpeechRate(SPEECH_RATE)
            instance.setPitch(PITCH)
            instance.setVolumeGain(VOLUME_GAIN)
            instance.setSampleRate(SAMPLE_RATE)
            if (!skelLibPath.isNullOrBlank()) instance.setSkelLibPath(skelLibPath)
            instance.init()
            modelDir = dir
            tts = instance
            return instance
        }
    }

    private fun prepareModelDir(): File {
        val target = File(context.filesDir, MODEL_DIR_NAME)
        val marker = File(target, ".installed")
        if (!marker.exists()) {
            target.mkdirs()
            val names = context.assets.list(ASSET_DIR).orEmpty()
            require(names.isNotEmpty()) { "MeloTTS-ZH VoiceAI assets are missing: $ASSET_DIR" }
            for (name in names) {
                context.assets.open("$ASSET_DIR/$name").use { input ->
                    File(target, name).outputStream().use { output -> input.copyTo(output) }
                }
            }
            marker.writeText("1")
        }
        return target
    }

    override fun close() {
        synchronized(lock) {
            active?.let { request ->
                request.deferred.completeExceptionally(IllegalStateException("VoiceAI TTS closed"))
                active = null
            }
            tts?.let { instance ->
                runCatching { instance.stop() }
                runCatching { instance.deInit() }
                runCatching { instance.release() }
            }
            tts = null
        }
    }
}
