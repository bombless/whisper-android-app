package com.example.whisperapp.qnn

import android.content.Context
import java.io.File
import java.security.MessageDigest

object WhisperReferenceRunner {
    private const val MODEL_ASSET = "models/whisper/ggml-tiny.bin"
    private const val SAMPLE_RATE = 16_000
    private const val SAMPLE_COUNT = 80_000

    init {
        System.loadLibrary("whisper_reference")
    }

    data class Result(
        val report: String,
        val pcmSha256: String,
    )

    fun run(context: Context, pcm16: ShortArray, sampleRate: Int, threads: Int = 4): Result {
        require(sampleRate == SAMPLE_RATE) { "Reference expects 16000 Hz, got $sampleRate" }
        require(pcm16.size == SAMPLE_COUNT) { "Reference expects exactly 80000 samples, got ${pcm16.size}" }

        val model = File(context.cacheDir, "whisper_reference/ggml-tiny.bin")
        if (!model.isFile || model.length() < 70_000_000L) {
            model.parentFile?.mkdirs()
            context.assets.open(MODEL_ASSET).use { input ->
                model.outputStream().use { output -> input.copyTo(output) }
            }
        }
        val sha256 = sha256Pcm16(pcm16)
        val report = runNative(pcm16, model.absolutePath, threads)
        return Result("WHISPER_REF_PCM_SHA256=$sha256\n$report", sha256)
    }

    private fun sha256Pcm16(samples: ShortArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = ByteArray(samples.size * 2)
        var p = 0
        for (sample in samples) {
            val v = sample.toInt()
            bytes[p++] = (v and 0xff).toByte()
            bytes[p++] = ((v ushr 8) and 0xff).toByte()
        }
        return digest.digest(bytes).joinToString("") { "%02x".format(it) }
    }

    fun transcribeFirst5s(context: Context, pcm16: ShortArray, sampleRate: Int, threads: Int = 1): String {
        require(sampleRate == SAMPLE_RATE) { "Reference expects 16000 Hz, got $sampleRate" }
        require(pcm16.size == SAMPLE_COUNT) { "Reference expects exactly 80000 samples, got ${pcm16.size}" }
        val model = File(context.cacheDir, "whisper_reference/ggml-tiny.bin")
        if (!model.isFile || model.length() < 70_000_000L) {
            model.parentFile?.mkdirs()
            context.assets.open(MODEL_ASSET).use { input -> model.outputStream().use { output -> input.copyTo(output) } }
        }
        return transcribeNative(pcm16, model.absolutePath, threads)
    }

    private external fun runNative(pcm: ShortArray, modelPath: String, threads: Int): String

    private external fun transcribeNative(pcm: ShortArray, modelPath: String, threads: Int): String
}
