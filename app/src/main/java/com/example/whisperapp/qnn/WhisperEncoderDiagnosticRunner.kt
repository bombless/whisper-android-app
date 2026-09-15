package com.example.whisperapp.qnn

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object WhisperEncoderDiagnosticRunner {
    private const val MODEL_ASSET = "models/whisper/ggml-tiny.bin"
    init { System.loadLibrary("whisper_reference") }
    data class Result(val logFile: File, val report: String, val result: String)
    fun run(context: Context, pcm16: ShortArray, sampleRate: Int, threads: Int = 1): Result {
        require(sampleRate == 16_000) { "Expected 16000 Hz, got $sampleRate" }
        require(pcm16.size == 80_000) { "Diagnostic expects exactly 80000 samples, got ${pcm16.size}" }
        val model = File(context.cacheDir, "whisper_reference/ggml-tiny.bin")
        if (!model.isFile || model.length() < 70_000_000L) {
            model.parentFile?.mkdirs()
            context.assets.open(MODEL_ASSET).use { input -> model.outputStream().use { output -> input.copyTo(output) } }
        }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.filesDir, "diagnostics").apply { mkdirs() }
        val file = File(dir, "encoder_diagnostic_$stamp.txt")
        val header = "Whisper Encoder Diagnostic v1\ntimestamp=$stamp\nsampleRate=$sampleRate\nsampleCount=${pcm16.size}\npaddedSampleCount=480000\nthreads=$threads\nmodelPath=${model.absolutePath}\n"
        val report = try { runNative(pcm16, model.absolutePath, threads) } catch (t: Throwable) { "ENCODER_TEST_END result=FAILED error=${t.stackTraceToString()}" }
        val full = header + report
        file.writeText(full)
        return Result(file, full, Regex("ENCODER_TEST_END result=(\\w+)").find(report)?.groupValues?.get(1) ?: "FAILED")
    }
    private external fun runNative(pcm: ShortArray, modelPath: String, threads: Int): String
}
