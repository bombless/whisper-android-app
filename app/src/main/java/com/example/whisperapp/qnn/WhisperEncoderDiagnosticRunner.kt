package com.example.whisperapp.qnn

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.util.Log

object WhisperEncoderDiagnosticRunner {
    private const val TAG = "WhisperEncoderDiagnostic"
    private const val MODEL_ASSET = "models/whisper/ggml-tiny.bin"
    init { System.loadLibrary("whisper_reference") }
    data class Result(val logFile: File, val report: String, val result: String)
    fun run(context: Context, pcm16: ShortArray, sampleRate: Int, threads: Int = 1): Result {
        Log.i(TAG, "DIAG_RUNNER_START samples=${pcm16.size} sampleRate=$sampleRate threads=$threads")
        require(sampleRate == 16_000) { "Expected 16000 Hz, got $sampleRate" }
        require(pcm16.size == 80_000) { "Diagnostic expects exactly 80000 samples, got ${pcm16.size}" }
        val model = File(context.cacheDir, "whisper_reference/ggml-tiny.bin")
        Log.i(TAG, "DIAG_MODEL_CHECK exists=${model.isFile} size=${if (model.isFile) model.length() else 0L}")
        if (!model.isFile || model.length() < 70_000_000L) {
            Log.i(TAG, "DIAG_MODEL_COPY_START")
            model.parentFile?.mkdirs()
            context.assets.open(MODEL_ASSET).use { input -> model.outputStream().use { output -> input.copyTo(output) } }
            Log.i(TAG, "DIAG_MODEL_COPY_DONE")
        } else {
            Log.i(TAG, "DIAG_MODEL_COPY_SKIP")
        }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.filesDir, "diagnostics").apply { mkdirs() }
        val file = File(dir, "encoder_diagnostic_$stamp.txt")
        Log.i(TAG, "DIAG_REPORT_FILE_CREATE path=${file.name}")
        val header = "Whisper Encoder Diagnostic v1\ntimestamp=$stamp\nsampleRate=$sampleRate\nsampleCount=${pcm16.size}\npaddedSampleCount=480000\nthreads=$threads\nmodelPath=${model.absolutePath}\n"
        Log.i(TAG, "DIAG_JNI_CALL_START")
        val report = try { runNative(pcm16, model.absolutePath, threads) } catch (t: Throwable) {
            Log.e(TAG, "DIAG_RUNNER_EXCEPTION type=${t::class.java.name} message=${t.message}")
            "ENCODER_TEST_END result=FAILED error=${t.stackTraceToString()}"
        }
        Log.i(TAG, "DIAG_JNI_CALL_DONE reportChars=${report.length}")
        val full = header + report
        file.writeText(full)
        Log.i(TAG, "DIAG_REPORT_WRITE_DONE file=${file.name} bytes=${file.length()}")
        val result = Regex("ENCODER_TEST_END result=(\\w+)").find(report)?.groupValues?.get(1) ?: "FAILED"
        return Result(file, full, result)
    }
    private external fun runNative(pcm: ShortArray, modelPath: String, threads: Int): String
}
