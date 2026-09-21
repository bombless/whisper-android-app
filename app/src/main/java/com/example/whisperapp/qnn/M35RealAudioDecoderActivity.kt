package com.example.whisperapp.qnn

import android.app.Activity
import android.os.Bundle
import android.util.Log
import com.example.whisperapp.asr.WhisperVariant
import com.example.whisperapp.audio.WavPcmReader
import java.io.File
/**
 * Runs a fixed WAV through the QNN Whisper path and writes a report, giving a
 * deterministic alternative to the live-microphone flow.
 *
 *   adb shell am start -n com.example.whisperapp/.qnn.M35RealAudioDecoderActivity --es variant turbo
 *
 * Reads `<externalFilesDir>/output.wav` (16 kHz mono PCM16).
 */
class M35RealAudioDecoderActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            var variant = WhisperVariant.TINY
            try {
                variant = when (intent.getStringExtra("variant")?.lowercase()) {
                    "turbo", "large", "large_v3_turbo", "large-v3-turbo" -> WhisperVariant.LARGE_V3_TURBO
                    else -> WhisperVariant.TINY
                }
                Log.i(TAG, "START variant=${variant.name} assetDir=${variant.assetDir}")
                // OEM ROMs on this device suppress this app's logcat output, so stage
                // breadcrumbs go to a file that survives a hang or a native crash.
                OpTrace.stageLogFile(File(getExternalFilesDir(null), "decoder_stages.log"))
                QnnWhisperDecoderJni.setStageLog(File(getExternalFilesDir(null), "decoder_stages.log").absolutePath)
                OpTrace.stage("activity.start variant=${variant.name}")
                // Diagnostic HTP/EP overrides, e.g.
                //   --es perf burst --ei vtcm 8 --ei fin 3 --ei rpc 100 --es prio high --ei repeats 2
                val htpOverrides = buildMap {
                    intent.getStringExtra("perf")?.let { put("htp_performance_mode", it) }
                    intent.getStringExtra("vtcm")?.let { put("vtcm_mb", it) }
                    intent.getStringExtra("fin")?.let { put("htp_graph_finalization_optimization_mode", it) }
                    intent.getStringExtra("rpc")?.let { put("rpc_control_latency", it) }
                    intent.getStringExtra("prio")?.let { put("qnn_context_priority", it) }
                    intent.getStringExtra("extra_key")?.let { key ->
                        put(key, intent.getStringExtra("extra_value").orEmpty())
                    }
                    // Run-level QNN option, e.g. --es runopt "qnn.perf_mode=burst"
                    intent.getStringExtra("runopt")?.let { kv ->
                        val split = kv.indexOf('=')
                        if (split > 0) put(kv.substring(0, split), kv.substring(split + 1))
                    }
                }
                val repeats = intent.getIntExtra("repeats", 0)
                // `--es decode java` forces the reference host-tensor loop; anything else uses
                // the native IoBinding loop when it is available. Runs must be separate
                // processes (see the note at the run() call below).
                // `--es decode java` forces the reference host-tensor loop; `--es decode
                // native` opts into the P1 IoBinding loop, which is off by default because it
                // measured no faster than Java (see the runner's nativeDecodeEnabled doc).
                // Runs must be separate processes (see the note at the run() call below).
                val forceJava = intent.getStringExtra("decode") == "java"
                val forceNative = intent.getStringExtra("decode") == "native"
                QnnWhisperRealAudioRunner.configureHtp(htpOverrides, repeats, nativeDecode = forceNative)
                OpTrace.stage("activity.configured repeats=$repeats forceJava=$forceJava forceNative=$forceNative")
                Log.i(TAG, "HTP_OVERRIDES=$htpOverrides repeats=$repeats forceJava=$forceJava forceNative=$forceNative")
                val wav = File(getExternalFilesDir(null), "output.wav")
                Log.i(TAG, "WAV path=${wav.absolutePath} exists=${wav.isFile} bytes=${if (wav.isFile) wav.length() else -1}")
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                OpTrace.stage("activity.wav_read samples=${pcm.samples.size} rate=${pcm.sampleRate}")
                Log.i(TAG, "PCM samples=${pcm.samples.size} rate=${pcm.sampleRate} channels=${pcm.channels}")
                // A/B mode (P1 acceptance): decode the same WAV through the native IoBinding
                // loop, reporting enough detail to compare against a separate java-path run.
                // The two paths are measured in *separate processes*: `run()` tears its
                // sessions down in `finally`, and re-entering `start()` re-registers the QNN
                // EP plugin, which ORT permits only once per process (it segfaults inside
                // libonnxruntime.so). Comparing across processes also removes any doubt about
                // one pass warming the other.
                val result = QnnWhisperRealAudioRunner.run(
                    context = this,
                    pcm16 = pcm.samples,
                    sampleRate = pcm.sampleRate,
                    requestedSteps = 128,
                    autoregressive = true,
                    variant = variant,
                )
                OpTrace.stage("activity.run_returned passed=${result.passed}")
                File(getExternalFilesDir(null), resultFileName(variant)).writeText(result.report)
                // `--es decode java` writes the reference the native run is compared against;
                // `--es decode native` produces the comparison. A default (no `decode`) run is
                // just a normal Java-path transcription and writes neither.
                if (forceJava) {
                    File(getExternalFilesDir(null), "p1_reference_${variant.name.lowercase()}.txt").writeText(result.report)
                    OpTrace.stage("activity.reference_saved")
                } else if (forceNative) {
                    compareWithReference(variant, result.report)
                }
                Log.i(TAG, "RESULT_WRITTEN variant=${variant.name} passed=${result.passed}")
                Log.i(TAG, "DECODED_TEXT variant=${variant.name} text=\"${result.text}\"")
            } catch (t: Throwable) {
                runCatching {
                    File(getExternalFilesDir(null), resultFileName(variant)).writeText("ACTIVITY EXCEPTION\n${t.stackTraceToString()}")
                }
                Log.e(TAG, "ACTIVITY EXCEPTION", t)
            }
            runOnUiThread { finish() }
        }.start()
    }

    private fun resultFileName(variant: WhisperVariant) =
        "m35_real_audio_decoder_result_${variant.name.lowercase()}.txt"

    /**
     * Compares a freshly written result report against a previously captured reference run.
     *
     * The two P1 decode paths cannot be measured in one process (see the note at the run()
     * call), so the reference is produced by a second invocation with `--es decode java` and
     * the comparison happens here on the file contents.
     */
    private fun compareWithReference(variant: WhisperVariant, current: String) {
        val referenceFile = File(getExternalFilesDir(null), "p1_reference_${variant.name.lowercase()}.txt")
        if (!referenceFile.isFile) {
            OpTrace.stage("ab.no_reference file=${referenceFile.name}")
            return
        }
        fun field(report: String, name: String): String =
            report.lineSequence().firstOrNull { it.startsWith("$name:") }?.substringAfter(": ")?.trim().orEmpty()
        val reference = referenceFile.readText()
        val currentTokens = field(current, "token_ids")
        val referenceTokens = field(reference, "token_ids")
        val out = buildString {
            appendLine("P1 A/B COMPARISON (variant=${variant.name})")
            appendLine("tokens_identical: ${currentTokens == referenceTokens}")
            appendLine("native_token_ids: $currentTokens")
            appendLine("java_token_ids: $referenceTokens")
            appendLine("native_text: ${field(current, "decoded_text")}")
            appendLine("java_text: ${field(reference, "decoded_text")}")
            for (name in listOf("decoder_path", "native_step_p50_ms", "native_step_min_ms", "native_step_max_ms", "native_bind_ms", "encoder_host_copy_ms", "encoder_ms", "steps_completed")) {
                appendLine("current_$name: ${field(current, name)}")
            }
            for (name in listOf("decoder_path", "encoder_host_copy_ms", "encoder_ms", "steps_completed")) {
                appendLine("reference_$name: ${field(reference, name)}")
            }
        }
        File(getExternalFilesDir(null), "p1_ab_comparison.txt").writeText(out)
        OpTrace.stage("ab.report_written identical=${currentTokens == referenceTokens}")
    }

    private companion object {
        const val TAG = "M35_REAL_DECODER"
    }
}
