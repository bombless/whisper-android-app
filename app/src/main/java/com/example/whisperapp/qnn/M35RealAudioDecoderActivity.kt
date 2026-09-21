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
                QnnWhisperRealAudioRunner.configureHtp(htpOverrides, repeats)
                Log.i(TAG, "HTP_OVERRIDES=$htpOverrides repeats=$repeats")
                val wav = File(getExternalFilesDir(null), "output.wav")
                Log.i(TAG, "WAV path=${wav.absolutePath} exists=${wav.isFile} bytes=${if (wav.isFile) wav.length() else -1}")
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                Log.i(TAG, "PCM samples=${pcm.samples.size} rate=${pcm.sampleRate} channels=${pcm.channels}")
                val result = QnnWhisperRealAudioRunner.run(
                    context = this,
                    pcm16 = pcm.samples,
                    sampleRate = pcm.sampleRate,
                    requestedSteps = 128,
                    autoregressive = true,
                    variant = variant,
                )
                File(getExternalFilesDir(null), resultFileName(variant)).writeText(result.report)
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

    private companion object {
        const val TAG = "M35_REAL_DECODER"
    }
}
