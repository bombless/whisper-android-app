package io.github.bombless.htp_playpen.asr

import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Debug-only sample-audio test entry point; does not request or use microphone permission. */
class SenseVoiceDebugSmokeActivity : ComponentActivity() {
    private lateinit var output: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        output = TextView(this).apply { textSize = 16f; setPadding(32, 48, 32, 48); text = "SenseVoice smoke test running…" }
        setContentView(ScrollView(this).apply { addView(output) })
        lifecycleScope.launch {
            val report = withContext(Dispatchers.IO) { runSmokeTest() }
            output.text = report
        }
    }

    private suspend fun runSmokeTest(): String {
        val engine = SenseVoiceAsrEngine(applicationContext, language = SenseVoiceAsrEngine.LANGUAGE_AUTO, numThreads = 2)
        val lines = mutableListOf("SenseVoiceSmall INT8 · CPU · auto")
        try {
            Log.i(TAG, "SMOKE_BEGIN")
            engine.prepare()
            for (file in listOf("zh.wav", "ja.wav")) {
                val bytes = assets.open("sensevoice_test_wavs/$file").use { it.readBytes() }
                val wav = parsePcm16Wav(bytes)
                val result = engine.transcribe(wav.samples, wav.sampleRate)
                check(result.text.isNotBlank()) { "Empty transcription for $file" }
                lines += "$file · ${result.latencyMs} ms\n${result.text}"
                Log.i(TAG, "SMOKE_RESULT file=$file latencyMs=${result.latencyMs} text=${result.text}")
            }
            lines += "PASS"
            Log.i(TAG, "SMOKE_PASS")
        } catch (failure: Throwable) {
            lines += "FAIL: ${failure.message ?: failure.javaClass.simpleName}"
            Log.e(TAG, "SMOKE_FAIL", failure)
        } finally {
            engine.close()
        }
        return lines.joinToString("\n\n")
    }

    private data class PcmWav(val samples: ShortArray, val sampleRate: Int)

    private fun parsePcm16Wav(bytes: ByteArray): PcmWav {
        require(bytes.size >= 44 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF")
        require(String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE")
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var dataOffset = -1
        var dataSize = 0
        var offset = 12
        while (offset + 8 <= bytes.size) {
            val id = String(bytes, offset, 4, Charsets.US_ASCII)
            val size = buffer.getInt(offset + 4).coerceAtLeast(0)
            val body = offset + 8
            if (body + size > bytes.size) break
            when (id) {
                "fmt " -> {
                    channels = buffer.getShort(body + 2).toInt()
                    sampleRate = buffer.getInt(body + 4)
                    bitsPerSample = buffer.getShort(body + 14).toInt()
                }
                "data" -> { dataOffset = body; dataSize = size; break }
            }
            offset = body + size + (size and 1)
        }
        require(channels == 1 && bitsPerSample == 16 && sampleRate > 0 && dataOffset >= 0) {
            "Expected mono PCM16 WAV"
        }
        val samples = ShortArray(dataSize / 2) { buffer.getShort(dataOffset + it * 2) }
        return PcmWav(samples, sampleRate)
    }

    companion object { private const val TAG = "SENSEVOICE_SMOKE" }
}