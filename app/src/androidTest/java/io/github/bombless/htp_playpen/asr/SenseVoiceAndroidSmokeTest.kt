package io.github.bombless.htp_playpen.asr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class SenseVoiceAndroidSmokeTest {
    @Test
    fun recognizesMandarinAndJapaneseReferenceAudio() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val engine = SenseVoiceAsrEngine(context, language = SenseVoiceAsrEngine.LANGUAGE_AUTO, numThreads = 2)
        try {
            engine.prepare()
            for (name in listOf("zh.wav", "ja.wav")) {
                val wav = instrumentation.context.assets.open("test_wavs/$name").use { it.readBytes() }
                val parsed = parseMonoPcm16Wav(wav)
                assertTrue("Unexpected sample rate in $name", parsed.sampleRate > 0)
                val result = engine.transcribe(parsed.samples, parsed.sampleRate)
                Log.i(TAG, "SMOKE_RESULT file=$name latencyMs=${result.latencyMs} text=${result.text}")
                assertFalse("SenseVoice returned empty text for $name", result.text.isBlank())
            }
        } finally {
            engine.close()
        }
    }

    private data class WavPcm(val samples: ShortArray, val sampleRate: Int)

    private fun parseMonoPcm16Wav(bytes: ByteArray): WavPcm {
        require(bytes.size >= 44) { "WAV header is truncated" }
        require(String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF")
        require(String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE")
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var sampleRate = 0
        var channels = 0
        var bitsPerSample = 0
        var dataOffset = -1
        var dataSize = 0
        var offset = 12
        while (offset + 8 <= bytes.size) {
            val chunkId = String(bytes, offset, 4, Charsets.US_ASCII)
            val chunkSize = buffer.getInt(offset + 4).coerceAtLeast(0)
            val body = offset + 8
            if (body + chunkSize > bytes.size) break
            when (chunkId) {
                "fmt " -> {
                    channels = buffer.getShort(body + 2).toInt()
                    sampleRate = buffer.getInt(body + 4)
                    bitsPerSample = buffer.getShort(body + 14).toInt()
                }
                "data" -> {
                    dataOffset = body
                    dataSize = chunkSize
                    break
                }
            }
            offset = body + chunkSize + (chunkSize and 1)
        }
        require(channels == 1 && bitsPerSample == 16 && dataOffset >= 0) { "Expected mono PCM16 WAV" }
        val count = dataSize / 2
        val samples = ShortArray(count) { buffer.getShort(dataOffset + it * 2) }
        return WavPcm(samples, sampleRate)
    }

    companion object { private const val TAG = "SENSEVOICE_SMOKE" }
}