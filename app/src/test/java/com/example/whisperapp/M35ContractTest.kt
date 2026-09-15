package com.example.whisperapp

import com.example.whisperapp.asr.WhisperTokenizer
import com.example.whisperapp.audio.WhisperFeatureExtractor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.PI
import kotlin.math.sin

class M35ContractTest {
    private val tokenizerDir = File("src/main/assets/models/whisper/tokenizer")

    @Test
    fun tokenizerAssetsEncodeDecodeEnglishSmoke() {
        val tokenizer = WhisperTokenizer.fromAssetTexts(
            File(tokenizerDir, "vocab.json").readText(),
            File(tokenizerDir, "merges.txt").readText(),
            File(tokenizerDir, "added_tokens.json").readText(),
        )

        assertEquals(50257, tokenizer.tokenToId("<|endoftext|>"))
        assertEquals(50258, tokenizer.tokenToId("<|startoftranscript|>"))
        assertEquals(50259, tokenizer.tokenToId("<|en|>"))
        assertEquals(50359, tokenizer.tokenToId("<|transcribe|>"))
        assertEquals(50363, tokenizer.tokenToId("<|notimestamps|>"))

        for (text in listOf("hello", "hello world", "test")) {
            val ids = tokenizer.encode(text)
            assertTrue("$text must encode to at least one token", ids.isNotEmpty())
            assertEquals(text, tokenizer.decode(ids))
        }

        assertEquals("", tokenizer.decode(intArrayOf(50258, 50259, 50359, 50363, 50257)))
    }

    @Test
    fun whisperFeatureExtractorMatchesReferenceSineStatistics() {
        val sampleRate = WhisperFeatureExtractor.SAMPLE_RATE
        val pcm = ShortArray(sampleRate)
        for (i in pcm.indices) {
            pcm[i] = (0.5 * 32768.0 * sin(2.0 * PI * 1000.0 * i / sampleRate)).toInt().toShort()
        }

        val features = WhisperFeatureExtractor().extract(pcm, sampleRate)
        assertTrue(features.data.all { it.isFinite() })
        assertEquals(1L, features.shape[0])
        assertEquals(80L, features.shape[1])
        assertEquals(3000L, features.shape[2])

        // Reference generated independently from Hugging Face's WhisperFeatureExtractor contract:
        // Slaney mel scale/filter normalization, Hann STFT, reflect centering, log10, 8 dB clamp, +4 / 4.
        assertEquals(0.79974797f, features.data[0], 1.0e-4f) // mel 0, frame 0
        assertEquals(0.28634482f, features.data[1], 1.0e-4f) // mel 0, frame 1
        assertEquals(0.83937876f, features.data[10 * 3000], 1.0e-4f)
        assertEquals(-0.56034781f, features.data[10 * 3000 + 50], 1.0e-4f)
        assertEquals(-0.55601221f, features.data.average().toFloat(), 1.0e-4f)
        assertEquals(-0.56034781f, features.data.minOrNull()!!, 1.0e-4f)
        assertEquals(1.43965219f, features.data.maxOrNull()!!, 1.0e-4f)
    }
}
