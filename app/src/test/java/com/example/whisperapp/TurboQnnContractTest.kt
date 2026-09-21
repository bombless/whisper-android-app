package com.example.whisperapp

import com.example.whisperapp.asr.WhisperTokenizer
import com.example.whisperapp.asr.WhisperVariant
import com.example.whisperapp.audio.WhisperMelConfig
import com.example.whisperapp.audio.WhisperTurboFeatureExtractor
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Contract tests for the Whisper-Large-V3-Turbo QNN variant.
 *
 * The QNN payload (encoder.bin / decoder.bin) is an opaque context blob, so it is not
 * parsed here. `tools/verify_turbo_epcontext.py` checks the generated EPContext wrappers
 * against the asset's metadata.json. These tests cover what the JVM can check on its own:
 * the Large-V3 special-token table, the variant's declared shapes, and the new 128-bin
 * incremental mel cache.
 */
class TurboQnnContractTest {

    private fun turboAssetDir(): File =
        sequenceOf(
            File("src/main/assets/models/whisper_large_v3_turbo"),
            File("../app/src/main/assets/models/whisper_large_v3_turbo"),
        ).firstOrNull { it.isDirectory }
            ?: error("turbo asset dir not found from ${File(".").absolutePath}")

    private fun loadTurboTokenizer(): WhisperTokenizer {
        val dir = File(turboAssetDir(), "tokenizer")
        return WhisperTokenizer.fromAssetTexts(
            File(dir, "vocab.json").readText(),
            File(dir, "merges.txt").readText(),
            File(dir, "added_tokens.json").readText(),
            WhisperVariant.LARGE_V3_TURBO.vocabSize,
        )
    }

    @Test
    fun turboAssetDirectoryIsComplete() {
        val dir = turboAssetDir()
        for (name in listOf("encoder.bin", "decoder.bin", "metadata.json", "encoder_ctx.onnx", "decoder_ctx.onnx")) {
            val file = File(dir, name)
            assertTrue("missing turbo asset $name at ${file.absolutePath}", file.isFile)
            assertTrue("empty turbo asset $name", file.length() > 0)
        }
        for (name in listOf("vocab.json", "merges.txt", "added_tokens.json")) {
            val file = File(dir, "tokenizer/$name")
            assertTrue("missing turbo tokenizer asset $name", file.isFile)
        }
    }

    @Test
    fun largeV3TokenizerShiftsTranscribeAndNoTimestamps() {
        val tokenizer = loadTurboTokenizer()
        assertEquals(51866, tokenizer.vocabularySize)

        // Base special tokens are shared with Tiny.
        assertEquals(50257, tokenizer.idFor("<|endoftext|>"))
        assertEquals(50258, tokenizer.idFor("<|startoftranscript|>"))
        assertEquals(50260, tokenizer.idFor("<|zh|>"))

        // Large-V3 adds <|yue|>, which shifts these two by one relative to Tiny
        // (Tiny: transcribe=50359, notimestamps=50363). Hard-coding Tiny's IDs here
        // would decode the wrong prompt tokens.
        assertEquals(50358, tokenizer.idFor("<|yue|>"))
        assertEquals(50360, tokenizer.idFor("<|transcribe|>"))
        assertEquals(50364, tokenizer.idFor("<|notimestamps|>"))

        // The text vocabulary is shared with Tiny, so the Simplified-Chinese blocklist
        // generated from Tiny's vocab.json stays valid for Turbo.
        for (text in listOf("你好世界", "hello world", "こんにちは")) {
            val ids = tokenizer.encode(text)
            assertTrue("$text must encode to at least one token", ids.isNotEmpty())
            assertEquals(text, tokenizer.decode(ids))
        }
    }

    @Test
    fun turboVariantShapesMatchMetadataContract() {
        val variant = WhisperVariant.LARGE_V3_TURBO
        assertEquals("models/whisper_large_v3_turbo", variant.assetDir)
        assertEquals(WhisperMelConfig.LargeV3Turbo, variant.melConfig)
        assertEquals(128, variant.nMels)
        assertEquals(3000, variant.nFrames)
        assertEquals(20, variant.nHeads)
        assertEquals(51866, variant.vocabSize)
        assertEquals(200, variant.maxDecodeLength)
        assertEquals(199, variant.selfCacheLength)
        assertEquals(listOf(20L, 1L, 64L, 1500L), variant.kvShape(isKey = true, slots = 1500).toList())
        assertEquals(listOf(20L, 1L, 1500L, 64L), variant.kvShape(isKey = false, slots = 1500).toList())
        assertEquals(listOf(20L, 1L, 64L, 199L), variant.selfInShape("k_cache_self_0_in").toList())
        assertEquals(listOf(20L, 1L, 199L, 64L), variant.selfInShape("v_cache_self_0_in").toList())
    }

    @Test
    fun tinyVariantKeepsItsOwnContract() {
        val variant = WhisperVariant.TINY
        assertEquals("models/whisper", variant.assetDir)
        assertEquals(80, variant.nMels)
        assertEquals(6, variant.nHeads)
        assertEquals(51865, variant.vocabSize)
        assertEquals(listOf(6L, 1L, 64L, 1500L), variant.kvShape(isKey = true, slots = 1500).toList())
        assertEquals(listOf(6L, 1L, 64L, 199L), variant.selfInShape("k_cache_self_0_in").toList())
    }

    @Test
    fun incrementalCacheMatchesFullTurboExtractionForGrowingAudio() {
        val extractor = WhisperTurboFeatureExtractor(WhisperMelConfig.LargeV3Turbo)
        val cache = extractor.newIncrementalCache()
        val first = ShortArray(16_000) { ((it * 17) % 32767).toShort() }
        val second = ShortArray(16_000) { (((it + 31_000) * 13) % 32767).toShort() }
        val third = ShortArray(16_000) { (((it + 62_000) * 19) % 32767).toShort() }

        cache.append(first, 16_000)
        cache.append(second, 16_000)
        val incremental = cache.append(third, 16_000)
        val full = extractor.extractHalf(first + second + third, 16_000)

        assertEquals(128 * 3000, incremental.size)
        assertArrayEquals("incremental 128-bin mel must match the full extraction", full, incremental)
    }

    /**
     * The live path appends every 100 ms, not every second, and it does so ~300 times for a 30 s
     * window. The normalization floor is a function of the max over the whole window, so it
     * *changes* as audio arrives and every already-written frame can be clamped by it later.
     *
     * This is the regression guard for the mel-cache optimization: the cache now reuses scratch
     * buffers and recomputes log10 into one of them instead of `rawMel.copyOf()`. If that reuse
     * ever leaked stale values — or if a frame were skipped because its raw energy did not
     * change while the floor did — the result would drift from the full extraction and this test
     * would catch it. Small 100 ms chunks are used deliberately so the floor moves many times
     * during the run.
     */
    @Test
    fun incrementalCacheStaysBitIdenticalAcrossManySmallAppends() {
        val extractor = WhisperTurboFeatureExtractor(WhisperMelConfig.LargeV3Turbo)
        val cache = extractor.newIncrementalCache()
        val chunkSamples = 1600 // 100 ms, matching the production read size
        val chunks = 60          // 6 s of audio in 100 ms steps

        val all = ShortArray(chunkSamples * chunks)
        var last: ShortArray? = null
        for (c in 0 until chunks) {
            // Speech-like amplitude that ramps, so the window max keeps rising and the
            // dynamic-range floor is recomputed differently on almost every append.
            val chunk = ShortArray(chunkSamples) { i ->
                val t = c * chunkSamples + i
                (((t * 37) % 8000) - 4000 + (c * 20)).toShort()
            }
            chunk.copyInto(all, c * chunkSamples)
            last = cache.append(chunk, 16_000)
        }

        val full = extractor.extractHalf(all, 16_000)
        assertArrayEquals(
            "incremental 128-bin mel must stay bit-identical across many small appends",
            full,
            last,
        )
    }

    /**
     * The returned array must not alias the cache's internal scratch buffer.
     *
     * The cache reuses its output buffer to avoid allocating 768 KB at 10 Hz. If that buffer
     * escaped, the producer's next append would overwrite the window a consumer is still
     * decoding — a silent corruption that would show up as wrong text, not as an exception.
     * This asserts the *cache* contract (each returned array is independent); the live path
     * additionally copies before publishing, see `LiveTranscriber.onAudioAppendedWithMel`.
     */
    @Test
    fun incrementalCacheReturnsIndependentArrays() {
        val cache = WhisperTurboFeatureExtractor(WhisperMelConfig.LargeV3Turbo).newIncrementalCache()
        val first = cache.append(ShortArray(16_000) { ((it * 17) % 32767).toShort() }, 16_000)
        val firstCopy = first.copyOf()
        cache.append(ShortArray(16_000) { (((it + 31_000) * 13) % 32767).toShort() }, 16_000)
        // The second append must not have mutated the array handed out by the first.
        assertArrayEquals("a returned mel window must not be mutated by a later append", firstCopy, first)
    }
}
