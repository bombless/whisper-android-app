package com.example.whisperapp

import com.example.whisperapp.asr.WhisperTokenizer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File

class WhisperTokenizerDecodeTest {
    private lateinit var tokenizer: WhisperTokenizer

    @Before
    fun setUp() {
        val tokenizerDir = File("src/main/assets/models/whisper/tokenizer")
        tokenizer = WhisperTokenizer.fromAssetTexts(
            File(tokenizerDir, "vocab.json").readText(),
            File(tokenizerDir, "merges.txt").readText(),
            File(tokenizerDir, "added_tokens.json").readText(),
        )
    }

    @Test
    fun eosOnlyIsFilteredToEmptyText() {
        assertEquals("", tokenizer.decode(intArrayOf(50257)))
    }

    @Test
    fun forcedPromptOnlyProducesNoUserText() {
        assertEquals("", tokenizer.decode(intArrayOf(50258, 50259, 50359, 50363)))
    }

    @Test
    fun specialTokensAreFilteredWhileTextTokensRemain() {
        val textIds = tokenizer.encode("hello, world!")
        assertTrue(textIds.isNotEmpty())
        val ids = intArrayOf(50258, 50259, 50359, 50363) + textIds + intArrayOf(50257)
        assertEquals("hello, world!", tokenizer.decode(ids))
    }

    @Test
    fun textFollowedByEosDoesNotExposeEos() {
        val textIds = tokenizer.encode("hello world")
        assertEquals("hello world", tokenizer.decode(textIds + intArrayOf(50257)))
    }

    @Test
    fun timestampTokensDoNotPolluteDecodedText() {
        val textIds = tokenizer.encode("hello")
        val ids = intArrayOf(50364) + textIds + intArrayOf(50365, 50257)
        assertEquals("hello", tokenizer.decode(ids))
    }

    @Test
    fun ordinaryUnicodeTextRoundTripsThroughBpeDecode() {
        for (text in listOf("hello world", "café", "你好世界", "こんにちは、世界！")) {
            val ids = tokenizer.encode(text)
            assertTrue("$text must encode to at least one token", ids.isNotEmpty())
            assertEquals(text, tokenizer.decode(ids))
        }
    }
}
