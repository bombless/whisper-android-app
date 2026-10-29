package com.example.whisperapp

import com.example.whisperapp.qnn.SdxlTokenizer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class SdxlTokenizerTest {
    @Test
    fun mergesTokensAndUsesEncoderSpecificPadding() {
        val tokenizer = SdxlTokenizer(
            mapOf("hello</w>" to 10, "1</w>" to 11, "2</w>" to 12, "!</w>" to 13),
            mapOf(("h" to "e") to 0, ("he" to "l") to 1, ("hel" to "l") to 2, ("hell" to "o</w>") to 3),
        )
        val encoded = tokenizer.encode("HELLO 12!", 0)
        assertArrayEquals(intArrayOf(49406, 10, 11, 12, 13, 49407), encoded.copyOfRange(0, 6))
        assertArrayEquals(encoded, tokenizer.encode("HELLO\u00a012!", 0))
        assertEquals(0, encoded[6])
        assertEquals(77, encoded.size)
    }

    @Test
    fun encodesUtf8BytesAndTruncatesBeforeEndToken() {
        val tokenizer = SdxlTokenizer(mapOf("Ã" to 20, "©</w>" to 21, "a</w>" to 22), emptyMap())
        assertArrayEquals(intArrayOf(49406, 20, 21, 49407), tokenizer.encode("é", 0).copyOfRange(0, 4))
        val truncated = tokenizer.encode("a ".repeat(100), 49407)
        assertEquals(49406, truncated.first())
        assertEquals(22, truncated[75])
        assertEquals(49407, truncated.last())
    }
}
