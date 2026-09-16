package com.example.whisperapp

import com.example.whisperapp.tts.ChineseTextProcessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChineseTextProcessorTest {
    @Test
    fun normalizeChineseTextRemovesNoiseAndSpaces() {
        assertEquals("你好，我是AI。", ChineseTextProcessor.normalizeChineseText(" 你好 ， 我是AI 。 "))
    }

    @Test
    fun incrementalTextRemovesChunkOverlap() {
        assertEquals("今天很好。", ChineseTextProcessor.incrementalNewText("你好我是张三", "我是张三今天很好。"))
        assertEquals("", ChineseTextProcessor.incrementalNewText("你好。", "你好。"))
    }

    @Test
    fun splitSentencesUsesChinesePunctuation() {
        assertEquals(listOf("第一句。", "第二句！", "第三句？"), ChineseTextProcessor.splitSentences("第一句。第二句！第三句？"))
        assertTrue(ChineseTextProcessor.splitSentences("一").isEmpty())
    }
}
