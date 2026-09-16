package com.example.whisperapp

import com.example.whisperapp.qnn.WhisperDecoderMask
import org.junit.Assert.assertEquals
import org.junit.Test

class WhisperDecoderMaskTest {
    @Test
    fun maskTracksActualAppendAndDropCacheAcrossAllPositions() {
        var cache = List(199) { false }
        for (position in 0 until 200) {
            val attentionKeys = cache + true
            val mask = WhisperDecoderMask.forPosition(position)
            assertEquals(200, mask.size)
            attentionKeys.forEachIndexed { index, valid ->
                assertEquals("position=$position index=$index", valid, mask[index] == 0.toShort())
                if (!valid) assertEquals(0xd640.toShort(), mask[index])
            }
            cache = attentionKeys.drop(1)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNegativePosition() { WhisperDecoderMask.forPosition(-1) }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsPositionOutsideGraph() { WhisperDecoderMask.forPosition(200) }
}
