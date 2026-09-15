package com.example.whisperapp

import com.example.whisperapp.qnn.LongAudioSingleContextRunner
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LongAudioSingleContextRunnerTest {
    @Test
    fun chunkResultPreservesIndependentChunkMetadata() {
        val result = LongAudioSingleContextRunner.ChunkResult(
            chunkIndex = 2,
            startSample = 960_000,
            endSample = 1_200_000,
            startTimeSeconds = 60.0,
            endTimeSeconds = 75.0,
            tokenIds = intArrayOf(100, 200, 50257),
            decodedText = "hello",
            eosReached = true,
        )

        assertEquals(2, result.chunkIndex)
        assertEquals(960_000, result.startSample)
        assertEquals(1_200_000, result.endSample)
        assertEquals(60.0, result.startTimeSeconds, 0.0)
        assertEquals(75.0, result.endTimeSeconds, 0.0)
        assertArrayEquals(intArrayOf(100, 200, 50257), result.tokenIds)
        assertEquals("hello", result.decodedText)
        assertTrue(result.eosReached)
    }
}
