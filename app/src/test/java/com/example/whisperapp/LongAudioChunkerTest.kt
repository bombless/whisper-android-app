package com.example.whisperapp

import com.example.whisperapp.audio.LongAudioChunker
import com.example.whisperapp.audio.WhisperFeatureExtractor
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LongAudioChunkerTest {
    @Test
    fun emptyAudioProducesNoChunks() {
        assertTrue(LongAudioChunker.split(ShortArray(0), 16_000).isEmpty())
    }

    @Test
    fun exactThirtySecondsProducesOneFullChunk() {
        val samples = ShortArray(WhisperFeatureExtractor.CHUNK_SAMPLES) { (it % 97).toShort() }

        val chunks = LongAudioChunker.split(samples, 16_000)

        assertEquals(1, chunks.size)
        assertEquals(0, chunks[0].index)
        assertEquals(0, chunks[0].startSample)
        assertEquals(480_000, chunks[0].endSampleExclusive)
        assertEquals(0.0, chunks[0].startTimeSeconds, 0.0)
        assertEquals(30.0, chunks[0].endTimeSeconds, 0.0)
        assertEquals(30.0, chunks[0].durationSeconds, 0.0)
        assertArrayEquals(samples, chunks[0].samples)
    }

    @Test
    fun seventyFiveSecondsProducesTwoFullAndOnePartialChunk() {
        val totalSamples = 75 * WhisperFeatureExtractor.SAMPLE_RATE
        val samples = ShortArray(totalSamples) { (it and 0x7fff).toShort() }

        val chunks = LongAudioChunker.split(samples, 16_000)

        assertEquals(3, chunks.size)
        assertEquals(listOf(0, 480_000, 960_000), chunks.map { it.startSample })
        assertEquals(listOf(480_000, 960_000, 1_200_000), chunks.map { it.endSampleExclusive })
        assertEquals(listOf(480_000, 480_000, 240_000), chunks.map { it.samples.size })
        assertEquals(listOf(30.0, 30.0, 15.0), chunks.map { it.durationSeconds })

        assertEquals(0.toShort(), chunks[0].samples.first())
        assertEquals(480_000.toShort(), chunks[1].samples.first())
        assertEquals((960_000 and 0x7fff).toShort(), chunks[2].samples.first())
        assertEquals((totalSamples - 1 and 0x7fff).toShort(), chunks[2].samples.last())
    }

    @Test
    fun chunksPreserveAllSamplesWithoutOverlapOrGap() {
        val samples = ShortArray(480_000 + 123) { (it % 251).toShort() }
        val chunks = LongAudioChunker.split(samples, 16_000)
        val reconstructed = chunks.flatMap { it.samples.asIterable() }.toShortArray()

        assertArrayEquals(samples, reconstructed)
        assertEquals(samples.size, chunks.sumOf { it.samples.size })
    }

    @Test(expected = IllegalArgumentException::class)
    fun nonSixteenKilohertzInputIsRejected() {
        LongAudioChunker.split(ShortArray(10), 44_100)
    }
}
