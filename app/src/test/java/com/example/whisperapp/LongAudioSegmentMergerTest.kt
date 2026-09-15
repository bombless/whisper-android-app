package com.example.whisperapp

import com.example.whisperapp.qnn.LongAudioSegmentMerger
import com.example.whisperapp.qnn.LongAudioSingleContextRunner
import org.junit.Assert.assertEquals
import org.junit.Test

class LongAudioSegmentMergerTest {
    private fun chunk(index: Int, text: String): LongAudioSingleContextRunner.ChunkResult =
        LongAudioSingleContextRunner.ChunkResult(
            chunkIndex = index,
            startSample = index * 480_000,
            endSample = (index + 1) * 480_000,
            startTimeSeconds = index * 30.0,
            endTimeSeconds = (index + 1) * 30.0,
            tokenIds = intArrayOf(),
            decodedText = text,
            eosReached = true,
        )

    @Test fun singleChunk() = assertEquals("hello", LongAudioSegmentMerger.merge(listOf(chunk(0, "hello"))))

    @Test fun multipleChunks() = assertEquals(
        "hello world test",
        LongAudioSegmentMerger.merge(listOf(chunk(0, "hello"), chunk(1, "world"), chunk(2, "test"))),
    )

    @Test fun emptyChunkIsIgnored() = assertEquals(
        "hello world",
        LongAudioSegmentMerger.merge(listOf(chunk(0, "hello"), chunk(1, ""), chunk(2, "world"))),
    )

    @Test fun allEmpty() = assertEquals(
        "",
        LongAudioSegmentMerger.merge(listOf(chunk(0, ""), chunk(1, "  "), chunk(2, ""))),
    )

    @Test fun partialFinalChunkIsIncluded() = assertEquals(
        "first second final",
        LongAudioSegmentMerger.merge(listOf(chunk(0, "first"), chunk(1, "second"), chunk(2, "final"))),
    )

    @Test fun preservesInputOrder() = assertEquals(
        "zero one two",
        LongAudioSegmentMerger.merge(listOf(chunk(0, "zero"), chunk(1, "one"), chunk(2, "two"))),
    )

    @Test fun trimsBoundaryWhitespaceOnly() = assertEquals(
        "hello world",
        LongAudioSegmentMerger.merge(listOf(chunk(0, "  hello  "), chunk(1, " world "))),
    )
}
