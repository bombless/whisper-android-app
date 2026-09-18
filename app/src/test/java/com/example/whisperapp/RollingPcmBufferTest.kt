package com.example.whisperapp

import com.example.whisperapp.audio.RollingPcmBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Test

class RollingPcmBufferTest {
    @Test
    fun snapshotPreservesChronologicalOrder() {
        val buffer = RollingPcmBuffer(5)
        buffer.append(shortArrayOf(1, 2, 3))
        assertArrayEquals(shortArrayOf(1, 2, 3), buffer.snapshot())
        buffer.append(shortArrayOf(4, 5, 6))
        assertArrayEquals(shortArrayOf(2, 3, 4, 5, 6), buffer.snapshot())
    }

    @Test
    fun partialAppendWorks() {
        val buffer = RollingPcmBuffer(4)
        val source = shortArrayOf(10, 11, 12, 13, 14)
        buffer.append(source, offset = 1, length = 3)
        assertArrayEquals(shortArrayOf(11, 12, 13), buffer.snapshot())
    }
}
