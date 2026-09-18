package com.example.whisperapp.audio

/**
 * Fixed-size PCM ring buffer for rolling-window ASR.
 *
 * The live UI keeps only the most recent [capacitySamples] samples. Appending a
 * one-second block therefore does not allocate or copy the whole recording.
 */
class RollingPcmBuffer(private val capacitySamples: Int) {
    init {
        require(capacitySamples > 0) { "capacitySamples must be positive" }
    }

    private val storage = ShortArray(capacitySamples)
    private var writeIndex = 0
    private var size = 0

    @Synchronized
    fun append(samples: ShortArray, offset: Int = 0, length: Int = samples.size - offset) {
        require(offset >= 0 && length >= 0 && offset + length <= samples.size) {
            "Invalid append range"
        }
        var src = offset
        var remaining = length
        while (remaining > 0) {
            val count = minOf(remaining, capacitySamples - writeIndex)
            samples.copyInto(storage, writeIndex, src, src + count)
            writeIndex = (writeIndex + count) % capacitySamples
            src += count
            remaining -= count
            size = minOf(capacitySamples, size + count)
        }
    }

    /**
     * Returns samples in chronological order. The returned array is a snapshot
     * and can safely be passed to the inference thread.
     */
    @Synchronized
    fun snapshot(): ShortArray {
        val result = ShortArray(size)
        val start = (writeIndex - size + capacitySamples) % capacitySamples
        if (size == 0) return result
        val first = minOf(size, capacitySamples - start)
        storage.copyInto(result, 0, start, start + first)
        if (first < size) {
            storage.copyInto(result, first, 0, size - first)
        }
        return result
    }

    @Synchronized
    fun clear() {
        storage.fill(0)
        writeIndex = 0
        size = 0
    }

    @Synchronized
    fun size(): Int = size
}
