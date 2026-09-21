package com.example.whisperapp.audio

/**
 * Lock-guarded PCM16 ring buffer that decouples the microphone reader from the
 * transcription consumer.
 *
 * Before this existed, `MainScreen` did `AudioRecord.read` and `transcribeChunk`
 * on the same coroutine: for the ~1 s a chunk took to transcribe nothing drained
 * the microphone, and every sample captured in that window was lost (AudioRecord's
 * internal buffer is far smaller than one inference). The reader now only ever
 * appends here — a bounded memory copy — while inference snapshots the tail.
 *
 * The buffer holds at most [capacity] samples (default 30 s, which is exactly the
 * Whisper context window), so it is naturally capped at the largest span the model
 * can consume; writing past capacity drops the oldest samples.
 *
 * Thread policy: [write] is called only from the recording coroutine, [snapshot] only
 * from the inference path, but both are synchronised so a future change cannot corrupt
 * the buffer. Neither method allocates while holding the lock beyond the copy it is
 * asked for.
 */
class PcmRingBuffer(val capacity: Int = DEFAULT_CAPACITY) {
    init {
        require(capacity > 0) { "ring capacity must be positive, got $capacity" }
    }

    private val data = ShortArray(capacity)
    private val lock = Any()

    /** Ring index of the oldest retained sample. */
    private var start = 0

    /** Number of retained samples, always in `0..capacity`. */
    private var count = 0

    /** Total samples ever accepted, including those already dropped as overwritten. */
    private var totalWritten = 0L

    /** Samples dropped because they were overwritten before the consumer read them. */
    private var overwritten = 0L

    val size: Int get() = synchronized(lock) { count }

    /** Every sample ever written; monotonic, so callers can detect gaps in their own reads. */
    val totalSamplesWritten: Long get() = synchronized(lock) { totalWritten }

    /** Samples lost to overwrite. A non-zero value means the consumer fell behind. */
    val overwrittenSamples: Long get() = synchronized(lock) { overwritten }

    /** Appends [samples] and returns how many were stored (less than `size` only at capacity). */
    fun write(samples: ShortArray): Int {
        if (samples.isEmpty()) return 0
        synchronized(lock) {
            totalWritten += samples.size
            // Keep the newest `toStore` samples when the write itself exceeds capacity.
            val toStore = minOf(samples.size, capacity)
            val sourceOffset = samples.size - toStore

            // Eviction happens first: the incoming samples occupy the tail of the buffer, so
            // anything they cannot fit on top of is dropped from the head. Accounting for it
            // here (rather than inside the copy loop) is what makes a write that exactly fills
            // a full buffer report its eviction instead of silently overwriting.
            val free = capacity - count
            val evicted = maxOf(0, toStore - free)
            if (evicted > 0) {
                start = (start + evicted) % capacity
                count -= evicted
                overwritten += evicted.toLong()
            }

            var written = 0
            while (written < toStore) {
                val destination = (start + count) % capacity
                val chunk = minOf(toStore - written, capacity - destination)
                samples.copyInto(data, destination, sourceOffset + written, sourceOffset + written + chunk)
                written += chunk
                count += chunk
            }
            // Samples the write itself could not hold because it outran capacity.
            overwritten += (samples.size - toStore).toLong()
            return toStore
        }
    }

    /**
     * Copies the newest `min([maxSamples], size)` samples into a fresh array.
     *
     * Returns an empty array when the buffer holds nothing. The caller owns the result,
     * so it may keep or mutate it freely; the buffer keeps its own copy.
     */
    fun snapshot(maxSamples: Int = Int.MAX_VALUE): ShortArray {
        synchronized(lock) {
            val take = minOf(count, maxSamples, capacity)
            if (take <= 0) return ShortArray(0)
            val out = ShortArray(take)
            // Oldest retained sample that is actually wanted.
            val from = (start + count - take + capacity) % capacity
            val firstChunk = minOf(take, capacity - from)
            data.copyInto(out, 0, from, from + firstChunk)
            if (firstChunk < take) {
                data.copyInto(out, firstChunk, 0, take - firstChunk)
            }
            return out
        }
    }

    /**
     * Snapshots like [snapshot] but also reports the absolute sample index of the first
     * returned sample, so a consumer can tell whether it missed audio between reads.
     */
    fun snapshotWithIndex(maxSamples: Int = Int.MAX_VALUE): Snapshot {
        synchronized(lock) {
            val take = minOf(count, maxSamples, capacity)
            val firstAbsoluteIndex = totalWritten - take
            return Snapshot(snapshotLocked(take), firstAbsoluteIndex, overwritten)
        }
    }

    private fun snapshotLocked(take: Int): ShortArray {
        if (take <= 0) return ShortArray(0)
        val out = ShortArray(take)
        val from = (start + count - take + capacity) % capacity
        val firstChunk = minOf(take, capacity - from)
        data.copyInto(out, 0, from, from + firstChunk)
        if (firstChunk < take) data.copyInto(out, firstChunk, 0, take - firstChunk)
        return out
    }

    /** Clears all retained audio and resets the drop counters (total stays monotonic per reset). */
    fun reset() {
        synchronized(lock) {
            start = 0
            count = 0
            overwritten = 0
        }
    }

    data class Snapshot(val samples: ShortArray, val firstSampleIndex: Long, val overwrittenSamples: Long)

    companion object {
        /** 30 s at 16 kHz: the Whisper context window, and the largest useful span. */
        const val DEFAULT_CAPACITY = 16_000 * 30
    }
}
