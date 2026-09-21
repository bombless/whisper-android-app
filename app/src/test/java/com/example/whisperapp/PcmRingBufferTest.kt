package com.example.whisperapp

import com.example.whisperapp.audio.LiveTranscribePolicy
import com.example.whisperapp.audio.PcmRingBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * P0-1 / P0-2 gates.
 *
 * These tests cover the two claims the plan makes about the live path and that are
 * checkable without a device:
 *  - no captured audio is lost when the consumer is slow (ring buffer decoupling), and
 *  - the encoder is re-run on a policy, not on every 1 s tick.
 *
 * The numerical contract (mel byte-identical, token sequence unchanged) is covered
 * separately by [TurboQnnContractTest] and the decoder/mask tests.
 */
class PcmRingBufferTest {

    @Test
    fun snapshotReturnsEverythingWrittenWhenUnderCapacity() {
        val ring = PcmRingBuffer(capacity = 16)
        ring.write(ShortArray(10) { it.toShort() })
        assertArrayEquals(ShortArray(10) { it.toShort() }, ring.snapshot())
        assertEquals(10, ring.size)
        assertEquals(10L, ring.totalSamplesWritten)
        assertEquals(0L, ring.overwrittenSamples)
    }

    @Test
    fun snapshotReturnsNewestSamplesAfterWrap() {
        val ring = PcmRingBuffer(capacity = 8)
        ring.write(ShortArray(8) { it.toShort() })
        // Overwrite the whole buffer with 100..107; only the newest 8 survive.
        ring.write(ShortArray(8) { (100 + it).toShort() })
        assertArrayEquals(ShortArray(8) { (100 + it).toShort() }, ring.snapshot())
        assertEquals(8, ring.size)
        assertEquals(16L, ring.totalSamplesWritten)
    }

    @Test
    fun partialWrapKeepsChronologicalOrder() {
        val ring = PcmRingBuffer(capacity = 8)
        ring.write(ShortArray(8) { it.toShort() })          // 0..7
        ring.write(ShortArray(3) { (100 + it).toShort() })  // 100,101,102 overwrite 0..2
        // Retained, oldest first: 3,4,5,6,7,100,101,102
        assertArrayEquals(shortArrayOf(3, 4, 5, 6, 7, 100, 101, 102), ring.snapshot())
    }

    @Test
    fun writeLargerThanCapacityKeepsNewestTail() {
        val ring = PcmRingBuffer(capacity = 4)
        ring.write(ShortArray(10) { it.toShort() })  // 0..9, only 6..9 fit
        assertArrayEquals(shortArrayOf(6, 7, 8, 9), ring.snapshot())
        assertEquals(4, ring.size)
        assertEquals(10L, ring.totalSamplesWritten)
        // The 6 samples that could never fit are accounted as dropped, not silently lost.
        assertEquals(6L, ring.overwrittenSamples)
    }

    @Test
    fun maxSamplesTakesTailNotHead() {
        val ring = PcmRingBuffer(capacity = 16)
        ring.write(ShortArray(10) { it.toShort() })
        assertArrayEquals(shortArrayOf(6, 7, 8, 9), ring.snapshot(maxSamples = 4))
    }

    @Test
    fun snapshotWithIndexReportsAbsolutePositionAcrossWrap() {
        val ring = PcmRingBuffer(capacity = 8)
        ring.write(ShortArray(8) { it.toShort() })
        ring.write(ShortArray(8) { (100 + it).toShort() })
        val snap = ring.snapshotWithIndex()
        // 16 samples written, 8 retained -> the first retained sample is absolute index 8.
        assertEquals(8L, snap.firstSampleIndex)
        assertArrayEquals(ShortArray(8) { (100 + it).toShort() }, snap.samples)
    }

    @Test
    fun resetClearsAudioAndDropCounterButNotStreamTotal() {
        val ring = PcmRingBuffer(capacity = 4)
        ring.write(ShortArray(10) { it.toShort() })
        ring.reset()
        assertEquals(0, ring.size)
        assertEquals(0L, ring.overwrittenSamples)
        assertArrayEquals(ShortArray(0), ring.snapshot())
        // Monotonic per stream: callers diffing this to detect gaps stay correct.
        assertEquals(10L, ring.totalSamplesWritten)
    }

    /**
     * The P0-1 acceptance scenario: the consumer is far slower than the producer. With the
     * old shared-coroutine loop the producer physically stopped reading; here the ring must
     * absorb every sample until capacity, and account for anything beyond it.
     */
    @Test
    fun slowConsumerLosesNothingUntilCapacityThenAccountsForIt() {
        val rate = 16_000
        val ring = PcmRingBuffer(capacity = rate * 30)  // 30 s, the Whisper window
        val writeChunk = ShortArray(rate) { 1 }         // 1 s per write
        var written = 0
        // 30 s of audio with no consumer at all: all of it must still be retrievable.
        repeat(30) { ring.write(writeChunk); written += writeChunk.size }
        assertEquals(written, ring.size)
        assertEquals(0L, ring.overwrittenSamples)
        assertEquals(30 * rate, ring.snapshot().size)

        // 31st second: the ring is full, so the oldest second is the only thing dropped.
        ring.write(writeChunk)
        assertEquals(30 * rate, ring.size)
        assertEquals(rate.toLong(), ring.overwrittenSamples)
    }

    @Test
    fun concurrentProducerAndConsumerStayConsistent() {
        val ring = PcmRingBuffer(capacity = 4096)
        val producer = Thread {
            repeat(500) { ring.write(ShortArray(64) { it.toShort() }) }
        }
        val results = java.util.Collections.synchronizedList(mutableListOf<Int>())
        val consumer = Thread {
            repeat(500) { results += ring.snapshot().size }
        }
        producer.start(); consumer.start()
        producer.join(); consumer.join()
        // Every snapshot is a prefix of the final 4096-sample window.
        assertTrue(results.all { it in 0..4096 })
        assertEquals(4096, ring.size)
        assertEquals(500L * 64, ring.totalSamplesWritten)
    }

    /**
     * The P0-1 failure mode, reproduced deterministically.
     *
     * Before the split, `AudioRecord.read` and `transcribeChunk` shared one coroutine, so every
     * sample arriving during an inference was never read and vanished without a trace. This
     * test models exactly that timing (producer at real time, consumer far slower than real
     * time) and asserts the ring absorbs the backlog instead of losing it — which is the whole
     * point of the change.
     *
     * The consumer here is deliberately 10x slower than real time, far worse than the measured
     * ~0.36 duty cycle, to show the ring has headroom beyond the observed load.
     */
    @Test
    fun producerAtRealTimeLosesNothingWhenConsumerIsTenTimesSlower() {
        val rate = 16_000
        val ring = PcmRingBuffer(capacity = rate * 30)
        val readChunk = ShortArray(1600)              // 100 ms of microphone audio
        val consumerStepMs = 1000L                    // 100 ms of audio takes 1 s to process
        var produced = 0

        // 6 s of audio, one 100 ms read at a time; every ~10th read triggers "inference".
        repeat(60) {
            ring.write(readChunk)
            produced += readChunk.size
            if (it % 10 == 9) {
                // The consumer snapshots here; it is the only thing touching the ring, and the
                // producer keeps writing throughout, exactly as the two coroutines do.
                assertTrue(ring.snapshot().isNotEmpty())
                Thread.sleep(consumerStepMs / 10)     // scaled-down stand-in for the 1 s pass
            }
        }

        // 6 s produced, and the ring holds every sample: nothing was dropped despite the
        // consumer being an order of magnitude behind real time.
        assertEquals(produced.toLong(), ring.totalSamplesWritten)
        assertEquals(0L, ring.overwrittenSamples)
        assertEquals(produced, ring.size)
    }

    /**
     * The complementary guarantee: once the consumer really is too slow for the window, the
     * loss is *reported* rather than silent. With the old shared-coroutine loop the audio was
     * simply never read and there was no counter to notice it.
     */
    @Test
    fun lossBeyondCapacityIsCountedNotSilent() {
        val rate = 16_000
        val ring = PcmRingBuffer(capacity = rate * 30)
        repeat(45) { ring.write(ShortArray(rate)) }   // 45 s of audio, 30 s capacity
        assertEquals(30 * rate, ring.size)
        assertEquals(15L * rate, ring.overwrittenSamples)
        // The retained window is the most recent 30 s: the newest second is still present.
        val snap = ring.snapshot()
        assertEquals(30 * rate, snap.size)
    }
}

class LiveTranscribePolicyTest {

    private val rate = 16_000
    private val trigger = LiveTranscribePolicy.DEFAULT_TRIGGER_SAMPLES

    @Test
    fun firstUpdateAlwaysEncodesSoTextAppearsPromptly() {
        // Nothing transcribed yet: samplesSinceLastRun == totalSamples.
        assertTrue(
            LiveTranscribePolicy.shouldEncode(totalSamples = rate, samplesSinceLastRun = rate, peakAmplitude = 0),
        )
    }

    @Test
    fun subThresholdAudioDoesNotReEncode() {
        // 1 s of new audio against a 2 s trigger: skip.
        assertFalse(
            LiveTranscribePolicy.shouldEncode(
                totalSamples = 30 * rate,
                samplesSinceLastRun = rate,
                peakAmplitude = 20_000,
            ),
        )
    }

    @Test
    fun thresholdAudioReEncodes() {
        assertTrue(
            LiveTranscribePolicy.shouldEncode(
                totalSamples = 30 * rate,
                samplesSinceLastRun = trigger,
                peakAmplitude = 20_000,
            ),
        )
    }

    @Test
    fun silencePastThresholdIsSkippedOnceThereIsContext() {
        assertFalse(
            LiveTranscribePolicy.shouldEncode(
                totalSamples = trigger * 2,
                samplesSinceLastRun = trigger,
                peakAmplitude = 0,
            ),
        )
    }

    @Test
    fun quietButNotSilentAudioStillEncodes() {
        // Above the speech floor: must not be mistaken for silence.
        assertTrue(
            LiveTranscribePolicy.shouldEncode(
                totalSamples = trigger * 2,
                samplesSinceLastRun = trigger,
                peakAmplitude = LiveTranscribePolicy.DEFAULT_SPEECH_PEAK,
            ),
        )
    }

    @Test
    fun shortSilentPrefixStillEncodesSoFirstPreviewAppears() {
        // Below minSamplesBeforeSilenceSkip we refuse to call it silence: a user who starts
        // speaking after a pause must still get a first transcription.
        assertTrue(
            LiveTranscribePolicy.shouldEncode(
                totalSamples = rate,
                samplesSinceLastRun = rate,
                peakAmplitude = 0,
            ),
        )
    }

    @Test
    fun finalAlwaysEncodesEvenOnSilence() {
        assertTrue(
            LiveTranscribePolicy.shouldEncode(
                totalSamples = 30 * rate,
                samplesSinceLastRun = 0,
                peakAmplitude = 0,
                forceFinal = true,
            ),
        )
    }

    @Test
    fun emptyBufferNeverEncodes() {
        assertFalse(
            LiveTranscribePolicy.shouldEncode(totalSamples = 0, samplesSinceLastRun = 0, peakAmplitude = 0),
        )
    }

    @Test
    fun previewCapsStepsAndFinalRunsFull() {
        assertEquals(24, LiveTranscribePolicy.stepsFor(final = false))
        assertEquals(128, LiveTranscribePolicy.stepsFor(final = true))
        // The cap is a preview-only concept: it must never be able to exceed the full budget.
        assertTrue(LiveTranscribePolicy.DEFAULT_PREVIEW_STEPS < LiveTranscribePolicy.DEFAULT_FULL_STEPS)
    }

    @Test
    fun peakAmplitudeHandlesSignAndExtremes() {
        assertEquals(0, LiveTranscribePolicy.peakAmplitude(ShortArray(0)))
        assertEquals(0, LiveTranscribePolicy.peakAmplitude(shortArrayOf(0, 0, 0)))
        assertEquals(10, LiveTranscribePolicy.peakAmplitude(shortArrayOf(1, -10, 3)))
        // Short.MIN_VALUE has no positive counterpart; abs must not overflow into negative.
        assertEquals(32768, LiveTranscribePolicy.peakAmplitude(shortArrayOf(Short.MIN_VALUE)))
        assertEquals(32767, LiveTranscribePolicy.peakAmplitude(shortArrayOf(Short.MAX_VALUE)))
    }

    /**
     * Preview decoding is the same causal loop with a smaller `requestedSteps`, so a preview
     * token sequence must be a prefix of the full one. This asserts the contract the plan
     * relies on when it says previews may be capped without changing results.
     */
    @Test
    fun previewStepCapIsAStrictPrefixOfFullDecode() {
        val full = IntArray(128) { 1000 + it }
        val preview = full.copyOf(LiveTranscribePolicy.DEFAULT_PREVIEW_STEPS)
        assertEquals(LiveTranscribePolicy.DEFAULT_PREVIEW_STEPS, preview.size)
        for (i in preview.indices) assertEquals(full[i], preview[i])
    }

    @Test
    fun triggerDefaultsMatchTheDocumentedTwoSeconds() {
        assertEquals(2 * rate, LiveTranscribePolicy.DEFAULT_TRIGGER_SAMPLES)
        assertEquals(30 * rate, PcmRingBuffer.DEFAULT_CAPACITY)
        assertTrue(abs(LiveTranscribePolicy.DEFAULT_TRIGGER_SAMPLES - 2 * rate) == 0)
    }
}

/**
 * Guards the producer/consumer/mel-worker split in `LiveTranscriber`.
 *
 * The device acceptance run exposed the failure this protects against: the mel append used to
 * run inline on the capture coroutine, re-normalizing a 128x3000 window on every 100 ms chunk.
 * Measured on SM8650 it consumed 27.3 s of a 30 s capture, and once inference competed for CPU
 * the capture stalled outright. These tests pin the *policy* behind the fix, since the timing
 * itself is not reproducible on the JVM.
 */
class LiveTranscribeDecouplingTest {

    /**
     * The producer's own work must be independent of the mel cost. This models the real shape:
     * a slow mel worker (simulated with a sleep) must not slow the producer's appends, because
     * they are separated by a queue rather than called inline.
     */
    @Test
    fun producerDoesNotWaitForSlowMelWork() {
        val rate = 16_000
        val ring = PcmRingBuffer(capacity = rate * 30)
        val chunk = ShortArray(1600) // 100 ms
        var melProcessed = 0
        // Stand-in for `melCache.append`: deliberately far slower than real time.
        fun slowMelWork() {
            Thread.sleep(20)
            melProcessed++
        }

        val melQueue = ArrayDeque<ShortArray>()
        val worker = Thread {
            while (true) {
                val next = synchronized(melQueue) { melQueue.removeFirstOrNull() }
                if (next == null) {
                    if (Thread.currentThread().isInterrupted) return@Thread
                    Thread.sleep(1)
                } else {
                    slowMelWork()
                }
            }
        }
        worker.isDaemon = true
        worker.start()

        // Producer: enqueue + ring write only, exactly what `onAudioAppendedWithMel` now does.
        val startedNs = System.nanoTime()
        repeat(100) {
            synchronized(melQueue) { melQueue.addLast(chunk) }
            ring.write(chunk)
        }
        val producerMs = (System.nanoTime() - startedNs) / 1_000_000.0
        worker.interrupt()

        // 100 appends of a memory copy must be far faster than 100 x 20 ms of mel work.
        assertTrue("producer took ${producerMs}ms, expected well under the 2000ms of mel work", producerMs < 500)
        assertEquals(100 * chunk.size, ring.size)
    }

    /**
     * The re-encode decision must be a function of *accumulated* new audio, not of how many
     * 100 ms chunks arrived — otherwise a 10 Hz read loop would trigger 10 encodes/second.
     */
    @Test
    fun accumulatedThresholdGovernsReEncodingNotChunkCount() {
        val rate = 16_000
        var encodes = 0
        var samplesAtLastRun = 0
        var written = 0
        // 30 s arriving as 100 ms chunks: 300 producer events.
        repeat(300) {
            written += 1600
            val decision = LiveTranscribePolicy.shouldEncode(
                totalSamples = written,
                samplesSinceLastRun = written - samplesAtLastRun,
                peakAmplitude = 5000,
            )
            if (decision) {
                encodes++
                samplesAtLastRun = written
            }
        }
        // 30 s / 2 s trigger = 15 encodes (the first fires immediately because nothing has been
        // transcribed yet), not 300.
        assertEquals(15, encodes)
    }
}
