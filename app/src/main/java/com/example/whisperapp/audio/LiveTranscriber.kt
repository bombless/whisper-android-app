package com.example.whisperapp.audio

import android.util.Log
import com.example.whisperapp.asr.WhisperVariant
import com.example.whisperapp.qnn.QnnWhisperRealAudioRunner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.max

/**
 * Live-transcription policy: when to re-run the encoder and how far to decode.
 *
 * This is the decision half of P0-2 (triggered re-encoding + preview step cap) and the
 * consumer half of P0-1 (ring-buffer decoupling). It is deliberately free of Android
 * recorder/UI types so the acceptance criteria — "no dropped audio", "re-encode only
 * when >= N seconds of new audio arrived", "preview caps the step count" — are unit
 * testable on the JVM.
 *
 * Contract:
 *  - [onAudioAppended] is called by the recording coroutine after each successful read.
 *    It never blocks on inference; it only publishes a request when the trigger says so.
 *  - [decide] is the pure policy function; [LiveTranscriber] applies it.
 *
 * Numerical contract: none of this touches PCM, mel or logits. A triggered encode runs
 * the exact same `transcribeChunk` call as the untriggered path, only less often, and a
 * preview decode is the same autoregressive loop with a smaller `requestedSteps`. Token
 * prefixes produced under a preview limit are identical to the corresponding prefix of a
 * full decode because decoding is causal and the prompt is fixed.
 */
object LiveTranscribePolicy {
    /** Minimum new audio before the encoder is re-run (≈2 s of speech). */
    const val DEFAULT_TRIGGER_SAMPLES = 16_000 * 2

    /** Preview decodes are capped here: enough for roughly half a sentence of Chinese. */
    const val DEFAULT_PREVIEW_STEPS = 24

    /** Full decode at end of utterance / stop. */
    const val DEFAULT_FULL_STEPS = 128

    /**
     * Peak amplitude (of 32767) that counts as speech. Used to skip re-encoding through
     * silence, where the transcript provably cannot change.
     */
    const val DEFAULT_SPEECH_PEAK = 400

    /**
     * How much audio must have arrived before we are willing to treat an update as
     * "silence" and skip re-encoding. Guards the very first update, where skipping would
     * mean never producing any preview at all.
     */
    const val DEFAULT_MIN_SAMPLES_BEFORE_SILENCE_SKIP = 16_000 * 4

    /**
     * Should the encoder + decoder run for this update?
     *
     * @param totalSamples       audio currently in the context window
     * @param samplesSinceLastRun audio appended since the last *completed* encode
     * @param peakAmplitude      max |sample| in the newly appended audio
     * @param forceFinal         true at end-of-utterance/stop, which always runs a full decode
     */
    fun shouldEncode(
        totalSamples: Int,
        samplesSinceLastRun: Int,
        peakAmplitude: Int,
        forceFinal: Boolean = false,
        triggerSamples: Int = DEFAULT_TRIGGER_SAMPLES,
        speechPeak: Int = DEFAULT_SPEECH_PEAK,
        minSamplesBeforeSilenceSkip: Int = DEFAULT_MIN_SAMPLES_BEFORE_SILENCE_SKIP,
    ): Boolean {
        if (totalSamples <= 0) return false
        // First ever update has nothing to diff against; always transcribe so the UI
        // shows text as soon as possible.
        if (samplesSinceLastRun >= totalSamples) return true
        if (forceFinal) return true
        if (samplesSinceLastRun < triggerSamples) return false
        // Enough new audio: re-encode unless it is provably silent and we already have
        // some context to fall back on.
        if (peakAmplitude < speechPeak && totalSamples >= minSamplesBeforeSilenceSkip) return false
        return true
    }

    /** Preview during speech, full decode when the utterance is finished. */
    fun stepsFor(final: Boolean, previewSteps: Int = DEFAULT_PREVIEW_STEPS, fullSteps: Int = DEFAULT_FULL_STEPS): Int =
        if (final) fullSteps else previewSteps

    /** Peak |sample| of a PCM16 buffer, as an int in 0..32768 (`abs(-32768)`). */
    fun peakAmplitude(samples: ShortArray): Int {
        var peak = 0
        for (s in samples) {
            val v = if (s < 0) -s.toInt() else s.toInt()
            if (v > peak) peak = v
        }
        return peak
    }
}

/**
 * Drives live transcription off a [PcmRingBuffer].
 *
 * Three coroutines, deliberately separate, because each stage has a very different cost:
 *
 *  1. **producer** (owned by the call site, e.g. the `AudioRecord` loop) calls
 *     [onAudioAppendedWithMel] once per captured chunk. It does only O(samples) memory work —
 *     a ring write and a peak scan — so a microphone read is never delayed.
 *  2. **mel worker** ([start]) drains `melInput` and runs the incremental mel append. That
 *     append re-normalizes the whole 128x3000 window (the dynamic-range floor depends on the
 *     window max), which measured 27.3 s of a 30 s capture when it ran inline on the producer;
 *     giving it its own coroutine is what keeps capture alive under inference load.
 *  3. **consumer** ([start]) applies [LiveTranscribePolicy] and runs inference. It never reads
 *     the microphone, so a slow decode cannot stall capture: the producer keeps filling the
 *     ring, and if inference falls further behind than the 30 s capacity the ring wraps and
 *     [overwrittenSamples] accounts for the loss instead of it being silent.
 */
class LiveTranscriber(
    private val context: android.content.Context,
    private val scope: CoroutineScope,
    private val variant: WhisperVariant,
    private val melCache: WhisperIncrementalMelCache,
    private val ring: PcmRingBuffer = PcmRingBuffer(),
    private val triggerSamples: Int = LiveTranscribePolicy.DEFAULT_TRIGGER_SAMPLES,
    private val previewSteps: Int = LiveTranscribePolicy.DEFAULT_PREVIEW_STEPS,
    private val fullSteps: Int = LiveTranscribePolicy.DEFAULT_FULL_STEPS,
    private val onResult: (Result) -> Unit,
) {
    /** One completed transcription, plus the bookkeeping the acceptance logs need. */
    data class Result(
        val updateId: Int,
        val text: String,
        val elapsedMs: Double,
        val steps: Int,
        val final: Boolean,
        val audioSamples: Int,
        /**
         * Inference seconds per second of **newly arrived** audio.
         *
         * This is the number that decides whether the loop keeps up: the loop only stays real
         * time while the work done per update is smaller than the audio that arrived since the
         * previous update. Reporting it against the whole context window instead would be
         * meaningless — a 30 s window re-encoded in 1.2 s looks like 0.04 either way, while the
         * update is actually 0.6x real time against the 2 s that triggered it.
         *
         * `null` when this update was not caused by new audio (a forced final decode with
         * nothing appended), because the ratio is undefined there.
         */
        val realTimeFactor: Double?,
        /** New audio that triggered this update; the denominator of [realTimeFactor]. */
        val newAudioSamples: Int,
        val cumulativeSamples: Long,
        val overwrittenSamples: Long,
        val error: Throwable? = null,
    )

    /** Queue depth 1: a newer trigger supersedes an unstarted one, but never merges into one running. */
    private val wake = Channel<Unit>(Channel.CONFLATED)

    /**
     * Scratch channel carrying captured PCM from the producer to the mel worker.
     *
     * The mel append is not cheap: it recomputes the FFTs of the affected frames *and*
     * re-normalizes the whole 128x3000 window (the dynamic-range floor depends on the window
     * max, so it must be revisited every time). Running it inline made the producer spend
     * 27.3 s of a 30 s capture inside this call, which starved the microphone read and stalled
     * capture outright once inference was also competing for the CPU.
     *
     * Only the ring write and the peak scan stay on the producer thread; both are O(samples)
     * memory operations. The PCM is forwarded here so the mel worker sees every sample exactly
     * once, in capture order, independently of how far behind inference is.
     *
     * Capacity is generous (0.5 s of 100 ms chunks) because the worker only falls behind while
     * it is itself blocked; if it ever did overflow, the drop is counted rather than silent.
     */
    private val melInput = Channel<ShortArray>(capacity = 32)

    /** Chunks dropped because the mel worker could not keep up; must stay 0 in practice. */
    private val melDropped = java.util.concurrent.atomic.AtomicLong(0)
    val melDroppedChunks: Long get() = melDropped.get()

    @Volatile private var consumerJob: Job? = null

    /**
     * Absolute stream position (in samples, monotonic from stream start) at which the last
     * transcription was taken. Compared against [PcmRingBuffer.totalSamplesWritten] to
     * measure how much *new* audio has arrived, which is the re-encode trigger.
     */
    private var samplesAtLastRun = 0L

    /** Peak amplitude seen since the last completed transcription, for the silence check. */
    private var pendingPeak = 0

    /** Samples the mel worker has consumed; compared against the ring total before a decode. */
    @Volatile private var melSamplesFed = 0L

    /** Signals that the mel worker finished a chunk. */
    private val melReady = Channel<Unit>(Channel.CONFLATED)

    @Volatile private var melJob: Job? = null

    /** Newest FP16 mel window, published by the mel worker. */
    @Volatile private var latestMel: ShortArray? = null

    private var updateId = 0

    val overwrittenSamples: Long get() = ring.overwrittenSamples

    /** Samples appended by the producer so far; the acceptance test's drop-free reference. */
    val totalSamplesWritten: Long get() = ring.totalSamplesWritten

    /** Encode decisions taken, split by outcome, for the "encoder is not re-run every tick" claim. */
    private var encodeCount = 0
    private var skipCount = 0
    val encodes: Int get() = encodeCount
    val skips: Int get() = skipCount

    fun start(): Job {
        consumerJob?.cancel()
        melJob?.cancel()
        val melWorker = scope.launch(Dispatchers.Default) {
            for (chunk in melInput) {
                // Runs in capture order on its own coroutine, so a slow mel pass can never
                // block the microphone read the way it used to.
                latestMel = melCache.append(chunk, SAMPLE_RATE)
                melSamplesFed += chunk.size
                melReady.trySend(Unit)
            }
        }
        melJob = melWorker
        val job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                wake.receive()
                // Drain any further signals that arrived while we were idle, then take a
                // single fresh decision: the trigger is a threshold on *accumulated* new
                // audio, so a signal that no longer passes it simply does nothing.
                while (wake.tryReceive().isSuccess) { /* coalesce */ }
                val decision = decide(forceFinal = false)
                if (decision.encode) {
                    encodeCount++
                    awaitMelFor(ring.totalSamplesWritten)
                    runOnce(decision)
                } else {
                    skipCount++
                }
            }
        }
        consumerJob = job
        return job
    }

    /**
     * Waits until the mel worker has consumed [sampleCount] samples.
     *
     * The mel window must cover the audio being decoded, and the worker runs ahead of or behind
     * the ring independently. Waiting here is safe precisely because the worker does nothing but
     * mel work: it cannot be blocked by inference, so this always terminates once the queued
     * chunks are processed.
     */
    private suspend fun awaitMelFor(sampleCount: Long) {
        while (melSamplesFed < sampleCount) {
            if (melInput.isEmpty && melSamplesFed < sampleCount) {
                // Worker is idle yet still behind: the chunks are gone (dropped), so waiting
                // would hang. Report it rather than spinning forever.
                Log.w(TAG, "MEL_STARVED fed=$melSamplesFed wanted=$sampleCount dropped=${melDropped.get()}")
                return
            }
            melReady.receive()
        }
    }

    /** Producer entry point: append captured PCM. Cheap and non-blocking. */
    fun onAudioAppended(samples: ShortArray) {
        if (samples.isEmpty()) return
        val before = ring.totalSamplesWritten
        ring.write(samples)
        val stored = (ring.totalSamplesWritten - before).toInt()
        if (stored > 0) {
            val peak = LiveTranscribePolicy.peakAmplitude(samples)
            if (peak > pendingPeak) pendingPeak = peak
        }
        wake.trySend(Unit)
    }

    /**
     * Producer entry point for a captured chunk: the ring write and peak scan happen here
     * (both O(samples) memory work), and the mel append is queued to [melInput] for the mel
     * worker to consume. The producer never does mel work, so a capture read can never be
     * delayed by it.
     *
     * Returns the newest available mel window, which may lag this chunk by a few hundred ms;
     * the decode path waits for the worker to catch up via `awaitMelFor`.
     */
    fun onAudioAppendedWithMel(samples: ShortArray): ShortArray? {
        if (samples.isEmpty()) return null
        onAudioAppended(samples)
        if (melInput.trySend(samples).isFailure) {
            melDropped.incrementAndGet()
            Log.w(TAG, "MEL_QUEUE_FULL dropped=${melDropped.get()}")
        }
        return latestMel
    }

    /**
     * Requests a final full decode (end of utterance / user pressed stop). The consumer
     * performs it before shutting down.
     */
    fun requestFinal(): Job? {
        val job = scope.launch(Dispatchers.Default) { runOnce(decide(forceFinal = true)) }
        return job
    }

    fun stop() {
        consumerJob?.cancel()
        consumerJob = null
        melJob?.cancel()
        melJob = null
        // The channel is intentionally NOT closed: the recording loop calls stop() and start()
        // repeatedly, and a closed channel would reject every subsequent chunk.
    }

    fun reset() {
        ring.reset()
        melCache.reset()
        melSamplesFed = 0
        melDropped.set(0)
        samplesAtLastRun = 0
        pendingPeak = 0
        latestMel = null
        updateId = 0
        encodeCount = 0
        skipCount = 0
    }

    private data class Decision(val encode: Boolean, val final: Boolean, val steps: Int, val totalSamples: Int)

    private fun decide(forceFinal: Boolean): Decision {
        val total = ring.size
        val sinceLastRun = (ring.totalSamplesWritten - samplesAtLastRun).toInt()
        val encode = LiveTranscribePolicy.shouldEncode(
            totalSamples = total,
            samplesSinceLastRun = sinceLastRun,
            peakAmplitude = pendingPeak,
            forceFinal = forceFinal,
            triggerSamples = triggerSamples,
        )
        return Decision(encode, forceFinal, LiveTranscribePolicy.stepsFor(forceFinal, previewSteps, fullSteps), total)
    }

    private fun runOnce(decision: Decision) {
        if (!decision.encode) {
            // Nothing to do, but remember the audio so a later trigger measures from here.
            samplesAtLastRun = max(samplesAtLastRun, ring.totalSamplesWritten)
            return
        }
        updateId++
        val currentId = updateId
        val snapshot = ring.snapshotWithIndex()
        val audio = snapshot.samples
        if (audio.isEmpty()) return
        // The mel window is produced by the producer side (see [onAudioAppendedForMel]) so
        // inference never has to replay audio that already left the ring.
        val melHalf = latestMel
        if (melHalf == null) {
            Log.w(TAG, "SKIP #$currentId no mel window available yet")
            return
        }
        val startedNs = System.nanoTime()
        var error: Throwable? = null
        var text = ""
        var completedSteps = 0
        val result = runCatching {
            QnnWhisperRealAudioRunner.transcribeChunk(
                context = context,
                pcm16 = audio,
                sampleRate = SAMPLE_RATE,
                requestedSteps = decision.steps,
                autoregressive = true,
                precomputedMelHalf = melHalf,
                debugUpdateId = currentId,
                variant = variant,
            ).also { check(it.passed) { it.report } }
        }
        val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000.0
        when {
            result.isSuccess -> {
                text = result.getOrThrow().text.trim()
                completedSteps = decision.steps
            }
            else -> {
                error = result.exceptionOrNull()
                Log.e(TAG, "TRANSCRIBE_ERROR #$currentId type=${error?.javaClass?.name} message=${error?.message}", error)
            }
        }
        val newAudioSamples = (ring.totalSamplesWritten - samplesAtLastRun).toInt().coerceAtLeast(0)
        samplesAtLastRun = ring.totalSamplesWritten
        pendingPeak = 0
        val newAudioSeconds = newAudioSamples / SAMPLE_RATE.toDouble()
        val rtf = if (newAudioSeconds > 0) elapsedMs / 1000.0 / newAudioSeconds else null
        Log.i(
            TAG,
            "TRANSCRIBE_DONE #$currentId final=${decision.final} steps=${decision.steps} elapsedMs=$elapsedMs " +
                "audioSamples=${audio.size} windowSec=${audio.size / SAMPLE_RATE.toDouble()} newAudioSamples=$newAudioSamples rtf=$rtf " +
                "cumulativeSamples=${snapshot.firstSampleIndex + audio.size} " +
                "overwritten=${snapshot.overwrittenSamples} text=\"${if (text.isBlank()) "<EMPTY>" else text.take(160)}\"",
        )
        onResult(
            Result(
                updateId = currentId,
                text = text,
                elapsedMs = elapsedMs,
                steps = completedSteps,
                final = decision.final,
                audioSamples = audio.size,
                realTimeFactor = rtf,
                newAudioSamples = newAudioSamples,
                cumulativeSamples = snapshot.firstSampleIndex + audio.size,
                overwrittenSamples = snapshot.overwrittenSamples,
                error = error,
            ),
        )
    }

    companion object {
        private const val TAG = "WHISPER_LIVE"
        const val SAMPLE_RATE = 16_000
    }
}
