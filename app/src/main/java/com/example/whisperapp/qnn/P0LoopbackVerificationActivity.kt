package com.example.whisperapp.qnn

import android.app.Activity
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import com.example.whisperapp.asr.WhisperVariant
import com.example.whisperapp.audio.LiveTranscriber
import com.example.whisperapp.audio.WavPcm16
import com.example.whisperapp.audio.WavPcmReader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * P0-1 acceptance on real hardware: record through the microphone while a known WAV plays
 * back, then check that **no captured sample was lost**.
 *
 * The plan's M1 exit criterion ("连续说话 30 s，丢样计数为 0") needs a live speaker, which is
 * neither repeatable nor available in an automated run. This activity replaces the human with
 * the device's own speaker: `output.wav` is played through [AudioTrack] while the same
 * `AudioRecord` → [LiveTranscriber] pipeline the app uses in production captures it.
 *
 * Why this is a valid test of the decoupling and not just of the recorder: the point of P0-1
 * was that inference used to block the read loop, so everything spoken during a ~1 s
 * transcription was dropped. That failure mode only shows up when the consumer is genuinely
 * slower than real time, which is exactly the case here — a full Turbo encoder + decoder pass
 * runs for every triggered update while playback keeps pushing samples.
 *
 * Two independent checks, both required:
 *  1. `ring.overwrittenSamples == 0` — the ring never evicted anything, i.e. the consumer kept
 *     up with the producer over a 30 s window;
 *  2. the recorder's own sample total matches the wall-clock duration within [TOLERANCE_MS],
 *     which catches the *other* failure mode — AudioRecord silently returning fewer samples
 *     than real time passed (an overrun), which no ring-buffer counter can see.
 *
 * Usage:
 *   adb shell am start -n com.example.whisperapp/.qnn.P0LoopbackVerificationActivity \
 *       --es variant turbo --ei seconds 30
 *
 * Writes `p0_loopback_result.txt` into the app's external files dir.
 */
class P0LoopbackVerificationActivity : Activity() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            // The report is flushed as it is produced, not only at the end: this activity runs a
            // real encoder/decoder load and the host has no other way to observe it (this ROM
            // drops the app's logcat). Writing only on success meant a run that died or was
            // killed mid-capture left no evidence at all.
            val out = StringBuilder()
            fun flush() = runCatching {
                File(getExternalFilesDir(null), RESULT_FILE).writeText(out.toString())
            }

            try {
                runVerification(out, ::flush)
            } catch (t: Throwable) {
                out.appendLine("P0-1 LOOPBACK: ERROR")
                out.appendLine(t.stackTraceToString())
                Log.e(TAG, "LOOPBACK FAILED", t)
            }
            flush()
            OpTrace.stage("p0_loopback.written")
            runOnUiThread { finish() }
        }.start()
    }

    private fun runVerification(out: StringBuilder, flush: () -> Unit) {
        val variant = when (intent.getStringExtra("variant")?.lowercase()) {
            "turbo", "large", "large_v3_turbo", "large-v3-turbo" -> WhisperVariant.LARGE_V3_TURBO
            else -> WhisperVariant.TINY
        }
        val requestedSeconds = intent.getIntExtra("seconds", 30).coerceIn(5, 30)
        // `--ez synthetic true` skips the microphone and paces a known PCM stream at exactly
        // real time. The ROM refuses microphone access to any app that is not the focused
        // window (a background game is enough to block it), which makes the mic path
        // unrunnable in automation. The synthetic mode still exercises the real ring buffer
        // and the real inference consumer at true real-time rates, so it answers the question
        // the acceptance test exists for: does the pipeline keep up, and is anything lost?
        val synthetic = intent.getBooleanExtra("synthetic", false)

        OpTrace.stageLogFile(File(getExternalFilesDir(null), "decoder_stages.log"))
        OpTrace.stage("p0_loopback.begin variant=${variant.name} seconds=$requestedSeconds")

        val wavFile = File(getExternalFilesDir(null), "output.wav")
        check(wavFile.isFile) { "missing ${wavFile.absolutePath}" }
        val wav: WavPcm16 = wavFile.inputStream().use { WavPcmReader.read(it) }
        // Loop the source if it is shorter than the requested window; the measurement is about
        // sample accounting, so repeating the utterance is harmless and keeps the run bounded.
        val targetSamples = SAMPLE_RATE * requestedSeconds
        targetSamplesForDeadline = targetSamples
        val playback = buildPlayback(wav, targetSamples)

        out.appendLine("P0-1 LOOPBACK ACCEPTANCE (variant=${variant.name})")
        out.appendLine("mode: ${if (synthetic) "synthetic_real_time" else "microphone_loopback"}")
        out.appendLine("source_wav_samples: ${wav.samples.size}")
        out.appendLine("target_seconds: $requestedSeconds")
        out.appendLine("target_samples: $targetSamples")

        // The recorder is only created in microphone mode; synthetic mode paces a known PCM
        // stream itself, which is the only way to run this on a ROM that blocks microphone
        // access for any app that is not the focused window.
        val minBuffer = if (synthetic) 0 else {
            val b = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            check(b > 0) { "AudioRecord unavailable (minBuffer=$b)" }
            b
        }

        val melCache = variant.newMelCache()
        val results = mutableListOf<LiveTranscriber.Result>()
        var transcriber: LiveTranscriber? = null
        var consumerJob: Job? = null
        var recorder: AudioRecord? = null
        var track: AudioTrack? = null
        var readTotal = 0L
        val readChunk = ShortArray(SAMPLE_RATE / 10) // 100 ms reads
        var readErrors = 0

        try {
            QnnWhisperRealAudioRunner.start(this, variant)
            OpTrace.stage("p0_loopback.session_ready")

            if (!synthetic) {
                recorder = AudioRecord(
                    MediaRecorder.AudioSource.MIC, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, SAMPLE_RATE),
                )
                check(recorder.state == AudioRecord.STATE_INITIALIZED) { "microphone init failed" }
            }

            val live = LiveTranscriber(
                context = applicationContext,
                scope = scope,
                variant = variant,
                melCache = melCache,
                onResult = { r -> synchronized(results) { results += r } },
            )
            transcriber = live
            consumerJob = live.start()

            track = buildTrack()
            if (!synthetic) recorder?.startRecording()

            // Playback runs on its own thread. It must NOT share a thread with the read loop:
            // `AudioTrack.write` blocks until the sink drains, so a single threaded
            // write-then-read loop advances at the speed of the slower of the two, and the
            // capture is no longer paced by the microphone. That measured 1/5 real time and
            // made the run meaningless. With playback decoupled, `recorder.read` blocks for
            // exactly the audio it returns (i.e. real time), which is the production condition.
            val playbackThread = if (synthetic) null else Thread {
                var offset = 0
                while (offset < playback.size && !Thread.currentThread().isInterrupted) {
                    val feed = minOf(trackBufferSamples, playback.size - offset)
                    val written = track.write(playback, offset, feed)
                    if (written <= 0) break
                    offset += written
                }
            }.also { it.start() }

            if (!synthetic) track.play()
            val wallStartNs = System.nanoTime()
            // Separate the two possible causes of wall-clock drift:
            //  - `readBlockedMs`: time spent inside AudioRecord.read. If this dominates, the
            //    microphone itself is the pacer (expected: real time).
            //  - `appendMs`: time spent in our own producer work (ring + mel). If THIS grows,
            //    the producer is doing too much work per read and would be a real P0-1 defect,
            //    because the mel append runs on the same thread as the microphone read.
            var readBlockedNs = 0L
            var appendNs = 0L
            // A read that never returns is the one failure the counters cannot describe: the
            // ROM can leave AudioRecord blocked with no data when the app is not the focused
            // window. Bound the whole capture so a stuck run reports a diagnosis instead of
            // hanging the activity forever.
            val captureDeadlineNs = wallStartNs + (expectedCaptureMs() * STALL_FACTOR * 1_000_000L).toLong()
            var syntheticOffset = 0
            var progressMark = 0L
            while (readTotal < targetSamples) {
                if (readTotal - progressMark >= SAMPLE_RATE * 5) {
                    progressMark = readTotal
                    out.appendLine("progress_samples: $readTotal")
                    flush()
                }
                if (System.nanoTime() > captureDeadlineNs) {
                    out.appendLine("capture_stalled: true")
                    OpTrace.stage("p0_loopback.stalled samples=$readTotal")
                    flush()
                    break
                }
                val readStart = System.nanoTime()
                val n: Int
                val chunk: ShortArray
                if (synthetic) {
                    // Emit exactly one read-chunk per its own duration, so the producer runs at
                    // true real time: `readChunk.size` samples every `readChunk.size / 16 kHz`.
                    val want = minOf(readChunk.size.toLong(), targetSamples - readTotal).toInt()
                    chunk = ShortArray(want)
                    for (i in 0 until want) chunk[i] = playback[(syntheticOffset + i) % playback.size]
                    syntheticOffset += want
                    n = want
                    val dueNs = wallStartNs + ((readTotal + n) * 1_000_000_000L) / SAMPLE_RATE
                    val sleepNs = dueNs - System.nanoTime()
                    if (sleepNs > 0) Thread.sleep(sleepNs / 1_000_000L, (sleepNs % 1_000_000L).toInt())
                } else {
                    n = recorder!!.read(readChunk, 0, readChunk.size)
                    chunk = if (n == readChunk.size) readChunk else readChunk.copyOf(maxOf(n, 0))
                }
                readBlockedNs += System.nanoTime() - readStart
                if (n <= 0) {
                    readErrors++
                    continue
                }
                val appendStart = System.nanoTime()
                live.onAudioAppendedWithMel(chunk)
                appendNs += System.nanoTime() - appendStart
                readTotal += n
            }
            val wallMs = (System.nanoTime() - wallStartNs) / 1_000_000.0
            val readBlockedMs = readBlockedNs / 1_000_000.0
            val appendMs = appendNs / 1_000_000.0
            OpTrace.stage("p0_loopback.capture_done samples=$readTotal wallMs=$wallMs readBlockedMs=$readBlockedMs appendMs=$appendMs")

            runCatching { if (!synthetic) track.stop() }
            runCatching { playbackThread?.interrupt() }
            runCatching { if (!synthetic) recorder?.stop() }

            // Drain the tail through a full decode, exactly as the production loop does.
            runCatching { transcriber.requestFinal()?.let { runBlocking { it.join() } } }
            runBlocking { consumerJob.cancelAndJoin() }

            val written = live.totalSamplesWritten
            val overwritten = live.overwrittenSamples
            val expectedMs = targetSamples * 1000.0 / SAMPLE_RATE
            val driftMs = wallMs - expectedMs
            val matched = written == readTotal.toLong()
            val monotonic = isMonotonic(results.map { it.cumulativeSamples })
            val maxUpdateMs = results.maxOfOrNull { it.elapsedMs } ?: 0.0
            val medianUpdateMs = median(results.map { it.elapsedMs })
            val previewSteps = results.filter { !it.final }.map { it.steps }.distinct()
            val finalSteps = results.filter { it.final }.map { it.steps }.distinct()

            out.appendLine("read_samples: $readTotal")
            out.appendLine("read_errors: $readErrors")
            out.appendLine("samples_written_to_ring: $written")
            out.appendLine("samples_match_read: $matched")
            out.appendLine("overwritten_samples: $overwritten")
            out.appendLine("capture_wall_ms: ${"%.1f".format(wallMs)}")
            out.appendLine("expected_wall_ms: ${"%.1f".format(expectedMs)}")
            out.appendLine("wall_drift_ms: ${"%.1f".format(driftMs)}")
            out.appendLine("read_blocked_ms: ${"%.1f".format(readBlockedMs)}")
            out.appendLine("producer_append_ms: ${"%.1f".format(appendMs)}")
            out.appendLine("updates_completed: ${results.size}")
            out.appendLine("encodes: ${live.encodes}")
            out.appendLine("skips: ${live.skips}")
            out.appendLine("cumulative_monotonic: $monotonic")
            out.appendLine("update_median_ms: ${"%.1f".format(medianUpdateMs)}")
            out.appendLine("update_max_ms: ${"%.1f".format(maxUpdateMs)}")
            out.appendLine("preview_steps_seen: $previewSteps")
            out.appendLine("final_steps_seen: $finalSteps")
            val rtfs = results.filter { !it.final }.mapNotNull { it.realTimeFactor }
            out.appendLine("preview_rtf_median: ${"%.4f".format(median(rtfs))}")
            out.appendLine("preview_rtf_max: ${"%.4f".format(rtfs.maxOrNull() ?: 0.0)}")
            out.appendLine("preview_rtf_note: inference_seconds / NEW_audio_seconds (the keep-up criterion)")
            // Total inference time against total captured audio: the end-to-end duty cycle.
            val totalInferenceMs = results.sumOf { it.elapsedMs }
            val dutyCycle = totalInferenceMs / wallMs
            out.appendLine("total_inference_ms: ${"%.1f".format(totalInferenceMs)}")
            out.appendLine("inference_duty_cycle: ${"%.4f".format(dutyCycle)}")
            out.appendLine("texts: ${results.map { it.text.take(40) }}")

            // Acceptance: nothing dropped, nothing invented, and the capture ran in real time.
            val noLoss = overwritten == 0L && matched
            val realTime = kotlin.math.abs(driftMs) <= TOLERANCE_MS
            out.appendLine("ACCEPT_NO_LOST_AUDIO: $noLoss")
            out.appendLine("ACCEPT_CAPTURE_REAL_TIME: $realTime (tolerance ${TOLERANCE_MS}ms)")
            out.appendLine(if (noLoss && realTime) "P0-1 LOOPBACK: PASS" else "P0-1 LOOPBACK: FAIL")
            flush()
            OpTrace.stage("p0_loopback.done noLoss=$noLoss realTime=$realTime overwritten=$overwritten")
        } finally {
            runCatching { track?.stop() }
            runCatching { track?.release() }
            runCatching { recorder?.stop() }
            runCatching { recorder?.release() }
            runCatching { transcriber?.stop() }
            runCatching { QnnWhisperRealAudioRunner.stop() }
        }
    }

    /** Repeats (or truncates) [wav] to exactly [targetSamples] so playback length is known. */
    private fun buildPlayback(wav: WavPcm16, targetSamples: Int): ShortArray {
        val out = ShortArray(targetSamples)
        var offset = 0
        while (offset < targetSamples) {
            val n = minOf(wav.samples.size, targetSamples - offset)
            wav.samples.copyInto(out, offset, 0, n)
            offset += n
        }
        return out
    }

    private var targetSamplesForDeadline = 0

    private fun expectedCaptureMs(): Double =
        targetSamplesForDeadline * 1000.0 / SAMPLE_RATE

    private fun buildTrack(): AudioTrack = AudioTrack.Builder()
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build(),
        )
        .setAudioFormat(
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(SAMPLE_RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
        )
        .setBufferSizeInBytes(SAMPLE_RATE * 2)
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()

    private fun isMonotonic(values: List<Long>): Boolean =
        values.zipWithNext().all { (a, b) -> b >= a }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    private companion object {
        const val TAG = "P0_LOOPBACK"
        const val SAMPLE_RATE = 16_000
        const val RESULT_FILE = "p0_loopback_result.txt"
        /** Wall-clock vs audio-clock drift allowed before we call it an overrun. */
        const val TOLERANCE_MS = 20.0
        /** Capture is abandoned after this multiple of the expected duration. */
        const val STALL_FACTOR = 4.0
        const val trackBufferSamples = 8_000
    }
}
