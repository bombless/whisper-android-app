package io.github.bombless.htp_playpen.asr

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/** Microphone producer and offline utterance worker are decoupled by a bounded PCM queue. */
class SenseVoiceRecordingSession(
    context: Context,
    private val engine: SenseVoiceAsrEngine,
    private val scope: CoroutineScope,
    private val onResult: (AsrResult) -> Unit,
    private val onStatus: (String) -> Unit,
) {
    private val appContext = context.applicationContext
    @Volatile var droppedChunks: Long = 0
        private set

    fun start(): Job = scope.launch(Dispatchers.IO) {
        val sampleRate = SenseVoiceAsrEngine.TARGET_SAMPLE_RATE
        val frameSamples = sampleRate / 10 // 100 ms
        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        check(minBuffer > 0) { "麦克风不支持 16 kHz PCM16" }
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuffer, frameSamples * 8),
        )
        check(recorder.state == AudioRecord.STATE_INITIALIZED) { "麦克风初始化失败" }
        val queue = Channel<ShortArray>(capacity = QUEUE_CAPACITY)
        val worker = scope.launch(Dispatchers.Default) {
            val utterance = ArrayList<ShortArray>()
            var utteranceSamples = 0
            var silenceFrames = 0
            var speechSeen = false
            suspend fun flush() {
                if (utteranceSamples == 0) return
                val pcm = ShortArray(utteranceSamples)
                var offset = 0
                for (part in utterance) {
                    part.copyInto(pcm, offset)
                    offset += part.size
                }
                utterance.clear()
                utteranceSamples = 0
                silenceFrames = 0
                speechSeen = false
                try {
                    val result = engine.transcribe(pcm, sampleRate)
                    if (result.text.isNotBlank()) onResult(result)
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (failure: Throwable) {
                    Log.e(TAG, "Utterance transcription failed", failure)
                    onStatus("SenseVoice 识别失败：${failure.message ?: failure.javaClass.simpleName}")
                }
            }
            for (frame in queue) {
                val rms = rms(frame)
                val voiced = rms >= VAD_RMS_THRESHOLD
                if (voiced) {
                    speechSeen = true
                    silenceFrames = 0
                } else if (speechSeen) {
                    silenceFrames++
                }
                if (speechSeen || utteranceSamples > 0) {
                    utterance += frame
                    utteranceSamples += frame.size
                }
                if (speechSeen && (silenceFrames >= SILENCE_FRAMES || utteranceSamples >= MAX_UTTERANCE_SAMPLES)) {
                    flush()
                }
            }
            // Process the last partial utterance when the user stops recording.
            flush()
        }
        val buffer = ShortArray(frameSamples)
        try {
            recorder.startRecording()
            check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "麦克风未进入录音状态" }
            onStatus("SenseVoice 正在监听…（静音分句）")
            while (currentCoroutineContext().isActive) {
                val count = recorder.read(buffer, 0, buffer.size)
                if (count < 0) error("麦克风读取失败：$count")
                if (count == 0) continue
                val frame = if (count == buffer.size) buffer.copyOf() else buffer.copyOf(count)
                if (queue.trySend(frame).isFailure) {
                    droppedChunks++
                    Log.w(TAG, "PCM_QUEUE_FULL droppedChunks=$droppedChunks")
                }
            }
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
            queue.close()
            withContext(NonCancellable) { worker.join() }
            onStatus("已停止 · 丢弃音频块 $droppedChunks")
        }
    }

    private fun rms(samples: ShortArray): Double {
        if (samples.isEmpty()) return 0.0
        var sum = 0.0
        for (sample in samples) {
            val value = sample.toDouble()
            sum += value * value
        }
        return sqrt(sum / samples.size)
    }

    companion object {
        private const val TAG = "SENSEVOICE_CAPTURE"
        private const val QUEUE_CAPACITY = 64 // 6.4 seconds of 100 ms frames
        private const val VAD_RMS_THRESHOLD = 420.0
        private const val SILENCE_FRAMES = 7 // 700 ms endpointing
        private const val MAX_UTTERANCE_SAMPLES = 15 * SenseVoiceAsrEngine.TARGET_SAMPLE_RATE
    }
}