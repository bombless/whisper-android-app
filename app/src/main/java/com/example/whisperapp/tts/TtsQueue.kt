package com.example.whisperapp.tts

import com.example.whisperapp.audio.AudioPlayer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

/** Serializes independent offline TTS synthesis and PCM playback. */
class TtsQueue(
    private val scope: CoroutineScope,
    private val engine: TtsEngine,
    private val player: AudioPlayer,
    private val onError: (Throwable) -> Unit = {},
) {
    private val queue = Channel<String>(Channel.UNLIMITED)
    private val worker: Job = scope.launch(Dispatchers.IO) {
        for (text in queue) {
            try {
                val result = engine.synthesize(text)
                if (result.pcm16.isNotEmpty() && result.sampleRate > 0) {
                    player.play(result.pcm16, result.sampleRate)
                } else {
                    throw IllegalStateException("TTS returned empty audio for ${text.length} characters")
                }
            } catch (t: Throwable) {
                onError(t)
            }
        }
    }

    fun offer(text: String) {
        val normalized = ChineseTextProcessor.normalizeChineseText(text)
        if (normalized.isNotBlank()) queue.trySend(normalized)
    }

    suspend fun prepare() {
        engine.prepare()
    }

    fun clearAndStop() {
        while (queue.tryReceive().isSuccess) Unit
        player.stop()
    }

    /** Wait for native synthesis to finish before releasing the native engine. */
    suspend fun closeAndWait() {
        while (queue.tryReceive().isSuccess) Unit
        queue.close()
        player.stop()
        worker.cancelAndJoin()
        engine.close()
    }

    /** Non-suspending disposal fallback; does not release native engine prematurely. */
    fun close() {
        queue.close()
        player.stop()
        worker.cancel()
    }
}
