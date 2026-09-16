package com.example.whisperapp.tts

import com.example.whisperapp.audio.AudioPlayer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

/** Serializes TTS synthesis and playback so ASR never waits on audio generation. */
class TtsQueue(
    private val scope: CoroutineScope,
    private val engine: TtsEngine,
    private val player: AudioPlayer,
    private val onError: (Throwable) -> Unit = {},
) {
    private val queue = Channel<String>(Channel.UNLIMITED)
    private var worker: Job = scope.launch(Dispatchers.IO) {
        for (text in queue) {
            try {
                val result = engine.synthesize(text)
                if (result.pcm16.isNotEmpty() && result.sampleRate > 0) {
                    player.play(result.pcm16, result.sampleRate)
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

    fun clearAndStop() {
        while (queue.tryReceive().isSuccess) Unit
        player.stop()
    }

    fun close() {
        queue.close()
        worker.cancel()
        player.stop()
        engine.close()
    }
}
