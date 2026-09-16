package com.example.whisperapp.tts

/**
 * QNN MeloTTS-ZH adapter boundary.
 *
 * The repository currently has no verified MeloTTS-ZH QNN/HTP model/context
 * binary or matching native wrapper, so inference deliberately remains
 * disabled until those frozen assets are added. Keeping the failure here
 * prevents TTS errors from cancelling the Whisper recording coroutine.
 */
class QnnTtsEngine : TtsEngine {
    override suspend fun synthesize(text: String): TtsResult =
        error("QNN TTS is not enabled: verified MeloTTS-ZH QNN asset/runtime is required")

    override fun close() = Unit
}
