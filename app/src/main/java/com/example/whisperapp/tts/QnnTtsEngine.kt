package com.example.whisperapp.tts

/** Placeholder until a verified AI Hub MeloTTS-ZH QNN/HTP asset and matching QAIRT SDK are present. */
class QnnTtsEngine : TtsEngine {
    override suspend fun synthesize(text: String): ShortArray =
        error("QNN TTS is not enabled: verified MeloTTS-ZH QNN asset/runtime is required")
    override fun close() = Unit
}
