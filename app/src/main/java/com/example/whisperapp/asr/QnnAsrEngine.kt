package com.example.whisperapp.asr

/** Placeholder until a verified AI Hub Whisper-Tiny QNN/HTP asset and matching QAIRT SDK are present. */
class QnnAsrEngine : AsrEngine {
    override suspend fun transcribe(pcm16: ShortArray, sampleRate: Int): AsrResult =
        error("QNN ASR is not enabled: verified Whisper-Tiny QNN asset/runtime is required")
    override fun close() = Unit
}
