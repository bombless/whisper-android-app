package io.github.bombless.htp_playpen.asr

data class AsrResult(val text: String, val latencyMs: Long)

interface AsrEngine {
    suspend fun transcribe(pcm16: ShortArray, sampleRate: Int): AsrResult
    fun close()
}

enum class SttEngineChoice(val displayName: String) {
    WHISPER_QNN("Whisper QNN"),
    SENSEVOICE("SenseVoice INT8 · CPU"),
}
