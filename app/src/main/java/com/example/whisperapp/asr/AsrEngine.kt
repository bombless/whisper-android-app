package com.example.whisperapp.asr

data class AsrResult(val text: String, val latencyMs: Long)

interface AsrEngine {
    suspend fun transcribe(pcm16: ShortArray, sampleRate: Int): AsrResult
    fun close()
}
