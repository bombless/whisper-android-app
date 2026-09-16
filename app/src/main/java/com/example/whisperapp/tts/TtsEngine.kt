package com.example.whisperapp.tts

interface TtsEngine {
    suspend fun synthesize(text: String): TtsResult
    fun close()
}

data class TtsResult(
    val pcm16: ShortArray,
    val sampleRate: Int,
    val latencyMs: Long,
)
