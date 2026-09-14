package com.example.whisperapp.tts

interface TtsEngine {
    suspend fun synthesize(text: String): ShortArray
    fun close()
}
