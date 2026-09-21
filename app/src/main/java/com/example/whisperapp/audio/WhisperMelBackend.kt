package com.example.whisperapp.audio

/**
 * The two mel operations the QNN Whisper runner needs, independent of which
 * variant's frontend provides them (Tiny 80-bin vs Large-v3-Turbo 128-bin).
 */
interface WhisperMelFrontend {
    fun extract(pcm16: ShortArray, sampleRate: Int): WhisperFeatures
    fun extractHalf(pcm16: ShortArray, sampleRate: Int): ShortArray
}

/** Incremental mel cache used by the live-transcription loop. */
interface WhisperIncrementalMelCache {
    /** Appends PCM and returns the latest full FP16 mel window ([nMels, nFrames]). */
    fun append(samples: ShortArray, sampleRate: Int): ShortArray
    fun reset()
}
