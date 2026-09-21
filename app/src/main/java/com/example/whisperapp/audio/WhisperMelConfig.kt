package com.example.whisperapp.audio

/**
 * Model-specific audio feature contract.
 *
 * Keep these values separate from the QNN decoder contract so a model variant
 * cannot accidentally inherit another variant's mel dimensions.
 */
data class WhisperMelConfig(
    val name: String,
    val sampleRate: Int = 16_000,
    val nFft: Int = 400,
    val hopLength: Int = 160,
    val nMels: Int,
    val nFrames: Int = 3000,
    val chunkSamples: Int = 480_000,
    val minFrequencyHz: Double = 0.0,
    val maxFrequencyHz: Double = 8_000.0,
    val logFloor: Float = 1e-10f,
    val logDynamicRange: Float = 8f,
) {
    companion object {
        val Tiny = WhisperMelConfig(name = "whisper-tiny", nMels = 80)
        val LargeV3Turbo = WhisperMelConfig(name = "whisper-large-v3-turbo", nMels = 128)
    }
}