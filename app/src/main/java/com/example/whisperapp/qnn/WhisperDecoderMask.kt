package com.example.whisperapp.qnn

/** Qualcomm appends the current KV on the right and drops the oldest slot. */
internal object WhisperDecoderMask {
    const val WIDTH = 200
    private val maskedHalf = 0xd640.toShort() // FP16 -100.0, Qualcomm MASK_NEG.

    fun forPosition(position: Int): ShortArray {
        require(position in 0 until WIDTH) { "decoder position=$position outside attention mask width" }
        val firstValidIndex = WIDTH - position - 1
        return ShortArray(WIDTH) { index -> if (index >= firstValidIndex) 0 else maskedHalf }
    }
}
