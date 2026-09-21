package com.example.whisperapp.asr

import com.example.whisperapp.audio.WhisperFeatureExtractor
import com.example.whisperapp.audio.WhisperIncrementalMelCache
import com.example.whisperapp.audio.WhisperMelConfig
import com.example.whisperapp.audio.WhisperMelFrontend
import com.example.whisperapp.audio.WhisperTurboFeatureExtractor

/**
 * Everything that differs between the shipped Whisper QNN variants.
 *
 * The encoder/decoder graph structure is identical for both (4 decoder layers,
 * 200-wide attention mask, 199 self-KV slots, 1500 cross-KV positions), so the
 * runner stays one code path and reads its variant-specific numbers here.
 *
 * Values are taken from each asset's `metadata.json`:
 *  - Tiny (`models/whisper`):          input_features [1,80,3000],  6 heads, vocab 51865
 *  - Turbo (`models/whisper_large_v3_turbo`): input_features [1,128,3000], 20 heads, vocab 51866
 *
 * Special-token IDs are deliberately NOT listed here: they are resolved from the
 * loaded tokenizer by name, because Large-V3 inserts `<|yue|>` and thereby shifts
 * `<|transcribe|>`/`<|notimestamps|>` by one relative to Tiny.
 */
enum class WhisperVariant(
    val displayName: String,
    val assetDir: String,
    val melConfig: WhisperMelConfig,
    val nHeads: Int,
    val vocabSize: Int,
) {
    TINY(
        displayName = "Whisper Tiny",
        assetDir = "models/whisper",
        melConfig = WhisperMelConfig.Tiny,
        nHeads = 6,
        vocabSize = 51865,
    ),
    LARGE_V3_TURBO(
        displayName = "Whisper Large v3 Turbo",
        assetDir = "models/whisper_large_v3_turbo",
        melConfig = WhisperMelConfig.LargeV3Turbo,
        nHeads = 20,
        vocabSize = 51866,
    );

    val nMels: Int get() = melConfig.nMels
    val nFrames: Int get() = melConfig.nFrames

    /** `attention_mask` width declared by both assets' metadata.json. */
    val maxDecodeLength: Int get() = 200

    /** Self-KV cache slots: the graph appends the new token and keeps the last 199. */
    val selfCacheLength: Int get() = maxDecodeLength - 1

    val encoderModelAsset: String get() = "$assetDir/encoder_ctx.onnx"
    val decoderModelAsset: String get() = "$assetDir/decoder_ctx.onnx"
    val encoderBinaryAsset: String get() = "$assetDir/encoder.bin"
    val decoderBinaryAsset: String get() = "$assetDir/decoder.bin"
    val tokenizerAssetDir: String get() = "$assetDir/tokenizer"

    fun newFrontend(): WhisperMelFrontend = when (this) {
        TINY -> WhisperFeatureExtractor()
        LARGE_V3_TURBO -> WhisperTurboFeatureExtractor(melConfig)
    }

    fun newMelCache(): WhisperIncrementalMelCache = when (this) {
        TINY -> WhisperFeatureExtractor().newIncrementalCache()
        LARGE_V3_TURBO -> WhisperTurboFeatureExtractor(melConfig).newIncrementalCache()
    }

    /** `k_cache_*` is [heads,1,64,slots]; `v_cache_*` is [heads,1,slots,64]. */
    fun kvShape(isKey: Boolean, slots: Int): LongArray =
        if (isKey) longArrayOf(nHeads.toLong(), 1, 64, slots.toLong())
        else longArrayOf(nHeads.toLong(), 1, slots.toLong(), 64)

    fun selfInShape(name: String): LongArray = kvShape(name.startsWith("k_"), selfCacheLength)
}
