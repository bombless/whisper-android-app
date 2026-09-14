package com.example.whisperapp.asr

import android.content.Context
import java.security.MessageDigest

/**
 * Verified-asset gate for Whisper-Tiny. Full encoder/decoder execution is intentionally
 * kept separate from the smoke-test path until the exact QAIRT context invocation ABI is wired.
 */
class QnnAsrEngine(private val context: Context) : AsrEngine {
    data class AssetInfo(val encoderBytes: Long, val decoderBytes: Long, val metadata: String)

    fun verifyAssets(): AssetInfo {
        val encoder = context.assets.open("models/whisper/encoder.bin").use { it.readBytes() }
        val decoder = context.assets.open("models/whisper/decoder.bin").use { it.readBytes() }
        val metadata = context.assets.open("models/whisper/whisper_metadata.json")
            .bufferedReader(Charsets.UTF_8).use { it.readText() }
        check(encoder.size == 19_832_832) { "Unexpected Whisper encoder size=${encoder.size}" }
        check(decoder.size == 97_452_032) { "Unexpected Whisper decoder size=${decoder.size}" }
        check(metadata.contains("qualcomm-snapdragon-8gen3")) { "Wrong Whisper target metadata" }
        check(metadata.contains("QAIRT 2.45.0.260326154327")) { "Unexpected Whisper QAIRT version" }
        return AssetInfo(encoder.size.toLong(), decoder.size.toLong(), metadata)
    }

    fun assetSha256(path: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        context.assets.open(path).use { input ->
            val buffer = ByteArray(64 * 1024)
            var count = input.read(buffer)
            while (count >= 0) {
                if (count > 0) digest.update(buffer, 0, count)
                count = input.read(buffer)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    override suspend fun transcribe(pcm16: ShortArray, sampleRate: Int): AsrResult =
        error("Whisper QNN graph invocation is the next integration step; verified encoder/decoder context assets are packaged")

    override fun close() = Unit
}
