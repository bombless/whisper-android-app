package com.example.whisperapp.qnn

import android.content.Context
import android.util.Log
import org.json.JSONObject

/**
 * Contract inspector for the Qualcomm QNN Context Binary asset.
 *
 * This deliberately does not open the .bin files with ONNX Runtime: the shipped
 * Qualcomm asset is QNN_CONTEXT_BINARY, not an ONNX model. The authoritative
 * I/O contract is read from the asset's metadata.json.
 */
object QnnWhisperLargeV3TurboContractRunner {
    private const val TAG = "TURBO_CONTRACT"
    const val ASSET_DIR = "models/whisper_large_v3_turbo"
    private const val METADATA = "$ASSET_DIR/metadata.json"

    data class Result(
        val passed: Boolean,
        val report: String,
        val manifest: WhisperLargeV3TurboManifest? = null,
    )

    fun run(context: Context): Result {
        return try {
            val names = context.assets.list(ASSET_DIR)?.toList().orEmpty().sorted()
            require("encoder.bin" in names) { "Missing encoder.bin in $ASSET_DIR: $names" }
            require("decoder.bin" in names) { "Missing decoder.bin in $ASSET_DIR: $names" }
            require("metadata.json" in names) { "Missing metadata.json in $ASSET_DIR: $names" }

            val json = context.assets.open(METADATA).bufferedReader().use { it.readText() }
            val manifest = WhisperLargeV3TurboManifest.fromMetadata(JSONObject(json))
            val report = manifest.toReport(names)
            Log.i(TAG, report)
            Result(true, report, manifest)
        } catch (t: Throwable) {
            Log.e(TAG, "Turbo contract inspection failed", t)
            Result(false, "TURBO CONTRACT: FAIL\n${t.javaClass.name}: ${t.message}")
        }
    }
}