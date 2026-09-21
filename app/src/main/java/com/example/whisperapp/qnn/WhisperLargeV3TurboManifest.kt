package com.example.whisperapp.qnn

import org.json.JSONArray
import org.json.JSONObject

data class WhisperLargeV3TurboTensorSpec(
    val shape: List<Long>,
    val dtype: String,
)

data class WhisperLargeV3TurboManifest(
    val model: String,
    val runtime: String,
    val precision: String,
    val qairtVersion: String,
    val chipsetAliases: List<String>,
    val chipsetMarketingName: String,
    val htpVersion: Int,
    val socModel: Int,
    val supportsFp16: Boolean,
    val referenceDevice: String,
    val encoderInputs: Map<String, WhisperLargeV3TurboTensorSpec>,
    val encoderOutputs: Map<String, WhisperLargeV3TurboTensorSpec>,
    val decoderInputs: Map<String, WhisperLargeV3TurboTensorSpec>,
    val decoderOutputs: Map<String, WhisperLargeV3TurboTensorSpec>,
) {
    val maxDecodeLength: Int
        get() = decoderInputs["attention_mask"]?.shape?.lastOrNull()?.toInt()
            ?: error("metadata.json does not expose decoder attention_mask length")

    fun toReport(assetNames: List<String>): String = buildString {
        appendLine("TURBO CONTRACT: PASS")
        appendLine("model: $model")
        appendLine("asset_dir: ${QnnWhisperLargeV3TurboContractRunner.ASSET_DIR}")
        appendLine("runtime: $runtime")
        appendLine("precision: $precision")
        appendLine("qairt: $qairtVersion")
        appendLine("chipset: $chipsetMarketingName")
        appendLine("chipset_aliases: ${chipsetAliases.joinToString(", ")}")
        appendLine("htp_version: $htpVersion")
        appendLine("soc_model: $socModel")
        appendLine("supports_fp16: $supportsFp16")
        appendLine("reference_device: $referenceDevice")
        appendLine("max_decode_length: $maxDecodeLength")
        appendLine()
        appendLine("=== ASSETS ===")
        assetNames.forEach(::appendLine)
        appendLine()
        appendLine("=== ENCODER ===")
        appendTensors("input", encoderInputs)
        appendTensors("output", encoderOutputs)
        appendLine()
        appendLine("=== DECODER ===")
        appendTensors("input", decoderInputs)
        appendTensors("output", decoderOutputs)
    }

    private fun StringBuilder.appendTensors(
        direction: String,
        tensors: Map<String, WhisperLargeV3TurboTensorSpec>,
    ) {
        tensors.toSortedMap().forEach { (name, spec) ->
            appendLine("$direction.$name dtype=${spec.dtype} shape=${spec.shape}")
        }
    }

    companion object {
        fun fromMetadata(root: JSONObject): WhisperLargeV3TurboManifest {
            require(root.optString("model_id") == "whisper_large_v3_turbo") {
                "Unexpected model_id: ${root.optString("model_id")}" }
            require(root.optString("runtime") == "qnn_context_binary") {
                "Unexpected runtime: ${root.optString("runtime")}" }
            require(root.optString("precision") == "float") {
                "Unexpected precision: ${root.optString("precision")}" }

            val files = root.getJSONObject("model_files")
            val enc = files.getJSONObject("encoder.bin")
            val dec = files.getJSONObject("decoder.bin")
            val chipset = root.getJSONObject("chipset_attributes")
            val versions = root.getJSONObject("tool_versions")

            return WhisperLargeV3TurboManifest(
                model = root.getString("model_name"),
                runtime = root.getString("runtime"),
                precision = root.getString("precision"),
                qairtVersion = versions.getString("qairt"),
                chipsetAliases = chipset.getJSONArray("aliases").strings(),
                chipsetMarketingName = chipset.getString("marketing_name"),
                htpVersion = chipset.getInt("htp_version"),
                socModel = chipset.getInt("soc_model"),
                supportsFp16 = chipset.getBoolean("supports_fp16"),
                referenceDevice = chipset.getString("reference_device"),
                encoderInputs = tensors(enc.getJSONObject("inputs")),
                encoderOutputs = tensors(enc.getJSONObject("outputs")),
                decoderInputs = tensors(dec.getJSONObject("inputs")),
                decoderOutputs = tensors(dec.getJSONObject("outputs")),
            ).also { it.validate() }
        }

        private fun WhisperLargeV3TurboManifest.validate() {
            require(encoderInputs.containsKey("input_features")) { "Missing encoder input_features" }
            require(decoderInputs.containsKey("input_ids")) { "Missing decoder input_ids" }
            require(decoderInputs.containsKey("position_ids")) { "Missing decoder position_ids" }
            require(decoderInputs.containsKey("attention_mask")) { "Missing decoder attention_mask" }
            require(decoderOutputs.containsKey("logits")) { "Missing decoder logits" }
            require(decoderOutputs.keys.count { it.startsWith("k_cache_self_") } == 4) {
                "Expected 4 decoder self-K outputs from metadata"
            }
            require(decoderOutputs.keys.count { it.startsWith("v_cache_self_") } == 4) {
                "Expected 4 decoder self-V outputs from metadata"
            }
        }

        private fun tensors(obj: JSONObject): Map<String, WhisperLargeV3TurboTensorSpec> =
            obj.keys().asSequence().associateWith { name ->
                val tensor = obj.getJSONObject(name)
                WhisperLargeV3TurboTensorSpec(
                    shape = tensor.getJSONArray("shape").longs(),
                    dtype = tensor.getString("dtype"),
                )
            }

        private fun JSONArray.strings(): List<String> =
            (0 until length()).map { getString(it) }

        private fun JSONArray.longs(): List<Long> =
            (0 until length()).map { getLong(it) }
    }
}