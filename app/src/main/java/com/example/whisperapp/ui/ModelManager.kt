package com.example.whisperapp.ui

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.zip.ZipInputStream

enum class ManagedModel(val title: String) {
    LFM2("LFM2.5 230M"),
    WHISPER_LARGE_V3("Whisper Large v3"),
}

data class ModelFileStatus(
    val model: ManagedModel,
    val path: String,
    val bundled: Boolean,
    val localFile: File,
    val downloadUrl: String?,
    val expectedSize: Long? = null,
) {
    val exists: Boolean
        get() = bundled || (localFile.isFile && localFile.length() > 0)
}

data class ModelStatus(
    val model: ManagedModel,
    val files: List<ModelFileStatus>,
) {
    val ready: Boolean get() = files.all { it.exists }
    val existingFiles: Int get() = files.count { it.exists }
    val missingFiles: List<String> get() = files.filterNot { it.exists }.map { it.path }
}

object ModelManager {
    private const val RAW_BASE =
        "https://raw.atomgit.com/bombless/android-multimodal-onnx-htp/raw/master/assets/models"
    private const val LFM2_MODEL_DIR = "models/LFM2.5-230M-ONNX"
    private const val LFM2_MODEL_ROOT = "LFM2.5-230M-ONNX"
    private const val TURBO_ASSET_URL =
        "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/" +
            "whisper_large_v3_turbo/releases/v0.62.2/" +
            "whisper_large_v3_turbo-qnn_context_binary-float-qualcomm_snapdragon_8gen3.zip"
    private const val TOKENIZER_BASE =
        "https://huggingface.co/openai/whisper-large-v3-turbo/resolve/main/"

    private val lfm2Files = listOf(
        "onnx/model_q4.onnx",
        "onnx/model_q4f32.onnx_data",
    )
    private val whisperFiles = listOf(
        "encoder.bin",
        "decoder.bin",
        "encoder_ctx.onnx",
        "decoder_ctx.onnx",
        "metadata.json",
        "tokenizer/vocab.json",
        "tokenizer/tokenizer.json",
        "tokenizer/added_tokens.json",
        "tokenizer/merges.txt",
    )
    private val whisperBinarySizes = mapOf(
        "encoder.bin" to 1_755_942_912L,
        "decoder.bin" to 452_481_024L,
    )

    fun modelDir(context: Context, model: ManagedModel): File =
        if (model == ManagedModel.LFM2) {
            File(context.filesDir, LFM2_MODEL_DIR)
        } else {
            File(
                context.getExternalFilesDir(null) ?: context.filesDir,
                "whisper_large_v3_turbo",
            )
        }.apply { mkdirs() }

    fun status(context: Context, model: ManagedModel): ModelStatus {
        val files = if (model == ManagedModel.LFM2) {
            lfm2Files.map { path ->
                ModelFileStatus(
                    model = model,
                    path = path,
                    bundled = false,
                    localFile = File(modelDir(context, model), path),
                    downloadUrl = RAW_BASE + "/" + LFM2_MODEL_ROOT + "/" + path,
                )
            }
        } else {
            whisperFiles.map { path ->
                val bundled =
                    assetExists(context, "models/whisper_large_v3_turbo/" + path) &&
                        path !in whisperBinarySizes
                val url = when {
                    path in whisperBinarySizes -> TURBO_ASSET_URL
                    path.startsWith("tokenizer/") ->
                        TOKENIZER_BASE + path.removePrefix("tokenizer/")
                    else -> null
                }
                ModelFileStatus(
                    model = model,
                    path = path,
                    bundled = bundled,
                    localFile = File(modelDir(context, model), path),
                    downloadUrl = url,
                    expectedSize = whisperBinarySizes[path],
                )
            }
        }
        return ModelStatus(model, files)
    }

    suspend fun downloadModel(
        context: Context,
        model: ManagedModel,
        onProgress: (file: String, current: Long, total: Long) -> Unit = { _, _, _ -> },
    ) = withContext(Dispatchers.IO) {
        val current = status(context, model)
        if (model == ManagedModel.WHISPER_LARGE_V3) {
            downloadWhisper(context, current, onProgress)
        } else {
            current.files.filterNot { it.exists }.forEach { file ->
                download(context, file) { done, total ->
                    onProgress(file.path, done, total)
                }
            }
        }
    }

    suspend fun download(
        context: Context,
        file: ModelFileStatus,
        onProgress: (Long, Long) -> Unit = { _, _ -> },
    ): File = withContext(Dispatchers.IO) {
        val target = file.localFile
        target.parentFile?.mkdirs()
        val temp = File(target.parentFile, target.name + ".part")
        val url = file.downloadUrl ?: error("此文件暂无可用下载源：" + file.path)
        val existing = temp.length()
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            if (existing > 0) setRequestProperty("Range", "bytes=" + existing + "-")
        }
        try {
            check(connection.responseCode in 200..299) { "HTTP " + connection.responseCode }
            val responseLength = connection.contentLengthLong
            val resumed = connection.responseCode == HttpURLConnection.HTTP_PARTIAL
            val start = if (resumed) existing else 0L
            if (!resumed && existing > 0) temp.delete()
            connection.inputStream.use { input ->
                FileOutputStream(temp, start > 0).use { output ->
                    val buffer = ByteArray(1024 * 1024)
                    var done = start
                    val total = if (responseLength > 0) responseLength + start else file.expectedSize ?: -1L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        output.write(buffer, 0, read)
                        done += read
                        onProgress(done, total)
                    }
                }
            }
            check(temp.length() > 0) { "下载结果为空：" + file.path }
            if (file.expectedSize != null) {
                check(temp.length() == file.expectedSize) {
                    file.path + " 大小不正确：" + temp.length()
                }
            }
            if (target.exists()) target.delete()
            check(temp.renameTo(target)) { "无法保存文件：" + target.absolutePath }
            target
        } finally {
            connection.disconnect()
            if (temp.exists()) temp.delete()
        }
    }

    private suspend fun downloadWhisper(
        context: Context,
        status: ModelStatus,
        onProgress: (String, Long, Long) -> Unit,
    ) {
        val missing = status.files.filterNot { it.exists }
        val missingBinaries = missing.filter { it.path in whisperBinarySizes }
        if (missingBinaries.isNotEmpty()) {
            val dir = modelDir(context, ManagedModel.WHISPER_LARGE_V3)
            val connection = (URL(TURBO_ASSET_URL).openConnection() as HttpURLConnection).apply {
                connectTimeout = 30_000
                readTimeout = 120_000
            }
            try {
                check(connection.responseCode in 200..299) {
                    "HTTP " + connection.responseCode
                }
                val total = connection.contentLengthLong
                connection.inputStream.use { raw ->
                    ZipInputStream(raw.buffered()).use { zip ->
                        while (true) {
                            val entry = zip.nextEntry ?: break
                            val name = entry.name.substringAfterLast('/')
                            if (name in whisperBinarySizes && missingBinaries.any { it.path == name }) {
                                val target = File(dir, name)
                                val part = File(dir, name + ".part")
                                FileOutputStream(part).use { output ->
                                    val buffer = ByteArray(1024 * 1024)
                                    var written = 0L
                                    while (true) {
                                        val read = zip.read(buffer)
                                        if (read < 0) break
                                        output.write(buffer, 0, read)
                                        written += read
                                        onProgress(name, written, total)
                                    }
                                }
                                check(part.length() == whisperBinarySizes.getValue(name)) {
                                    name + " 大小不正确：" + part.length()
                                }
                                if (target.exists()) target.delete()
                                check(part.renameTo(target)) { "无法保存文件：" + target.absolutePath }
                            }
                            zip.closeEntry()
                        }
                    }
                }
            } finally {
                connection.disconnect()
            }
        }

        missing
            .filter { it.path !in whisperBinarySizes && it.downloadUrl != null }
            .forEach { file ->
                download(context, file) { done, total ->
                    onProgress(file.path, done, total)
                }
            }
    }

    private fun assetExists(context: Context, path: String): Boolean =
        runCatching { context.assets.open(path).use { true } }.getOrDefault(false)
}
