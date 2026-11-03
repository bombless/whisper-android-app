package io.github.bombless.htp_playpen.tts

import android.content.Context
import android.util.JsonReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlin.coroutines.coroutineContext

/** Downloads the exact, hash-pinned Kokoro bundle declared in the APK manifest. */
internal class KokoroModelRepository(private val context: Context) {
    data class Progress(
        val file: String,
        val bytesDownloaded: Long,
        val totalBytesDownloaded: Long,
        val totalBytes: Long,
        val completed: Int,
        val totalFiles: Int,
    )

    private data class ModelFile(val path: String, val size: Long, val sha256: String)
    private data class Manifest(
        val model: String,
        val revision: String,
        val license: String,
        val totalBytes: Long,
        val files: List<ModelFile>,
    )

    private val manifest: Manifest by lazy { readManifest() }

    suspend fun ensureModels(onProgress: (Progress) -> Unit): String = withContext(Dispatchers.IO) {
        val m = manifest
        val directory = File(context.filesDir, "models_tts/kokoro-${m.revision}")
        check(directory.mkdirs() || directory.isDirectory) { "Cannot create Kokoro model directory: $directory" }
        val marker = File(directory, ".verified-manifest-sha256")
        val manifestHash = sha256(context.assets.open(MANIFEST_ASSET))
        val markerMatches = marker.isFile && marker.readText().trim() == manifestHash
        val allPresent = m.files.all { spec ->
            val file = safeChild(directory, spec.path)
            file.isFile && file.length() == spec.size
        }
        if (markerMatches && allPresent) return@withContext directory.absolutePath

        var completedBytes = 0L
        var completedFiles = 0
        for (spec in m.files) {
            coroutineContext.ensureActive()
            val target = safeChild(directory, spec.path)
            if (target.isFile && target.length() == spec.size && sha256(target) == spec.sha256) {
                completedBytes += spec.size
                completedFiles++
                onProgress(Progress(spec.path, spec.size, completedBytes, m.totalBytes, completedFiles, m.files.size))
                continue
            }

            target.parentFile?.let { parent ->
                check(parent.mkdirs() || parent.isDirectory) { "Cannot create model directory: $parent" }
            }
            val partial = File(target.parentFile, target.name + ".part")
            var lastFailure: Throwable? = null
            var verified = false
            if (partial.isFile && partial.length() == spec.size) {
                if (sha256(partial) == spec.sha256) {
                    if (target.exists()) check(target.delete()) { "Cannot replace invalid model file: $target" }
                    check(partial.renameTo(target)) { "Cannot finalize downloaded model file: $target" }
                    verified = true
                } else {
                    partial.delete()
                }
            }
            for (attempt in 1..4) {
                if (verified) break
                coroutineContext.ensureActive()
                try {
                    downloadFile(spec, partial, completedBytes, m, completedFiles, onProgress)
                    check(partial.length() == spec.size) {
                        "Kokoro asset size mismatch for ${spec.path}: ${partial.length()} != ${spec.size}"
                    }
                    val actualHash = sha256(partial)
                    check(actualHash == spec.sha256) {
                        "Kokoro SHA-256 mismatch for ${spec.path}: expected ${spec.sha256}, got $actualHash"
                    }
                    if (target.exists()) check(target.delete()) { "Cannot replace invalid model file: $target" }
                    check(partial.renameTo(target)) { "Cannot finalize downloaded model file: $target" }
                    verified = true
                    lastFailure = null
                    break
                } catch (failure: Throwable) {
                    lastFailure = failure
                    if (attempt == 4) break
                    // A hash mismatch is not recoverable by resuming the same bytes.
                    if (failure.message?.contains("SHA-256 mismatch") == true ||
                        failure.message?.contains("size mismatch") == true
                    ) partial.delete()
                    delay(1_000L * attempt)
                }
            }
            if (!verified) throw IllegalStateException("Failed to install Kokoro asset ${spec.path}", lastFailure)

            completedBytes += spec.size
            completedFiles++
            onProgress(Progress(spec.path, spec.size, completedBytes, m.totalBytes, completedFiles, m.files.size))
        }

        marker.writeText(manifestHash)
        onProgress(Progress("模型校验完成", 0L, m.totalBytes, m.totalBytes, m.files.size, m.files.size))
        directory.absolutePath
    }

    private suspend fun downloadFile(
        spec: ModelFile,
        partial: File,
        completedBytes: Long,
        manifest: Manifest,
        completedFiles: Int,
        onProgress: (Progress) -> Unit,
    ) {
        if (partial.isFile && partial.length() > spec.size) partial.delete()
        val resumeAt = partial.takeIf { it.isFile }?.length() ?: 0L
        val url = "https://huggingface.co/${manifest.model}/resolve/${manifest.revision}/${spec.path}"
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 30_000
            readTimeout = 90_000
            instanceFollowRedirects = true
            requestMethod = "GET"
            setRequestProperty("User-Agent", "whisper-app-kokoro/0.2")
            if (resumeAt > 0L) setRequestProperty("Range", "bytes=$resumeAt-")
        }
        try {
            val status = connection.responseCode
            check(status == HttpURLConnection.HTTP_OK || status == HttpURLConnection.HTTP_PARTIAL) {
                "Kokoro download HTTP $status for ${spec.path}"
            }
            val append = resumeAt > 0L && status == HttpURLConnection.HTTP_PARTIAL
            val startAt = if (append) resumeAt else 0L
            var fileBytes = startAt
            connection.inputStream.use { input ->
                FileOutputStream(partial, append).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE * 4)
                    while (true) {
                        coroutineContext.ensureActive()
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        fileBytes += count
                        onProgress(
                            Progress(
                                spec.path,
                                fileBytes,
                                completedBytes + fileBytes,
                                manifest.totalBytes,
                                completedFiles,
                                manifest.files.size,
                            ),
                        )
                    }
                    output.fd.sync()
                }
            }
        } finally {
            connection.disconnect()
        }
    }

    private fun readManifest(): Manifest {
        context.assets.open(MANIFEST_ASSET).use { stream ->
            JsonReader(InputStreamReader(stream, Charsets.UTF_8)).use { reader ->
                var model = ""
                var revision = ""
                var license = ""
                var totalBytes = 0L
                val files = mutableListOf<ModelFile>()
                reader.beginObject()
                while (reader.hasNext()) {
                    when (reader.nextName()) {
                        "model" -> model = reader.nextString()
                        "revision" -> revision = reader.nextString()
                        "license" -> license = reader.nextString()
                        "totalBytes" -> totalBytes = reader.nextLong()
                        "files" -> {
                            reader.beginArray()
                            while (reader.hasNext()) {
                                var path = ""
                                var size = 0L
                                var hash = ""
                                reader.beginObject()
                                while (reader.hasNext()) {
                                    when (reader.nextName()) {
                                        "path" -> path = reader.nextString()
                                        "size" -> size = reader.nextLong()
                                        "sha256" -> hash = reader.nextString()
                                        else -> reader.skipValue()
                                    }
                                }
                                reader.endObject()
                                require(path.isNotBlank() && !path.startsWith('/') && !path.contains(".."))
                                require(size > 0 && hash.matches(Regex("[a-f0-9]{64}")))
                                files += ModelFile(path, size, hash)
                            }
                            reader.endArray()
                        }
                        else -> reader.skipValue()
                    }
                }
                reader.endObject()
                require(model == MODEL_ID && revision == MODEL_REVISION) { "Unexpected Kokoro model identity or revision" }
                require(license == "Apache-2.0") { "Unexpected Kokoro model license: $license" }
                require(files.isNotEmpty() && files.sumOf { it.size } == totalBytes) { "Invalid Kokoro manifest byte total" }
                return Manifest(model, revision, license, totalBytes, files)
            }
        }
    }

    private fun safeChild(root: File, relativePath: String): File {
        val child = File(root, relativePath).canonicalFile
        require(child.path.startsWith(root.canonicalPath + File.separator)) { "Invalid model path: $relativePath" }
        return child
    }

    private fun sha256(file: File): String = FileInputStream(file).use(::sha256)

    private fun sha256(input: java.io.InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        input.use { stream ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val MANIFEST_ASSET = "models/tts/kokoro/manifest.json"
        private const val MODEL_ID = "soniqo/Kokoro-82M-ONNX"
        private const val MODEL_REVISION = "b561586381f1b212a8d6bd2a46483e04f54f4ebf"
    }
}