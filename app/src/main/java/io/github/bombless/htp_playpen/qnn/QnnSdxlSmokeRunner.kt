package io.github.bombless.htp_playpen.qnn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

internal object QnnSdxlSmokeRunner {
    private const val TAG = "SDXL_QNN_SMOKE"

    fun prepare(context: Context): File {
        val directory = File(context.getExternalFilesDir(null), "sdxl")
        for (name in listOf("unet.bin", "unet_ctx.onnx", "tokenizer.json", "vae.onnx",
            "text_encoder/model.onnx", "text_encoder/model.onnx_data",
            "text_encoder_2/model.onnx", "text_encoder_2/model.onnx_data")) {
            val file = File(directory, name)
            check(file.parentFile?.let { it.isDirectory || it.mkdirs() } == true) { "Cannot create parent of $file" }
            if (!file.exists()) check(file.createNewFile()) { "Cannot create $file" }
        }
        return directory
    }

    fun runText(context: Context): String {
        return try {
            val directory = File(context.getExternalFilesDir(null), "sdxl")
            val tokenizer = SdxlTokenizer.fromFile(File(directory, "tokenizer.json"))
            val ids = tokenizer.encode("Hello world! café 123", 49407)
            check(ids.copyOfRange(0, 9).contentEquals(intArrayOf(49406, 3306, 1002, 256, 15304, 272, 273, 274, 49407))) {
                "Tokenizer mismatch: ${ids.copyOfRange(0, 9).contentToString()}"
            }
            val env = OrtEnvironment.getEnvironment()
            val reports = listOf("text_encoder", "text_encoder_2").mapIndexed { index, name ->
                val model = File(directory, "$name/model.onnx")
                check(model.length() > 0 && File(directory, "$name/model.onnx_data").length() > 0) { "Missing $name" }
                OrtSession.SessionOptions().use { options ->
                    options.setIntraOpNumThreads(4)
                    val started = System.nanoTime()
                    env.createSession(model.absolutePath, options).use { session ->
                        val tokenIds = tokenizer.encode("a red apple", if (index == 0) 49407 else 0)
                        val input = if (index == 0) {
                            OnnxTensor.createTensor(env, IntBuffer.wrap(tokenIds), longArrayOf(1, 77))
                        } else {
                            OnnxTensor.createTensor(env, LongBuffer.wrap(tokenIds.map(Int::toLong).toLongArray()), longArrayOf(1, 77))
                        }
                        input.use {
                            val outputNames = if (index == 0) setOf("hidden_states.11") else setOf("hidden_states.31", "text_embeds")
                            session.run(mapOf("input_ids" to input), outputNames).use { result ->
                                for (output in outputNames) {
                                    val tensor = result.get(output).orElseThrow() as OnnxTensor
                                    val buffer = tensor.floatBuffer
                                    check(buffer.hasRemaining()) { "Empty $name/$output" }
                                    while (buffer.hasRemaining()) check(buffer.get().isFinite()) { "Invalid $name/$output" }
                                }
                            }
                        }
                    }
                    "$name=${(System.nanoTime() - started) / 1_000_000}ms"
                }
            }
            "PASS: tokenizer and CPU CLIP ${reports.joinToString()}"
        } catch (failure: Throwable) {
            Log.e(TAG, "SDXL text smoke test failed", failure)
            "FAIL: ${failure.javaClass.name}: ${failure.message}"
        }
    }

    fun run(context: Context): String {
        val tensors = mutableListOf<OnnxTensor>()
        return try {
            val directory = File(context.getExternalFilesDir(null), "sdxl")
            check(directory.isDirectory || directory.mkdirs()) { "Cannot create $directory" }
            val model = File(directory, "unet_ctx.onnx")
            val binary = File(directory, "unet.bin")
            if (!model.exists()) model.createNewFile()
            if (!binary.exists()) binary.createNewFile()
            check(model.length() > 0 && binary.length() > 0) { "Missing UNet assets in $directory" }
            val env = OrtEnvironment.getEnvironment()
            QnnEpRegistration.ensureRegistered(env)
            val device = env.epDevices.first { it.epName == QnnEpRegistration.EP_NAME }
            val backend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(backend.isFile) { "Missing QNN HTP backend: $backend" }
            val options = OrtSession.SessionOptions()
            try {
                options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                options.addExecutionProvider(listOf(device), mapOf(
                    "backend_path" to backend.absolutePath,
                    "soc_model" to "57",
                    "htp_arch" to "75",
                    "htp_performance_mode" to "burst",
                ))
                env.createSession(model.absolutePath, options).use { session ->
                    fun floats(shape: LongArray): OnnxTensor {
                        val count = shape.fold(1L) { product, dimension -> product * dimension }.toInt()
                        return OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(count)), shape)
                            .also { tensors.add(it) }
                    }
                    val inputs = mapOf(
                        "sample" to floats(longArrayOf(1, 4, 128, 128)),
                        "encoder_hidden_states" to floats(longArrayOf(1, 77, 2048)),
                        "timestamp" to OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(999)), longArrayOf(1))
                            .also { tensors.add(it) },
                        "time_ids" to floats(longArrayOf(1, 6)),
                        "text_embeds" to floats(longArrayOf(1, 1280)),
                    )
                    val started = System.nanoTime()
                    session.run(inputs).use { result ->
                        val values = (result[0] as OnnxTensor).floatBuffer
                        var count = 0
                        var nonzero = 0
                        while (values.hasRemaining()) {
                            val value = values.get()
                            check(value.isFinite()) { "Non-finite output at $count" }
                            if (value != 0f) nonzero++
                            count++
                        }
                        check(count == 4 * 128 * 128 && nonzero > 0) { "Unexpected output count=$count nonzero=$nonzero" }
                        "PASS: QNN UNet output=$count nonzero=$nonzero duration_ms=${(System.nanoTime() - started) / 1_000_000}"
                    }
                }
            } finally {
                options.close()
            }
        } catch (failure: Throwable) {
            Log.e(TAG, "SDXL QNN smoke test failed", failure)
            "FAIL: ${failure.javaClass.name}: ${failure.message}"
        } finally {
            tensors.asReversed().forEach { runCatching { it.close() } }
        }
    }
}
