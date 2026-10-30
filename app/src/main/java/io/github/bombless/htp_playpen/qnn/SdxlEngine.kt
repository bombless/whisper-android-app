package io.github.bombless.htp_playpen.qnn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.roundToInt

internal object SdxlEngine {
    private const val LATENT_SIZE = 4 * 128 * 128
    private const val PIXELS = 1024 * 1024

    data class Conditioning(val hidden: FloatArray, val pooled: FloatArray)

    fun generate(
        context: Context,
        prompt: String,
        negativePrompt: String,
        steps: Int,
        seed: Long,
        progress: (String) -> Unit,
    ): Bitmap {
        val directory = File(context.getExternalFilesDir(null), "sdxl")
        val required = listOf("unet.bin", "unet_ctx.onnx", "tokenizer.json", "vae.onnx",
            "text_encoder/model.onnx", "text_encoder/model.onnx_data",
            "text_encoder_2/model.onnx", "text_encoder_2/model.onnx_data")
        val missing = required.filter { File(directory, it).let { file -> !file.isFile || file.length() == 0L } }
        check(missing.isEmpty()) { "缺少 SDXL 模型文件: ${missing.joinToString()}；目录: $directory" }
        val env = OrtEnvironment.getEnvironment()
        progress("正在编码提示词…")
        val tokenizer = SdxlTokenizer.fromFile(File(directory, "tokenizer.json"))
        val conditioning = encodePrompts(env, directory, tokenizer, listOf(negativePrompt, prompt), progress)
        val scheduler = SdxlEulerScheduler(steps)
        val random = Random(seed)
        val latents = FloatArray(LATENT_SIZE) { random.nextGaussian().toFloat() * scheduler.initialNoiseSigma }
        progress("正在加载 NPU 模型…")
        denoise(context, env, directory, scheduler, conditioning, latents, progress)
        progress("正在解码图像…")
        return decode(env, directory, latents)
    }

    private fun encodePrompts(
        env: OrtEnvironment,
        directory: File,
        tokenizer: SdxlTokenizer,
        prompts: List<String>,
        progress: (String) -> Unit,
    ): List<Conditioning> {
        val hiddenParts = Array(2) { arrayOfNulls<FloatArray>(2) }
        val pooled = arrayOfNulls<FloatArray>(2)
        for (encoder in 0..1) {
            progress("正在编码提示词 (${encoder + 1}/2)…")
            val name = if (encoder == 0) "text_encoder" else "text_encoder_2"
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(4)
                env.createSession(File(directory, "$name/model.onnx").absolutePath, options).use { session ->
                    for (index in prompts.indices) {
                        val ids = tokenizer.encode(prompts[index], if (encoder == 0) 49407 else 0)
                        val tokenTensor = if (encoder == 0) {
                            OnnxTensor.createTensor(env, IntBuffer.wrap(ids), longArrayOf(1, 77))
                        } else {
                            OnnxTensor.createTensor(env, LongBuffer.wrap(ids.map(Int::toLong).toLongArray()), longArrayOf(1, 77))
                        }
                        tokenTensor.use { tensor ->
                            val hiddenName = if (encoder == 0) "hidden_states.11" else "hidden_states.31"
                            val outputNames = if (encoder == 0) setOf(hiddenName) else setOf(hiddenName, "text_embeds")
                            session.run(mapOf("input_ids" to tensor), outputNames).use { result ->
                                hiddenParts[encoder][index] = readFloats(result.get(hiddenName).orElseThrow() as OnnxTensor)
                                if (encoder == 1) pooled[index] = readFloats(result.get("text_embeds").orElseThrow() as OnnxTensor)
                            }
                        }
                        progress("正在编码提示词 (${encoder + 1}/2)…")
                    }
                }
            }
        }
        return prompts.indices.map { index ->
            val first = checkNotNull(hiddenParts[0][index])
            val second = checkNotNull(hiddenParts[1][index])
            val pooledOutput = checkNotNull(pooled[index])
            check(first.size == 77 * 768 && second.size == 77 * 1280 && pooledOutput.size == 1280)
            val hidden = FloatArray(77 * 2048)
            for (token in 0 until 77) {
                first.copyInto(hidden, token * 2048, token * 768, (token + 1) * 768)
                second.copyInto(hidden, token * 2048 + 768, token * 1280, (token + 1) * 1280)
            }
            Conditioning(hidden, pooledOutput)
        }
    }

    private fun denoise(
        context: Context,
        env: OrtEnvironment,
        directory: File,
        scheduler: SdxlEulerScheduler,
        conditioning: List<Conditioning>,
        latents: FloatArray,
        progress: (String) -> Unit,
    ) {
        QnnEpRegistration.ensureRegistered(env)
        val device = env.epDevices.first { it.epName == QnnEpRegistration.EP_NAME }
        val backend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
        check(backend.isFile) { "缺少 QNN HTP 后端: $backend" }
        OrtSession.SessionOptions().use { options ->
            options.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            options.addExecutionProvider(listOf(device), mapOf(
                "backend_path" to backend.absolutePath,
                "soc_model" to "57",
                "htp_arch" to "75",
                "htp_performance_mode" to "burst",
            ))
            env.createSession(File(directory, "unet_ctx.onnx").absolutePath, options).use { session ->
                val sampleBuffer = directFloats(LATENT_SIZE)
                val timeBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder()).asIntBuffer()
                val tensors = mutableListOf<OnnxTensor>()
                try {
                    val sample = OnnxTensor.createTensor(env, sampleBuffer, longArrayOf(1, 4, 128, 128)).also { tensors.add(it) }
                    val timestamp = OnnxTensor.createTensor(env, timeBuffer, longArrayOf(1)).also { tensors.add(it) }
                    val timeIds = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(1024f, 1024f, 0f, 0f, 1024f, 1024f)), longArrayOf(1, 6)).also { tensors.add(it) }
                    val hidden = conditioning.map { OnnxTensor.createTensor(env, FloatBuffer.wrap(it.hidden), longArrayOf(1, 77, 2048)).also { tensor -> tensors.add(tensor) } }
                    val pooled = conditioning.map { OnnxTensor.createTensor(env, FloatBuffer.wrap(it.pooled), longArrayOf(1, 1280)).also { tensor -> tensors.add(tensor) } }
                    val scaled = FloatArray(LATENT_SIZE)
                    for (step in 0 until scheduler.steps) {
                        progress("NPU 去噪中 (${step + 1}/${scheduler.steps})…")
                        scheduler.scaleInput(latents, step, scaled)
                        sampleBuffer.clear()
                        sampleBuffer.put(scaled)
                        sampleBuffer.rewind()
                        timeBuffer.put(0, scheduler.timesteps[step])
                        val predictions = (0..1).map { index ->
                            session.run(mapOf(
                                "sample" to sample,
                                "encoder_hidden_states" to hidden[index],
                                "timestamp" to timestamp,
                                "time_ids" to timeIds,
                                "text_embeds" to pooled[index],
                            )).use { result -> readFloats(result[0] as OnnxTensor) }
                        }
                        scheduler.advance(latents, predictions[0], predictions[1], step, 7f)
                    }
                } finally {
                    tensors.asReversed().forEach { it.close() }
                }
            }
        }
    }

    private fun decode(env: OrtEnvironment, directory: File, latents: FloatArray): Bitmap {
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(4)
            env.createSession(File(directory, "vae.onnx").absolutePath, options).use { session ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(latents), longArrayOf(1, 4, 128, 128)).use { input ->
                    session.run(mapOf("latent_sample" to input)).use { result ->
                        val channels = readFloats(result[0] as OnnxTensor)
                        check(channels.size == PIXELS * 3) { "VAE 输出大小异常: ${channels.size}" }
                        val pixels = IntArray(PIXELS) { index ->
                            fun channel(offset: Int): Int = ((channels[index + offset] * 0.5f + 0.5f).coerceIn(0f, 1f) * 255f).roundToInt()
                            (0xff shl 24) or (channel(0) shl 16) or (channel(PIXELS) shl 8) or channel(PIXELS * 2)
                        }
                        return Bitmap.createBitmap(pixels, 1024, 1024, Bitmap.Config.ARGB_8888)
                    }
                }
            }
        }
    }

    private fun directFloats(size: Int): FloatBuffer =
        ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    private fun readFloats(tensor: OnnxTensor): FloatArray {
        val buffer = tensor.floatBuffer
        return FloatArray(buffer.remaining()).also { values ->
            buffer.get(values)
            check(values.all { it.isFinite() }) { "模型输出包含非有限值" }
        }
    }
}
