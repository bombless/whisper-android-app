package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtEpDevice
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.IntBuffer
import java.nio.ShortBuffer

object QnnDecoderSmokeRunner {
    private const val TAG = "QNN_DECODER_SMOKE"
    private const val EP_NAME = "QNNExecutionProvider"
    private val expectedOutputs = (0 until 4).flatMap { listOf("k_cache_self_${it}_out", "v_cache_self_${it}_out") } + "logits"
    data class Result(val passed: Boolean, val report: String)

    fun run(context: Context): Result {
        val model = copyAsset(context, "models/whisper/decoder_ctx.onnx")
        val binary = copyAsset(context, "models/whisper/decoder.bin")
        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, TAG)
        var session: OrtSession? = null
        var options: OrtSession.SessionOptions? = null
        val tensors = mutableListOf<OnnxTensor>()
        return try {
            System.loadLibrary("onnxruntime_providers_qnn")
            env.registerExecutionProviderLibrary(EP_NAME, "libonnxruntime_providers_qnn.so")
            val device = env.epDevices.firstOrNull { it.epName == EP_NAME } ?: error("QNN EP device not exposed")
            val backend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
            options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.disable_cpu_ep_fallback", "1")
                setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                addExecutionProvider(listOf(device as OrtEpDevice), mapOf("backend_path" to backend.absolutePath, "soc_model" to "57", "htp_arch" to "75", "offload_graph_io_quantization" to "0"))
            }
            session = env.createSession(model.absolutePath, options)
            check(session.inputNames.size == 19) { "inputs=${session.inputNames}" }
            check(session.outputNames.toSet() == expectedOutputs.toSet()) { "outputs=${session.outputNames}" }
            fun i32(shape: LongArray, value: Int): OnnxTensor {
                val n = shape.fold(1L) { a, b -> a * b }.toInt()
                return OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(n) { value }), shape).also { tensors += it }
            }
            fun f16(shape: LongArray): OnnxTensor {
                val n = shape.fold(1L) { a, b -> a * b }.toInt()
                return OnnxTensor.createTensor(
                    env,
                    ShortBuffer.wrap(ShortArray(n)),
                    shape,
                    OnnxJavaType.FLOAT16,
                ).also { tensors += it }
            }
            val inputs = linkedMapOf<String, OnnxTensor>("input_ids" to i32(longArrayOf(1, 1), 50258), "position_ids" to i32(longArrayOf(1), 0))
            for (i in 0 until 4) {
                inputs["k_cache_self_${i}_in"] = f16(longArrayOf(6, 1, 64, 199))
                inputs["v_cache_self_${i}_in"] = f16(longArrayOf(6, 1, 199, 64))
            }
            inputs["attention_mask"] = f16(longArrayOf(1, 1, 1, 200))
            for (i in 0 until 4) {
                inputs["k_cache_cross_$i"] = f16(longArrayOf(6, 1, 64, 1500))
                inputs["v_cache_cross_$i"] = f16(longArrayOf(6, 1, 1500, 64))
            }
            val start = System.nanoTime()
            session.run(inputs).use { result ->
                check(result.size() == 9) { "returned outputs=${result.size()}" }
                for (index in session.outputNames.indices) {
                    val name = session.outputNames.elementAt(index)
                    val tensor = result[index] as? OnnxTensor ?: error("missing output $name")
                    val info = session.outputInfo[name]!!.info as TensorInfo
                    check(info.type == OnnxJavaType.FLOAT16) { "$name dtype=${info.type}" }
                    val expected = when { name == "logits" -> longArrayOf(1, 51865, 1, 1); name.startsWith("k_cache_self") -> longArrayOf(6, 1, 64, 199); else -> longArrayOf(6, 1, 199, 64) }
                    check(info.shape.contentEquals(expected)) { "$name shape=${info.shape.contentToString()}" }
                    val b = tensor.getShortBuffer().duplicate(); var nan = 0; var inf = 0
                    while (b.hasRemaining()) { val v = halfToFloat(b.get().toInt() and 0xffff); if (v.isNaN()) nan++ else if (v.isInfinite()) inf++ }
                    check(nan == 0 && inf == 0) { "$name NaN=$nan Inf=$inf" }
                    Log.i(TAG, "$name shape=${info.shape.contentToString()} finite=${b.position()}")
                }
            }
            val ms = (System.nanoTime() - start) / 1_000_000.0
            Result(true, "Decoder execution: SUCCESS\nQNN EP: $EP_NAME\nBackend: HTP\nSoC: SM8650 / soc_model=57 / HTP v75\nContext binary bytes: ${binary.length()}\nExecution ms: $ms\nInput IDs: [1,1] INT32\nKV/mask: deterministic zero FP16\nOutputs: 8 self-KV + logits [1,51865,1,1] FP16\nCPU fallback: DISABLED")
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "Decoder execution: FAIL\n${t.javaClass.name}: ${t.message}")
        } finally {
            tensors.forEach { try { it.close() } catch (_: Throwable) {} }
            try { session?.close() } catch (_: Throwable) {}
            try { options?.close() } catch (_: Throwable) {}
            try { env.unregisterExecutionProviderLibrary(EP_NAME) } catch (_: Throwable) {}
        }
    }

    private fun halfToFloat(bits: Int): Float {
        val sign = (bits ushr 15) and 1; val exp = (bits ushr 10) and 31; val frac = bits and 1023
        val v = when (exp) { 0 -> frac / 1024.0f * Math.pow(2.0, -14.0).toFloat(); 31 -> if (frac == 0) Float.POSITIVE_INFINITY else Float.NaN; else -> (1f + frac / 1024f) * Math.pow(2.0, exp - 15.0).toFloat() }
        return if (sign == 0) v else -v
    }
    private fun copyAsset(context: Context, asset: String): File { val f = File(context.cacheDir, asset.substringAfterLast('/')); context.assets.open(asset).use { input -> f.outputStream().use { input.copyTo(it) } }; return f }
}
