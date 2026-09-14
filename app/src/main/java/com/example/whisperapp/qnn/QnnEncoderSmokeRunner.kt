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
import java.nio.ShortBuffer
import kotlin.math.max

object QnnEncoderSmokeRunner {
    private const val TAG = "QNN_ENCODER_SMOKE"
    private const val EP_NAME = "QNNExecutionProvider"
    private val expectedOutputs = listOf(
        "k_cache_cross_0", "v_cache_cross_0",
        "k_cache_cross_1", "v_cache_cross_1",
        "k_cache_cross_2", "v_cache_cross_2",
        "k_cache_cross_3", "v_cache_cross_3",
    )

    data class Result(val passed: Boolean, val report: String)

    fun run(context: Context): Result {
        val model = copyAsset(context, "models/whisper/encoder_ctx.onnx")
        val contextBinary = copyAsset(context, "models/whisper/encoder.bin")
        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, TAG)
        var session: OrtSession? = null
        var options: OrtSession.SessionOptions? = null
        var input: OnnxTensor? = null
        return try {
            System.loadLibrary("onnxruntime_providers_qnn")
            env.registerExecutionProviderLibrary(EP_NAME, "libonnxruntime_providers_qnn.so")
            val device = env.epDevices.firstOrNull { it.epName == EP_NAME }
                ?: error("QNN EP device not exposed")
            val backend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }

            options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.disable_cpu_ep_fallback", "1")
                setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                addExecutionProvider(
                    listOf(device as OrtEpDevice),
                    mapOf(
                        "backend_path" to backend.absolutePath,
                        "soc_model" to "57",
                        "htp_arch" to "75",
                        "offload_graph_io_quantization" to "0",
                    ),
                )
            }

            Log.i(TAG, "ENCODER_ASSET size=${contextBinary.length()} path=${contextBinary.absolutePath}")
            Log.i(TAG, "ENCODER_MODEL path=${model.absolutePath}")
            Log.i(TAG, "QNN_EP=${device.epName} vendor=${device.epVendor} options=${device.epOptions}")
            Log.i(TAG, "QNN_BACKEND=${backend.absolutePath}")

            val sessionStart = System.nanoTime()
            session = env.createSession(model.absolutePath, options)
            val sessionMs = (System.nanoTime() - sessionStart) / 1_000_000.0
            check(session.inputNames == setOf("input_features")) { "inputs=${session.inputNames}" }
            check(session.outputNames == expectedOutputs.toSet()) { "outputs=${session.outputNames}" }

            val inputInfo = session.inputInfo["input_features"]!!.info as TensorInfo
            check(inputInfo.shape.contentEquals(longArrayOf(1, 80, 3000))) { "input shape=${inputInfo.shape.contentToString()}" }
            check(inputInfo.type == OnnxJavaType.FLOAT16) { "input dtype=${inputInfo.type}" }

            val outputInfo = session.outputInfo
            check(outputInfo.size == 8) { "output count=${outputInfo.size}" }
            for (i in 0 until 4) {
                assertTensorInfo(outputInfo, "k_cache_cross_$i", longArrayOf(6, 1, 64, 1500))
                assertTensorInfo(outputInfo, "v_cache_cross_$i", longArrayOf(6, 1, 1500, 64))
            }

            input = OnnxTensor.createTensor(
                env,
                ShortBuffer.wrap(ShortArray(80 * 3000)),
                longArrayOf(1, 80, 3000),
                OnnxJavaType.FLOAT16,
            )

            val runStart = System.nanoTime()
            session.run(mapOf("input_features" to input)).use { result ->
                check(result.size() == 8) { "returned outputs=${result.size()}" }
                val orderedNames = session.outputNames.toList()
                for (index in orderedNames.indices) {
                    val name = orderedNames[index]
                    val tensor = result[index] as OnnxTensor
                    val info = outputInfo[name]!!.info as TensorInfo
                    val expectedShape = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 1500) else longArrayOf(6, 1, 1500, 64)
                    check(info.type == OnnxJavaType.FLOAT16) { "$name dtype=${info.type}" }
                    check(info.shape.contentEquals(expectedShape)) { "$name shape=${info.shape.contentToString()}" }
                    val stats = halfStats(tensor.getShortBuffer())
                    check(stats.nan == 0) { "$name NaN=${stats.nan}" }
                    check(stats.inf == 0) { "$name Inf=${stats.inf}" }
                    check(stats.nonZero > 0) { "$name all zero" }
                    Log.i(TAG, "$name shape=${info.shape.contentToString()} dtype=FP16 min=${stats.min} max=${stats.max} mean=${stats.mean} finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}")
                }
            }
            val runMs = (System.nanoTime() - runStart) / 1_000_000.0
            val report = buildString {
                appendLine("Encoder execution: SUCCESS")
                appendLine("QNN EP: $EP_NAME")
                appendLine("Backend: HTP")
                appendLine("SoC: SM8650 / soc_model=57 / HTP v75")
                appendLine("Context binary bytes: ${contextBinary.length()}")
                appendLine("Session creation ms: $sessionMs")
                appendLine("Execution ms: $runMs")
                appendLine("Input: input_features [1,80,3000] FP16")
                for (i in 0 until 4) {
                    appendLine("k_cache_cross_$i [6,1,64,1500] FP16")
                    appendLine("v_cache_cross_$i [6,1,1500,64] FP16")
                }
                appendLine("NaN = 0")
                appendLine("Inf = 0")
            }
            Log.i(TAG, "Encoder execution: SUCCESS")
            Result(true, report)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "Encoder execution: FAIL\n${t.javaClass.name}: ${t.message}")
        } finally {
            try { input?.close() } catch (_: Throwable) {}
            try { session?.close() } catch (_: Throwable) {}
            try { options?.close() } catch (_: Throwable) {}
            try { env.unregisterExecutionProviderLibrary(EP_NAME) } catch (_: Throwable) {}
        }
    }

    private fun assertTensorInfo(outputInfo: Map<String, ai.onnxruntime.NodeInfo>, name: String, shape: LongArray) {
        val info = outputInfo[name]?.info as? TensorInfo ?: error("Missing output $name")
        check(info.shape.contentEquals(shape)) { "$name shape=${info.shape.contentToString()}" }
        check(info.type == OnnxJavaType.FLOAT16) { "$name dtype=${info.type}" }
    }

    private data class HalfStats(val min: Float, val max: Float, val mean: Double, val finite: Int, val nan: Int, val inf: Int, val nonZero: Int)

    private fun halfStats(buffer: ShortBuffer): HalfStats {
        val copy = buffer.duplicate()
        var min = Float.POSITIVE_INFINITY
        var max = Float.NEGATIVE_INFINITY
        var sum = 0.0
        var finite = 0
        var nan = 0
        var inf = 0
        var nonZero = 0
        while (copy.hasRemaining()) {
            val value = halfToFloat(copy.get().toInt() and 0xffff)
            when {
                value.isNaN() -> nan++
                value.isInfinite() -> inf++
                else -> { finite++; min = kotlin.math.min(min, value); max = max(max, value); sum += value.toDouble(); if (value != 0f) nonZero++ }
            }
        }
        return HalfStats(min, max, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf, nonZero)
    }

    private fun halfToFloat(bits: Int): Float {
        val sign = (bits ushr 15) and 1
        val exponent = (bits ushr 10) and 0x1f
        val fraction = bits and 0x3ff
        val value = when (exponent) {
            0 -> fraction / 1024.0f * Math.pow(2.0, -14.0).toFloat()
            31 -> if (fraction == 0) Float.POSITIVE_INFINITY else Float.NaN
            else -> (1.0f + fraction / 1024.0f) * Math.pow(2.0, exponent - 15.0).toFloat()
        }
        return if (sign == 0) value else -value
    }

    private fun copyAsset(context: Context, asset: String): File {
        val file = File(context.cacheDir, asset.substringAfterLast('/'))
        context.assets.open(asset).use { input -> file.outputStream().use { output -> input.copyTo(output) } }
        return file
    }
}

