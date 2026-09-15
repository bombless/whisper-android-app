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

object QnnWhisperStepRunner {
    private const val TAG = "QNN_WHISPER_STEP"
    private const val EP_NAME = "QNNExecutionProvider"
    private const val BOS_TOKEN = 50258
    private val crossNames = (0 until 4).flatMap { listOf("k_cache_cross_$it", "v_cache_cross_$it") }
    private val selfOutNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_out", "v_cache_self_${it}_out") }

    data class Result(val passed: Boolean, val report: String)

    fun run(context: Context): Result {
        val encoderModel = copyAsset(context, "models/whisper/encoder_ctx.onnx")
        val encoderBinary = copyAsset(context, "models/whisper/encoder.bin")
        val decoderModel = copyAsset(context, "models/whisper/decoder_ctx.onnx")
        val decoderBinary = copyAsset(context, "models/whisper/decoder.bin")
        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, TAG)
        var encoderSession: OrtSession? = null
        var decoderSession: OrtSession? = null
        var encoderOptions: OrtSession.SessionOptions? = null
        var decoderOptions: OrtSession.SessionOptions? = null
        val tensors = mutableListOf<OnnxTensor>()
        return try {
            System.loadLibrary("onnxruntime_providers_qnn")
            env.registerExecutionProviderLibrary(EP_NAME, "libonnxruntime_providers_qnn.so")
            val device = env.epDevices.firstOrNull { it.epName == EP_NAME } ?: error("QNN EP device not exposed")
            val backend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
            val epOptions = mapOf(
                "backend_path" to backend.absolutePath,
                "soc_model" to "57",
                "htp_arch" to "75",
                "offload_graph_io_quantization" to "0",
            )
            encoderOptions = sessionOptions(device, epOptions)
            decoderOptions = sessionOptions(device, epOptions)
            encoderSession = env.createSession(encoderModel.absolutePath, encoderOptions)
            decoderSession = env.createSession(decoderModel.absolutePath, decoderOptions)
            check(encoderSession.inputNames == setOf("input_features")) { "encoder inputs=${encoderSession.inputNames}" }
            check(encoderSession.outputNames.toSet() == crossNames.toSet()) { "encoder outputs=${encoderSession.outputNames}" }
            check(decoderSession.inputNames.size == 19) { "decoder inputs=${decoderSession.inputNames}" }
            check(decoderSession.outputNames.toSet() == (selfOutNames + "logits").toSet()) { "decoder outputs=${decoderSession.outputNames}" }

            val encoderInputInfo = encoderSession.inputInfo["input_features"]!!.info as TensorInfo
            check(encoderInputInfo.type == OnnxJavaType.FLOAT16 && encoderInputInfo.shape.contentEquals(longArrayOf(1, 80, 3000)))
            val decoderInfo = decoderSession.inputInfo
            assertInfo(decoderInfo, "input_ids", longArrayOf(1, 1), OnnxJavaType.INT32)
            assertInfo(decoderInfo, "position_ids", longArrayOf(1), OnnxJavaType.INT32)
            for (i in 0 until 4) {
                assertInfo(decoderInfo, "k_cache_self_${i}_in", longArrayOf(6, 1, 64, 199), OnnxJavaType.FLOAT16)
                assertInfo(decoderInfo, "v_cache_self_${i}_in", longArrayOf(6, 1, 199, 64), OnnxJavaType.FLOAT16)
                assertInfo(decoderInfo, "k_cache_cross_$i", longArrayOf(6, 1, 64, 1500), OnnxJavaType.FLOAT16)
                assertInfo(decoderInfo, "v_cache_cross_$i", longArrayOf(6, 1, 1500, 64), OnnxJavaType.FLOAT16)
            }
            assertInfo(decoderInfo, "attention_mask", longArrayOf(1, 1, 1, 200), OnnxJavaType.FLOAT16)

            val inputFeatures = f16(env, longArrayOf(1, 80, 3000), tensors)
            val encoderStart = System.nanoTime()
            val crossInputs = LinkedHashMap<String, OnnxTensor>()
            encoderSession.run(mapOf("input_features" to inputFeatures)).use { result ->
                check(result.size() == 8) { "encoder result size=${result.size()}" }
                for (name in crossNames) {
                    val index = encoderSession.outputNames.toList().indexOf(name)
                    check(index >= 0) { "missing encoder output $name" }
                    val out = result[index] as OnnxTensor
                    val info = encoderSession.outputInfo[name]!!.info as TensorInfo
                    val expected = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 1500) else longArrayOf(6, 1, 1500, 64)
                    check(info.type == OnnxJavaType.FLOAT16 && info.shape.contentEquals(expected)) { "$name contract mismatch" }
                    val stats = halfStats(out.getShortBuffer())
                    check(stats.nan == 0 && stats.inf == 0 && stats.nonZero > 0) { "$name invalid stats=$stats" }
                    crossInputs[name] = copyHalfTensor(env, out, expected, tensors)
                    Log.i(TAG, "ENCODER_CROSS $name shape=${expected.contentToString()} dtype=FP16 min=${stats.min} max=${stats.max} mean=${stats.mean} finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}")
                }
            }
            val encoderMs = (System.nanoTime() - encoderStart) / 1_000_000.0

            val decoderInputs = LinkedHashMap<String, OnnxTensor>()
            decoderInputs["input_ids"] = i32(env, longArrayOf(1, 1), BOS_TOKEN, tensors)
            decoderInputs["position_ids"] = i32(env, longArrayOf(1), 0, tensors)
            for (i in 0 until 4) {
                decoderInputs["k_cache_self_${i}_in"] = f16(env, longArrayOf(6, 1, 64, 199), tensors)
                decoderInputs["v_cache_self_${i}_in"] = f16(env, longArrayOf(6, 1, 199, 64), tensors)
            }
            decoderInputs["attention_mask"] = f16(env, longArrayOf(1, 1, 1, 200), tensors)
            decoderInputs.putAll(crossInputs)

            val decoderStart = System.nanoTime()
            var logitsStats: HalfStats? = null
            decoderSession.run(decoderInputs).use { result ->
                check(result.size() == 9) { "decoder result size=${result.size()}" }
                for (name in selfOutNames + "logits") {
                    val index = decoderSession.outputNames.toList().indexOf(name)
                    check(index >= 0) { "missing decoder output $name" }
                    val out = result[index] as OnnxTensor
                    val info = decoderSession.outputInfo[name]!!.info as TensorInfo
                    val expected = when {
                        name == "logits" -> longArrayOf(1, 51865, 1, 1)
                        name.startsWith("k_cache_self") -> longArrayOf(6, 1, 64, 199)
                        else -> longArrayOf(6, 1, 199, 64)
                    }
                    check(info.type == OnnxJavaType.FLOAT16 && info.shape.contentEquals(expected)) { "$name contract mismatch" }
                    val stats = halfStats(out.getShortBuffer())
                    check(stats.nan == 0 && stats.inf == 0) { "$name NaN=${stats.nan} Inf=${stats.inf}" }
                    if (name == "logits") logitsStats = stats
                    Log.i(TAG, "DECODER_OUT $name shape=${expected.contentToString()} dtype=FP16 min=${stats.min} max=${stats.max} mean=${stats.mean} finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}")
                }
            }
            val decoderMs = (System.nanoTime() - decoderStart) / 1_000_000.0
            val ls = logitsStats ?: error("logits stats missing")
            val report = buildString {
                appendLine("Encoder -> Cross-KV -> Decoder -> logits: PASS")
                appendLine("Encoder execution: SUCCESS")
                appendLine("Encoder execution ms: $encoderMs")
                appendLine("Encoder context binary bytes: ${encoderBinary.length()}")
                for (i in 0 until 4) {
                    appendLine("k_cache_cross_$i [6,1,64,1500] FP16 NaN=0 Inf=0")
                    appendLine("v_cache_cross_$i [6,1,1500,64] FP16 NaN=0 Inf=0")
                }
                appendLine("Decoder execution: SUCCESS")
                appendLine("Decoder execution ms: $decoderMs")
                appendLine("Decoder context binary bytes: ${decoderBinary.length()}")
                appendLine("input_ids: [$BOS_TOKEN]")
                appendLine("position_ids: [0]")
                appendLine("attention_mask: [1,1,1,200] FP16 deterministic zero")
                appendLine("self-KV inputs: 8 tensors, deterministic zero FP16")
                appendLine("self-KV outputs: 8 tensors, FP16, NaN=0, Inf=0")
                appendLine("logits: [1,51865,1,1] FP16 min=${ls.min} max=${ls.max} mean=${ls.mean} NaN=${ls.nan} Inf=${ls.inf}")
                appendLine("QNN EP: $EP_NAME")
                appendLine("Backend: HTP")
                appendLine("SoC: SM8650 / soc_model=57 / HTP v75")
                appendLine("CPU fallback: DISABLED")
                appendLine("HTP backend path: ${backend.absolutePath}")
            }
            Log.i(TAG, "Encoder -> Cross-KV -> Decoder -> logits: PASS")
            Result(true, report)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "Encoder -> Cross-KV -> Decoder -> logits: FAIL\n${t.javaClass.name}: ${t.message}")
        } finally {
            tensors.asReversed().forEach { try { it.close() } catch (_: Throwable) {} }
            try { encoderSession?.close() } catch (_: Throwable) {}
            try { decoderSession?.close() } catch (_: Throwable) {}
            try { encoderOptions?.close() } catch (_: Throwable) {}
            try { decoderOptions?.close() } catch (_: Throwable) {}
            try { env.unregisterExecutionProviderLibrary(EP_NAME) } catch (_: Throwable) {}
        }
    }

    private fun sessionOptions(device: OrtEpDevice, epOptions: Map<String, String>) = OrtSession.SessionOptions().apply {
        addConfigEntry("session.disable_cpu_ep_fallback", "1")
        setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
        addExecutionProvider(listOf(device as OrtEpDevice), epOptions)
    }

    private fun assertInfo(inputInfo: Map<String, ai.onnxruntime.NodeInfo>, name: String, shape: LongArray, type: OnnxJavaType) {
        val info = inputInfo[name]?.info as? TensorInfo ?: error("Missing input $name")
        check(info.type == type && info.shape.contentEquals(shape)) { "$name type=${info.type} shape=${info.shape.contentToString()}" }
    }

    private fun i32(env: OrtEnvironment, shape: LongArray, value: Int, owner: MutableList<OnnxTensor>): OnnxTensor =
        OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(shape.fold(1L) { a, b -> a * b }.toInt()) { value }), shape).also { owner += it }

    private fun f16(env: OrtEnvironment, shape: LongArray, owner: MutableList<OnnxTensor>): OnnxTensor =
        OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(shape.fold(1L) { a, b -> a * b }.toInt())), shape, OnnxJavaType.FLOAT16).also { owner += it }

    private fun copyHalfTensor(env: OrtEnvironment, source: OnnxTensor, shape: LongArray, owner: MutableList<OnnxTensor>): OnnxTensor {
        val src = source.getShortBuffer().duplicate()
        val copy = ShortArray(src.remaining())
        src.get(copy)
        return OnnxTensor.createTensor(env, ShortBuffer.wrap(copy), shape, OnnxJavaType.FLOAT16).also { owner += it }
    }

    private data class HalfStats(val min: Float, val max: Float, val mean: Double, val finite: Int, val nan: Int, val inf: Int, val nonZero: Int)

    private fun halfStats(buffer: ShortBuffer): HalfStats {
        val copy = buffer.duplicate()
        var min = Float.POSITIVE_INFINITY; var max = Float.NEGATIVE_INFINITY; var sum = 0.0
        var finite = 0; var nan = 0; var inf = 0; var nonZero = 0
        while (copy.hasRemaining()) {
            val value = halfToFloat(copy.get().toInt() and 0xffff)
            when {
                value.isNaN() -> nan++
                value.isInfinite() -> inf++
                else -> { finite++; if (value < min) min = value; if (value > max) max = value; sum += value.toDouble(); if (value != 0f) nonZero++ }
            }
        }
        return HalfStats(min, max, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf, nonZero)
    }

    private fun halfToFloat(bits: Int): Float {
        val sign = (bits ushr 15) and 1; val exp = (bits ushr 10) and 31; val frac = bits and 1023
        val v = when (exp) { 0 -> frac / 1024.0f * Math.pow(2.0, -14.0).toFloat(); 31 -> if (frac == 0) Float.POSITIVE_INFINITY else Float.NaN; else -> (1f + frac / 1024f) * Math.pow(2.0, exp - 15.0).toFloat() }
        return if (sign == 0) v else -v
    }

    private fun copyAsset(context: Context, asset: String): File {
        val f = File(context.cacheDir, asset.substringAfterLast('/'))
        context.assets.open(asset).use { input -> f.outputStream().use { input.copyTo(it) } }
        return f
    }
}
