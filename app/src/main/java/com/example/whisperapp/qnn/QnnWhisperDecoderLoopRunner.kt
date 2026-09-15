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

object QnnWhisperDecoderLoopRunner {
    private const val TAG = "QNN_WHISPER_LOOP"
    private const val EP_NAME = "QNNExecutionProvider"
    private const val BOS_TOKEN = 50258
    private const val MAX_STEPS = 16
    private val crossNames = (0 until 4).flatMap { listOf("k_cache_cross_$it", "v_cache_cross_$it") }
    private val selfInNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_in", "v_cache_self_${it}_in") }
    private val selfOutNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_out", "v_cache_self_${it}_out") }

    data class Result(val passed: Boolean, val report: String)
    private data class HalfStats(val min: Float, val max: Float, val mean: Double, val finite: Int, val nan: Int, val inf: Int)

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
            val epOptions = mapOf("backend_path" to backend.absolutePath, "soc_model" to "57", "htp_arch" to "75", "offload_graph_io_quantization" to "0")
            encoderOptions = sessionOptions(device, epOptions)
            decoderOptions = sessionOptions(device, epOptions)
            encoderSession = env.createSession(encoderModel.absolutePath, encoderOptions)
            decoderSession = env.createSession(decoderModel.absolutePath, decoderOptions)
            check(encoderSession.inputNames == setOf("input_features")) { "encoder inputs=${encoderSession.inputNames}" }
            check(encoderSession.outputNames.toSet() == crossNames.toSet()) { "encoder outputs=${encoderSession.outputNames}" }
            check(decoderSession.inputNames.size == 19) { "decoder inputs=${decoderSession.inputNames}" }
            check(decoderSession.outputNames.toSet() == (selfOutNames + "logits").toSet()) { "decoder outputs=${decoderSession.outputNames}" }
            val encoderInfo = encoderSession.inputInfo["input_features"]!!.info as TensorInfo
            check(encoderInfo.type == OnnxJavaType.FLOAT16 && encoderInfo.shape.contentEquals(longArrayOf(1, 80, 3000)))
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
            val crossInputs = LinkedHashMap<String, OnnxTensor>()
            val encoderStart = System.nanoTime()
            encoderSession.run(mapOf("input_features" to inputFeatures)).use { result ->
                check(result.size() == 8)
                for (name in crossNames) {
                    val out = result[encoderSession.outputNames.toList().indexOf(name)] as OnnxTensor
                    val shape = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 1500) else longArrayOf(6, 1, 1500, 64)
                    val stats = halfStats(out.getShortBuffer())
                    check(stats.nan == 0 && stats.inf == 0) { "$name NaN=${stats.nan} Inf=${stats.inf}" }
                    crossInputs[name] = copyHalfTensor(env, out, shape, tensors)
                }
            }
            val encoderMs = (System.nanoTime() - encoderStart) / 1_000_000.0

            val selfKV = LinkedHashMap<String, OnnxTensor>()
            for (name in selfInNames) selfKV[name] = f16(env, selfShape(name), tensors)
            val attentionMask = f16(env, longArrayOf(1, 1, 1, 200), tensors)
            var currentToken = BOS_TOKEN
            var position = 0
            val steps = mutableListOf<String>()
            repeat(MAX_STEPS) { step ->
                val stepTensors = mutableListOf<OnnxTensor>()
                val inputs = LinkedHashMap<String, OnnxTensor>()
                inputs["input_ids"] = i32(env, longArrayOf(1, 1), currentToken, stepTensors)
                inputs["position_ids"] = i32(env, longArrayOf(1), position, stepTensors)
                inputs.putAll(selfKV)
                inputs["attention_mask"] = attentionMask
                inputs.putAll(crossInputs)
                val start = System.nanoTime()
                var logitsStats: HalfStats? = null
                var nextToken = -1
                val nextSelfKV = LinkedHashMap<String, OnnxTensor>()
                try {
                    decoderSession.run(inputs).use { result ->
                        check(result.size() == 9)
                        for (name in selfOutNames + "logits") {
                            val out = result[decoderSession.outputNames.toList().indexOf(name)] as OnnxTensor
                            val shape = when { name == "logits" -> longArrayOf(1, 51865, 1, 1); name.startsWith("k_") -> longArrayOf(6, 1, 64, 199); else -> longArrayOf(6, 1, 199, 64) }
                            val stats = halfStats(out.getShortBuffer())
                            check(stats.nan == 0 && stats.inf == 0 && stats.finite > 0) { "$name invalid finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}" }
                            if (name == "logits") {
                                logitsStats = stats
                                nextToken = argmaxHalf(out.getShortBuffer())
                            } else {
                                val inName = name.removeSuffix("_out") + "_in"
                                nextSelfKV[inName] = copyHalfTensor(env, out, shape, tensors)
                            }
                        }
                    }
                } finally {
                    stepTensors.forEach { try { it.close() } catch (_: Throwable) {} }
                }
                val elapsedMs = (System.nanoTime() - start) / 1_000_000.0
                val ls = logitsStats ?: error("step=$step logits missing")
                check(nextToken >= 0) { "step=$step argmax failed" }
                check(nextSelfKV.size == selfOutNames.size) { "step=$step selfKV outputs=${nextSelfKV.keys}" }
                var selfFinite = 0
                var selfNan = 0
                var selfInf = 0
                nextSelfKV.values.forEach { s -> val st = halfStats(s.getShortBuffer()); selfFinite += st.finite; selfNan += st.nan; selfInf += st.inf }
                check(selfNan == 0 && selfInf == 0) { "step=$step selfKV nan=$selfNan inf=$selfInf" }
                selfKV.clear()
                selfKV.putAll(nextSelfKV)
                val line = "STEP $step token=$currentToken position=$position latencyMs=$elapsedMs logitsFinite=${ls.finite} logitsMin=${ls.min} logitsMax=${ls.max} argmax=$nextToken selfKVFinite=$selfFinite selfKVNaN=$selfNan selfKVInf=$selfInf"
                steps += line
                Log.i(TAG, line)
                currentToken = nextToken
                position++
            }
            val report = buildString {
                appendLine("M3.4 autoregressive Decoder loop: PASS")
                appendLine("Encoder execution: SUCCESS ($encoderMs ms)")
                appendLine("Encoder context binary bytes: ${encoderBinary.length()}")
                appendLine("Decoder context binary bytes: ${decoderBinary.length()}")
                appendLine("Cross-KV: computed once and reused for all $MAX_STEPS Decoder steps")
                appendLine("Self-KV: initialized once to zeros, then previous outputs reused")
                appendLine("Attention mask: [1,1,1,200] FP16")
                appendLine("Position IDs: 0..${MAX_STEPS - 1}")
                appendLine("Initial token: $BOS_TOKEN")
                steps.forEach(::appendLine)
                appendLine("QNN EP: $EP_NAME")
                appendLine("Backend: HTP")
                appendLine("SoC: SM8650 / soc_model=57 / HTP v75")
                appendLine("CPU fallback: DISABLED")
                appendLine("HTP backend path: ${backend.absolutePath}")
            }
            Result(true, report)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "M3.4 autoregressive Decoder loop: FAIL\n${t.javaClass.name}: ${t.message}")
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
    private fun assertInfo(info: Map<String, ai.onnxruntime.NodeInfo>, name: String, shape: LongArray, type: OnnxJavaType) {
        val actual = info[name]?.info as? TensorInfo ?: error("Missing input $name")
        check(actual.type == type && actual.shape.contentEquals(shape)) { "$name type=${actual.type} shape=${actual.shape.contentToString()}" }
    }
    private fun selfShape(name: String) = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 199) else longArrayOf(6, 1, 199, 64)
    private fun i32(env: OrtEnvironment, shape: LongArray, value: Int, owner: MutableList<OnnxTensor>) = OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(shape.fold(1L) { a, b -> a * b }.toInt()) { value }), shape).also { owner += it }
    private fun f16(env: OrtEnvironment, shape: LongArray, owner: MutableList<OnnxTensor>) = OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(shape.fold(1L) { a, b -> a * b }.toInt())), shape, OnnxJavaType.FLOAT16).also { owner += it }
    private fun copyHalfTensor(env: OrtEnvironment, source: OnnxTensor, shape: LongArray, owner: MutableList<OnnxTensor>): OnnxTensor { val src = source.getShortBuffer().duplicate(); val copy = ShortArray(src.remaining()); src.get(copy); return OnnxTensor.createTensor(env, ShortBuffer.wrap(copy), shape, OnnxJavaType.FLOAT16).also { owner += it } }
    private fun halfStats(buffer: ShortBuffer): HalfStats { val c = buffer.duplicate(); var min = Float.POSITIVE_INFINITY; var max = Float.NEGATIVE_INFINITY; var sum = 0.0; var finite = 0; var nan = 0; var inf = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); when { v.isNaN() -> nan++; v.isInfinite() -> inf++; else -> { finite++; if (v < min) min = v; if (v > max) max = v; sum += v.toDouble() } } }; return HalfStats(min, max, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf) }
    private fun argmaxHalf(buffer: ShortBuffer): Int { val c = buffer.duplicate(); var best = Float.NEGATIVE_INFINITY; var bestIndex = -1; var i = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); if (v > best) { best = v; bestIndex = i }; i++ }; return bestIndex }
    private fun halfToFloat(bits: Int): Float { val sign = (bits ushr 15) and 1; val exp = (bits ushr 10) and 31; val frac = bits and 1023; val v = when (exp) { 0 -> frac / 1024.0f * Math.pow(2.0, -14.0).toFloat(); 31 -> if (frac == 0) Float.POSITIVE_INFINITY else Float.NaN; else -> (1f + frac / 1024f) * Math.pow(2.0, exp - 15.0).toFloat() }; return if (sign == 0) v else -v }
    private fun copyAsset(context: Context, asset: String): File { val f = File(context.cacheDir, asset.substringAfterLast('/')); context.assets.open(asset).use { input -> f.outputStream().use { input.copyTo(it) } }; return f }
}
