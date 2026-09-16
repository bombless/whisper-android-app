package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.IntBuffer
import java.nio.ShortBuffer

object QnnWhisperDecoderLoopRunner {
    private const val TAG = "QNN_WHISPER_LOOP"
    private const val EP_NAME = "QNNExecutionProvider"
    private const val VALIDATED_BOS_TOKEN = 50258
    private const val MAX_STEPS = 16
    private val crossNames = (0 until 4).flatMap { listOf("k_cache_cross_$it", "v_cache_cross_$it") }
    private val selfInNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_in", "v_cache_self_${it}_in") }
    private val selfOutNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_out", "v_cache_self_${it}_out") }

    data class Result(val passed: Boolean, val report: String)
    private data class DecoderStepResult(val logits: OnnxTensor, val nextSelfKv: LinkedHashMap<String, OnnxTensor>)
    private data class HalfStats(val min: Float, val max: Float, val mean: Double, val finite: Int, val nan: Int, val inf: Int, val nonZero: Int)

    fun run(context: Context): Result {
        return try {
            Result(true, runLoop(context, MAX_STEPS))
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "M3.4 status: FAIL\n${t.javaClass.name}: ${t.message}")
        }
    }

    private fun runLoop(context: Context, requestedSteps: Int): String {
        require(requestedSteps in 1..MAX_STEPS)
        val encoderModel = copyAsset(context, "models/whisper/encoder_ctx.onnx")
        val encoderBinary = copyAsset(context, "models/whisper/encoder.bin")
        val decoderModel = copyAsset(context, "models/whisper/decoder_ctx.onnx")
        val decoderBinary = copyAsset(context, "models/whisper/decoder.bin")
        val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, TAG)
        val tensors = mutableListOf<OnnxTensor>()
        var encoderSession: OrtSession? = null
        var decoderSession: OrtSession? = null
        var encoderOptions: OrtSession.SessionOptions? = null
        var decoderOptions: OrtSession.SessionOptions? = null
        try {
            // Keep the validated QnnWhisperStepRunner initialization and session contract.
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
            fun makeOptions() = OrtSession.SessionOptions().apply {
                addConfigEntry("session.disable_cpu_ep_fallback", "1")
                setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                addExecutionProvider(listOf(device), epOptions)
            }
            encoderOptions = makeOptions()
            decoderOptions = makeOptions()
            encoderSession = env.createSession(encoderModel.absolutePath, encoderOptions)
            decoderSession = env.createSession(decoderModel.absolutePath, decoderOptions)

            check(encoderSession.inputNames == setOf("input_features")) { "encoder inputs=${encoderSession.inputNames}" }
            check(encoderSession.outputNames.toSet() == crossNames.toSet()) { "encoder outputs=${encoderSession.outputNames}" }
            check(decoderSession.inputNames.size == 19) { "decoder inputs=${decoderSession.inputNames}" }
            check(decoderSession.outputNames.toSet() == (selfOutNames + "logits").toSet()) { "decoder outputs=${decoderSession.outputNames}" }
            validateDecoderContract(decoderSession)

            val inputFeatures = f16(env, longArrayOf(1, 80, 3000), tensors)
            val crossInputs = LinkedHashMap<String, OnnxTensor>()
            val encoderStart = System.nanoTime()
            encoderSession.run(mapOf("input_features" to inputFeatures)).use { result ->
                check(result.size() == 8) { "encoder result size=${result.size()}" }
                for (name in crossNames) {
                    val index = encoderSession.outputNames.toList().indexOf(name)
                    check(index >= 0) { "missing encoder output $name" }
                    val out = result[index] as OnnxTensor
                    val shape = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 1500) else longArrayOf(6, 1, 1500, 64)
                    val stats = halfStats(out.getShortBuffer())
                    check(stats.nan == 0 && stats.inf == 0 && stats.nonZero > 0) { "$name invalid stats=$stats" }
                    crossInputs[name] = copyHalfTensor(env, out, shape, tensors)
                    Log.i(TAG, "ENCODER_CROSS $name shape=${shape.contentToString()} dtype=FP16 finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}")
                }
            }
            val encoderMs = (System.nanoTime() - encoderStart) / 1_000_000.0

            var selfKv = LinkedHashMap<String, OnnxTensor>()
            for (name in selfInNames) selfKv[name] = f16(env, selfShape(name), tensors)
            var currentToken = VALIDATED_BOS_TOKEN
            var position = 0
            var completedSteps = 0
            val stepLines = mutableListOf<String>()

            repeat(requestedSteps) { step ->
                val stepStart = System.nanoTime()
                val stepTensors = mutableListOf<OnnxTensor>()
                val attentionMask = causalAttentionMask(env, position, stepTensors)
                val stepResult = runDecoderStep(
                    env = env,
                    session = decoderSession,
                    step = step,
                    inputToken = currentToken,
                    position = position,
                    selfKvInputs = selfKv,
                    crossKvInputs = crossInputs,
                    attentionMask = attentionMask,
                    owner = stepTensors,
                )
                val logitsStats = halfStats(stepResult.logits.getShortBuffer())
                val nextToken = argmaxHalf(stepResult.logits.getShortBuffer())
                check(nextToken >= 0) { "step=$step argmax failed" }

                var selfFinite = 0
                var selfNan = 0
                var selfInf = 0
                stepResult.nextSelfKv.values.forEach {
                    val stats = halfStats(it.getShortBuffer())
                    selfFinite += stats.finite
                    selfNan += stats.nan
                    selfInf += stats.inf
                }
                check(selfNan == 0 && selfInf == 0) { "step=$step selfKV nan=$selfNan inf=$selfInf" }

                selfKv = stepResult.nextSelfKv
                completedSteps++
                val line = "STEP $step token_in=$currentToken position=$position logitsFinite=${logitsStats.finite} logitsMin=${logitsStats.min} logitsMax=${logitsStats.max} argmax=$nextToken selfKVFinite=$selfFinite selfKVNaN=$selfNan selfKVInf=$selfInf latencyMs=${(System.nanoTime() - stepStart) / 1_000_000.0}"
                stepLines += line
                Log.i(TAG, line)
                currentToken = nextToken
                position++
                stepTensors.forEach { tensor ->
                    if (tensor !== stepResult.logits && !stepResult.nextSelfKv.values.contains(tensor)) {
                        try { tensor.close() } catch (_: Throwable) {}
                    }
                }
            }

            check(completedSteps >= 8) { "M3.4 PASS requires >=8 completed steps, got $completedSteps" }
            return buildString {
                appendLine("M3.4 status: PASS")
                appendLine("device: Snapdragon 8 Gen 3 / SM8650")
                appendLine("soc_model: 57")
                appendLine("htp_arch: 75")
                appendLine("encoder: HTP SUCCESS")
                appendLine("decoder: HTP SUCCESS")
                appendLine("steps_requested: $requestedSteps")
                appendLine("steps_completed: $completedSteps")
                appendLine("cross_kv: computed_once=true")
                appendLine("cross_kv: finite=true")
                appendLine("self_kv: reused=true")
                appendLine("self_kv: finite=true")
                appendLine("argmax: enabled=true")
                appendLine("cpu_fallback: disabled=true")
                appendLine("dsp_crash: false")
                appendLine("encoder_ms: $encoderMs")
                appendLine("encoder_context_binary_bytes: ${encoderBinary.length()}")
                appendLine("decoder_context_binary_bytes: ${decoderBinary.length()}")
                appendLine("attention_mask: causal [1,1,1,200] FP16 per step")
                appendLine("position_ids: 0..${completedSteps - 1}")
                appendLine("BOS: validated existing runner value $VALIDATED_BOS_TOKEN")
                appendLine("EOS detection: unavailable / not configured")
                stepLines.forEach(::appendLine)
            }
        } finally {
            tensors.asReversed().forEach { try { it.close() } catch (_: Throwable) {} }
            try { encoderSession?.close() } catch (_: Throwable) {}
            try { decoderSession?.close() } catch (_: Throwable) {}
            try { encoderOptions?.close() } catch (_: Throwable) {}
            try { decoderOptions?.close() } catch (_: Throwable) {}
            try { env.unregisterExecutionProviderLibrary(EP_NAME) } catch (_: Throwable) {}
        }
    }

    private fun runDecoderStep(
        env: OrtEnvironment,
        session: OrtSession,
        step: Int,
        inputToken: Int,
        position: Int,
        selfKvInputs: Map<String, OnnxTensor>,
        crossKvInputs: Map<String, OnnxTensor>,
        attentionMask: OnnxTensor,
        owner: MutableList<OnnxTensor>,
    ): DecoderStepResult {
        val inputs = LinkedHashMap<String, OnnxTensor>()
        inputs["input_ids"] = i32(env, longArrayOf(1, 1), inputToken, owner)
        inputs["position_ids"] = i32(env, longArrayOf(1), position, owner)
        inputs.putAll(selfKvInputs)
        inputs["attention_mask"] = attentionMask
        inputs.putAll(crossKvInputs)
        var logits: OnnxTensor? = null
        val nextSelfKv = LinkedHashMap<String, OnnxTensor>()
        session.run(inputs).use { result ->
            check(result.size() == 9) { "step=$step decoder result size=${result.size()}" }
            for (name in selfOutNames + "logits") {
                val index = session.outputNames.toList().indexOf(name)
                check(index >= 0) { "missing decoder output $name" }
                val out = result[index] as OnnxTensor
                val shape = when {
                    name == "logits" -> longArrayOf(1, 51865, 1, 1)
                    name.startsWith("k_cache_self") -> longArrayOf(6, 1, 64, 199)
                    else -> longArrayOf(6, 1, 199, 64)
                }
                val stats = halfStats(out.getShortBuffer())
                check(stats.nan == 0 && stats.inf == 0 && stats.finite > 0) { "step=$step $name invalid finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}" }
                if (name == "logits") {
                    logits = copyHalfTensor(env, out, shape, owner)
                } else {
                    val nextInputName = name.removeSuffix("_out") + "_in"
                    nextSelfKv[nextInputName] = copyHalfTensor(env, out, shape, owner)
                }
            }
        }
        check(nextSelfKv.size == selfOutNames.size) { "step=$step selfKV outputs=${nextSelfKv.keys}" }
        return DecoderStepResult(logits ?: error("step=$step logits missing"), nextSelfKv)
    }

    private fun validateDecoderContract(session: OrtSession) {
        val inputInfo = session.inputInfo
        fun assertInfo(name: String, shape: LongArray, type: OnnxJavaType) {
            val info = inputInfo[name]?.info as? TensorInfo ?: error("Missing input $name")
            check(info.type == type && info.shape.contentEquals(shape)) { "$name type=${info.type} shape=${info.shape.contentToString()}" }
        }
        assertInfo("input_ids", longArrayOf(1, 1), OnnxJavaType.INT32)
        assertInfo("position_ids", longArrayOf(1), OnnxJavaType.INT32)
        for (i in 0 until 4) {
            assertInfo("k_cache_self_${i}_in", longArrayOf(6, 1, 64, 199), OnnxJavaType.FLOAT16)
            assertInfo("v_cache_self_${i}_in", longArrayOf(6, 1, 199, 64), OnnxJavaType.FLOAT16)
            assertInfo("k_cache_cross_$i", longArrayOf(6, 1, 64, 1500), OnnxJavaType.FLOAT16)
            assertInfo("v_cache_cross_$i", longArrayOf(6, 1, 1500, 64), OnnxJavaType.FLOAT16)
        }
        assertInfo("attention_mask", longArrayOf(1, 1, 1, 200), OnnxJavaType.FLOAT16)
    }

    private fun selfShape(name: String) = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 199) else longArrayOf(6, 1, 199, 64)
    private fun causalAttentionMask(env: OrtEnvironment, position: Int, owner: MutableList<OnnxTensor>): OnnxTensor {
        val values = WhisperDecoderMask.forPosition(position)
        return OnnxTensor.createTensor(env, ShortBuffer.wrap(values), longArrayOf(1, 1, 1, 200), OnnxJavaType.FLOAT16).also { owner += it }
    }

    private fun i32(env: OrtEnvironment, shape: LongArray, value: Int, owner: MutableList<OnnxTensor>) = OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(shape.fold(1L) { a, b -> a * b }.toInt()) { value }), shape).also { owner += it }
    private fun f16(env: OrtEnvironment, shape: LongArray, owner: MutableList<OnnxTensor>) = OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(shape.fold(1L) { a, b -> a * b }.toInt())), shape, OnnxJavaType.FLOAT16).also { owner += it }
    private fun copyHalfTensor(env: OrtEnvironment, source: OnnxTensor, shape: LongArray, owner: MutableList<OnnxTensor>): OnnxTensor { val src = source.getShortBuffer().duplicate(); val copy = ShortArray(src.remaining()); src.get(copy); return OnnxTensor.createTensor(env, ShortBuffer.wrap(copy), shape, OnnxJavaType.FLOAT16).also { owner += it } }
    private fun halfStats(buffer: ShortBuffer): HalfStats { val c = buffer.duplicate(); var min = Float.POSITIVE_INFINITY; var max = Float.NEGATIVE_INFINITY; var sum = 0.0; var finite = 0; var nan = 0; var inf = 0; var nonZero = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); when { v.isNaN() -> nan++; v.isInfinite() -> inf++; else -> { finite++; if (v < min) min = v; if (v > max) max = v; sum += v.toDouble(); if (v != 0f) nonZero++ } } }; return HalfStats(min, max, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf, nonZero) }
    private fun argmaxHalf(buffer: ShortBuffer): Int { val c = buffer.duplicate(); var best = Float.NEGATIVE_INFINITY; var bestIndex = -1; var i = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); if (v.isFinite() && v > best) { best = v; bestIndex = i }; i++ }; return bestIndex }
    private fun halfToFloat(bits: Int): Float { val sign = (bits ushr 15) and 1; val exp = (bits ushr 10) and 31; val frac = bits and 1023; val v = when (exp) { 0 -> frac / 1024.0f * Math.pow(2.0, -14.0).toFloat(); 31 -> if (frac == 0) Float.POSITIVE_INFINITY else Float.NaN; else -> (1f + frac / 1024f) * Math.pow(2.0, exp - 15.0).toFloat() }; return if (sign == 0) v else -v }
    private fun copyAsset(context: Context, asset: String): File { val f = File(context.cacheDir, asset.substringAfterLast('/')); context.assets.open(asset).use { input -> f.outputStream().use { output -> input.copyTo(output) } }; return f }
}
