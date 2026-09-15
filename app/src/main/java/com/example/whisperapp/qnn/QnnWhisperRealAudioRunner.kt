package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import com.example.whisperapp.asr.WhisperTokenizer
import com.example.whisperapp.audio.WhisperFeatureExtractor
import java.io.File
import java.nio.IntBuffer
import java.nio.ShortBuffer
import kotlin.math.max
import kotlin.math.min

object QnnWhisperRealAudioRunner {
    private const val TAG = "QNN_WHISPER_REAL"
    private const val EP_NAME = "QNNExecutionProvider"
    private const val EOS_TOKEN = 50257
    private const val MAX_STEPS = 16
    private val forcedPrompt = intArrayOf(50258, 50259, 50359, 50363)
    private val crossNames = (0 until 4).flatMap { listOf("k_cache_cross_$it", "v_cache_cross_$it") }
    private val selfInNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_in", "v_cache_self_${it}_in") }
    private val selfOutNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_out", "v_cache_self_${it}_out") }

    data class Result(val passed: Boolean, val report: String, val tokenIds: IntArray = IntArray(0), val text: String = "")
    private data class DecoderStepResult(val logits: OnnxTensor, val nextSelfKv: LinkedHashMap<String, OnnxTensor>)
    private data class HalfStats(val min: Float, val max: Float, val mean: Double, val finite: Int, val nan: Int, val inf: Int, val nonZero: Int)

    fun run(context: Context, pcm16: ShortArray, sampleRate: Int, requestedSteps: Int = MAX_STEPS): Result {
        require(requestedSteps in 1..MAX_STEPS)
        require(sampleRate == WhisperFeatureExtractor.SAMPLE_RATE) { "Whisper expects 16000 Hz, got $sampleRate" }
        return try {
            Log.i(TAG, "START requestedSteps=$requestedSteps samples=${pcm16.size} sampleRate=$sampleRate")
            val tokenizer = WhisperTokenizer.fromAssets(context.assets)
            Log.i(TAG, "TOKENIZER_READY")
            val features = WhisperFeatureExtractor().extract(pcm16, sampleRate)
            Log.i(TAG, "MEL_READY shape=${features.shape.contentToString()}")
            val output = runLoop(context, features.data, pcm16.size, sampleRate, tokenizer, requestedSteps)
            Result(true, output.first, output.second, output.third)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "M3.5 status: FAIL\n${t.javaClass.name}: ${t.message}")
        }
    }

    private fun runLoop(context: Context, features: FloatArray, pcmSamples: Int, sampleRate: Int, tokenizer: WhisperTokenizer, requestedSteps: Int): Triple<String, IntArray, String> {
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

            check(features.size == 1 * 80 * 3000) { "feature element count=${features.size}" }
            val featureStats = floatStats(features)
            check(featureStats.nan == 0 && featureStats.inf == 0 && featureStats.nonZero > 0) { "feature stats=$featureStats" }
            val inputFeatures = f16FromFloat(env, features, longArrayOf(1, 80, 3000), tensors)
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
            val attentionMask = f16(env, longArrayOf(1, 1, 1, 200), tensors)
            var currentToken = forcedPrompt[0]
            var completedSteps = 0
            var eosReached = false
            val generated = mutableListOf<Int>()
            val stepLines = mutableListOf<String>()

            for (step in 0 until requestedSteps) {
                val inputToken = if (step < forcedPrompt.size) forcedPrompt[step] else currentToken
                val stepStart = System.nanoTime()
                val stepTensors = mutableListOf<OnnxTensor>()
                val stepResult = runDecoderStep(
                    env = env,
                    session = decoderSession,
                    step = step,
                    inputToken = inputToken,
                    position = step,
                    selfKvInputs = selfKv,
                    crossKvInputs = crossInputs,
                    attentionMask = attentionMask,
                    owner = stepTensors,
                )
                val logitsStats = halfStats(stepResult.logits.getShortBuffer())
                check(logitsStats.nan == 0 && logitsStats.inf == 0) { "step=$step logits NaN=${logitsStats.nan} Inf=${logitsStats.inf}" }
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
                if (step >= forcedPrompt.size - 1) {
                    if (nextToken == EOS_TOKEN) {
                        eosReached = true
                    } else {
                        generated += nextToken
                        currentToken = nextToken
                    }
                }
                val line = "STEP $step token_in=$inputToken position=$step logitsFinite=${logitsStats.finite} logitsMin=${logitsStats.min} logitsMax=${logitsStats.max} argmax=$nextToken selfKVFinite=$selfFinite selfKVNaN=$selfNan selfKVInf=$selfInf latencyMs=${(System.nanoTime() - stepStart) / 1_000_000.0}"
                stepLines += line
                Log.i(TAG, line)
                stepTensors.forEach { tensor ->
                    if (tensor !== stepResult.logits && !stepResult.nextSelfKv.values.contains(tensor)) {
                        try { tensor.close() } catch (_: Throwable) {}
                    }
                }
                try { stepResult.logits.close() } catch (_: Throwable) {}
                if (eosReached) break
            }

            check(completedSteps >= 1) { "M3.5 requires decoder step 0 to complete" }
            val decodedText = tokenizer.decode(generated.toIntArray())
            val report = buildString {
                appendLine("M3.5 status: PASS")
                appendLine("audio: real PCM16")
                appendLine("sample_rate: $sampleRate")
                appendLine("channels: mono")
                appendLine("pcm_samples: $pcmSamples")
                appendLine("features: [1,80,3000] FP32")
                appendLine("feature_min: ${featureStats.min}")
                appendLine("feature_max: ${featureStats.max}")
                appendLine("feature_mean: ${featureStats.mean}")
                appendLine("feature_finite: ${featureStats.finite}")
                appendLine("feature_nan: ${featureStats.nan}")
                appendLine("feature_inf: ${featureStats.inf}")
                appendLine("encoder_input_adapter: FP32->FP16")
                appendLine("encoder: HTP SUCCESS")
                appendLine("decoder: HTP SUCCESS")
                appendLine("decoder_step_0: PASS")
                appendLine("steps_requested: $requestedSteps")
                appendLine("steps_completed: $completedSteps")
                appendLine("cross_kv_computed_once: true")
                appendLine("cross_kv_finite: true")
                appendLine("self_kv_recursive: true")
                appendLine("self_kv_finite: true")
                appendLine("forced_prompt: ${forcedPrompt.contentToString()}")
                appendLine("eos_reached: $eosReached")
                appendLine("eos_token: $EOS_TOKEN")
                appendLine("token_ids: ${generated.joinToString(",")}")
                appendLine("decoded_text: $decodedText")
                appendLine("cpu_fallback: disabled=true")
                appendLine("dsp_crash: false")
                appendLine("encoder_ms: $encoderMs")
                appendLine("encoder_context_binary_bytes: ${encoderBinary.length()}")
                appendLine("decoder_context_binary_bytes: ${decoderBinary.length()}")
                stepLines.forEach(::appendLine)
            }
            return Triple(report, generated.toIntArray(), decodedText)
        } finally {
            tensors.asReversed().forEach { try { it.close() } catch (_: Throwable) {} }
            try { encoderSession?.close() } catch (_: Throwable) {}
            try { decoderSession?.close() } catch (_: Throwable) {}
            try { encoderOptions?.close() } catch (_: Throwable) {}
            try { decoderOptions?.close() } catch (_: Throwable) {}
            try { env.unregisterExecutionProviderLibrary(EP_NAME) } catch (_: Throwable) { Unit }
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
    private fun i32(env: OrtEnvironment, shape: LongArray, value: Int, owner: MutableList<OnnxTensor>) = OnnxTensor.createTensor(env, IntBuffer.wrap(IntArray(shape.fold(1L) { a, b -> a * b }.toInt()) { value }), shape).also { owner += it }
    private fun f16(env: OrtEnvironment, shape: LongArray, owner: MutableList<OnnxTensor>) = OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(shape.fold(1L) { a, b -> a * b }.toInt())), shape, OnnxJavaType.FLOAT16).also { owner += it }
    private fun f16FromFloat(env: OrtEnvironment, values: FloatArray, shape: LongArray, owner: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env, ShortBuffer.wrap(ShortArray(values.size) { floatToHalf(values[it]) }), shape, OnnxJavaType.FLOAT16).also { owner += it }
    private fun copyHalfTensor(env: OrtEnvironment, source: OnnxTensor, shape: LongArray, owner: MutableList<OnnxTensor>): OnnxTensor { val src = source.getShortBuffer().duplicate(); val copy = ShortArray(src.remaining()); src.get(copy); return OnnxTensor.createTensor(env, ShortBuffer.wrap(copy), shape, OnnxJavaType.FLOAT16).also { owner += it } }
    private fun floatStats(values: FloatArray): HalfStats {
        var minValue = Float.POSITIVE_INFINITY
        var maxValue = Float.NEGATIVE_INFINITY
        var sum = 0.0
        var finite = 0
        var nan = 0
        var inf = 0
        var nonZero = 0
        for (value in values) {
            when {
                value.isNaN() -> nan++
                value.isInfinite() -> inf++
                else -> { finite++; minValue = kotlin.math.min(minValue, value); maxValue = kotlin.math.max(maxValue, value); sum += value.toDouble(); if (value != 0f) nonZero++ }
            }
        }
        return HalfStats(minValue, maxValue, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf, nonZero)
    }

    private fun floatToHalf(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exponent = ((bits ushr 23) and 0xff) - 127 + 15
        val mantissa = bits and 0x7fffff
        return when {
            exponent <= 0 -> if (exponent < -10) sign.toShort() else (sign or ((mantissa or 0x800000) shr (1 - exponent + 13))).toShort()
            exponent >= 31 -> (sign or 0x7c00 or if (mantissa == 0) 0 else 0x0200).toShort()
            else -> (sign or (exponent shl 10) or ((mantissa + 0x1000) shr 13)).toShort()
        }
    }

    private fun halfStats(buffer: ShortBuffer): HalfStats { val c = buffer.duplicate(); var min = Float.POSITIVE_INFINITY; var max = Float.NEGATIVE_INFINITY; var sum = 0.0; var finite = 0; var nan = 0; var inf = 0; var nonZero = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); when { v.isNaN() -> nan++; v.isInfinite() -> inf++; else -> { finite++; if (v < min) min = v; if (v > max) max = v; sum += v.toDouble(); if (v != 0f) nonZero++ } } }; return HalfStats(min, max, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf, nonZero) }
    private fun argmaxHalf(buffer: ShortBuffer): Int { val c = buffer.duplicate(); var best = Float.NEGATIVE_INFINITY; var bestIndex = -1; var i = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); if (v.isFinite() && v > best) { best = v; bestIndex = i }; i++ }; return bestIndex }
    private fun halfToFloat(bits: Int): Float { val sign = (bits ushr 15) and 1; val exp = (bits ushr 10) and 31; val frac = bits and 1023; val v = when (exp) { 0 -> frac / 1024.0f * Math.pow(2.0, -14.0).toFloat(); 31 -> if (frac == 0) Float.POSITIVE_INFINITY else Float.NaN; else -> (1f + frac / 1024f) * Math.pow(2.0, exp - 15.0).toFloat() }; return if (sign == 0) v else -v }
    private fun copyAsset(context: Context, asset: String): File { val f = File(context.cacheDir, asset.substringAfterLast('/')); context.assets.open(asset).use { input -> f.outputStream().use { output -> input.copyTo(output) } }; return f }
}
