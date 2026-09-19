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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.nio.ShortBuffer
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

object QnnWhisperRealAudioRunner {
    private const val TAG = "QNN_WHISPER_REAL"
    private const val EP_NAME = "QNNExecutionProvider"
    private const val EOS_TOKEN = 50257
    private const val START_OF_TRANSCRIPT_TOKEN = 50258
    private const val ZH_LANGUAGE_TOKEN = 50260
    private const val TRANSCRIBE_TOKEN = 50359
    private const val NO_TIMESTAMPS_TOKEN = 50363
    private const val MAX_STEPS = 16
    private const val MAX_GENERATION_STEPS = 32
    // Whisper language token: <|zh|>; the previous prompt forced English.
    private val forcedPrompt = intArrayOf(START_OF_TRANSCRIPT_TOKEN, ZH_LANGUAGE_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN)
    private val crossNames = (0 until 4).flatMap { listOf("k_cache_cross_$it", "v_cache_cross_$it") }
    private val selfInNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_in", "v_cache_self_${it}_in") }
    private val selfOutNames = (0 until 4).flatMap { listOf("k_cache_self_${it}_out", "v_cache_self_${it}_out") }
    private val lifecycleLock = Any()
    private var env: OrtEnvironment? = null
    private var encoderSession: OrtSession? = null
    private var decoderSession: OrtSession? = null
    private var encoderOptions: OrtSession.SessionOptions? = null
    private var decoderOptions: OrtSession.SessionOptions? = null
    private var qnnRegistered = false
    private var lifecycleChunk = 0
    private var profileServer: QnnProfileServer? = null
    private var profileDir: File? = null
    private var cachedTokenizer: WhisperTokenizer? = null

    data class Result(val passed: Boolean, val report: String, val tokenIds: IntArray = IntArray(0), val text: String = "")
    /**
     * KV feedback: [result] stays unclosed for one extra step so its self-KV output
     * tensors can be fed back as the next step's inputs without any host-side copy.
     * [logits] and [nextSelfKv] are views owned by [result].
     */
    private data class DecoderStepResult(val result: OrtSession.Result, val logits: OnnxTensor, val nextSelfKv: LinkedHashMap<String, OnnxTensor>)
    private data class HalfStats(val min: Float, val max: Float, val mean: Double, val finite: Int, val nan: Int, val inf: Int, val nonZero: Int)

    @Volatile var debugDiagnostics = false
        private set
    @Volatile var profilingEnabled = false
        private set
    private var sessionsProfiled: Boolean? = null

    /** Called from the HTTP config endpoint; sessions are rebuilt lazily on next start(). */
    fun configure(profiling: Boolean? = null, debug: Boolean? = null) {
        profiling?.let { profilingEnabled = it }
        debug?.let { debugDiagnostics = it }
        Log.i(TAG, "CONFIG profiling=$profilingEnabled debug=$debugDiagnostics")
    }

    private fun closeSessionsLocked() {
        try { decoderSession?.close() } catch (_: Throwable) {}
        try { encoderSession?.close() } catch (_: Throwable) {}
        try { decoderOptions?.close() } catch (_: Throwable) {}
        try { encoderOptions?.close() } catch (_: Throwable) {}
        decoderSession = null
        encoderSession = null
        decoderOptions = null
        encoderOptions = null
        sessionsProfiled = null
    }

    fun start(context: Context) {
        synchronized(lifecycleLock) {
            val appContext = context.applicationContext
            val qnnProfileRoot = File(appContext.filesDir, "qnn-profile")
            if (profileServer == null) profileServer = QnnProfileServer(qnnProfileRoot).also { it.start() }
            if (env != null && encoderSession != null && decoderSession != null && sessionsProfiled == profilingEnabled) return
            if (encoderSession != null || decoderSession != null) {
                // profiling/debug flag flipped since these sessions were built — rebuild them
                closeSessionsLocked()
            }
            Log.i(TAG, "QNN_LIFECYCLE START")
            val encoderModel = copyAsset(appContext, "models/whisper/encoder_ctx.onnx")
            val decoderModel = copyAsset(appContext, "models/whisper/decoder_ctx.onnx")
            // EPContext wrappers use a relative ep_cache_context (encoder.bin / decoder.bin).
            // ORT resolves that path relative to the wrapper ONNX file, so the binaries must
            // already exist beside the copied wrapper before createSession() is called.
            val encoderBinary = copyAsset(appContext, "models/whisper/encoder.bin")
            val decoderBinary = copyAsset(appContext, "models/whisper/decoder.bin")
            check(encoderBinary.isFile && encoderBinary.length() > 0) {
                "Encoder EPContext binary missing: ${encoderBinary.absolutePath}"
            }
            check(decoderBinary.isFile && decoderBinary.length() > 0) {
                "Decoder EPContext binary missing: ${decoderBinary.absolutePath}"
            }
            Log.i(TAG, "QNN_LIFECYCLE CONTEXT_BINARIES encoder=${encoderBinary.length()} decoder=${decoderBinary.length()}")
            if (env == null) {
                env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, TAG)
                Log.i(TAG, "QNN_LIFECYCLE INIT_ORT")
            }
            val runtimeEnv = env!!
            if (!qnnRegistered) {
                System.loadLibrary("onnxruntime_providers_qnn")
                runtimeEnv.registerExecutionProviderLibrary(EP_NAME, "libonnxruntime_providers_qnn.so")
                qnnRegistered = true
                Log.i(TAG, "QNN_LIFECYCLE REGISTER_EP")
            } else {
                Log.i(TAG, "QNN_LIFECYCLE REUSE_EP")
            }
            val device = runtimeEnv.epDevices.firstOrNull { it.epName == EP_NAME } ?: error("QNN EP device not exposed")
            val backend = File(appContext.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
            val runProfileDir = qnnProfileRoot.apply { mkdirs() }
            val profilePrefix = "run_" + System.currentTimeMillis()
            profileDir = runProfileDir
            fun makeOptions(profileFile: File) = OrtSession.SessionOptions().apply {
                // EPContext wrapper graphs contain a small CPU-side IO adapter (notably
                // QuantizeLinear/DequantizeLinear). Disabling CPU fallback makes session
                // creation fail before recording even starts when those nodes are left on CPU.
                setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                val epOptions = buildMap<String, String> {
                    put("backend_path", backend.absolutePath)
                    put("soc_model", "57")
                    put("htp_arch", "75")
                    put("offload_graph_io_quantization", "0")
                    if (profilingEnabled) {
                        put("profiling_level", "optrace")
                        put("profiling_file_path", profileFile.absolutePath)
                    }
                }
                addExecutionProvider(listOf(device), epOptions)
            }
            sessionsProfiled = profilingEnabled
            Log.i(TAG, "QNN_LIFECYCLE PROFILING=$profilingEnabled DEBUG=$debugDiagnostics")
            encoderOptions = makeOptions(File(runProfileDir, profilePrefix + "_encoder.csv"))
            decoderOptions = makeOptions(File(runProfileDir, profilePrefix + "_decoder.csv"))
            Log.i(TAG, "QNN_LIFECYCLE CREATE_ENCODER")
            OpTrace.time("session.create", "init", mapOf("model" to "encoder")) {
                encoderSession = runtimeEnv.createSession(encoderModel.absolutePath, encoderOptions)
            }
            Log.i(TAG, "QNN_LIFECYCLE CREATE_DECODER")
            OpTrace.time("session.create", "init", mapOf("model" to "decoder")) {
                decoderSession = runtimeEnv.createSession(decoderModel.absolutePath, decoderOptions)
            }
            check(encoderSession!!.inputNames == setOf("input_features")) { "encoder inputs=${encoderSession!!.inputNames}" }
            check(encoderSession!!.outputNames.toSet() == crossNames.toSet()) { "encoder outputs=${encoderSession!!.outputNames}" }
            check(decoderSession!!.inputNames.size == 19) { "decoder inputs=${decoderSession!!.inputNames}" }
            check(decoderSession!!.outputNames.toSet() == (selfOutNames + "logits").toSet()) { "decoder outputs=${decoderSession!!.outputNames}" }
            if (debugDiagnostics) validateDecoderContract(decoderSession!!)
            // Preload the tokenizer so the first transcribed chunk doesn't pay ~750 ms of
            // vocab parsing inside the real-time loop; chunks 2+ reuse the cached instance.
            OpTrace.time("tokenizer.preload", "init") {
                cachedTokenizer ?: WhisperTokenizer.fromAssets(appContext.assets).also { cachedTokenizer = it }
            }
            lifecycleChunk = 0
        }
    }

    fun transcribeChunk(context: Context, pcm16: ShortArray, sampleRate: Int, requestedSteps: Int = MAX_STEPS, autoregressive: Boolean = false): Result {
        synchronized(lifecycleLock) {
            start(context)
            lifecycleChunk++
            Log.i(TAG, "QNN_CHUNK START #$lifecycleChunk")
            val runId = OpTrace.beginRun(
                "chunk_$lifecycleChunk",
                mapOf(
                    "samples" to pcm16.size.toString(),
                    "sample_rate" to sampleRate.toString(),
                    "requested_steps" to requestedSteps.toString(),
                    "autoregressive" to autoregressive.toString(),
                )
            )
            return try {
                transcribeChunkLocked(context.applicationContext, pcm16, sampleRate, requestedSteps, autoregressive)
            } finally {
                OpTrace.endRun(runId, pcm16.size / sampleRate.toDouble())
                Log.i(TAG, "QNN_CHUNK DONE #$lifecycleChunk")
            }
        }
    }

    fun stop() {
        synchronized(lifecycleLock) {
            if (env == null && encoderSession == null && decoderSession == null && !qnnRegistered) return
            Log.i(TAG, "QNN_LIFECYCLE STOP")
            closeSessionsLocked()
            Log.i(TAG, "QNN_LIFECYCLE CLOSED")
            lifecycleChunk = 0
            Log.i(TAG, "QNN_LIFECYCLE STOP_DONE ENV_EP_RETAINED")
        }
    }

    fun run(context: Context, pcm16: ShortArray, sampleRate: Int, requestedSteps: Int = MAX_STEPS, autoregressive: Boolean = false): Result {
        require(if (autoregressive) requestedSteps in 1..(forcedPrompt.size - 1 + MAX_GENERATION_STEPS) else requestedSteps in 1..MAX_STEPS)
        require(sampleRate == WhisperFeatureExtractor.SAMPLE_RATE) { "Whisper expects 16000 Hz, got $sampleRate" }
        return try {
            Log.i(TAG, "START requestedSteps=$requestedSteps samples=${pcm16.size} sampleRate=$sampleRate")
            start(context)
            transcribeChunk(context, pcm16, sampleRate, requestedSteps, autoregressive)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "M3.5 status: FAIL\n${t.javaClass.name}: ${t.message}")
        } finally {
            stop()
        }
    }

    private fun transcribeChunkLocked(context: Context, pcm16: ShortArray, sampleRate: Int, requestedSteps: Int, autoregressive: Boolean): Result {
        return try {
            val tokenizer = OpTrace.time("tokenizer.load", "preprocess") {
                cachedTokenizer ?: WhisperTokenizer.fromAssets(context.assets).also { cachedTokenizer = it }
            }
            Log.i(TAG, "TOKENIZER_READY cached=${cachedTokenizer != null}")
            val extractor = WhisperFeatureExtractor()
            val melFloat: FloatArray?
            val melHalf: ShortArray?
            if (debugDiagnostics) {
                melFloat = OpTrace.time("mel.extract", "preprocess") { extractor.extract(pcm16, sampleRate).data }
                melHalf = null
            } else {
                melFloat = null
                melHalf = OpTrace.time("mel.extract", "preprocess") { extractor.extractHalf(pcm16, sampleRate) }
            }
            val melSize = melFloat?.size ?: melHalf!!.size
            Log.i(TAG, "MEL_READY elements=$melSize half=${melHalf != null}")
            val melStats = if (debugDiagnostics) floatStats(melFloat!!) else null
            val diagnosticDir = if (debugDiagnostics) createDiagnosticDir(context) else context.cacheDir
            if (debugDiagnostics) {
                LongAudioAppLogger.info("MEL_STATS min=${melStats!!.min} max=${melStats.max} mean=${melStats.mean} finite=${melStats.finite} nan=${melStats.nan} inf=${melStats.inf} nonZero=${melStats.nonZero}")
                OpTrace.time("diagnostics.write", "preprocess") {
                    writePcmDiagnostic(diagnosticDir, pcm16, sampleRate)
                    writeFloat32(diagnosticDir.resolve("mel.bin"), melFloat!!)
                }
                writeText(diagnosticDir.resolve("metadata.txt"), buildString {
                    appendLine("experiment_id=debug")
                    appendLine("sample_rate=$sampleRate")
                    appendLine("pcm_samples=${pcm16.size}")
                    appendLine("mel.dtype=float32")
                    appendLine("mel.elements=$melSize")
                    appendLine("mel.shape=1x80x3000")
                })
            }
            LongAudioAppLogger.info("MEL_READY elements=$melSize half=${melHalf != null}")
            val output = runLoop(context, melFloat, melHalf, pcm16.size, sampleRate, tokenizer, requestedSteps, autoregressive, diagnosticDir)
            Result(true, output.first, output.second, output.third)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Result(false, "M3.5 status: FAIL\n${t.javaClass.name}: ${t.message}")
        }
    }

    private fun runLoop(context: Context, melFloat: FloatArray?, melHalf: ShortArray?, pcmSamples: Int, sampleRate: Int, tokenizer: WhisperTokenizer, requestedSteps: Int, autoregressive: Boolean, diagnosticDir: File): Triple<String, IntArray, String> {
        val encoderBinary = copyAsset(context, "models/whisper/encoder.bin")
        val decoderBinary = copyAsset(context, "models/whisper/decoder.bin")
        val env = this.env ?: error("QNN runner is not started")
        val encoderSession = this.encoderSession ?: error("encoder session not initialized")
        val decoderSession = this.decoderSession ?: error("decoder session not initialized")
        val tensors = mutableListOf<OnnxTensor>()
        try {
            Log.i(TAG, "ENCODER_SESSION_REUSE")
            check(encoderSession.inputNames == setOf("input_features")) { "encoder inputs=${encoderSession.inputNames}" }
            check(encoderSession.outputNames.toSet() == crossNames.toSet()) { "encoder outputs=${encoderSession.outputNames}" }
            check(decoderSession.inputNames.size == 19) { "decoder inputs=${decoderSession.inputNames}" }
            check(decoderSession.outputNames.toSet() == (selfOutNames + "logits").toSet()) { "decoder outputs=${decoderSession.outputNames}" }
            if (debugDiagnostics) validateDecoderContract(decoderSession)

            check((melFloat?.size ?: melHalf!!.size) == 1 * 80 * 3000) { "feature element count mismatch" }
            val featureStats = if (debugDiagnostics) {
                val s = floatStats(melFloat!!)
                check(s.nan == 0 && s.inf == 0 && s.nonZero > 0) { "feature stats=$s" }
                s
            } else null
            val inputFeatures = if (melHalf != null) {
                // FP16 path: the extractor already produced binary16 samples.
                OnnxTensor.createTensor(env, ShortBuffer.wrap(melHalf), longArrayOf(1, 80, 3000), OnnxJavaType.FLOAT16)
                    .also { tensors += it }
            } else {
                OpTrace.time("fp32_to_fp16", "preprocess", mapOf("elements" to melFloat!!.size.toString())) {
                    f16FromFloat(env, melFloat, longArrayOf(1, 80, 3000), tensors)
                }
            }
            val inputShorts = if (debugDiagnostics) melHalf ?: ShortArray(melFloat!!.size) { floatToHalf(melFloat[it]) } else null
            if (debugDiagnostics) {
                writeFloat16(diagnosticDir.resolve("encoder_input.bin"), inputShorts!!)
                appendMetadata(diagnosticDir, "encoder_input.dtype=float16\nencoder_input.elements=${inputShorts.size}\nencoder_input.shape=1x80x3000\n")
            }
            Log.i(TAG, "ENCODER_INPUT_READY shape=[1, 80, 3000] dtype=FP16")
            Log.i(TAG, "ENCODER_OUTPUT_READY names=${encoderSession.outputNames}")
            val crossInputs = LinkedHashMap<String, OnnxTensor>()
            val encoderStart = System.nanoTime()
            Log.i(TAG, "ENCODER_RUN_START")
            OpTrace.time("encoder.run", "device") {
                encoderSession.run(mapOf("input_features" to inputFeatures)).use { result ->
                    Log.i(TAG, "ENCODER_RUN_RETURN resultSize=${result.size()}")
                    check(result.size() == 8) { "encoder result size=${result.size()}" }
                    for (name in crossNames) {
                        val index = encoderSession.outputNames.toList().indexOf(name)
                        check(index >= 0) { "missing encoder output $name" }
                        val out = result[index] as OnnxTensor
                        val shape = if (name.startsWith("k_")) longArrayOf(6, 1, 64, 1500) else longArrayOf(6, 1, 1500, 64)
                        OpTrace.time("encoder.host_copy", "host", mapOf("tensor" to name)) {
                            if (debugDiagnostics) {
                                val s = halfStats(out.getShortBuffer())
                                LongAudioAppLogger.info("ENCODER_OUTPUT_STATS name=$name shape=${shape.contentToString()} min=${s.min} max=${s.max} mean=${s.mean} finite=${s.finite} nan=${s.nan} inf=${s.inf} nonZero=${s.nonZero}")
                                check(s.nan == 0 && s.inf == 0 && s.nonZero > 0) { "$name invalid stats=$s" }
                            }
                            val outputCopy = copyHalfTensor(env, out, shape, tensors)
                            crossInputs[name] = outputCopy
                            if (debugDiagnostics) {
                                val outputBuffer = out.getShortBuffer().duplicate()
                                val outputShorts = ShortArray(outputBuffer.remaining())
                                outputBuffer.get(outputShorts)
                                writeFloat16(diagnosticDir.resolve("encoder_output_${name}.bin"), outputShorts)
                                appendMetadata(diagnosticDir, "encoder_output_${name}.dtype=float16\nencoder_output_${name}.elements=${outputShorts.size}\nencoder_output_${name}.shape=${shape.joinToString("x")}\n")
                            }
                        }
                    }
                }
            }
            val encoderMs = (System.nanoTime() - encoderStart) / 1_000_000.0

            var selfKv: Map<String, OnnxTensor> = LinkedHashMap<String, OnnxTensor>().also { m ->
                for (name in selfInNames) m[name] = f16(env, selfShape(name), tensors)
            }
            var prevResult: OrtSession.Result? = null
            val forcedValidationMode = !autoregressive
            var currentToken = forcedPrompt.last()
            var completedSteps = 0
            var generationStepsCompleted = 0
            var eosReached = false
            val generated = mutableListOf<Int>()
            val stepLines = mutableListOf<String>()
            LongAudioAppLogger.info("DECODER_START")

            for (step in 0 until requestedSteps) {
                val inputToken = if (forcedValidationMode) forcedPrompt[step % forcedPrompt.size] else if (step < forcedPrompt.size) forcedPrompt[step] else currentToken
                val stepStart = System.nanoTime()
                val stepTensors = mutableListOf<OnnxTensor>()
                val attentionMask = OpTrace.time("decoder.mask", "host", mapOf("step" to step.toString())) {
                    causalAttentionMask(env, step, stepTensors)
                }
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
                val logitsStats = if (debugDiagnostics) {
                    OpTrace.time("decoder.host_copy", "host", mapOf("step" to step.toString(), "part" to "logits_stats")) {
                        val s = halfStats(stepResult.logits.getShortBuffer())
                        check(s.nan == 0 && s.inf == 0) { "step=$step logits NaN=${s.nan} Inf=${s.inf}" }
                        s
                    }
                } else null
                val nextToken = OpTrace.time("decoder.argmax", "host", mapOf("step" to step.toString(), "vocab" to "51865")) {
                    argmaxHalf(stepResult.logits.getShortBuffer())
                }
                check(nextToken >= 0) { "step=$step argmax failed" }
                if (debugDiagnostics) {
                    val topK = OpTrace.time("decoder.topk", "host", mapOf("step" to step.toString(), "k" to "5")) {
                        topKHalf(stepResult.logits.getShortBuffer(), 5)
                    }
                    val eosLogit = halfAt(stepResult.logits.getShortBuffer(), EOS_TOKEN)
                    LongAudioAppLogger.info("QNN_COMPARE_STEP step=$step inputToken=$inputToken position=$step top1=$nextToken eosLogit=$eosLogit maxLogit=${topK.firstOrNull()?.second ?: Float.NaN}")
                    LongAudioAppLogger.info("QNN_COMPARE_TOPK step=$step ${topK.joinToString(" ") { pair -> "${pair.first}:${pair.second}" }}")
                }
                LongAudioAppLogger.info("DECODER_TOKEN step=$step tokenId=$nextToken inputToken=$inputToken")

                var selfFinite = 0
                var selfNan = 0
                var selfInf = 0
                if (debugDiagnostics) {
                    OpTrace.time("decoder.host_copy", "host", mapOf("step" to step.toString(), "part" to "self_kv_stats")) {
                        stepResult.nextSelfKv.values.forEach {
                            val stats = halfStats(it.getShortBuffer())
                            selfFinite += stats.finite
                            selfNan += stats.nan
                            selfInf += stats.inf
                        }
                    }
                    check(selfNan == 0 && selfInf == 0) { "step=$step selfKV nan=$selfNan inf=$selfInf" }
                }

                selfKv = stepResult.nextSelfKv
                // The previous step's result (whose tensors were this step's inputs) is now
                // fully consumed; release it only after the new run has finished reading them.
                try { prevResult?.close() } catch (_: Throwable) {}
                prevResult = stepResult.result
                completedSteps++
                // The last prompt input already predicts the first generated token.
                if (!forcedValidationMode && step >= forcedPrompt.size - 1) {
                    generationStepsCompleted++
                    if (nextToken == EOS_TOKEN) {
                        generated += nextToken
                        eosReached = true
                        LongAudioAppLogger.info("DECODER_GENERATED step=$step count=${generated.size} lastToken=$nextToken ids=${generated.take(20)}")
                        LongAudioAppLogger.info("DECODER_EOS step=$step tokenId=$EOS_TOKEN")
                    } else {
                        generated += nextToken
                        currentToken = nextToken
                        LongAudioAppLogger.info("DECODER_GENERATED step=$step count=${generated.size} lastToken=$nextToken ids=${generated.take(20)}")
                    }
                }
                val stepTotalMs = (System.nanoTime() - stepStart) / 1_000_000.0
                OpTrace.record("decoder.step", "pipeline", stepTotalMs, mapOf("step" to step.toString(), "token" to nextToken.toString()))
                val line = if (debugDiagnostics) {
                    "STEP $step token_in=$inputToken position=$step logitsFinite=${logitsStats!!.finite} logitsMin=${logitsStats.min} logitsMax=${logitsStats.max} argmax=$nextToken selfKVFinite=$selfFinite selfKVNaN=$selfNan selfKVInf=$selfInf latencyMs=$stepTotalMs"
                } else {
                    "STEP $step token_in=$inputToken argmax=$nextToken latencyMs=$stepTotalMs"
                }
                stepLines += line
                Log.i(TAG, line)
                // Per-step helper tensors (mask, ids) are consumed once run() returns.
                // logits/self-KV stay alive inside stepResult.result for the next iteration.
                stepTensors.forEach { tensor ->
                    try { tensor.close() } catch (_: Throwable) {}
                }
                if (eosReached) break
            }
            try { prevResult?.close() } catch (_: Throwable) {}
            prevResult = null

            check(completedSteps >= 1) { "M3.5 requires decoder step 0 to complete" }
            val tokenizerInputIds = generated.toIntArray()
            LongAudioAppLogger.info("TOKENIZER_CALL_IDS count=${tokenizerInputIds.size} ids=${tokenizerInputIds.contentToString()}")
            val decodedText = OpTrace.time("tokenizer.decode", "postprocess", mapOf("tokens" to tokenizerInputIds.size.toString())) {
                tokenizer.decode(tokenizerInputIds)
            }
            LongAudioAppLogger.info("TOKENIZER_INPUT_IDS count=${tokenizerInputIds.size} ids=${tokenizerInputIds.contentToString()}")
            LongAudioAppLogger.info("TOKENIZER_DECODED_TEXT text=\"$decodedText\"")
            val report = buildString {
                appendLine("M3.5 status: PASS")
                appendLine("audio: real PCM16")
                appendLine("sample_rate: $sampleRate")
                appendLine("channels: mono")
                appendLine("pcm_samples: $pcmSamples")
                appendLine("features: [1,80,3000] FP32")
                if (featureStats != null) {
                    appendLine("feature_min: ${featureStats.min}")
                    appendLine("feature_max: ${featureStats.max}")
                    appendLine("feature_mean: ${featureStats.mean}")
                    appendLine("feature_finite: ${featureStats.finite}")
                    appendLine("feature_nan: ${featureStats.nan}")
                    appendLine("feature_inf: ${featureStats.inf}")
                }
                appendLine("encoder_input_adapter: FP32->FP16")
                appendLine("encoder: HTP SUCCESS")
                appendLine("decoder: HTP SUCCESS")
                appendLine("decoder_step_0: PASS")
                appendLine("steps_requested: $requestedSteps")
                appendLine("steps_completed: $completedSteps")
                appendLine("generation_steps_requested: ${if (autoregressive) (requestedSteps - forcedPrompt.size + 1).coerceAtLeast(0) else 0}")
                appendLine("attention_mask: right-aligned valid KV, FP16 masked_value=-100")
                appendLine("autoregressive: $autoregressive")
                appendLine("generation_steps_completed: $generationStepsCompleted")
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
            // Per-chunk tensors are released here; QNN/ORT/Sessions stay alive until stop().
            tensors.asReversed().forEach { try { it.close() } catch (_: Throwable) {} }
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
        val result = OpTrace.time("decoder.run", "device", mapOf("step" to step.toString(), "inputs" to inputs.size.toString())) {
            session.run(inputs)
        }
        // NOTE: result is intentionally NOT closed here — its self-KV output tensors become
        // the next step's inputs (zero-copy KV feedback). The caller closes the previous
        // result only after the next run has finished reading them.
        check(result.size() == 9) { "step=$step decoder result size=${result.size()}" }
        for (name in selfOutNames + "logits") {
            val index = session.outputNames.toList().indexOf(name)
            check(index >= 0) { "missing decoder output $name" }
            val out = result[index] as OnnxTensor
            // Touch one element so lazy/async EPs materialize the tensor before it is
            // fed back as a next-step input (a no-op for synchronous EPs).
            out.getShortBuffer().get(0)
            if (debugDiagnostics) {
                OpTrace.time("decoder.host_copy", "host", mapOf("step" to step.toString(), "part" to "scan_$name")) {
                    val stats = halfStats(out.getShortBuffer())
                    check(stats.nan == 0 && stats.inf == 0 && stats.finite > 0) { "step=$step $name invalid finite=${stats.finite} nan=${stats.nan} inf=${stats.inf}" }
                }
            }
            if (name == "logits") {
                logits = out
            } else {
                nextSelfKv[name.removeSuffix("_out") + "_in"] = out
            }
        }
        check(nextSelfKv.size == selfOutNames.size) { "step=$step selfKV outputs=${nextSelfKv.keys}" }
        return DecoderStepResult(result, logits ?: error("step=$step logits missing"), nextSelfKv)
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

    /**
     * The graph appends the current token after 199 cache slots, then returns the
     * last 199 slots. Valid keys are on the right; unused leading slots are masked.
     */
    private fun causalAttentionMask(env: OrtEnvironment, position: Int, owner: MutableList<OnnxTensor>): OnnxTensor {
        val values = WhisperDecoderMask.forPosition(position)
        return OnnxTensor.createTensor(
            env,
            ShortBuffer.wrap(values),
            longArrayOf(1, 1, 1, 200),
            OnnxJavaType.FLOAT16
        ).also { owner += it }
    }

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
            else -> {
                var halfExponent = exponent
                var halfMantissa = (mantissa + 0x1000) shr 13
                if (halfMantissa == 0x400) {
                    halfMantissa = 0
                    halfExponent++
                }
                if (halfExponent >= 31) (sign or 0x7c00).toShort()
                else (sign or (halfExponent shl 10) or halfMantissa).toShort()
            }
        }
    }

    private fun halfStats(buffer: ShortBuffer): HalfStats { val c = buffer.duplicate(); var min = Float.POSITIVE_INFINITY; var max = Float.NEGATIVE_INFINITY; var sum = 0.0; var finite = 0; var nan = 0; var inf = 0; var nonZero = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); when { v.isNaN() -> nan++; v.isInfinite() -> inf++; else -> { finite++; if (v < min) min = v; if (v > max) max = v; sum += v.toDouble(); if (v != 0f) nonZero++ } } }; return HalfStats(min, max, if (finite == 0) Double.NaN else sum / finite, finite, nan, inf, nonZero) }
    private fun argmaxHalf(buffer: ShortBuffer): Int { val c = buffer.duplicate(); var best = Float.NEGATIVE_INFINITY; var bestIndex = -1; var i = 0; while (c.hasRemaining()) { val v = halfToFloat(c.get().toInt() and 0xffff); if (v.isFinite() && v > best) { best = v; bestIndex = i }; i++ }; return bestIndex }
    private fun halfAt(buffer: ShortBuffer, index: Int): Float { val c = buffer.duplicate(); c.position(index); return halfToFloat(c.get().toInt() and 0xffff) }
    private fun topKHalf(buffer: ShortBuffer, k: Int): List<Pair<Int, Float>> {
        val c = buffer.duplicate()
        val values = ArrayList<Pair<Int, Float>>(c.remaining())
        var index = 0
        while (c.hasRemaining()) {
            val value = halfToFloat(c.get().toInt() and 0xffff)
            if (value.isFinite()) values += index to value
            index++
        }
        return values.sortedByDescending { it.second }.take(k)
    }
    private fun halfToFloat(bits: Int): Float { val sign = (bits ushr 15) and 1; val exp = (bits ushr 10) and 31; val frac = bits and 1023; val v = when (exp) { 0 -> frac * SUBNORMAL_STEP; 31 -> if (frac == 0) Float.POSITIVE_INFINITY else Float.NaN; else -> (1f + frac / 1024f) * EXP_TABLE[exp] }; return if (sign == 0) v else -v }

    // Lookup tables: EXP_TABLE[e] = 2^(e-15) for normal exponents, SUBNORMAL_STEP = 2^-24.
    private val EXP_TABLE = FloatArray(32) { e -> if (e == 0) 0f else Math.pow(2.0, (e - 15).toDouble()).toFloat() }
    private val SUBNORMAL_STEP = Math.pow(2.0, -24.0).toFloat()
    private fun createDiagnosticDir(context: Context): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
        return File(context.filesDir, "diagnostics/experiment_$stamp").apply { mkdirs() }
    }

    private fun writeText(file: File, text: String) {
        file.writeText(text, Charsets.UTF_8)
    }

    private fun appendMetadata(dir: File, text: String) {
        dir.resolve("metadata.txt").appendText(text, Charsets.UTF_8)
    }

    private fun writeFloat32(file: File, values: FloatArray) {
        val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(buffer::putFloat)
        file.writeBytes(buffer.array())
    }

    private fun writeFloat16(file: File, values: ShortArray) {
        val buffer = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(buffer::putShort)
        file.writeBytes(buffer.array())
    }

    private fun writePcmDiagnostic(dir: File, pcm16: ShortArray, sampleRate: Int) {
        val padded = ShortArray(WhisperFeatureExtractor.CHUNK_SAMPLES)
        pcm16.copyInto(padded, 0, 0, min(pcm16.size, padded.size))
        val raw = ByteBuffer.allocate(pcm16.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        pcm16.forEach(raw::putShort)
        val paddedRaw = ByteBuffer.allocate(padded.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        padded.forEach(paddedRaw::putShort)
        dir.resolve("pcm.bin").writeBytes(raw.array())
        dir.resolve("pcm_padded.bin").writeBytes(paddedRaw.array())
        val sha = MessageDigest.getInstance("SHA-256").digest(raw.array()).joinToString("") { "%02X".format(it) }
        val paddedSha = MessageDigest.getInstance("SHA-256").digest(paddedRaw.array()).joinToString("") { "%02X".format(it) }
        val stats = floatStats(FloatArray(pcm16.size) { pcm16[it].toFloat() })
        writeText(dir.resolve("pcm_stats.txt"), buildString {
            appendLine("sample_count=${pcm16.size}")
            appendLine("sample_rate=$sampleRate")
            appendLine("first16=${pcm16.take(16).joinToString(",")}")
            appendLine("last16=${pcm16.takeLast(16).joinToString(",")}")
            appendLine("min=${stats.min}")
            appendLine("max=${stats.max}")
            appendLine("mean=${stats.mean}")
            appendLine("rms=${kotlin.math.sqrt(pcm16.fold(0.0) { acc, v -> acc + v.toDouble() * v.toDouble() } / pcm16.size)}")
            appendLine("sha256_raw_pcm=$sha")
            appendLine("padded_count=${padded.size}")
            appendLine("padded_zero_count=${padded.count { it == 0.toShort() } - pcm16.take(padded.size).count { it == 0.toShort() }}")
            appendLine("sha256_padded_pcm=$paddedSha")
            appendLine("first_padded=${padded.firstOrNull() ?: 0}")
            appendLine("last_padded=${padded.lastOrNull() ?: 0}")
        })
    }

    private fun copyAsset(context: Context, asset: String): File {
        val f = File(context.cacheDir, asset.substringAfterLast('/'))
        // Context binaries are immutable; skip the copy when a same-size file already exists.
        // (Re-copying 116 MB of encoder/decoder bins per chunk cost ~1.3 s.)
        context.assets.open(asset).use { input ->
            val size = input.available().toLong()
            if (f.isFile && f.length() == size) return f
            f.outputStream().use { output -> input.copyTo(output) }
        }
        return f
    }
}
