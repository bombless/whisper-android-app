package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** Native LFM3 runner: ONNX Runtime QNN EP + HTP, with no WebView/WASM. */
object QnnLfmChatRunner {
    private const val TAG = "QNN_LFM3"
    private const val EP = "QNNExecutionProvider"
    private const val MODEL_DIR = "LFM3-1.7B-ONNX/onnx"
    private const val MODEL_FILE = "model_q4.onnx"
    /** Must match external_data.location embedded in model_q4.onnx. */
    private const val EXTERNAL_DATA_NAME = "model_q4.onnx_data"
    private const val EXTERNAL_DATA_SIZE = 1_625_944_064L
    const val DEFAULT_MAX_TOKENS = 4096
    private val CONV_LAYERS = intArrayOf(0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
    private val KV_LAYERS = intArrayOf(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    private const val CONV_CHANNELS = 2048
    private const val CONV_KERNEL = 7
    private const val KV_HEADS = 4
    private const val HEAD_DIM = 128

    @Volatile
    private var cancelRequested = false

    fun cancelGeneration() {
        cancelRequested = true
    }

    data class Status(val ready: Boolean, val message: String)

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: LfmTokenizer? = null
    private val lock = Any()

    fun start(context: Context): Status = synchronized(lock) {
        var stage = "begin"
        try {
            val app = context.applicationContext
            LfmChatLogger.init(app)
            stage = "check_model_files"
            val modelDir = File(app.filesDir, "models/$MODEL_DIR")
            val model = File(modelDir, MODEL_FILE)
            val data = File(modelDir, EXTERNAL_DATA_NAME)
            if (data.isFile && data.length() != EXTERNAL_DATA_SIZE) {
                val err = "LFM3 Q4 权重大小不匹配：${data.length()}，预期 $EXTERNAL_DATA_SIZE；请重新下载正确的 model_q4.onnx_data"
                LfmChatLogger.error(err)
                error(err)
            }
            LfmChatLogger.info("init begin filesDir=${app.filesDir.absolutePath} model=${model.absolutePath} exists=${model.isFile} bytes=${model.length()} externalData=${data.absolutePath} exists=${data.isFile} bytes=${data.length()}")
            check(model.isFile) { "LFM3 Q4 模型未安装：" + model.absolutePath }
            check(data.isFile) { "LFM3 Q4 权重未安装：" + data.absolutePath }
            stage = "get_ort_environment"
            if (env == null) {
                LfmChatLogger.info("creating ONNX Runtime environment")
                env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, TAG)
                LfmChatLogger.info("ONNX Runtime environment created")
            } else LfmChatLogger.info("reusing ONNX Runtime environment")
            stage = "register_qnn_provider"
            LfmChatLogger.info("ensuring QNN EP registration")
            QnnEpRegistration.ensureRegistered(env!!)
            LfmChatLogger.info("QNN EP registration check complete")
            stage = "inspect_ep_devices"
            val availableEps = runCatching { env!!.epDevices.joinToString { it.epName } }.getOrElse { "<query failed: ${it.message}>" }
            LfmChatLogger.info("available EP devices=$availableEps")
            if (session == null) {
                val device = env!!.epDevices.firstOrNull { it.epName == EP }
                    ?: error("QNNExecutionProvider device not exposed")
                stage = "check_htp_backend"
                val backend = File(app.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                LfmChatLogger.info("HTP backend path=${backend.absolutePath} exists=${backend.isFile} bytes=${backend.length()}")
                check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
                stage = "configure_session_options"
                val options = OrtSession.SessionOptions().apply {
                    setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
                    addExecutionProvider(listOf(device), mapOf(
                        "backend_path" to backend.absolutePath,
                        "soc_model" to "57",
                        "htp_arch" to "75",
                        "enable_htp_fp16_precision" to "1",
                        "enable_block_quant_weight_optimization" to "1",
                    ))
                }
                stage = "create_onnx_session"
                LfmChatLogger.info("creating ONNX session with QNN HTP backend...")
                session = env!!.createSession(model.absolutePath, options)
                options.close()
                LfmChatLogger.info("session created with QNN HTP inputs=${session!!.inputNames.size} outputs=${session!!.outputNames.size}")
            } else {
                LfmChatLogger.info("reusing existing ONNX session")
            }
            stage = "load_tokenizer"
            tokenizer = tokenizer ?: LfmTokenizer.fromAssets(app.assets)
            LfmChatLogger.info("tokenizer ready")
            Status(true, "LFM3 已就绪 · QNN HTP")
        } catch (t: Throwable) {
            LfmChatLogger.error("init failed stage=$stage type=${t::class.java.name} message=${t.message}", t)
            Status(false, "QNN 加载失败[$stage]：${t.message ?: t::class.java.simpleName}")
        }
    }

    /**
     * Generates assistant reply with streaming token support.
     * @param onToken Called whenever new tokens are decoded; provides accumulated generated string.
     */
    fun generate(
        context: Context,
        history: List<Pair<String, String>>,
        maxTokens: Int = DEFAULT_MAX_TOKENS,
        isCancelled: (() -> Boolean)? = null,
        onToken: ((partialText: String) -> Unit)? = null,
    ): String = synchronized(lock) {
        cancelRequested = false
        val status = start(context)
        check(status.ready) { status.message }
        val tok = tokenizer!!
        val s = session!!
        val promptText = buildPrompt(history)
        val promptIds = tok.encode(promptText).toMutableList()
        require(promptIds.isNotEmpty()) { "Prompt tokenization produced no tokens" }
        val startedNs = System.nanoTime()
        LfmChatLogger.info("GENERATE_START historyLen=${history.size} promptChars=${promptText.length} promptTokens=${promptIds.size} maxTokens=$maxTokens eosId=${LfmTokenizer.EOS_ID}")
        LfmChatLogger.info("PROMPT_PREVIEW ${promptText.take(200).replace("\n", "\\n")}")

        val past = HashMap<String, FloatArray>()
        var pastLen = 0
        var inputIds = promptIds
        val generatedIds = ArrayList<Int>()
        var stopReason = "MAX_TOKENS_REACHED"
        var lastDecodeText = ""

        for (step in 0 until maxTokens) {
            if (cancelRequested || isCancelled?.invoke() == true || Thread.currentThread().isInterrupted) {
                stopReason = "USER_CANCELLED"
                LfmChatLogger.info("GENERATE_CANCELLED at step=$step generated=${generatedIds.size}")
                break
            }
            val sequenceLength = inputIds.size
            val inputs = LinkedHashMap<String, OnnxTensor>()
            val owned = ArrayList<OnnxTensor>()
            try {
                inputs["input_ids"] = longTensor(inputIds.map { it.toLong() }.toLongArray(), longArrayOf(1, sequenceLength.toLong()), owned)
                inputs["attention_mask"] = longTensor(LongArray(pastLen + sequenceLength) { 1L }, longArrayOf(1, (pastLen + sequenceLength).toLong()), owned)
                inputs["position_ids"] = longTensor(LongArray(sequenceLength) { (pastLen + it).toLong() }, longArrayOf(1, sequenceLength.toLong()), owned)
                inputs["logits_to_keep"] = longTensor(longArrayOf(1L), longArrayOf(1), owned)

                // Conv state cache (13 layers)
                for (layer in CONV_LAYERS) {
                    inputs["past_conv.$layer"] = floatTensor(
                        past["past_conv.$layer"] ?: FloatArray(CONV_CHANNELS * CONV_KERNEL),
                        longArrayOf(1, CONV_CHANNELS.toLong(), CONV_KERNEL.toLong()), owned
                    )
                }
                // KV attention cache (11 layers)
                for (layer in KV_LAYERS) {
                    val k = "past_key_values.$layer.key"
                    val v = "past_key_values.$layer.value"
                    inputs[k] = floatTensor(
                        past[k] ?: FloatArray(KV_HEADS * pastLen * HEAD_DIM),
                        longArrayOf(1, KV_HEADS.toLong(), pastLen.toLong(), HEAD_DIM.toLong()), owned
                    )
                    inputs[v] = floatTensor(
                        past[v] ?: FloatArray(KV_HEADS * pastLen * HEAD_DIM),
                        longArrayOf(1, KV_HEADS.toLong(), pastLen.toLong(), HEAD_DIM.toLong()), owned
                    )
                }

                val stepStartedNs = System.nanoTime()
                s.run(inputs).use { result ->
                    val outputs = s.outputNames.associateWith { name -> result[s.outputNames.indexOf(name)] as OnnxTensor }
                    val logits = outputs.getValue("logits")
                    val nextToken = argmaxLastPosition(logits)
                    val stepMs = (System.nanoTime() - stepStartedNs) / 1_000_000
                    generatedIds += nextToken

                    // Update past cache from output tensors
                    for (layer in CONV_LAYERS) {
                        past["past_conv.$layer"] = readFloats(outputs.getValue("present_conv.$layer"))
                    }
                    for (layer in KV_LAYERS) {
                        past["past_key_values.$layer.key"] = readFloats(outputs.getValue("present.$layer.key"))
                        past["past_key_values.$layer.value"] = readFloats(outputs.getValue("present.$layer.value"))
                    }

                    if (step == 0 || step % 32 == 0 || nextToken == LfmTokenizer.EOS_ID) {
                        val currentToks = generatedIds.size
                        val totalElapsedMs = (System.nanoTime() - startedNs) / 1_000_000
                        val speed = if (totalElapsedMs > 0) String.format("%.1f", currentToks * 1000.0 / totalElapsedMs) else "0.0"
                        LfmChatLogger.info("STEP step=$step token=$nextToken pastLen=$pastLen seqLen=$sequenceLength stepMs=${stepMs}ms generated=${currentToks} speed=${speed}tok/s")
                    }

                    // Decode streaming text for UI
                    lastDecodeText = tok.decode(generatedIds.toIntArray())
                    onToken?.invoke(lastDecodeText)
                }
            } finally {
                owned.forEach { runCatching { it.close() } }
            }

            pastLen += sequenceLength
            if (generatedIds.last() == LfmTokenizer.EOS_ID) {
                stopReason = "EOS_TOKEN"
                break
            }
            inputIds = mutableListOf(generatedIds.last())
        }

        val totalElapsedMs = (System.nanoTime() - startedNs) / 1_000_000
        val avgSpeed = if (totalElapsedMs > 0) String.format("%.1f", generatedIds.size * 1000.0 / totalElapsedMs) else "0.0"
        LfmChatLogger.info("GENERATE_FINISH totalTokens=${generatedIds.size} elapsedMs=${totalElapsedMs}ms avgSpeed=${avgSpeed}tok/s stopReason=$stopReason logFile=${LfmChatLogger.getLogPath()}")
        return lastDecodeText
    }

    fun stop() = synchronized(lock) {
        runCatching { session?.close() }
        session = null
    }

    /** Mirrors the model's chat template, including BOS and direct response prompt without internal monologue. */
    private fun buildPrompt(history: List<Pair<String, String>>): String = buildString {
        append("<|startoftext|>")
        for ((role, content) in history) {
            append("<|im_start|>").append(role).append('\n')
            append(content)
            append("<|im_end|>\n")
        }
        append("<|im_start|>assistant\n<think>\n</think>")
    }

    private fun argmaxLastPosition(tensor: OnnxTensor): Int {
        val shape = tensor.info.shape
        val vocab = shape[shape.size - 1].toInt()
        val b = tensor.floatBuffer
        var best = Float.NEGATIVE_INFINITY
        var bestIndex = 0
        for (i in 0 until vocab) {
            val v = b.get(b.limit() - vocab + i)
            if (v > best) { best = v; bestIndex = i }
        }
        return bestIndex
    }

    private fun readFloats(tensor: OnnxTensor): FloatArray {
        val b = tensor.floatBuffer.duplicate()
        val out = FloatArray(b.remaining())
        b.get(out)
        return out
    }

    private fun longTensor(data: LongArray, shape: LongArray, owned: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env!!, LongBuffer.wrap(data), shape).also { owned += it }

    private fun floatTensor(data: FloatArray, shape: LongArray, owned: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env!!, FloatBuffer.wrap(data), shape).also { owned += it }
}

typealias QnnLfm2ChatRunner = QnnLfmChatRunner
