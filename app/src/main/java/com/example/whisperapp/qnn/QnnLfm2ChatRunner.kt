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

/** Native LFM2.5 runner: ONNX Runtime QNN EP + HTP, with no WebView/WASM. */
object QnnLfm2ChatRunner {
    private const val TAG = "QNN_LFM2"
    private const val EP = "QNNExecutionProvider"
    private const val MODEL_DIR = "LFM2.5-230M-ONNX/onnx"
    private const val MODEL_FILE = "model_q4.onnx"
    /** Must match external_data.location embedded in model_q4.onnx. */
    private const val EXTERNAL_DATA_NAME = "model_q4.onnx_data"
    private const val EXTERNAL_DATA_SIZE = 211_111_936L
    private const val MAX_TOKENS = 256
    private val CONV_LAYERS = intArrayOf(0, 1, 3, 5, 7, 9, 11, 13)
    private val KV_LAYERS = intArrayOf(2, 4, 6, 8, 10, 12)

    data class Status(val ready: Boolean, val message: String)

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: Lfm2Tokenizer? = null
    private val lock = Any()

    fun start(context: Context): Status = synchronized(lock) {
        var stage = "begin"
        try {
            val app = context.applicationContext
            stage = "check_model_files"
            val modelDir = File(app.filesDir, "models/$MODEL_DIR")
            val model = File(modelDir, MODEL_FILE)
            val data = File(modelDir, EXTERNAL_DATA_NAME)
            if (data.isFile && data.length() != EXTERNAL_DATA_SIZE) {
                error("LFM2.5 Q4 权重大小不匹配：${data.length()}，预期 $EXTERNAL_DATA_SIZE；请重新下载正确的 model_q4.onnx_data")
            }
            Log.i(TAG, "init begin filesDir=${app.filesDir.absolutePath} nativeLibraryDir=${app.applicationInfo.nativeLibraryDir} model=${model.absolutePath} exists=${model.isFile} bytes=${model.length()} externalData=${data.absolutePath} exists=${data.isFile} bytes=${data.length()}")
            check(model.isFile) { "LFM2.5 Q4 模型未安装：" + model.absolutePath }
            check(data.isFile) { "LFM2.5 Q4 权重未安装：" + data.absolutePath }
            stage = "get_ort_environment"
            if (env == null) {
                Log.i(TAG, "creating ONNX Runtime environment")
                env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, TAG)
                Log.i(TAG, "ONNX Runtime environment created")
            } else Log.i(TAG, "reusing ONNX Runtime environment")
            // The environment (and therefore the QNN EP registration) is shared with the Whisper
            // ASR runner, which keeps its registration alive after recording stops; the gate below
            // is idempotent and also re-registers if a smoke-test activity removed the provider.
            stage = "register_qnn_provider"
            Log.i(TAG, "ensuring QNN EP registration")
            QnnEpRegistration.ensureRegistered(env!!)
            Log.i(TAG, "QNN EP registration check complete")
            stage = "inspect_ep_devices"
            Log.i(TAG, "available EP devices=${runCatching { env!!.epDevices.joinToString { it.epName } }.getOrElse { "<query failed: ${it.message}>" }}")
            if (session == null) {
                val device = env!!.epDevices.firstOrNull { it.epName == EP }
                    ?: error("QNNExecutionProvider device not exposed")
                stage = "check_htp_backend"
                val backend = File(app.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                Log.i(TAG, "HTP backend path=${backend.absolutePath} exists=${backend.isFile} bytes=${backend.length()}")
                check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
                stage = "configure_session_options"
                val options = OrtSession.SessionOptions().apply {
                    setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
                    // CPU fallback must stay enabled: the graph declares int64 index inputs, which
                    // QNN/HTP does not consume natively (ORT inserts int64->int32 casts on CPU).
                    // Disabling fallback makes createSession() fail outright, exactly like the
                    // Whisper EPContext wrappers documented in QnnWhisperRealAudioRunner.
                    addExecutionProvider(listOf(device), mapOf(
                        "backend_path" to backend.absolutePath,
                        "soc_model" to "57",
                        "htp_arch" to "75",
                        "enable_htp_fp16_precision" to "1",
                        "enable_block_quant_weight_optimization" to "1",
                    ))
                }
                stage = "create_onnx_session"
                Log.i(TAG, "creating ONNX session modelBytes=${model.length()} externalDataBytes=${data.length()}")
                session = env!!.createSession(model.absolutePath, options)
                options.close()
                Log.i(TAG, "session created with QNN HTP inputs=${session!!.inputNames.size} outputs=${session!!.outputNames.size}")
            } else {
                Log.i(TAG, "reusing existing ONNX session")
            }
            stage = "load_tokenizer"
            tokenizer = tokenizer ?: Lfm2Tokenizer.fromAssets(app.assets)
            Log.i(TAG, "tokenizer ready")
            Status(true, "LFM2.5 已就绪 · QNN HTP")
        } catch (t: Throwable) {
            Log.e(TAG, "init failed stage=$stage type=${t::class.java.name} message=${t.message}", t)
            Status(false, "QNN 加载失败[$stage]：${t.message ?: t::class.java.simpleName}")
        }
    }

    fun generate(context: Context, history: List<Pair<String, String>>): String = synchronized(lock) {
        val status = start(context)
        check(status.ready) { status.message }
        val tok = tokenizer!!
        val s = session!!
        val promptText = buildPrompt(history)
        val promptIds = tok.encode(promptText).toMutableList()
        require(promptIds.isNotEmpty()) { "Prompt tokenization produced no tokens" }
        val startedNs = System.nanoTime()
        Log.i(TAG, "GENERATE_START history=${history.size} promptChars=${promptText.length} promptTokens=${promptIds.size} promptPreview=${promptText.take(500).replace("\n", "\\n")} promptIds=${promptIds.joinToString(",")}")
        Log.i(TAG, "MODEL_META inputNames=${s.inputNames} outputNames=${s.outputNames} maxTokens=$MAX_TOKENS eosId=${Lfm2Tokenizer.EOS_ID}")

        val past = HashMap<String, FloatArray>()
        var pastLen = 0
        var inputIds = promptIds
        val generatedIds = ArrayList<Int>()

        repeat(MAX_TOKENS) {
            val sequenceLength = inputIds.size
            val inputs = LinkedHashMap<String, OnnxTensor>()
            val owned = ArrayList<OnnxTensor>()
            try {
                // The exported graph declares input_ids/attention_mask/position_ids as int64.
                inputs["input_ids"] = longTensor(inputIds.map { it.toLong() }.toLongArray(), longArrayOf(1, sequenceLength.toLong()), owned)
                inputs["attention_mask"] = longTensor(LongArray(pastLen + sequenceLength) { 1L }, longArrayOf(1, (pastLen + sequenceLength).toLong()), owned)
                inputs["position_ids"] = longTensor(LongArray(sequenceLength) { (pastLen + it).toLong() }, longArrayOf(1, sequenceLength.toLong()), owned)

                for (layer in CONV_LAYERS) {
                    inputs["past_conv.$layer"] = floatTensor(
                        past["past_conv.$layer"] ?: FloatArray(1024 * 3),
                        longArrayOf(1, 1024, 3), owned
                    )
                }
                for (layer in KV_LAYERS) {
                    val k = "past_key_values.$layer.key"
                    val v = "past_key_values.$layer.value"
                    inputs[k] = floatTensor(
                        past[k] ?: FloatArray(8 * pastLen * 64),
                        longArrayOf(1, 8, pastLen.toLong(), 64), owned
                    )
                    inputs[v] = floatTensor(
                        past[v] ?: FloatArray(8 * pastLen * 64),
                        longArrayOf(1, 8, pastLen.toLong(), 64), owned
                    )
                }

                if (it == 0 || it % 16 == 0) {
                    Log.i(TAG, "STEP_INPUT step=$it inputIds=${inputIds.joinToString(",")} sequenceLength=$sequenceLength pastLen=$pastLen attentionMaskLen=${pastLen + sequenceLength} positionIds=${(pastLen until pastLen + sequenceLength).joinToString(",")}")
                }
                val stepStartedNs = System.nanoTime()
                s.run(inputs).use { result ->
                    val outputs = s.outputNames.associateWith { name -> result[s.outputNames.indexOf(name)] as OnnxTensor }
                    val logits = outputs.getValue("logits")
                    val nextToken = argmaxLastPosition(logits)
                    val logitsBuffer = logits.floatBuffer.duplicate()
                    val vocabSize = logits.info.shape.last().toInt()
                    val finiteCount = (0 until vocabSize).count { idx -> logitsBuffer.get(logitsBuffer.limit() - vocabSize + idx).isFinite() }
                    generatedIds += nextToken
                    if (it < 12 || it % 16 == 0 || nextToken == Lfm2Tokenizer.EOS_ID) {
                        Log.i(TAG, "STEP_OUTPUT step=$it token=$nextToken tokenText=${tok.decode(intArrayOf(nextToken)).replace("\n", "\\n")} logitsShape=${logits.info.shape.contentToString()} vocab=$vocabSize finiteLastLogits=$finiteCount stepMs=${(System.nanoTime() - stepStartedNs) / 1_000_000} pastConvLens=${CONV_LAYERS.joinToString(",") { layer -> past["past_conv.$layer"]?.size?.toString() ?: "0" }} pastKvLens=${KV_LAYERS.joinToString(",") { layer -> past["past_key_values.$layer.key"]?.size?.toString() ?: "0" }}")
                    }

                    for (layer in CONV_LAYERS) {
                        past["past_conv.$layer"] = readFloats(outputs.getValue("present_conv.$layer"))
                    }
                    for (layer in KV_LAYERS) {
                        past["past_key_values.$layer.key"] = readFloats(outputs.getValue("present.$layer.key"))
                        past["past_key_values.$layer.value"] = readFloats(outputs.getValue("present.$layer.value"))
                    }
                }
            } finally {
                owned.forEach { runCatching { it.close() } }
            }

            pastLen += sequenceLength
            if (generatedIds.last() == Lfm2Tokenizer.EOS_ID) {
                return@synchronized finish(tok, generatedIds, startedNs)
            }
            inputIds = mutableListOf(generatedIds.last())
        }
        return@synchronized finish(tok, generatedIds, startedNs)
    }

    private fun finish(tok: Lfm2Tokenizer, ids: List<Int>, startedNs: Long): String {
        val text = tok.decode(ids.toIntArray())
        val elapsedMs = (System.nanoTime() - startedNs) / 1_000_000
        Log.i(TAG, "GENERATE_DONE tokens=${ids.size} elapsedMs=$elapsedMs text=\"${text.take(200)}\"")
        return text
    }

    fun stop() = synchronized(lock) {
        runCatching { session?.close() }
        session = null
    }

    /** Mirrors the model's own chat template (tokenizer_config.json), including its exact BOS. */
    private fun buildPrompt(history: List<Pair<String, String>>): String = buildString {
        append("<|startoftext|>")
        for ((role, content) in history) {
            append("<|im_start|>").append(role).append('\n')
            append(content)
            append("<|im_end|>\n")
        }
        append("<|im_start|>assistant\n")
    }

    /**
     * `logits` has shape [1, sequence_length, vocab]; only the last position predicts the next
     * token. Reading the maximum over the whole buffer would also consider the prompt positions.
     */
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
