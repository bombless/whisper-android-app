package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.example.whisperapp.js.JsRuntime
import com.example.whisperapp.js.LfmToolCallParser
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
    const val THINKING_PROMPT_PREFIX = "<think>让我把完整的方案想清楚再输出\n"
    const val TOOL_SYSTEM_PROMPT = """你是一个具备代码运行和自我验证能力的 AI 助手。你可以调用内置的 JavaScript 运行时工具来运行和测试代码，以确保算法和代码实现的绝对正确性。

【工具调用规则】
当你需要验证算法（例如红黑树、二叉搜索树、图算法、排序、复杂数学计算等）时，请在回复中调用工具：
<tool_call>
```javascript
// 编写待验证的代码与自动化测试用例
console.log("测试结果:", ...);
```
</tool_call>

系统执行后会自动将控制台输出返回在 <tool_response> 中。
请根据运行结果检验算法：如果测试通过，请结合验证结果输出最终方案；如果测试未通过，请分析原因并修正。"""
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
     * Generates assistant reply with streaming token support and optional JavaScript tool execution.
     * @param onToken Called whenever new tokens are decoded; provides accumulated generated string.
     */
    fun generate(
        context: Context,
        history: List<Pair<String, String>>,
        maxTokens: Int = DEFAULT_MAX_TOKENS,
        isCancelled: (() -> Boolean)? = null,
        enableTools: Boolean = true,
        onToken: ((partialText: String) -> Unit)? = null,
    ): String = synchronized(lock) {
        cancelRequested = false
        val status = start(context)
        check(status.ready) { status.message }
        val tok = tokenizer!!
        val s = session!!
        val promptText = buildPrompt(history, enableTools = enableTools)
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
        var lastDecodeText = THINKING_PROMPT_PREFIX
        var lastExecutedToolEnd = 0
        var toolExecutionCount = 0
        val maxToolCalls = 3
        onToken?.invoke(lastDecodeText)

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
                    var nextToken = argmaxLastPosition(logits)
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
                    val decoded = tok.decode(generatedIds.toIntArray())
                    lastDecodeText = if (decoded.startsWith("<think>")) decoded else (THINKING_PROMPT_PREFIX + decoded)
                    onToken?.invoke(lastDecodeText)

                    // Detect and execute JavaScript tool call if emitted by model
                    if (enableTools && toolExecutionCount < maxToolCalls) {
                        val toolCall = LfmToolCallParser.findNextToolCall(lastDecodeText, fromIndex = lastExecutedToolEnd)
                        if (toolCall != null && toolCall.code.isNotBlank()) {
                            toolExecutionCount++
                            lastExecutedToolEnd = toolCall.endIndex
                            LfmChatLogger.info("TOOL_CALL_DETECTED tool=${toolCall.name} codeChars=${toolCall.code.length} iteration=$toolExecutionCount")

                            // Inform UI that the JS tool is actively running
                            onToken?.invoke(lastDecodeText + "\n[⚙️ JavaScript 运行时正在执行并测试代码...]\n")

                            val execResult = JsRuntime.execute(toolCall.code)
                            LfmChatLogger.info("TOOL_CALL_RESULT success=${execResult.success} timeMs=${execResult.executionTimeMs}ms")

                            val responseText = LfmToolCallParser.formatToolResponse(execResult)
                            val respIds = tok.encode(responseText)
                            if (respIds.isNotEmpty()) {
                                generatedIds += respIds.toList()

                                // Ingest tool response tokens into model KV cache
                                for (respTok in respIds) {
                                    val feedInputs = LinkedHashMap<String, OnnxTensor>()
                                    val feedOwned = ArrayList<OnnxTensor>()
                                    try {
                                        feedInputs["input_ids"] = longTensor(longArrayOf(respTok.toLong()), longArrayOf(1, 1), feedOwned)
                                        feedInputs["attention_mask"] = longTensor(LongArray(pastLen + 1) { 1L }, longArrayOf(1, (pastLen + 1).toLong()), feedOwned)
                                        feedInputs["position_ids"] = longTensor(longArrayOf(pastLen.toLong()), longArrayOf(1, 1), feedOwned)
                                        feedInputs["logits_to_keep"] = longTensor(longArrayOf(1L), longArrayOf(1), feedOwned)

                                        for (l in CONV_LAYERS) {
                                            feedInputs["past_conv.$l"] = floatTensor(
                                                past["past_conv.$l"] ?: FloatArray(CONV_CHANNELS * CONV_KERNEL),
                                                longArrayOf(1, CONV_CHANNELS.toLong(), CONV_KERNEL.toLong()), feedOwned
                                            )
                                        }
                                        for (l in KV_LAYERS) {
                                            feedInputs["past_key_values.$l.key"] = floatTensor(
                                                past["past_key_values.$l.key"] ?: FloatArray(KV_HEADS * pastLen * HEAD_DIM),
                                                longArrayOf(1, KV_HEADS.toLong(), pastLen.toLong(), HEAD_DIM.toLong()), feedOwned
                                            )
                                            feedInputs["past_key_values.$l.value"] = floatTensor(
                                                past["past_key_values.$l.value"] ?: FloatArray(KV_HEADS * pastLen * HEAD_DIM),
                                                longArrayOf(1, KV_HEADS.toLong(), pastLen.toLong(), HEAD_DIM.toLong()), feedOwned
                                            )
                                        }

                                        s.run(feedInputs).use { feedRes ->
                                            val outMap = s.outputNames.associateWith { name -> feedRes[s.outputNames.indexOf(name)] as OnnxTensor }
                                            for (l in CONV_LAYERS) {
                                                past["past_conv.$l"] = readFloats(outMap.getValue("present_conv.$l"))
                                            }
                                            for (l in KV_LAYERS) {
                                                past["past_key_values.$l.key"] = readFloats(outMap.getValue("present.$l.key"))
                                                past["past_key_values.$l.value"] = readFloats(outMap.getValue("present.$l.value"))
                                            }
                                            nextToken = argmaxLastPosition(outMap.getValue("logits"))
                                        }
                                        pastLen += 1
                                    } finally {
                                        feedOwned.forEach { runCatching { it.close() } }
                                    }
                                }

                                val fullDecoded = tok.decode(generatedIds.toIntArray())
                                lastDecodeText = if (fullDecoded.startsWith("<think>")) fullDecoded else (THINKING_PROMPT_PREFIX + fullDecoded)
                                lastExecutedToolEnd = lastDecodeText.length
                                onToken?.invoke(lastDecodeText)
                            }
                        }
                    }
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

    /** Mirrors the model's chat template, stripping old thinking and opening reasoning prompt. */
    private fun buildPrompt(history: List<Pair<String, String>>, enableTools: Boolean = true): String = buildString {
        append("<|startoftext|>")
        val hasSystem = history.any { it.first == "system" }
        if (!hasSystem && enableTools) {
            append("<|im_start|>system\n")
            append(TOOL_SYSTEM_PROMPT)
            append("<|im_end|>\n")
        }
        for ((role, content) in history) {
            append("<|im_start|>").append(role).append('\n')
            if (role == "assistant") {
                // In official LFM3 Jinja template, prior assistant turns strip thinking:
                val stripped = if (content.contains("</think>")) {
                    "<think></think>" + content.substringAfter("</think>").trim()
                } else {
                    content
                }
                append(stripped)
            } else {
                append(content)
            }
            append("<|im_end|>\n")
        }
        append("<|im_start|>assistant\n")
        append(THINKING_PROMPT_PREFIX)
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
