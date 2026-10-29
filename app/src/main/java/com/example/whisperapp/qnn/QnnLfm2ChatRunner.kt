package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.PowerManager
import android.util.Log
import com.example.whisperapp.js.JsRuntime
import com.example.whisperapp.js.LfmToolCallParser
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
    const val TOOL_SYSTEM_PROMPT = """你是一个具备代码运行和自我验证能力的 AI 助手。你可以调用内置的 JavaScript 运行时工具（run_javascript）来执行并验证代码，以确保算法实现、数学计算与复杂逻辑的绝对正确性。

【工具定义】
你可以调用如下工具：
- 工具名称: run_javascript
- 功能说明: 执行并验证 JavaScript (ES6) 脚本，支持 console.log() 与 assert(condition, message)。
- 参数列表:
  - code (string, 必需): 要执行的 JavaScript 代码。

【工具调用规范】
当你需要验证算法（如红黑树、二叉树、图算法、排序、复杂数学计算等）时，请使用原生格式调用工具：
<tool_call>
<function=run_javascript>
<parameter=code>
// 在此处编写合法的 JavaScript (ES6) 脚本
// 可使用 console.log() 输出结果，可使用 assert(condition, message) 进行断言验证
</parameter>
</function>
</tool_call>

系统会自动执行脚本并将控制台输出返回在 <tool_response> 中。你必须根据 <tool_response> 的测试结果输出最终正确方案。

【调用示例 (One-Shot Example)】
用户：请验证 1 + 1 是否等于 2。
助手：
让我调用 JavaScript 运行时来验证这个计算：
<tool_call>
<function=run_javascript>
<parameter=code>
const result = 1 + 1;
console.log("计算结果:", result);
assert(result === 2, "1 + 1 应该等于 2");
</parameter>
</function>
</tool_call>
<tool_response>
[输出]
计算结果: 2
[状态] 执行成功 (无输出)
</tool_response>
经过 JavaScript 运行时验证，计算结果 1 + 1 = 2，断言验证通过。"""
    private val CONV_LAYERS = intArrayOf(0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)
    private val KV_LAYERS = intArrayOf(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    private const val CONV_CHANNELS = 2048
    private const val CONV_KERNEL = 7
    private const val KV_HEADS = 4
    private const val HEAD_DIM = 128

    private class ConvCachePool {
        private val buffers = HashMap<Int, FloatBuffer>()
        init {
            for (l in CONV_LAYERS) {
                val bb = ByteBuffer.allocateDirect(CONV_CHANNELS * CONV_KERNEL * 4)
                    .order(ByteOrder.nativeOrder())
                buffers[l] = bb.asFloatBuffer()
            }
        }
        fun get(layer: Int): FloatBuffer {
            val b = buffers.getValue(layer)
            b.clear()
            return b
        }
        fun reset() {
            for (b in buffers.values) {
                b.clear()
                while (b.hasRemaining()) b.put(0f)
                b.clear()
            }
        }
    }

    private class KvLayerCache(initialCapacityTokens: Int = 1024) {
        private var capacity = initialCapacityTokens
        var directBuffer: FloatBuffer = ByteBuffer.allocateDirect(512 * capacity * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            private set

        fun reset() {}

        private fun ensureCapacity(tokens: Int) {
            if (tokens > capacity) {
                var newCap = capacity * 2
                while (newCap < tokens) newCap *= 2
                val newBb = ByteBuffer.allocateDirect(512 * newCap * 4)
                    .order(ByteOrder.nativeOrder())
                val newFb = newBb.asFloatBuffer()
                val oldFb = directBuffer.duplicate()
                oldFb.position(0)
                oldFb.limit(512 * capacity)
                newFb.put(oldFb)
                directBuffer = newFb
                capacity = newCap
            }
        }

        fun wrap(tokens: Int): FloatBuffer {
            val fb = directBuffer.duplicate()
            fb.position(0)
            fb.limit(512 * tokens)
            return fb
        }

        fun updateFrom(tensor: OnnxTensor, totalTokens: Int) {
            ensureCapacity(totalTokens)
            val srcFb = tensor.floatBuffer
            val dstFb = directBuffer.duplicate()
            dstFb.position(0)
            dstFb.limit(512 * totalTokens)
            srcFb.position(0)
            srcFb.limit(512 * totalTokens)
            dstFb.put(srcFb)
        }
    }

    private class KvCachePool {
        private val keyCaches = HashMap<Int, KvLayerCache>()
        private val valCaches = HashMap<Int, KvLayerCache>()
        init {
            for (l in KV_LAYERS) {
                keyCaches[l] = KvLayerCache()
                valCaches[l] = KvLayerCache()
            }
        }
        fun getKey(layer: Int): KvLayerCache = keyCaches.getValue(layer)
        fun getValue(layer: Int): KvLayerCache = valCaches.getValue(layer)
        fun reset() {
            for (c in keyCaches.values) c.reset()
            for (c in valCaches.values) c.reset()
        }
    }

    private class SessionOutputBindings(session: OrtSession) {
        val logitsIndex: Int
        val convIndices = IntArray(CONV_LAYERS.maxOrNull()?.plus(1) ?: 24) { -1 }
        val kvKeyIndices = IntArray(KV_LAYERS.maxOrNull()?.plus(1) ?: 24) { -1 }
        val kvValIndices = IntArray(KV_LAYERS.maxOrNull()?.plus(1) ?: 24) { -1 }

        init {
            val names = session.outputNames.toList()
            logitsIndex = names.indexOf("logits")
            for (l in CONV_LAYERS) {
                convIndices[l] = names.indexOf("present_conv.$l")
            }
            for (l in KV_LAYERS) {
                kvKeyIndices[l] = names.indexOf("present.$l.key")
                kvValIndices[l] = names.indexOf("present.$l.value")
            }
        }
    }

    private val convPool = ConvCachePool()
    private val kvPool = KvCachePool()
    private var outputBindings: SessionOutputBindings? = null

    private val staticAllOnesBuffer: LongBuffer = ByteBuffer.allocateDirect(8192 * 8)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer().apply {
            for (i in 0 until 8192) put(1L)
            clear()
        }
    private val posDirectBuffer: LongBuffer = ByteBuffer.allocateDirect(DEFAULT_MAX_TOKENS * 8)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer()
    private val inputIdDirectBuffer: LongBuffer = ByteBuffer.allocateDirect(DEFAULT_MAX_TOKENS * 8)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer()
    private val logitsToKeepBuffer: LongBuffer = ByteBuffer.allocateDirect(8)
        .order(ByteOrder.nativeOrder())
        .asLongBuffer().apply {
            put(1L)
            clear()
        }

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
            val modelDir = listOfNotNull(
                File(app.filesDir, "models/$MODEL_DIR"),
                app.getExternalFilesDir(null)?.let { File(it, "models/$MODEL_DIR") }
            ).firstOrNull { File(it, MODEL_FILE).isFile } ?: File(app.filesDir, "models/$MODEL_DIR")
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
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    addExecutionProvider(listOf(device), mapOf(
                        "backend_path" to backend.absolutePath,
                        "soc_model" to "57",
                        "htp_arch" to "75",
                        "enable_htp_fp16_precision" to "1",
                        "enable_block_quant_weight_optimization" to "1",
                        "htp_performance_mode" to "sustained_high_performance",
                        "qnn_context_priority" to "high",
                        "vtcm_mb" to "8",
                        "htp_graph_finalization_optimization_mode" to "3",
                    ))
                }
                stage = "create_onnx_session"
                LfmChatLogger.info("creating ONNX session with QNN HTP backend...")
                session = env!!.createSession(model.absolutePath, options)
                options.close()
                outputBindings = SessionOutputBindings(session!!)
                LfmChatLogger.info("session created with QNN HTP inputs=${session!!.inputNames.size} outputs=${session!!.outputNames.size}")
            } else {
                LfmChatLogger.info("reusing existing ONNX session")
                if (outputBindings == null) {
                    outputBindings = SessionOutputBindings(session!!)
                }
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
        val runId = OpTrace.beginRun("lfm3_chat", mapOf("prompt_tokens" to promptIds.size.toString()))

        LfmChatLogger.info("GENERATE_START historyLen=${history.size} promptChars=${promptText.length} promptTokens=${promptIds.size} maxTokens=$maxTokens eosId=${LfmTokenizer.EOS_ID}")
        LfmChatLogger.info("PROMPT_PREVIEW ${promptText.take(200).replace("\n", "\\n")}")

        convPool.reset()
        kvPool.reset()

        val streamDecoder = tok.createStreamDecoder()
        var pastLen = 0
        var inputIds = promptIds
        val generatedIds = ArrayList<Int>()
        var stopReason = "MAX_TOKENS_REACHED"
        var lastDecodeText = THINKING_PROMPT_PREFIX
        var lastExecutedToolEnd = 0
        var toolExecutionCount = 0
        val maxToolCalls = 3
        val stepLatencies = ArrayList<Double>()
        onToken?.invoke(lastDecodeText)

        val powerManager = context.applicationContext.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val wakeLock = powerManager?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "whisperapp:LFM3_INFERENCE")?.apply {
            setReferenceCounted(false)
            runCatching { acquire(10 * 60 * 1000L) }
        }

        val runOptions = OrtSession.RunOptions().apply {
            addRunConfigEntry("qnn.htp_perf_mode", "sustained_high_performance")
            addRunConfigEntry("qnn.perf_mode", "sustained_high_performance")
        }

        try {
            for (step in 0 until maxTokens) {
                if (cancelRequested || isCancelled?.invoke() == true || Thread.currentThread().isInterrupted) {
                    stopReason = "USER_CANCELLED"
                    LfmChatLogger.info("GENERATE_CANCELLED at step=$step generated=${generatedIds.size}")
                    break
                }
                val stepT0 = System.nanoTime()
                val sequenceLength = inputIds.size
                val inputs = LinkedHashMap<String, OnnxTensor>(64)
                val owned = ArrayList<OnnxTensor>(48)
                try {
                    val totalMaskLen = pastLen + sequenceLength
                    val maskBuffer = if (totalMaskLen <= staticAllOnesBuffer.capacity()) {
                        val b = staticAllOnesBuffer.duplicate()
                        b.position(0)
                        b.limit(totalMaskLen)
                        b
                    } else {
                        LongBuffer.wrap(LongArray(totalMaskLen) { 1L })
                    }
                    val posBuf = if (sequenceLength <= posDirectBuffer.capacity()) {
                        posDirectBuffer.clear()
                        for (i in 0 until sequenceLength) posDirectBuffer.put((pastLen + i).toLong())
                        posDirectBuffer.flip()
                        posDirectBuffer.duplicate()
                    } else {
                        LongBuffer.wrap(LongArray(sequenceLength) { (pastLen + it).toLong() })
                    }
                    val inputBuf = if (sequenceLength <= inputIdDirectBuffer.capacity()) {
                        inputIdDirectBuffer.clear()
                        for (id in inputIds) inputIdDirectBuffer.put(id.toLong())
                        inputIdDirectBuffer.flip()
                        inputIdDirectBuffer.duplicate()
                    } else {
                        LongBuffer.wrap(inputIds.map { it.toLong() }.toLongArray())
                    }
                    val logitsToKeepBuf = logitsToKeepBuffer.duplicate().apply { position(0); limit(1) }

                    inputs["input_ids"] = longTensor(inputBuf, longArrayOf(1, sequenceLength.toLong()), owned)
                    inputs["attention_mask"] = longTensor(maskBuffer, longArrayOf(1, totalMaskLen.toLong()), owned)
                    inputs["position_ids"] = longTensor(posBuf, longArrayOf(1, sequenceLength.toLong()), owned)
                    inputs["logits_to_keep"] = longTensor(logitsToKeepBuf, longArrayOf(1), owned)

                    // Conv state cache (13 layers) - zero heap allocation via direct buffers
                    for (layer in CONV_LAYERS) {
                        inputs["past_conv.$layer"] = floatTensor(
                            convPool.get(layer),
                            longArrayOf(1, CONV_CHANNELS.toLong(), CONV_KERNEL.toLong()), owned
                        )
                    }
                    // KV attention cache (11 layers) - zero heap allocation via direct buffers
                    for (layer in KV_LAYERS) {
                        val k = "past_key_values.$layer.key"
                        val v = "past_key_values.$layer.value"
                        inputs[k] = floatTensor(
                            kvPool.getKey(layer).wrap(pastLen),
                            longArrayOf(1, KV_HEADS.toLong(), pastLen.toLong(), HEAD_DIM.toLong()), owned
                        )
                        inputs[v] = floatTensor(
                            kvPool.getValue(layer).wrap(pastLen),
                            longArrayOf(1, KV_HEADS.toLong(), pastLen.toLong(), HEAD_DIM.toLong()), owned
                        )
                    }

                    val prepNs = System.nanoTime()
                    s.run(inputs, runOptions).use { result ->
                        val runNs = System.nanoTime()
                        val outBindings = outputBindings ?: SessionOutputBindings(s).also { outputBindings = it }
                        val logits = result[outBindings.logitsIndex] as OnnxTensor
                        var nextToken = argmaxLastPosition(logits)
                        generatedIds += nextToken

                        // Update past cache from output tensors directly into pooled direct buffers
                        val newTotalLen = pastLen + sequenceLength
                        for (layer in CONV_LAYERS) {
                            val t = result[outBindings.convIndices[layer]] as OnnxTensor
                            val dst = convPool.get(layer)
                            val src = t.floatBuffer
                            src.position(0)
                            dst.position(0)
                            dst.put(src)
                            dst.clear()
                        }
                        for (layer in KV_LAYERS) {
                            val keyTensor = result[outBindings.kvKeyIndices[layer]] as OnnxTensor
                            val valTensor = result[outBindings.kvValIndices[layer]] as OnnxTensor
                            kvPool.getKey(layer).updateFrom(keyTensor, newTotalLen)
                            kvPool.getValue(layer).updateFrom(valTensor, newTotalLen)
                        }
                        val extractNs = System.nanoTime()

                        // Incremental streaming text decode
                        val streamText = streamDecoder.append(nextToken)
                        lastDecodeText = if (streamText.startsWith("<think>")) streamText else (THINKING_PROMPT_PREFIX + streamText)
                        onToken?.invoke(lastDecodeText)
                        val decodeNs = System.nanoTime()

                        val prepMs = (prepNs - stepT0) / 1_000_000.0
                        val runMs = (runNs - prepNs) / 1_000_000.0
                        val extractMs = (extractNs - runNs) / 1_000_000.0
                        val decMs = (decodeNs - extractNs) / 1_000_000.0
                        val totalStepMs = (decodeNs - stepT0) / 1_000_000.0
                        stepLatencies.add(totalStepMs)

                        OpTrace.record("lfm.step.prep", "host", prepMs)
                        OpTrace.record("lfm.step.run", "device", runMs, mapOf("step" to step.toString(), "seqLen" to sequenceLength.toString(), "pastLen" to pastLen.toString()))
                        OpTrace.record("lfm.step.extract", "host", extractMs)
                        OpTrace.record("lfm.step.decode", "postprocess", decMs)
                        OpTrace.record("lfm.step.total", "pipeline", totalStepMs)

                        if (step == 0 || step % 16 == 0 || nextToken == LfmTokenizer.EOS_ID) {
                            val currentToks = generatedIds.size
                            val totalElapsedMs = (System.nanoTime() - startedNs) / 1_000_000
                            val speed = if (totalElapsedMs > 0) String.format("%.1f", currentToks * 1000.0 / totalElapsedMs) else "0.0"
                            val heapMb = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024)
                            LfmChatLogger.info(
                                "STEP step=%4d tok=%6d seqLen=%3d pastLen=%4d | prep=%4.1fms run=%4.1fms ext=%4.1fms dec=%4.1fms total=%5.1fms | speed=%4s tok/s heap=%3dMB".format(
                                    step, nextToken, sequenceLength, pastLen, prepMs, runMs, extractMs, decMs, totalStepMs, speed, heapMb
                                )
                            )
                        }

                        // Fast probe before scanning full text for tool call tags
                        val activeSub = if (lastExecutedToolEnd < lastDecodeText.length) lastDecodeText.substring(lastExecutedToolEnd) else ""
                        val hasToolCallEnd = activeSub.isNotEmpty() &&
                                (activeSub.endsWith('>') || activeSub.takeLast(32).contains("</")) &&
                                (
                                    activeSub.contains("</tool_call>") ||
                                    activeSub.contains("</run_javascript>") ||
                                    (!activeSub.contains("<tool_call") && activeSub.contains("</function>")) ||
                                    (activeSub.contains("</function>") && activeSub.substringAfterLast("</function>").length > 20)
                                )
                        if (enableTools && toolExecutionCount < maxToolCalls && hasToolCallEnd) {
                            val toolCall = LfmToolCallParser.findNextToolCall(lastDecodeText, fromIndex = lastExecutedToolEnd)
                            if (toolCall != null && toolCall.code.isNotBlank()) {
                                toolExecutionCount++
                                lastExecutedToolEnd = toolCall.endIndex
                                LfmChatLogger.info("TOOL_CALL_DETECTED tool=${toolCall.name} codeChars=${toolCall.code.length} iteration=$toolExecutionCount")

                                onToken?.invoke(lastDecodeText + "\n[⚙️ JavaScript 运行时正在执行并测试代码...]\n")

                                val execResult = OpTrace.time("lfm.tool.exec", "tools") {
                                    JsRuntime.execute(toolCall.code)
                                }
                                LfmChatLogger.info("TOOL_CALL_RESULT success=${execResult.success} timeMs=${execResult.executionTimeMs}ms error=${execResult.error}")

                                val responseText = LfmToolCallParser.formatToolResponse(execResult)
                                val respIds = tok.encode(responseText)
                                if (respIds.isNotEmpty()) {
                                    generatedIds += respIds.toList()

                                    // Batch Prefill: Ingest all tool response tokens into model KV cache in a single forward pass
                                    val respLen = respIds.size
                                    val currentPast = pastLen + sequenceLength
                                    val feedInputs = LinkedHashMap<String, OnnxTensor>(64)
                                    val feedOwned = ArrayList<OnnxTensor>(48)
                                    try {
                                        val feedMaskLen = currentPast + respLen
                                        val feedMaskBuffer = if (feedMaskLen <= staticAllOnesBuffer.capacity()) {
                                            val b = staticAllOnesBuffer.duplicate()
                                            b.position(0)
                                            b.limit(feedMaskLen)
                                            b
                                        } else {
                                            LongBuffer.wrap(LongArray(feedMaskLen) { 1L })
                                        }
                                        val feedPosBuf = if (respLen <= posDirectBuffer.capacity()) {
                                            posDirectBuffer.clear()
                                            for (i in 0 until respLen) posDirectBuffer.put((currentPast + i).toLong())
                                            posDirectBuffer.flip()
                                            posDirectBuffer.duplicate()
                                        } else {
                                            LongBuffer.wrap(LongArray(respLen) { (currentPast + it).toLong() })
                                        }
                                        val feedInputBuf = if (respLen <= inputIdDirectBuffer.capacity()) {
                                            inputIdDirectBuffer.clear()
                                            for (id in respIds) inputIdDirectBuffer.put(id.toLong())
                                            inputIdDirectBuffer.flip()
                                            inputIdDirectBuffer.duplicate()
                                        } else {
                                            LongBuffer.wrap(respIds.map { it.toLong() }.toLongArray())
                                        }
                                        val feedLogitsToKeep = logitsToKeepBuffer.duplicate().apply { position(0); limit(1) }

                                        feedInputs["input_ids"] = longTensor(feedInputBuf, longArrayOf(1, respLen.toLong()), feedOwned)
                                        feedInputs["attention_mask"] = longTensor(feedMaskBuffer, longArrayOf(1, feedMaskLen.toLong()), feedOwned)
                                        feedInputs["position_ids"] = longTensor(feedPosBuf, longArrayOf(1, respLen.toLong()), feedOwned)
                                        feedInputs["logits_to_keep"] = longTensor(feedLogitsToKeep, longArrayOf(1), feedOwned)

                                        for (l in CONV_LAYERS) {
                                            feedInputs["past_conv.$l"] = floatTensor(
                                                convPool.get(l),
                                                longArrayOf(1, CONV_CHANNELS.toLong(), CONV_KERNEL.toLong()), feedOwned
                                            )
                                        }
                                        for (l in KV_LAYERS) {
                                            feedInputs["past_key_values.$l.key"] = floatTensor(
                                                kvPool.getKey(l).wrap(currentPast),
                                                longArrayOf(1, KV_HEADS.toLong(), currentPast.toLong(), HEAD_DIM.toLong()), feedOwned
                                            )
                                            feedInputs["past_key_values.$l.value"] = floatTensor(
                                                kvPool.getValue(l).wrap(currentPast),
                                                longArrayOf(1, KV_HEADS.toLong(), currentPast.toLong(), HEAD_DIM.toLong()), feedOwned
                                            )
                                        }

                                        val toolPrefillStartNs = System.nanoTime()
                                        s.run(feedInputs, runOptions).use { feedRes ->
                                            val updatedPastLen = currentPast + respLen
                                            for (l in CONV_LAYERS) {
                                                val t = feedRes[outBindings.convIndices[l]] as OnnxTensor
                                                val dst = convPool.get(l)
                                                val src = t.floatBuffer
                                                src.position(0)
                                                dst.position(0)
                                                dst.put(src)
                                                dst.clear()
                                            }
                                            for (l in KV_LAYERS) {
                                                val keyTensor = feedRes[outBindings.kvKeyIndices[l]] as OnnxTensor
                                                val valTensor = feedRes[outBindings.kvValIndices[l]] as OnnxTensor
                                                kvPool.getKey(l).updateFrom(keyTensor, updatedPastLen)
                                                kvPool.getValue(l).updateFrom(valTensor, updatedPastLen)
                                            }
                                            val feedLogits = feedRes[outBindings.logitsIndex] as OnnxTensor
                                            nextToken = argmaxLastPosition(feedLogits)
                                        }
                                        val toolPrefillMs = (System.nanoTime() - toolPrefillStartNs) / 1_000_000.0
                                        LfmChatLogger.info("TOOL_BATCH_PREFILL tokens=$respLen timeMs=${String.format("%.1f", toolPrefillMs)}ms speed=${String.format("%.1f", respLen * 1000.0 / toolPrefillMs.coerceAtLeast(1.0))}tok/s")
                                        OpTrace.record("lfm.tool.batch_prefill", "pipeline", toolPrefillMs, mapOf("tokens" to respLen.toString()))
                                        pastLen += respLen
                                    } finally {
                                        feedOwned.forEach { runCatching { it.close() } }
                                    }

                                    streamDecoder.appendAll(respIds.toList())
                                    val fullDecoded = streamDecoder.text()
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
        } finally {
            runCatching {
                if (wakeLock?.isHeld == true) {
                    wakeLock.release()
                }
            }
            runOptions.close()
        }

        val totalElapsedMs = (System.nanoTime() - startedNs) / 1_000_000
        val sortedLatencies = stepLatencies.sorted()
        val minStepMs = sortedLatencies.firstOrNull() ?: 0.0
        val maxStepMs = sortedLatencies.lastOrNull() ?: 0.0
        val p50StepMs = if (sortedLatencies.isNotEmpty()) sortedLatencies[sortedLatencies.size / 2] else 0.0
        val p90StepMs = if (sortedLatencies.isNotEmpty()) sortedLatencies[(sortedLatencies.size * 0.9).toInt().coerceAtMost(sortedLatencies.size - 1)] else 0.0
        val p99StepMs = if (sortedLatencies.isNotEmpty()) sortedLatencies[(sortedLatencies.size * 0.99).toInt().coerceAtMost(sortedLatencies.size - 1)] else 0.0

        val promptTokensCount = promptIds.size
        val generatedTokensCount = generatedIds.size
        val prefillMs = stepLatencies.firstOrNull() ?: 0.0
        val decodeElapsedMs = (totalElapsedMs - prefillMs).coerceAtLeast(0.0)
        val decodeTokens = (generatedTokensCount - 1).coerceAtLeast(0)
        val prefillSpeed = if (prefillMs > 0) String.format("%.1f", promptTokensCount * 1000.0 / prefillMs) else "0.0"
        val decodeSpeed = if (decodeElapsedMs > 0) String.format("%.1f", decodeTokens * 1000.0 / decodeElapsedMs) else "0.0"
        val avgSpeed = if (totalElapsedMs > 0) String.format("%.1f", generatedTokensCount * 1000.0 / totalElapsedMs) else "0.0"

        LfmChatLogger.info(
            "GENERATE_SUMMARY totalTokens=$generatedTokensCount promptTokens=$promptTokensCount elapsedMs=${totalElapsedMs}ms " +
            "ttftMs=${String.format("%.1f", prefillMs)} prefillSpeed=${prefillSpeed}tok/s " +
            "decodeSpeed=${decodeSpeed}tok/s avgSpeed=${avgSpeed}tok/s " +
            "latency=[min=%.1f, p50=%.1f, p90=%.1f, p99=%.1f, max=%.1f]ms toolCalls=$toolExecutionCount stopReason=$stopReason".format(
                minStepMs, p50StepMs, p90StepMs, p99StepMs, maxStepMs
            )
        )
        OpTrace.endRun(runId, 0.0, mapOf(
            "total_tokens" to generatedTokensCount.toString(),
            "elapsed_ms" to totalElapsedMs.toString(),
            "avg_speed" to avgSpeed,
            "stop_reason" to stopReason
        ))
        return lastDecodeText
    }

    fun stop() = synchronized(lock) {
        runCatching { session?.close() }
        session = null
        outputBindings = null
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
        b.position(b.limit() - vocab)
        var best = Float.NEGATIVE_INFINITY
        var bestIndex = 0
        var i = 0
        while (i < vocab) {
            val v = b.get()
            if (v > best) {
                best = v
                bestIndex = i
            }
            i++
        }
        return bestIndex
    }

    private fun longTensor(data: LongArray, shape: LongArray, owned: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env!!, LongBuffer.wrap(data), shape).also { owned += it }

    private fun longTensor(buffer: LongBuffer, shape: LongArray, owned: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env!!, buffer, shape).also { owned += it }

    private fun floatTensor(buffer: FloatBuffer, shape: LongArray, owned: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env!!, buffer, shape).also { owned += it }

    private fun floatTensor(data: FloatArray, shape: LongArray, owned: MutableList<OnnxTensor>) =
        OnnxTensor.createTensor(env!!, FloatBuffer.wrap(data), shape).also { owned += it }
}

typealias QnnLfm2ChatRunner = QnnLfmChatRunner
