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
    private const val MODEL_ASSET = "chat/models/LFM2.5-230M-ONNX/onnx/model_q4.onnx"
    private const val DATA_PREFIX = "chat/models/LFM2.5-230M-ONNX/onnx/model_q4.onnx_data.part"
    /**
     * Name the graph itself declares for its external data (`external_data.location` of every
     * external initializer). ORT resolves it relative to the model file, so the concatenated
     * chunks must land under exactly this name — not after the .onnx asset name.
     */
    private const val EXTERNAL_DATA_NAME = "model_q4f32.onnx_data"
    private const val MAX_TOKENS = 256
    private val CONV_LAYERS = intArrayOf(0, 1, 3, 5, 7, 9, 11, 13)
    private val KV_LAYERS = intArrayOf(2, 4, 6, 8, 10, 12)

    data class Status(val ready: Boolean, val message: String)

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var tokenizer: Lfm2Tokenizer? = null
    private val lock = Any()

    fun start(context: Context): Status = synchronized(lock) {
        try {
            val app = context.applicationContext
            val model = copyAsset(app, MODEL_ASSET)
            copyExternalData(app, model.parentFile ?: app.cacheDir)
            if (env == null) env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, TAG)
            // The environment (and therefore the QNN EP registration) is shared with the Whisper
            // ASR runner, which keeps its registration alive after recording stops; the gate below
            // is idempotent and also re-registers if a smoke-test activity removed the provider.
            QnnEpRegistration.ensureRegistered(env!!)
            if (session == null) {
                val device = env!!.epDevices.firstOrNull { it.epName == EP }
                    ?: error("QNNExecutionProvider device not exposed")
                val backend = File(app.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
                check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
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
                session = env!!.createSession(model.absolutePath, options)
                options.close()
                Log.i(TAG, "session created with QNN HTP inputs=${session!!.inputNames.size} outputs=${session!!.outputNames.size}")
            }
            tokenizer = tokenizer ?: Lfm2Tokenizer.fromAssets(app.assets)
            Status(true, "LFM2.5 已就绪 · QNN HTP")
        } catch (t: Throwable) {
            Log.e(TAG, "init failed", t)
            Status(false, "QNN 加载失败：${t.message ?: t::class.java.simpleName}")
        }
    }

    fun generate(context: Context, history: List<Pair<String, String>>): String = synchronized(lock) {
        val status = start(context)
        check(status.ready) { status.message }
        val tok = tokenizer!!
        val s = session!!
        val promptIds = tok.encode(buildPrompt(history)).toMutableList()
        require(promptIds.isNotEmpty()) { "Prompt tokenization produced no tokens" }
        val startedNs = System.nanoTime()
        Log.i(TAG, "GENERATE_START history=${history.size} promptTokens=${promptIds.size} firstIds=${promptIds.take(4)}")

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

                s.run(inputs).use { result ->
                    val outputs = s.outputNames.associateWith { name -> result[s.outputNames.indexOf(name)] as OnnxTensor }
                    val logits = outputs.getValue("logits")
                    val nextToken = argmaxLastPosition(logits)
                    generatedIds += nextToken

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

    private fun copyAsset(context: Context, asset: String, dir: File = context.cacheDir): File {
        val target = File(dir, asset.substringAfterLast('/'))
        context.assets.open(asset).use { input ->
            val size = input.available().toLong()
            if (!target.isFile || target.length() != size) {
                target.outputStream().use { input.copyTo(it) }
            }
        }
        return target
    }

    private fun copyExternalData(context: Context, dir: File): File {
        val target = File(dir, EXTERNAL_DATA_NAME)
        val assetDir = DATA_PREFIX.substringBeforeLast('/')
        val prefix = DATA_PREFIX.substringAfterLast('/')
        val names = context.assets.list(assetDir)?.filter { it.startsWith(prefix) }?.sorted() ?: emptyList()
        check(names.isNotEmpty()) { "Missing LFM2 external-data chunks" }
        val expected = names.sumOf { name -> context.assets.open("$assetDir/$name").use { it.available().toLong() } }
        if (target.isFile && target.length() == expected) return target
        target.outputStream().use { out ->
            for (name in names) context.assets.open("$assetDir/$name").use { it.copyTo(out, 1024 * 1024) }
        }
        return target
    }
}
