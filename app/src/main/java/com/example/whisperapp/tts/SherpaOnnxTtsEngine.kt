package com.example.whisperapp.tts

import android.content.Context
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import kotlin.math.roundToInt

/** Direct offline sherpa-onnx TTS. No Android TextToSpeech or Qualcomm TTS SDK. */
class SherpaOnnxTtsEngine(
    context: Context,
    private val variant: Variant = Variant.PIPER_XIAO_YA_INT8,
) : TtsEngine {
    enum class Variant {
        VITS_ZH_LL, PIPER_XIAO_YA_INT8;
        val displayName: String get() = when (this) {
            VITS_ZH_LL -> "VITS 中文（5 声线）"
            PIPER_XIAO_YA_INT8 -> "Piper 小雅（int8）"
        }
    }

    private val assets = context.applicationContext.assets
    @Volatile private var tts: OfflineTts? = null
    @Volatile private var closed = false

    private fun create(): OfflineTts {
        val model = OfflineTtsModelConfig()
        val ruleFsts: String
        when (variant) {
            Variant.VITS_ZH_LL -> {
                val d = "models/tts/vits_zh_ll"
                model.vits = OfflineTtsVitsModelConfig().apply {
                    this.model = "$d/model.onnx"
                    lexicon = "$d/lexicon.txt"
                    tokens = "$d/tokens.txt"
                }
                ruleFsts = "$d/phone.fst,$d/date.fst,$d/number.fst"
            }
            Variant.PIPER_XIAO_YA_INT8 -> {
                val d = "models/tts/piper_xiao_ya_int8"
                model.vits = OfflineTtsVitsModelConfig().apply {
                    this.model = "$d/zh_CN-xiao_ya-medium.onnx"
                    lexicon = "$d/lexicon.txt"
                    tokens = "$d/tokens.txt"
                }
                ruleFsts = "$d/phone.fst,$d/date.fst,$d/number.fst"
            }
        }
        model.numThreads = 2
        return OfflineTts(assets, OfflineTtsConfig(model = model, ruleFsts = ruleFsts))
    }

    override suspend fun prepare() {
        check(!closed) { "TTS engine is closed: $variant" }
        synchronized(this) {
            check(!closed) { "TTS engine is closed: $variant" }
            if (tts == null) tts = create()
        }
    }

    override suspend fun synthesize(text: String): TtsResult {
        check(!closed) { "TTS engine is closed: $variant" }
        val start = System.currentTimeMillis()
        val engine = tts ?: synchronized(this) {
            check(!closed) { "TTS engine is closed: $variant" }
            tts ?: create().also { tts = it }
        }
        val audio = engine.generate(text, 0, 1.0f)
        val samples = audio.samples
        val pcm = ShortArray(samples.size) { i ->
            (samples[i].coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt().toShort()
        }
        return TtsResult(pcm, audio.sampleRate, System.currentTimeMillis() - start)
    }

    override fun close() {
        synchronized(this) {
            closed = true
            tts?.release()
            tts = null
        }
    }
}
