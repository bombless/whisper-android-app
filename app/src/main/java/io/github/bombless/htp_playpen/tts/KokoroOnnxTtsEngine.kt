package io.github.bombless.htp_playpen.tts

import android.content.Context
import audio.soniqo.speech.SpeechSynthesizer
import audio.soniqo.speech.SpeechSynthesizerConfig
import audio.soniqo.speech.TtsModel
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Kokoro 82M multilingual ONNX adapter for the app's stable TtsEngine contract. */
internal class KokoroOnnxTtsEngine internal constructor(
    context: Context,
    private val onProgress: (KokoroModelRepository.Progress) -> Unit = {},
) : TtsEngine {
    private val appContext = context.applicationContext
    private val repository = KokoroModelRepository(appContext)
    private val prepareMutex = Mutex()

    @Volatile private var synthesizer: SpeechSynthesizer? = null
    @Volatile private var closed = false
    @Volatile private var languagePreference: String = LANGUAGE_AUTO
    @Volatile private var voicePreference: String = VOICE_AUTO

    fun setLanguagePreference(language: String) {
        require(language in setOf(LANGUAGE_AUTO, LANGUAGE_CHINESE, LANGUAGE_JAPANESE))
        languagePreference = language
    }

    fun setVoicePreference(voice: String) {
        require(voice in setOf(VOICE_AUTO, VOICE_JAPANESE, VOICE_CHINESE))
        voicePreference = voice
    }

    override suspend fun prepare() {
        check(!closed) { "Kokoro TTS engine is closed" }
        if (synthesizer != null) return
        prepareMutex.withLock {
            check(!closed) { "Kokoro TTS engine is closed" }
            if (synthesizer != null) return
            val modelDir = repository.ensureModels(onProgress)
            check(!closed) { "Kokoro TTS engine was closed while downloading its model" }
            val candidate = withContext(Dispatchers.IO) {
                SpeechSynthesizer(
                    SpeechSynthesizerConfig(
                        modelDir = modelDir,
                        useNnapi = false,
                        ttsModel = TtsModel.KOKORO_SHORT_TURN,
                    ),
                )
            }
            synchronized(this) {
                if (closed) {
                    candidate.close()
                    error("Kokoro TTS engine was closed while initializing its native session")
                }
                synthesizer = candidate
            }
        }
    }

    override suspend fun synthesize(text: String): TtsResult {
        check(!closed) { "Kokoro TTS engine is closed" }
        if (synthesizer == null) prepare()
        val engine = synthesizer ?: error("Kokoro synthesizer failed to initialize")
        val startedAt = System.nanoTime()
        val segments = splitSentences(text)
        require(segments.isNotEmpty()) { "请输入要朗读的文字" }

        val chunks = ArrayList<ShortArray>(segments.size)
        var sampleRate = 0
        for (segment in segments) {
            val language = resolveLanguage(segment, languagePreference)
            val phoneticText = when (language) {
                LANGUAGE_CHINESE -> KokoroTextNormalizer.chinese(segment)
                LANGUAGE_JAPANESE -> KokoroTextNormalizer.japanese(segment)
                else -> error("Unsupported Kokoro language: $language")
            }
            val voice = when (voicePreference) {
                VOICE_JAPANESE -> if (language == LANGUAGE_JAPANESE) VOICE_JAPANESE else VOICE_CHINESE
                VOICE_CHINESE -> if (language == LANGUAGE_CHINESE) VOICE_CHINESE else VOICE_JAPANESE
                else -> if (language == LANGUAGE_JAPANESE) VOICE_JAPANESE else VOICE_CHINESE
            }
            val result = engine.synthesize(phoneticText, language, voice)
            check(result.pcm16.isNotEmpty()) { "Kokoro returned empty audio for $language" }
            check(result.pcm16.size % 2 == 0) { "Kokoro returned malformed PCM16 data" }
            if (sampleRate == 0) sampleRate = result.sampleRate
            check(result.sampleRate == sampleRate) { "Kokoro sample rate changed between language segments" }
            val shortBuffer = ByteBuffer.wrap(result.pcm16).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            chunks += ShortArray(shortBuffer.remaining()).also(shortBuffer::get)
        }

        val silenceSamples = (sampleRate * INTER_SEGMENT_SILENCE_MS / 1000).coerceAtLeast(0)
        val totalSamples = chunks.sumOf { it.size } + silenceSamples * (chunks.size - 1).coerceAtLeast(0)
        val pcm = ShortArray(totalSamples)
        var offset = 0
        chunks.forEachIndexed { index, chunk ->
            chunk.copyInto(pcm, offset)
            offset += chunk.size
            if (index != chunks.lastIndex) offset += silenceSamples
        }
        return TtsResult(pcm, sampleRate, (System.nanoTime() - startedAt) / 1_000_000L)
    }

    override fun close() {
        synchronized(this) {
            closed = true
            synthesizer?.close()
            synthesizer = null
        }
    }

    private fun resolveLanguage(text: String, preference: String): String = when (preference) {
        LANGUAGE_CHINESE -> LANGUAGE_CHINESE
        LANGUAGE_JAPANESE -> LANGUAGE_JAPANESE
        else -> if (text.any(::isJapaneseKana)) LANGUAGE_JAPANESE else LANGUAGE_CHINESE
    }

    private fun isJapaneseKana(char: Char): Boolean =
        char in '\u3040'..'\u309f' || char in '\u30a0'..'\u30ff'

    private fun splitSentences(text: String): List<String> = text
        .replace(Regex("\\s+"), " ")
        .trim()
        .split(Regex("(?<=[。！？!?；;\\n])"))
        .map(String::trim)
        .filter(String::isNotEmpty)

    companion object {
        const val LANGUAGE_AUTO = "auto"
        const val LANGUAGE_CHINESE = "zh"
        const val LANGUAGE_JAPANESE = "ja"
        const val VOICE_AUTO = "auto"
        const val VOICE_JAPANESE = "jf_alpha"
        const val VOICE_CHINESE = "zf_xiaobei"
        private const val INTER_SEGMENT_SILENCE_MS = 90
    }
}