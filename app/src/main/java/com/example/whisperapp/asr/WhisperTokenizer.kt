package com.example.whisperapp.asr

/** Offline token vocabulary used by the Whisper decoder integration. */
class WhisperTokenizer(private val idToToken: Map<Int, String>) {
    fun token(id: Int): String? = idToToken[id]

    fun decode(tokenIds: IntArray): String = buildString {
        for (id in tokenIds) {
            val token = idToToken[id] ?: continue
            if (token.startsWith("<|")) break
            append(token)
        }
    }.replace(Regex("\\s+"), " ").trim()

    companion object {
        fun fromIdMap(idToToken: Map<Int, String>): WhisperTokenizer = WhisperTokenizer(idToToken)
    }
}
