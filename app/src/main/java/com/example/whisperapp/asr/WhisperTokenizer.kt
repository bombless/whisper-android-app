package com.example.whisperapp.asr

import android.content.res.AssetManager
import java.nio.charset.StandardCharsets
import java.util.regex.Pattern

/** Offline Whisper GPT-2 byte-level BPE tokenizer used by the decoder integration. */
class WhisperTokenizer private constructor(
    private val tokenToIdMap: Map<String, Int>,
    private val idToToken: Map<Int, String>,
    private val mergeRanks: Map<Pair<String, String>, Int>,
) {
    private val byteEncoder: Map<Int, Char> = buildByteEncoder()
    private val byteDecoder: Map<Char, Int> = buildByteDecoder(byteEncoder)

    fun token(id: Int): String? = idToToken[id]

    /** Number of vocabulary entries, including special/added tokens. */
    val vocabularySize: Int get() = tokenToIdMap.size

    fun tokenToId(token: String): Int? = tokenToIdMap[token]

    /** Resolves a special token by name; variant-independent prompt/stop-token lookup. */
    fun idFor(token: String): Int =
        tokenToIdMap[token] ?: error("Tokenizer vocabulary missing token: $token")

    fun encode(text: String): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val ids = ArrayList<Int>()
        for (match in TOKEN_PATTERN.matcher(text).results().toList()) {
            val piece = match.group()
            val byteEncoded = piece.toByteArray(StandardCharsets.UTF_8).joinToString("") { b ->
                val unsigned = b.toInt() and 0xFF
                (byteEncoder[unsigned] ?: error("Missing byte encoder entry for $unsigned")).toString()
            }
            val merged = bpe(byteEncoded)
            for (token in merged.split(" ")) {
                val id = tokenToIdMap[token]
                    ?: error("Tokenizer vocabulary missing BPE token: $token")
                ids += id
            }
        }
        return ids.toIntArray()
    }

    /** Decode normal Whisper tokens while filtering special/control tokens. */
    fun decode(tokenIds: IntArray): String {
        val source = buildString {
            for (id in tokenIds) {
                val token = idToToken[id] ?: continue
                if (token.startsWith("<|")) continue
                append(token)
            }
        }
        val byteBuffer = ByteArray(source.length * 4)
        var offset = 0
        for (ch in source) {
            val byte = byteDecoder[ch]
                ?: error("Tokenizer byte decoder missing codepoint U+${ch.code.toString(16)}")
            byteBuffer[offset++] = byte.toByte()
        }
        return String(byteBuffer, 0, offset, StandardCharsets.UTF_8).trim()
    }

    private fun bpe(token: String): String {
        if (token.length <= 1) return token
        var symbols = token.map { it.toString() }.toMutableList()
        while (symbols.size > 1) {
            var best: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until symbols.size - 1) {
                val pair = symbols[i] to symbols[i + 1]
                val rank = mergeRanks[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    best = pair
                }
            }
            val pair = best ?: break
            val merged = ArrayList<String>(symbols.size)
            var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == pair.first && symbols[i + 1] == pair.second) {
                    merged += pair.first + pair.second
                    i += 2
                } else {
                    merged += symbols[i]
                    i++
                }
            }
            symbols = merged
        }
        return symbols.joinToString(" ")
    }

    companion object {
        private val TOKEN_PATTERN = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        )

        fun fromIdMap(idToToken: Map<Int, String>): WhisperTokenizer {
            val tokenToId = idToToken.entries.associate { (id, token) -> token to id }
            return WhisperTokenizer(tokenToId, idToToken, emptyMap())
        }

        fun fromAssets(
            assets: AssetManager,
            basePath: String = "models/whisper/tokenizer",
            expectedVocabSize: Int = 51865,
        ): WhisperTokenizer {
            return fromAssetTexts(
                readAsset(assets, "$basePath/vocab.json"),
                readAsset(assets, "$basePath/merges.txt"),
                readAsset(assets, "$basePath/added_tokens.json"),
                expectedVocabSize,
            )
        }

        internal fun fromAssetTexts(
            vocabText: String,
            mergesText: String,
            addedTokensText: String,
            expectedVocabSize: Int = 51865,
        ): WhisperTokenizer {
            val tokenToId = LinkedHashMap<String, Int>(50258)
            tokenToId.putAll(parseStringIntObject(vocabText))
            tokenToId.putAll(parseStringIntObject(addedTokensText))

            val merges = mergesText
                .lineSequence()
                .drop(1)
                .filter { it.isNotBlank() }
                .map { line ->
                    val parts = line.trim().split(Regex("\\s+"), limit = 2)
                    require(parts.size == 2) { "Invalid Whisper BPE merge: $line" }
                    parts[0] to parts[1]
                }
                .toList()
            val mergeRanks = merges.withIndex().associate { it.value to it.index }

            require(tokenToId.size == expectedVocabSize) {
                "Whisper tokenizer vocabulary must contain $expectedVocabSize entries, got ${tokenToId.size}"
            }
            // Base special tokens are stable across the multilingual variants.
            require(tokenToId["<|endoftext|>"] == 50257)
            require(tokenToId["<|startoftranscript|>"] == 50258)
            require(tokenToId["<|en|>"] == 50259)
            require(tokenToId["<|zh|>"] == 50260)
            // <|transcribe|>/<|notimestamps|> are NOT stable: Large-V3 inserts <|yue|>,
            // which shifts them by one. Require presence only; callers resolve IDs by name.
            require(tokenToId.containsKey("<|transcribe|>")) { "Tokenizer is missing <|transcribe|>" }
            require(tokenToId.containsKey("<|notimestamps|>")) { "Tokenizer is missing <|notimestamps|>" }

            val idToToken = tokenToId.entries.associate { (token, id) -> id to token }
            return WhisperTokenizer(tokenToId, idToToken, mergeRanks)
        }

        private fun parseStringIntObject(json: String): Map<String, Int> {
            var i = 0
            fun skipWs() { while (i < json.length && json[i].isWhitespace()) i++ }
            fun expect(ch: Char) {
                skipWs()
                require(i < json.length && json[i] == ch) { "Invalid JSON near index $i" }
                i++
            }
            fun readString(): String {
                skipWs()
                expect('"')
                val out = StringBuilder()
                while (i < json.length) {
                    val ch = json[i++]
                    if (ch == '"') return out.toString()
                    if (ch != '\\') {
                        out.append(ch)
                        continue
                    }
                    require(i < json.length) { "Invalid JSON escape" }
                    when (val esc = json[i++]) {
                        '"' -> out.append('"')
                        '\\' -> out.append('\\')
                        '/' -> out.append('/')
                        'b' -> out.append('\b')
                        'f' -> out.append('\u000C')
                        'n' -> out.append('\n')
                        'r' -> out.append('\r')
                        't' -> out.append('\t')
                        'u' -> {
                            require(i + 4 <= json.length) { "Invalid unicode escape" }
                            out.append(json.substring(i, i + 4).toInt(16).toChar())
                            i += 4
                        }
                        else -> error("Unsupported JSON escape: $esc")
                    }
                }
                error("Unterminated JSON string")
            }
            fun readInt(): Int {
                skipWs()
                val start = i
                if (i < json.length && (json[i] == '-' || json[i] == '+')) i++
                while (i < json.length && json[i].isDigit()) i++
                require(i > start) { "Expected integer near index $i" }
                return json.substring(start, i).toInt()
            }

            val result = LinkedHashMap<String, Int>()
            skipWs()
            expect('{')
            skipWs()
            if (i < json.length && json[i] == '}') {
                i++
                return result
            }
            while (true) {
                val key = readString()
                expect(':')
                result[key] = readInt()
                skipWs()
                if (i < json.length && json[i] == ',') {
                    i++
                    continue
                }
                expect('}')
                break
            }
            return result
        }

        private fun readAsset(assets: AssetManager, path: String): String =
            assets.open(path).bufferedReader(StandardCharsets.UTF_8).use { it.readText() }

        private fun buildByteDecoder(encoder: Map<Int, Char>): Map<Char, Int> {
            val decoder = HashMap<Char, Int>(encoder.size)
            for ((byte, char) in encoder) decoder[char] = byte
            require(decoder[288.toChar()] == 32) { "Whisper byte-level space marker must map U+0120 to byte 32" }
            return decoder
        }

        private fun buildByteEncoder(): Map<Int, Char> {
            val bytes = ArrayList<Int>()
            for (b in 33..126) bytes += b
            for (b in 161..172) bytes += b
            for (b in 174..255) bytes += b
            val encoder = LinkedHashMap<Int, Char>(256)
            for (b in bytes) encoder[b] = b.toChar()
            var extra = 0
            for (b in 0..255) {
                if (b !in encoder) {
                    encoder[b] = (256 + extra).toChar()
                    extra++
                }
            }
            return encoder
        }
    }
}
