package com.example.whisperapp.qnn

import android.content.res.AssetManager
import org.json.JSONObject
import java.util.Locale
import java.util.regex.Pattern

/**
 * Small offline byte-level BPE tokenizer for LFM2.5.
 * It intentionally reads the Hugging Face tokenizer.json shipped with the app;
 * no JavaScript/WebView runtime is involved.
 */
class Lfm2Tokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val merges: Map<Pair<String, String>, Int>,
    private val special: Map<String, Int>,
) {
    private val splitPattern = Pattern.compile(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    )
    private val byteEncoder: Map<Int, Char> = buildByteEncoder()

    fun encode(text: String): IntArray {
        val ids = ArrayList<Int>()
        val matcher = splitPattern.matcher(text)
        while (matcher.find()) {
            val token = matcher.group()
            val symbols = token.toByteArray(Charsets.UTF_8).map { byteEncoder[it.toInt() and 0xff].toString() }.toMutableList()
            while (symbols.size > 1) {
                var best: Pair<String, String>? = null
                var bestRank = Int.MAX_VALUE
                for (i in 0 until symbols.size - 1) {
                    val pair = symbols[i] to symbols[i + 1]
                    val rank = merges[pair] ?: continue
                    if (rank < bestRank) { bestRank = rank; best = pair }
                }
                if (best == null) break
                val merged = best.first + best.second
                var i = 0
                while (i < symbols.size - 1) {
                    if (symbols[i] == best.first && symbols[i + 1] == best.second) {
                        symbols[i] = merged
                        symbols.removeAt(i + 1)
                    } else i++
                }
            }
            for (s in symbols) ids += vocab[s] ?: error("LFM tokenizer token missing from vocab: $s")
        }
        return ids.toIntArray()
    }

    fun decode(ids: IntArray): String {
        val inverse = inverseVocab
        val bytes = ArrayList<Byte>()
        for (id in ids) {
            if (id == EOS_ID || id < 0) continue
            val token = inverse[id] ?: continue
            for (ch in token) bytes += byteDecoder[ch]?.toByte() ?: continue
        }
        return bytes.toByteArray().toString(Charsets.UTF_8)
            .replace("<|im_end|>", "")
            .trim()
    }

    fun specialId(token: String): Int = special[token] ?: vocab[token] ?: error("Missing special token $token")

    private val inverseVocab: Map<Int, String> by lazy { vocab.entries.associate { it.value to it.key } }
    private val byteDecoder: Map<Char, Int> by lazy { byteEncoder.entries.associate { it.value to it.key } }

    companion object {
        const val EOS_ID = 7
        fun fromAssets(assets: AssetManager, base: String = "chat/models/LFM2.5-230M-ONNX/tokenizer.json"): Lfm2Tokenizer {
            val root = JSONObject(assets.open(base).bufferedReader().use { it.readText() })
            val model = root.getJSONObject("model")
            val vocabJson = model.getJSONObject("vocab")
            val vocab = HashMap<String, Int>(vocabJson.length())
            vocabJson.keys().forEach { vocab[it] = vocabJson.getInt(it) }
            val mergeMap = HashMap<Pair<String, String>, Int>()
            val merges = model.getJSONArray("merges")
            for (i in 0 until merges.length()) {
                val raw = merges.get(i)
                val p = if (raw is org.json.JSONArray) {
                    listOf(raw.getString(0), raw.getString(1))
                } else {
                    raw.toString().split(' ')
                }
                if (p.size == 2) mergeMap[p[0] to p[1]] = i
            }
            val special = HashMap<String, Int>()
            val added = root.optJSONArray("added_tokens")
            if (added != null) for (i in 0 until added.length()) {
                val o = added.getJSONObject(i)
                special[o.getString("content")] = o.getInt("id")
            }
            return Lfm2Tokenizer(vocab, mergeMap, special)
        }

        private fun buildByteEncoder(): Map<Int, Char> {
            val bs = ArrayList<Int>()
            for (i in 33..126) bs += i
            for (i in 161..172) bs += i
            for (i in 174..255) bs += i
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) if (b !in bs) { bs += b; cs += 256 + n; n++ }
            return bs.zip(cs).associate { it.first to it.second.toChar() }
        }
    }
}