package com.example.whisperapp.qnn

import org.json.JSONObject
import java.io.File
import java.text.Normalizer
import java.util.Locale

internal class SdxlTokenizer internal constructor(
    private val vocabulary: Map<String, Int>,
    private val mergeRanks: Map<Pair<String, String>, Int>,
) {
    fun encode(prompt: String, padToken: Int): IntArray {
        val normalized = Normalizer.normalize(prompt, Normalizer.Form.NFC)
            .replace(whitespace, " ")
            .lowercase(Locale.ROOT)
        val ids = ArrayList<Int>(MAX_LENGTH)
        ids.add(START_TOKEN)
        for (match in pattern.findAll(normalized)) {
            val pieces = match.value.toByteArray(Charsets.UTF_8).map { byteEncoder[it.toInt() and 255] }.toMutableList()
            if (pieces.isEmpty()) continue
            pieces[pieces.lastIndex] += "</w>"
            while (pieces.size > 1) {
                var lowestRank = Int.MAX_VALUE
                var pair: Pair<String, String>? = null
                for (index in 0 until pieces.lastIndex) {
                    val candidate = pieces[index] to pieces[index + 1]
                    val rank = mergeRanks[candidate] ?: continue
                    if (rank < lowestRank) {
                        lowestRank = rank
                        pair = candidate
                    }
                }
                val selected = pair ?: break
                var index = 0
                while (index < pieces.lastIndex) {
                    if (pieces[index] == selected.first && pieces[index + 1] == selected.second) {
                        pieces[index] += pieces[index + 1]
                        pieces.removeAt(index + 1)
                    }
                    index++
                }
            }
            for (piece in pieces) {
                if (ids.size >= MAX_LENGTH - 1) break
                ids.add(vocabulary[piece] ?: error("CLIP vocabulary missing token: $piece"))
            }
            if (ids.size >= MAX_LENGTH - 1) break
        }
        ids.add(END_TOKEN)
        return IntArray(MAX_LENGTH) { index -> ids.getOrElse(index) { padToken } }
    }

    companion object {
        private const val MAX_LENGTH = 77
        private const val START_TOKEN = 49406
        private const val END_TOKEN = 49407
        private val whitespace = Regex("[\\s\\p{Z}]+")
        private val pattern = Regex("'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{Z}\\p{L}\\p{N}]+")
        private val byteEncoder = Array(256) { "" }.also { result ->
            val visible = (33..126) + (161..172) + (174..255)
            val visibleSet = visible.toSet()
            var extra = 0
            for (value in 0..255) {
                val codePoint = if (value in visibleSet) value else 256 + extra++
                result[value] = codePoint.toChar().toString()
            }
        }

        fun fromFile(file: File): SdxlTokenizer {
            val model = JSONObject(file.readText(Charsets.UTF_8)).getJSONObject("model")
            val vocab = model.getJSONObject("vocab")
            val vocabulary = HashMap<String, Int>(vocab.length())
            val keys = vocab.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                vocabulary[key] = vocab.getInt(key)
            }
            val merges = model.getJSONArray("merges")
            val mergeRanks = HashMap<Pair<String, String>, Int>(merges.length())
            for (index in 0 until merges.length()) {
                val entry = merges.getJSONArray(index)
                mergeRanks[entry.getString(0) to entry.getString(1)] = index
            }
            return SdxlTokenizer(vocabulary, mergeRanks)
        }
    }
}
