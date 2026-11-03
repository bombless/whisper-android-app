package io.github.bombless.htp_playpen.tts

import android.icu.text.Transliterator
import java.text.Normalizer
import java.util.Locale

/** Converts Han characters to the phonetic scripts expected by Kokoro's G2P frontends. */
internal object KokoroTextNormalizer {
    private val hanToPinyin: Transliterator by lazy { Transliterator.getInstance("Han-Latin") }
    private val anyToHiragana: Transliterator by lazy { Transliterator.getInstance("Any-Hiragana") }

    fun chinese(text: String): String {
        val markedPinyin = hanToPinyin.transliterate(text)
        val numbered = markedPinyinToNumbers(markedPinyin)
        require(numbered.isNotBlank()) { "中文文本未能转换为拼音" }
        return numbered
    }

    fun japanese(text: String): String {
        val hiragana = anyToHiragana.transliterate(text)
        val stillKanji = hiragana.codePoints().anyMatch { cp ->
            Character.UnicodeScript.of(cp) == Character.UnicodeScript.HAN
        }
        require(!stillKanji) {
            "日文中有无法自动转换读音的汉字，请将该词改为平假名/片假名后重试"
        }
        require(hiragana.any { it in '\u3040'..'\u30ff' }) {
            "日文前端没有得到假名，请检查输入文本或显式填写假名读音"
        }
        return hiragana
    }

    private fun markedPinyinToNumbers(input: String): String {
        val decomposed = Normalizer.normalize(input.lowercase(Locale.ROOT), Normalizer.Form.NFD)
        val result = StringBuilder(decomposed.length + 8)
        val syllable = StringBuilder()
        var tone = 0

        fun flushSyllable() {
            if (syllable.isNotEmpty()) {
                result.append(syllable)
                if (tone in 1..4) result.append(tone)
                syllable.setLength(0)
                tone = 0
            }
        }

        var index = 0
        while (index < decomposed.length) {
            val cp = decomposed.codePointAt(index)
            index += Character.charCount(cp)
            when (cp) {
                0x0304 -> tone = 1 // macron
                0x0301 -> tone = 2 // acute
                0x030C -> tone = 3 // caron
                0x0300 -> tone = 4 // grave
                0x0308 -> if (syllable.isNotEmpty() && syllable.last() == 'u') {
                    syllable.setCharAt(syllable.lastIndex, 'v') // ü -> v in numbered Pinyin
                }
                else -> when {
                    Character.isLetterOrDigit(cp) -> syllable.appendCodePoint(cp)
                    Character.isWhitespace(cp) -> {
                        flushSyllable()
                        if (result.isNotEmpty() && result.last() != ' ') result.append(' ')
                    }
                    cp == 0x3001 || cp == 0xFF0C || cp == ','.code || cp == ';'.code || cp == 0xFF1B -> {
                        flushSyllable(); result.append(',')
                    }
                    cp == 0x3002 || cp == 0xFF0E || cp == '.'.code -> {
                        flushSyllable(); result.append('.')
                    }
                    cp == 0xFF01 || cp == '!'.code -> { flushSyllable(); result.append('!') }
                    cp == 0xFF1F || cp == '?'.code -> { flushSyllable(); result.append('?') }
                    else -> flushSyllable()
                }
            }
        }
        flushSyllable()
        return result.toString().replace(Regex("\\s+"), " ").trim()
    }
}