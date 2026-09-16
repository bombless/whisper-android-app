package com.example.whisperapp.tts

object ChineseTextProcessor {
    private val controlChars = Regex("[\\u0000-\\u001F\\u007F]")
    private val spacesBeforePunctuation = Regex("\\s+([，。！？；：、])")
    private val spacesAfterPunctuation = Regex("([，。！？；：、])\\s+")
    private val repeatedPunctuation = Regex("([，。！？；：、])\\1+")

    fun normalizeChineseText(text: String): String = text
        .replace(controlChars, "")
        .replace(Regex("\\s+"), " ")
        .trim()
        .replace(spacesBeforePunctuation, "$1")
        .replace(spacesAfterPunctuation, "$1")
        .replace(repeatedPunctuation, "$1")
        .trim()

    fun splitSentences(text: String): List<String> {
        val normalized = normalizeChineseText(text)
        if (normalized.isBlank()) return emptyList()
        return normalized
            .split(Regex("(?<=[。！？；])"))
            .map { it.trim() }
            .filter { it.length >= 2 }
    }

    fun incrementalNewText(previous: String, current: String): String {
        val oldText = normalizeChineseText(previous)
        val newText = normalizeChineseText(current)
        if (newText.isBlank()) return ""
        if (oldText.isBlank()) return newText
        if (newText == oldText) return ""
        if (newText.startsWith(oldText)) return newText.removePrefix(oldText).trim()

        val maxOverlap = minOf(oldText.length, newText.length)
        for (length in maxOverlap downTo 2) {
            if (oldText.takeLast(length) == newText.take(length)) {
                return newText.drop(length).trim()
            }
        }
        return newText
    }
}
