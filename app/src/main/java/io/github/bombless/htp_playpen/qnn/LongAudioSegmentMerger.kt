package io.github.bombless.htp_playpen.qnn

/** Minimal, order-preserving Long Audio segment merger. */
object LongAudioSegmentMerger {
    fun merge(chunks: List<LongAudioSingleContextRunner.ChunkResult>): String =
        chunks.asSequence()
            .map { it.decodedText.trim() }
            .filter { it.isNotEmpty() }
            .joinToString(" ")
}
