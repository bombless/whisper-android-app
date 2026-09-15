package com.example.whisperapp.audio

/**
 * Splits 16 kHz mono PCM16 audio into independent Whisper 30-second contexts.
 * The final chunk is kept at its real length; WhisperFeatureExtractor supplies the
 * zero-padded 30-second model context when that chunk is later processed.
 */
object LongAudioChunker {
    data class Chunk(
        val index: Int,
        val startSample: Int,
        val endSampleExclusive: Int,
        val samples: ShortArray,
    ) {
        val startTimeSeconds: Double
            get() = startSample.toDouble() / WhisperFeatureExtractor.SAMPLE_RATE
        val endTimeSeconds: Double
            get() = endSampleExclusive.toDouble() / WhisperFeatureExtractor.SAMPLE_RATE
        val durationSeconds: Double
            get() = samples.size.toDouble() / WhisperFeatureExtractor.SAMPLE_RATE
    }

    fun split(pcm16: ShortArray, sampleRate: Int): List<Chunk> {
        require(sampleRate == WhisperFeatureExtractor.SAMPLE_RATE) {
            "Whisper expects 16000 Hz, got $sampleRate"
        }
        if (pcm16.isEmpty()) return emptyList()

        val chunkSize = WhisperFeatureExtractor.CHUNK_SAMPLES
        val chunkCount = (pcm16.size + chunkSize - 1) / chunkSize
        return List(chunkCount) { index ->
            val start = index * chunkSize
            val end = minOf(start + chunkSize, pcm16.size)
            Chunk(
                index = index,
                startSample = start,
                endSampleExclusive = end,
                samples = pcm16.copyOfRange(start, end),
            )
        }
    }
}
