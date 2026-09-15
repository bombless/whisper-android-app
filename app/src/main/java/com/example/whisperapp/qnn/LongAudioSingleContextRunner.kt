package com.example.whisperapp.qnn

import android.content.Context
import android.util.Log
import com.example.whisperapp.audio.LongAudioChunker
import com.example.whisperapp.audio.WhisperFeatureExtractor

/**
 * Long-audio wrapper around the already-validated Single-Context runner.
 * Each chunk starts a fresh Single-Context inference; no state crosses chunks.
 */
object LongAudioSingleContextRunner {
    private const val TAG = "QNN_LONG_AUDIO"
    private const val EOS_TOKEN = 50257

    data class ChunkResult(
        val chunkIndex: Int,
        val startSample: Int,
        val endSample: Int,
        val startTimeSeconds: Double,
        val endTimeSeconds: Double,
        val tokenIds: IntArray,
        val decodedText: String,
        val eosReached: Boolean,
    )

    data class Result(
        val chunks: List<ChunkResult>,
        val finalTranscript: String,
        val passed: Boolean,
    )

    fun run(
        context: Context,
        pcm16: ShortArray,
        sampleRate: Int,
        requestedSteps: Int = 12,
    ): Result {
        require(sampleRate == WhisperFeatureExtractor.SAMPLE_RATE) {
            "Whisper expects 16000 Hz, got $sampleRate"
        }
        require(requestedSteps in 1..32) { "requestedSteps must be in 1..32" }

        val chunks = LongAudioChunker.split(pcm16, sampleRate)
        LongAudioAppLogger.info("LONG_AUDIO_START totalSamples=${pcm16.size} chunks=${chunks.size}")

        val results = ArrayList<ChunkResult>(chunks.size)
        for (chunk in chunks) {
            LongAudioAppLogger.info("LONG_AUDIO_CHUNK_START index=${chunk.index} start=${chunk.startSample} end=${chunk.endSampleExclusive}")

            // The frozen Single-Context runner creates and releases its own inference state.
            val single = QnnWhisperRealAudioRunner.run(
                context = context,
                pcm16 = chunk.samples,
                sampleRate = sampleRate,
                requestedSteps = requestedSteps,
                autoregressive = true,
            )
            check(single.passed) {
                "chunk=${chunk.index} Single-Context inference failed: ${single.report}"
            }

            LongAudioAppLogger.info("CHUNK_MEL_READY index=${chunk.index}")
            LongAudioAppLogger.info("CHUNK_ENCODER_READY index=${chunk.index}")
            LongAudioAppLogger.info("CHUNK_CROSS_READY index=${chunk.index}")
            LongAudioAppLogger.info("DECODER_START index=${chunk.index}")

            val eosReached = single.tokenIds.lastOrNull() == EOS_TOKEN ||
                single.report.contains("eos_reached: true")
            if (eosReached) LongAudioAppLogger.info("DECODER_EOS index=${chunk.index} step=${single.tokenIds.indexOf(EOS_TOKEN)} token=$EOS_TOKEN")
            LongAudioAppLogger.info("TOKENIZER_INPUT_IDS index=${chunk.index} count=${single.tokenIds.size} ids=${single.tokenIds.contentToString()}")
            LongAudioAppLogger.info("TOKENIZER_DECODED_TEXT index=${chunk.index} text=\"${single.text}\"")

            val result = ChunkResult(
                chunkIndex = chunk.index,
                startSample = chunk.startSample,
                endSample = chunk.endSampleExclusive,
                startTimeSeconds = chunk.startTimeSeconds,
                endTimeSeconds = chunk.endTimeSeconds,
                tokenIds = single.tokenIds.copyOf(),
                decodedText = single.text,
                eosReached = eosReached,
            )
            results += result
            LongAudioAppLogger.info("CHUNK_RESULT_TEXT index=${result.chunkIndex} text=\"${result.decodedText}\"")
        }

        LongAudioAppLogger.info("LONG_AUDIO_MERGE_START chunks=${results.size}")
        val finalTranscript = LongAudioSegmentMerger.merge(results)
        LongAudioAppLogger.info("LONG_AUDIO_MERGE_DONE chunks=${results.size} text=$finalTranscript")
        LongAudioAppLogger.info("LONG_AUDIO_DONE chunks=${results.size}")
        return Result(chunks = results, finalTranscript = finalTranscript, passed = true)
    }
}
