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
        Log.i(TAG, "LONG_AUDIO_START totalSamples=${pcm16.size} chunks=${chunks.size}")

        val results = ArrayList<ChunkResult>(chunks.size)
        for (chunk in chunks) {
            Log.i(TAG, "LONG_AUDIO_CHUNK_START index=${chunk.index} start=${chunk.startSample} end=${chunk.endSampleExclusive}")

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

            Log.i(TAG, "CHUNK_MEL_READY index=${chunk.index}")
            Log.i(TAG, "CHUNK_ENCODER_READY index=${chunk.index}")
            Log.i(TAG, "CHUNK_CROSS_READY index=${chunk.index}")
            Log.i(TAG, "CHUNK_DECODER_START index=${chunk.index}")

            val eosReached = single.tokenIds.lastOrNull() == EOS_TOKEN ||
                single.report.contains("eos_reached: true")
            if (eosReached) Log.i(TAG, "CHUNK_EOS index=${chunk.index} token=$EOS_TOKEN")
            Log.i(TAG, "CHUNK_TOKENIZER_DONE index=${chunk.index}")

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
            Log.i(TAG, "CHUNK_RESULT index=${result.chunkIndex} text=${result.decodedText} eos=${result.eosReached}")
        }

        Log.i(TAG, "LONG_AUDIO_DONE chunks=${results.size}")
        return Result(chunks = results, passed = true)
    }
}
