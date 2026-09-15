package com.example.whisperapp.qnn

import android.app.Activity
import android.os.Bundle
import android.util.Log
import com.example.whisperapp.audio.WavPcmReader
import java.io.File

/** Device-only verification harness for the first Long Audio milestone. */
class LongAudioVerificationActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            try {
                val wav = File(getExternalFilesDir(null), "long_audio_input.wav")
                Log.i("QNN_LONG_AUDIO", "VERIFICATION_START wav=${wav.absolutePath} exists=${wav.isFile}")
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                Log.i("QNN_LONG_AUDIO", "VERIFICATION_PCM samples=${pcm.samples.size} rate=${pcm.sampleRate} channels=${pcm.channels}")
                val result = LongAudioSingleContextRunner.run(this, pcm.samples, pcm.sampleRate)
                val report = buildString {
                    appendLine("LONG_AUDIO status=${if (result.passed) "PASS" else "FAIL"}")
                    appendLine("chunks=${result.chunks.size}")
                    result.chunks.forEach { chunk ->
                        appendLine("chunk=${chunk.chunkIndex} start=${chunk.startTimeSeconds} end=${chunk.endTimeSeconds} samples=${chunk.endSample - chunk.startSample} eos=${chunk.eosReached} text=${chunk.decodedText}")
                    }
                }
                File(getExternalFilesDir(null), "long_audio_verification_result.txt").writeText(report)
                Log.i("QNN_LONG_AUDIO", "RESULT_WRITTEN")
            } catch (t: Throwable) {
                File(getExternalFilesDir(null), "long_audio_verification_result.txt").writeText("LONG_AUDIO FAIL\n${t.stackTraceToString()}")
                Log.e("QNN_LONG_AUDIO", "VERIFICATION_EXCEPTION", t)
            }
            runOnUiThread { finish() }
        }.start()
    }
}
