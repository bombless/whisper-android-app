package com.example.whisperapp.qnn

import android.app.Activity
import android.os.Bundle
import android.util.Log
import com.example.whisperapp.audio.WavPcmReader
import java.io.File

class M35RealAudioDecoderActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            try {
                val wav = File(getExternalFilesDir(null), "output.wav")
                Log.i("M35_REAL_DECODER", "START wav=${wav.absolutePath} exists=${wav.isFile}")
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                Log.i("M35_REAL_DECODER", "PCM16 samples=${pcm.samples.size} rate=${pcm.sampleRate} channels=${pcm.channels}")
                val result = QnnWhisperRealAudioRunner.run(this, pcm.samples, pcm.sampleRate, requestedSteps = 16)
                File(getExternalFilesDir(null), "m35_real_audio_decoder_result.txt").writeText(result.report)
                Log.e("M35_REAL_DECODER", "RESULT_WRITTEN passed=${result.passed}")
            } catch (t: Throwable) {
                File("/sdcard/Download/m35_real_audio_decoder_result.txt").writeText("ACTIVITY EXCEPTION\n${t.stackTraceToString()}")
                Log.e("M35_REAL_DECODER", "ACTIVITY EXCEPTION", t)
            }
            runOnUiThread { finish() }
        }.start()
    }
}

