package com.example.whisperapp.qnn

import android.app.Activity
import android.os.Bundle
import android.util.Log
import java.io.File

class M35RealAudioEncoderActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            try {
                val result=M35RealAudioEncoderRunner.run(this,File(getExternalFilesDir(null),"m35_real_speech_16khz.wav"))
                File("/sdcard/Download/m35_encoder_result.txt").writeText(result.report)
                Log.e("M35_REAL_ENCODER","RESULT_FILE_WRITTEN passed=${result.passed}")
            } catch(t:Throwable) {
                File("/sdcard/Download/m35_encoder_result.txt").writeText("ACTIVITY EXCEPTION\n${t.stackTraceToString()}")
                Log.e("M35_REAL_ENCODER","ACTIVITY EXCEPTION",t)
            }
            runOnUiThread { finish() }
        }.start()
    }
}
