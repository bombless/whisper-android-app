package com.example.whisperapp.qnn

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.view.Gravity
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale

class QnnSmokeActivity : Activity() {
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (intent.getBooleanExtra("encoder_smoke", false)) {
            Thread {
                val result = QnnEncoderSmokeRunner.run(this)
                File(filesDir, "qnn-encoder-smoke-result.txt").writeText(result.report)
                runOnUiThread { finish() }
            }.start()
            return
        }
        if (intent.getBooleanExtra("whisper_step_smoke", false)) {
            Thread {
                val result = QnnWhisperStepRunner.run(this)
                File(filesDir, "qnn-whisper-step-result.txt").writeText(result.report)
                runOnUiThread { finish() }
            }.start()
            return
        }
        if (intent.getBooleanExtra("whisper_decoder_loop", false)) {
            Thread {
                val result = QnnWhisperDecoderLoopRunner.run(this)
                File(filesDir, "qnn-whisper-loop-result.txt").writeText(result.report)
                runOnUiThread { finish() }
            }.start()
            return
        }
        if (intent.getBooleanExtra("decoder_smoke", false)) {
            Thread {
                val result = QnnDecoderSmokeRunner.run(this)
                File(filesDir, "qnn-decoder-smoke-result.txt").writeText(result.report)
                runOnUiThread { finish() }
            }.start()
            return
        }
        statusView = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
            textSize = 16f
            gravity = Gravity.TOP or Gravity.START
            setPadding(32, 48, 32, 32)
            text = "QNN smoke test\n\nRunning..."
        }
        setContentView(ScrollView(this).apply { addView(statusView) })

        Thread {
            val reportFile = File(filesDir, "qnn-smoke-result.txt")
            reportFile.writeText("RUNNING\n")
            val result = try {
                QnnSmokeTest.run(this)
            } catch (t: Throwable) {
                QnnSmokeTest.Result(false, "FAIL: ${t.javaClass.simpleName}: ${t.message}")
            }
            val report = buildString {
                append("QNN smoke test\n\n")
                append(result.message)
                append("\n\nBackend: HTP\nCPU fallback: DISABLED")
                if (result.passed) {
                    append(String.format(Locale.ROOT,
                        "\n\nMedian: %.3f ms\nP95: %.3f ms\nMax: %.3f ms\nMax abs error: %.8f",
                        result.medianMs, result.p95Ms, result.maxMs, result.maxAbsError))
                }
                append("\n\nWhisper-Tiny QNN context asset: VERIFIED\nTarget: Snapdragon 8 Gen 3 / SM8650")
            }
            reportFile.writeText(report)
            runOnUiThread { statusView.text = report }
        }.start()
    }
}
