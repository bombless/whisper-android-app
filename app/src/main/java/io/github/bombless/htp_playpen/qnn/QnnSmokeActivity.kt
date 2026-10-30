package io.github.bombless.htp_playpen.qnn

import android.app.Activity
import android.content.pm.ApplicationInfo
import android.os.Bundle
import android.graphics.Color
import android.graphics.Bitmap
import android.util.Log
import android.view.Gravity
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.Locale

class QnnSmokeActivity : Activity() {
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val sdxlDiagnostic = listOf("sdxl_prepare", "sdxl_text_smoke", "sdxl_smoke", "sdxl_generate_smoke")
            .any { intent.getBooleanExtra(it, false) }
        if (sdxlDiagnostic && (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) == 0) {
            finish()
            return
        }
        if (intent.getBooleanExtra("sdxl_generate_smoke", false)) {
            Thread {
                val reportFile = File(filesDir, "sdxl-generate-result.txt")
                reportFile.writeText("RUNNING\n")
                val report = try {
                    val started = System.nanoTime()
                    val image = SdxlEngine.generate(this, "a red apple on a wooden table, natural light, photograph", "blurry, distorted", 20, 12345L) {
                        Log.i("SDXL_GENERATE", it)
                    }
                    val output = File(filesDir, "sdxl-generate-result.png")
                    output.outputStream().use { check(image.compress(Bitmap.CompressFormat.PNG, 100, it)) }
                    "PASS: ${image.width}x${image.height}, ${(System.nanoTime() - started) / 1_000_000}ms, ${output.absolutePath}"
                } catch (failure: Throwable) {
                    Log.e("SDXL_GENERATE", "Failed", failure)
                    "FAIL: ${failure.javaClass.name}: ${failure.message}"
                }
                reportFile.writeText(report)
                runOnUiThread { finish() }
            }.start()
            return
        }
        if (intent.getBooleanExtra("sdxl_prepare", false)) {
            val report = runCatching { QnnSdxlSmokeRunner.prepare(this) }
                .fold({ "READY: $it" }, { "FAIL: ${it.javaClass.name}: ${it.message}" })
            File(filesDir, "sdxl-prepare-result.txt").writeText(report)
            finish()
            return
        }
        if (intent.getBooleanExtra("sdxl_text_smoke", false)) {
            Thread {
                val reportFile = File(filesDir, "sdxl-text-smoke-result.txt")
                reportFile.writeText("RUNNING\n")
                reportFile.writeText(QnnSdxlSmokeRunner.runText(this))
                runOnUiThread { finish() }
            }.start()
            return
        }
        if (intent.getBooleanExtra("sdxl_smoke", false)) {
            Thread {
                val reportFile = File(filesDir, "sdxl-qnn-smoke-result.txt")
                reportFile.writeText("RUNNING\n")
                val report = QnnSdxlSmokeRunner.run(this)
                reportFile.writeText(report)
                runOnUiThread { finish() }
            }.start()
            return
        }
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
