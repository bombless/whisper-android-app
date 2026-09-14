package com.example.whisperapp.qnn

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.view.Gravity
import android.widget.ScrollView
import android.widget.TextView
import android.util.Log
import java.io.File
import java.util.Locale

class QnnSmokeActivity : Activity() {
    private lateinit var statusView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.i("QNN_SMOKE", "ACTIVITY_ON_CREATE")

        statusView = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
            textSize = 16f
            gravity = Gravity.TOP or Gravity.START
            setPadding(32, 48, 32, 32)
            text = "QNN smoke test\n\n正在初始化..."
        }
        setContentView(ScrollView(this).apply { addView(statusView) })
        Log.i("QNN_SMOKE", "STARTING_WORKER")

        Thread {
            val reportFile = File(filesDir, "qnn-smoke-result.txt")
            reportFile.writeText("RUNNING\n")
            Log.i("QNN_SMOKE", "WORKER_STARTED")
            val result = try {
                QnnSmokeTest.run(this)
            } catch (t: Throwable) {
                Log.e("QNN_SMOKE", "ACTIVITY_EXCEPTION", t)
                QnnSmokeTest.Result(false, "FAIL: ${t.javaClass.simpleName}: ${t.message}")
            }
            Log.i("QNN_SMOKE", "ACTIVITY_RESULT=${result.message}")
            val report = buildString {
                append("QNN smoke test\n\n")
                append(result.message)
                append("\n\nBackend: HTP\nCPU fallback: 禁用")
                if (result.passed) {
                    append(String.format(Locale.ROOT,
                        "\n\nMedian: %.3f ms\nP95: %.3f ms\nMax: %.3f ms\n最大绝对误差: %.8f",
                        result.medianMs, result.p95Ms, result.maxMs, result.maxAbsError))
                }
                append("\n\n测试结束。此页面不会自动关闭。")
            }
            // Persist the result because verbose vendor logs can evict logcat evidence.
            reportFile.writeText(report)
            runOnUiThread {
                statusView.text = report
            }
        }.start()
    }
}
