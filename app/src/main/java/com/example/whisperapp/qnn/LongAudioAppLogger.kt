package com.example.whisperapp.qnn

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object LongAudioAppLogger {
    private const val TAG = "QNN_LONG_AUDIO"
    private const val MAX_UI_LINES = 2000
    private val lock = Any()
    private val formatter = SimpleDateFormat("HH:mm:ss.SSS", Locale.US)
    private val _lines = MutableStateFlow<List<String>>(emptyList())
    val lines: StateFlow<List<String>> = _lines
    @Volatile private var logFile: File? = null

    fun startTest(context: Context) {
        synchronized(lock) {
            logFile = File(context.filesDir, "long_audio_verification.log")
            logFile!!.writeText("")
            _lines.value = emptyList()
        }
    }

    fun clear() {
        synchronized(lock) {
            _lines.value = emptyList()
            try { logFile?.writeText("") } catch (_: Throwable) { }
        }
    }

    fun info(message: String) = append(message, null)
    fun error(message: String, throwable: Throwable? = null) = append(message, throwable)
    fun snapshot(): String = synchronized(lock) { _lines.value.joinToString("\n") }

    private fun append(message: String, throwable: Throwable?) {
        val line = "[${formatter.format(Date())}] $message"
        synchronized(lock) {
            if (throwable == null) Log.i(TAG, message) else Log.e(TAG, message, throwable)
            _lines.value = (_lines.value + line).takeLast(MAX_UI_LINES)
            try {
                logFile?.appendText(line + "\n")
                if (throwable != null) logFile?.appendText(throwable.stackTraceToString())
            } catch (t: Throwable) {
                Log.e(TAG, "ERROR type=log_file_write message=${t.message}", t)
            }
        }
    }
}
