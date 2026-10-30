package io.github.bombless.htp_playpen.qnn

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object LfmChatLogger {
    private const val TAG = "QNN_LFM3"
    private const val MAX_FILE_BYTES = 5 * 1024 * 1024 // 5 MB
    private val lock = Any()
    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)
    @Volatile private var logFile: File? = null

    fun init(context: Context) {
        synchronized(lock) {
            if (logFile == null) {
                val f = File(context.filesDir, "lfm_chat.log")
                if (f.exists() && f.length() > MAX_FILE_BYTES) {
                    val backup = File(context.filesDir, "lfm_chat.old.log")
                    if (backup.exists()) backup.delete()
                    f.renameTo(backup)
                }
                logFile = f
            }
        }
    }

    fun info(message: String) = log(Log.INFO, message, null)
    fun warn(message: String) = log(Log.WARN, message, null)
    fun error(message: String, throwable: Throwable? = null) = log(Log.ERROR, message, throwable)

    private fun log(priority: Int, message: String, throwable: Throwable?) {
        val time = formatter.format(Date())
        val levelStr = when (priority) {
            Log.WARN -> "WARN"
            Log.ERROR -> "ERROR"
            else -> "INFO"
        }
        val fullMsg = "[$time] [$levelStr] $message"
        when (priority) {
            Log.WARN -> Log.w(TAG, message)
            Log.ERROR -> Log.e(TAG, message, throwable)
            else -> Log.i(TAG, message)
        }
        synchronized(lock) {
            try {
                logFile?.appendText(fullMsg + "\n")
                if (throwable != null) {
                    logFile?.appendText(throwable.stackTraceToString() + "\n")
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Failed writing to lfm_chat.log: ${t.message}")
            }
        }
    }

    fun getLogPath(): String = logFile?.absolutePath ?: "<not initialized>"
}
