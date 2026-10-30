package io.github.bombless.htp_playpen.ui

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ChatHistoryManager {
    private const val TAG = "ChatHistoryManager"
    private const val HISTORY_DIR_NAME = "chat_history"
    private const val SESSIONS_DIR_NAME = "sessions"
    private const val INDEX_FILE_NAME = "index.json"

    fun getBaseDir(context: Context): File = getBaseDir(context.filesDir)

    fun getBaseDir(filesDir: File): File {
        return File(filesDir, HISTORY_DIR_NAME).apply {
            if (!exists()) mkdirs()
        }
    }

    private fun getSessionsDir(filesDir: File): File {
        return File(getBaseDir(filesDir), SESSIONS_DIR_NAME).apply {
            if (!exists()) mkdirs()
        }
    }

    private fun getIndexFile(filesDir: File): File {
        return File(getBaseDir(filesDir), INDEX_FILE_NAME)
    }

    private fun getSessionFile(filesDir: File, id: String): File {
        return File(getSessionsDir(filesDir), "$id.json")
    }

    /**
     * Lists all chat sessions, sorted by last update time descending (newest first).
     */
    fun listSessions(context: Context): List<ChatSessionSummary> = listSessions(context.filesDir)

    @Synchronized
    fun listSessions(filesDir: File): List<ChatSessionSummary> {
        val indexFile = getIndexFile(filesDir)
        if (indexFile.exists()) {
            try {
                val array = JSONArray(indexFile.readText())
                val list = ArrayList<ChatSessionSummary>(array.length())
                for (i in 0 until array.length()) {
                    val obj = array.optJSONObject(i)
                    if (obj != null) {
                        list.add(ChatSessionSummary.fromJson(obj))
                    }
                }
                return list.sortedByDescending { it.updatedAt }
            } catch (e: Exception) {
                try { Log.w(TAG, "Corrupt index file, rebuilding from session files", e) } catch (_: Throwable) {}
            }
        }
        return rebuildIndexFromDisk(filesDir)
    }

    /**
     * Loads a full chat session with all messages.
     */
    fun loadSession(context: Context, id: String): ChatSession? = loadSession(context.filesDir, id)

    @Synchronized
    fun loadSession(filesDir: File, id: String): ChatSession? {
        val file = getSessionFile(filesDir, id)
        if (!file.exists()) return null
        return try {
            val json = JSONObject(file.readText())
            ChatSession.fromJson(json)
        } catch (e: Exception) {
            try { Log.e(TAG, "Failed to load session id=$id", e) } catch (_: Throwable) {}
            null
        }
    }

    /**
     * Saves or updates a chat session and updates the index file.
     */
    fun saveSession(context: Context, session: ChatSession) = saveSession(context.filesDir, session)

    @Synchronized
    fun saveSession(filesDir: File, session: ChatSession) {
        try {
            val sessionFile = getSessionFile(filesDir, session.id)
            val tmpFile = File(sessionFile.parentFile, "${session.id}.json.tmp")
            tmpFile.writeText(session.toJson().toString())
            if (!tmpFile.renameTo(sessionFile)) {
                sessionFile.delete()
                tmpFile.renameTo(sessionFile)
            }

            // Update index
            val summary = session.toSummary()
            val currentList = listSessions(filesDir).toMutableList()
            val existingIndex = currentList.indexOfFirst { it.id == session.id }
            if (existingIndex >= 0) {
                currentList[existingIndex] = summary
            } else {
                currentList.add(0, summary)
            }
            currentList.sortByDescending { it.updatedAt }
            writeIndex(filesDir, currentList)
            try { Log.d(TAG, "Saved session id=${session.id} title=${session.title} msgs=${session.messages.size}") } catch (_: Throwable) {}
        } catch (e: Exception) {
            try { Log.e(TAG, "Failed to save session id=${session.id}", e) } catch (_: Throwable) {}
        }
    }

    /**
     * Deletes a session by ID.
     */
    fun deleteSession(context: Context, id: String): Boolean = deleteSession(context.filesDir, id)

    @Synchronized
    fun deleteSession(filesDir: File, id: String): Boolean {
        return try {
            val sessionFile = getSessionFile(filesDir, id)
            if (sessionFile.exists()) {
                sessionFile.delete()
            }
            val currentList = listSessions(filesDir).filter { it.id != id }
            writeIndex(filesDir, currentList)
            try { Log.i(TAG, "Deleted session id=$id") } catch (_: Throwable) {}
            true
        } catch (e: Exception) {
            try { Log.e(TAG, "Failed to delete session id=$id", e) } catch (_: Throwable) {}
            false
        }
    }

    /**
     * Clears all chat history.
     */
    fun clearAll(context: Context): Boolean = clearAll(context.filesDir)

    @Synchronized
    fun clearAll(filesDir: File): Boolean {
        return try {
            val dir = getSessionsDir(filesDir)
            dir.listFiles()?.forEach { it.delete() }
            val indexFile = getIndexFile(filesDir)
            if (indexFile.exists()) indexFile.delete()
            try { Log.i(TAG, "Cleared all chat history") } catch (_: Throwable) {}
            true
        } catch (e: Exception) {
            try { Log.e(TAG, "Failed to clear all sessions", e) } catch (_: Throwable) {}
            false
        }
    }

    /**
     * Renames an existing session.
     */
    fun renameSession(context: Context, id: String, newTitle: String): Boolean = renameSession(context.filesDir, id, newTitle)

    @Synchronized
    fun renameSession(filesDir: File, id: String, newTitle: String): Boolean {
        val session = loadSession(filesDir, id) ?: return false
        val updated = session.copy(title = newTitle.trim(), updatedAt = System.currentTimeMillis())
        saveSession(filesDir, updated)
        return true
    }

    /**
     * Rebuilds index by reading all session files on disk.
     */
    @Synchronized
    private fun rebuildIndexFromDisk(filesDir: File): List<ChatSessionSummary> {
        val dir = getSessionsDir(filesDir)
        val files = dir.listFiles { _, name -> name.endsWith(".json") } ?: return emptyList()
        val list = mutableListOf<ChatSessionSummary>()
        for (f in files) {
            try {
                val json = JSONObject(f.readText())
                val session = ChatSession.fromJson(json)
                list.add(session.toSummary())
            } catch (e: Exception) {
                try { Log.w(TAG, "Skipping corrupted session file: ${f.name}", e) } catch (_: Throwable) {}
            }
        }
        val sorted = list.sortedByDescending { it.updatedAt }
        writeIndex(filesDir, sorted)
        return sorted
    }

    private fun writeIndex(filesDir: File, list: List<ChatSessionSummary>) {
        try {
            val arr = JSONArray()
            for (item in list) {
                arr.put(item.toJson())
            }
            getIndexFile(filesDir).writeText(arr.toString())
        } catch (e: Exception) {
            try { Log.e(TAG, "Failed to write index file", e) } catch (_: Throwable) {}
        }
    }

    /**
     * Generates a clean, human-readable session title from the first prompt.
     */
    fun generateTitle(prompt: String): String {
        val clean = prompt
            .replace(Regex("<tool_call[\\s\\S]*?</tool_call>"), "")
            .replace(Regex("```[\\s\\S]*?```"), "")
            .trim()
        val firstLine = clean.lines().firstOrNull { it.isNotBlank() } ?: prompt.trim()
        val stripped = firstLine.trim().removePrefix("#").trim()
        if (stripped.isBlank()) return "新对话"
        return if (stripped.length > 24) {
            stripped.take(24) + "…"
        } else {
            stripped
        }
    }

    /**
     * Helper to format timestamp for UI display.
     */
    fun formatTime(timestamp: Long): String {
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        return when {
            diff < 60_000L -> "刚刚"
            diff < 3600_000L -> "${diff / 60_000L} 分钟前"
            diff < 24 * 3600_000L -> {
                val sdf = SimpleDateFormat("HH:mm", Locale.getDefault())
                "今天 " + sdf.format(Date(timestamp))
            }
            diff < 48 * 3600_000L -> {
                val sdf = SimpleDateFormat("HH:mm", Locale.getDefault())
                "昨天 " + sdf.format(Date(timestamp))
            }
            else -> {
                val sdf = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())
                sdf.format(Date(timestamp))
            }
        }
    }
}
