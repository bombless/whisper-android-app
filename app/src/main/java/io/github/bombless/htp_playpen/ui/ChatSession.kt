package io.github.bombless.htp_playpen.ui

import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * Single turn message in a chat session.
 */
data class ChatMessage(
    val role: String, // "user" or "assistant"
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val stats: String? = null
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("role", role)
        put("content", content)
        put("timestamp", timestamp)
        if (stats != null) {
            put("stats", stats)
        }
    }

    companion object {
        fun fromJson(json: JSONObject): ChatMessage = ChatMessage(
            role = json.optString("role", "user"),
            content = json.optString("content", ""),
            timestamp = json.optLong("timestamp", System.currentTimeMillis()),
            stats = if (json.has("stats")) json.optString("stats") else null
        )
    }
}

/**
 * Full chat session with all messages.
 */
data class ChatSession(
    val id: String = UUID.randomUUID().toString(),
    val title: String,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
    val messages: List<ChatMessage> = emptyList()
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("title", title)
        put("createdAt", createdAt)
        put("updatedAt", updatedAt)
        val arr = JSONArray()
        for (m in messages) {
            arr.put(m.toJson())
        }
        put("messages", arr)
    }

    fun toSummary(): ChatSessionSummary {
        val lastMsg = messages.lastOrNull()
        val snippet = when {
            lastMsg == null -> "（空对话）"
            lastMsg.role == "assistant" -> {
                val clean = if (lastMsg.content.contains("</think>")) {
                    lastMsg.content.substringAfter("</think>").trim()
                } else {
                    lastMsg.content.trim()
                }
                clean.take(80).replace('\n', ' ')
            }
            else -> lastMsg.content.take(80).replace('\n', ' ')
        }
        return ChatSessionSummary(
            id = id,
            title = title,
            createdAt = createdAt,
            updatedAt = updatedAt,
            messageCount = messages.size,
            snippet = snippet
        )
    }

    companion object {
        fun fromJson(json: JSONObject): ChatSession {
            val id = json.optString("id", UUID.randomUUID().toString())
            val title = json.optString("title", "未命名对话")
            val createdAt = json.optLong("createdAt", System.currentTimeMillis())
            val updatedAt = json.optLong("updatedAt", createdAt)
            val msgArr = json.optJSONArray("messages") ?: JSONArray()
            val msgs = ArrayList<ChatMessage>(msgArr.length())
            for (i in 0 until msgArr.length()) {
                val item = msgArr.optJSONObject(i)
                if (item != null) {
                    msgs.add(ChatMessage.fromJson(item))
                }
            }
            return ChatSession(
                id = id,
                title = title,
                createdAt = createdAt,
                updatedAt = updatedAt,
                messages = msgs
            )
        }
    }
}

/**
 * Lightweight summary of a chat session for listing and preview.
 */
data class ChatSessionSummary(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val messageCount: Int,
    val snippet: String
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("title", title)
        put("createdAt", createdAt)
        put("updatedAt", updatedAt)
        put("messageCount", messageCount)
        put("snippet", snippet)
    }

    companion object {
        fun fromJson(json: JSONObject): ChatSessionSummary = ChatSessionSummary(
            id = json.optString("id", ""),
            title = json.optString("title", "未命名对话"),
            createdAt = json.optLong("createdAt", System.currentTimeMillis()),
            updatedAt = json.optLong("updatedAt", System.currentTimeMillis()),
            messageCount = json.optInt("messageCount", 0),
            snippet = json.optString("snippet", "")
        )
    }
}
