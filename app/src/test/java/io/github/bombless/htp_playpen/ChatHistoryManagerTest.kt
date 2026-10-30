package io.github.bombless.htp_playpen

import io.github.bombless.htp_playpen.ui.ChatHistoryManager
import io.github.bombless.htp_playpen.ui.ChatMessage
import io.github.bombless.htp_playpen.ui.ChatSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ChatHistoryManagerTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    @Test
    fun testChatMessageSerialization() {
        val msg = ChatMessage("user", "Hello world! 测试消息", 123456789L, stats = "16.8 词元/秒 · 150 词元")
        val json = msg.toJson()
        val restored = ChatMessage.fromJson(json)

        assertEquals("user", restored.role)
        assertEquals("Hello world! 测试消息", restored.content)
        assertEquals(123456789L, restored.timestamp)
        assertEquals("16.8 词元/秒 · 150 词元", restored.stats)
    }

    @Test
    fun testChatSessionSerialization() {
        val messages = listOf(
            ChatMessage("user", "用 JS 写快速排序"),
            ChatMessage("assistant", "<think>思考中</think>\n```javascript\nfunction qsort(){}\n```")
        )
        val session = ChatSession(
            id = "test-session-123",
            title = "快速排序算法",
            createdAt = 1000L,
            updatedAt = 2000L,
            messages = messages
        )

        val json = session.toJson()
        val restored = ChatSession.fromJson(json)

        assertEquals("test-session-123", restored.id)
        assertEquals("快速排序算法", restored.title)
        assertEquals(1000L, restored.createdAt)
        assertEquals(2000L, restored.updatedAt)
        assertEquals(2, restored.messages.size)
        assertEquals("user", restored.messages[0].role)
        assertEquals("用 JS 写快速排序", restored.messages[0].content)
        assertEquals("assistant", restored.messages[1].role)

        val summary = session.toSummary()
        assertEquals("test-session-123", summary.id)
        assertEquals("快速排序算法", summary.title)
        assertEquals(2, summary.messageCount)
        assertTrue(summary.snippet.contains("function qsort"))
        assertFalse(summary.snippet.contains("<think>"))
    }

    @Test
    fun testGenerateTitle() {
        assertEquals("新对话", ChatHistoryManager.generateTitle(""))
        assertEquals("新对话", ChatHistoryManager.generateTitle("   \n   "))

        val normal = "请用 JavaScript 实现一个红黑树"
        assertEquals("请用 JavaScript 实现一个红黑树", ChatHistoryManager.generateTitle(normal))

        val longPrompt = "这是一个非常非常非常非常非常长的一段提示词，用于测试标题截断逻辑是否正确工作"
        val title = ChatHistoryManager.generateTitle(longPrompt)
        assertTrue(title.endsWith("…"))
        assertEquals(25, title.length) // 24 chars + …

        val multiline = "# 任务标题\n这是正文内容"
        assertEquals("任务标题", ChatHistoryManager.generateTitle(multiline))
    }

    @Test
    fun testSaveLoadDeleteAndIndexRebuild() {
        val filesDir = tempFolder.newFolder("app_files")

        // 1. Initially empty
        val initialList = ChatHistoryManager.listSessions(filesDir)
        assertTrue(initialList.isEmpty())

        // 2. Save first session
        val session1 = ChatSession(
            id = "sess-1",
            title = "会话一：红黑树",
            createdAt = 1000L,
            updatedAt = 1000L,
            messages = listOf(
                ChatMessage("user", "实现红黑树"),
                ChatMessage("assistant", "这是红黑树代码")
            )
        )
        ChatHistoryManager.saveSession(filesDir, session1)

        val listAfter1 = ChatHistoryManager.listSessions(filesDir)
        assertEquals(1, listAfter1.size)
        assertEquals("sess-1", listAfter1[0].id)
        assertEquals("会话一：红黑树", listAfter1[0].title)
        assertEquals(2, listAfter1[0].messageCount)

        // 3. Save second session with newer updatedAt
        val session2 = ChatSession(
            id = "sess-2",
            title = "会话二：快速排序",
            createdAt = 2000L,
            updatedAt = 3000L,
            messages = listOf(
                ChatMessage("user", "快速排序怎么写？"),
                ChatMessage("assistant", "先选基准，再分区。")
            )
        )
        ChatHistoryManager.saveSession(filesDir, session2)

        val listAfter2 = ChatHistoryManager.listSessions(filesDir)
        assertEquals(2, listAfter2.size)
        // Newest updated first
        assertEquals("sess-2", listAfter2[0].id)
        assertEquals("sess-1", listAfter2[1].id)

        // 4. Load session by ID and check content
        val loaded1 = ChatHistoryManager.loadSession(filesDir, "sess-1")
        assertNotNull(loaded1)
        assertEquals(2, loaded1?.messages?.size)
        assertEquals("实现红黑树", loaded1?.messages?.get(0)?.content)

        val loaded2 = ChatHistoryManager.loadSession(filesDir, "sess-2")
        assertNotNull(loaded2)
        assertEquals("快速排序怎么写？", loaded2?.messages?.get(0)?.content)

        // 5. Rename session
        val renamed = ChatHistoryManager.renameSession(filesDir, "sess-1", "红黑树高级算法")
        assertTrue(renamed)
        val loadedRenamed = ChatHistoryManager.loadSession(filesDir, "sess-1")
        assertEquals("红黑树高级算法", loadedRenamed?.title)

        // 6. Test index rebuild when index.json is deleted
        val indexFile = tempFolder.root.resolve("app_files/chat_history/index.json")
        assertTrue(indexFile.exists())
        indexFile.delete()
        assertFalse(indexFile.exists())

        val rebuiltList = ChatHistoryManager.listSessions(filesDir)
        assertEquals(2, rebuiltList.size)
        assertTrue(indexFile.exists()) // Index should be reconstructed

        // 7. Delete single session
        val deleted = ChatHistoryManager.deleteSession(filesDir, "sess-2")
        assertTrue(deleted)
        val listAfterDelete = ChatHistoryManager.listSessions(filesDir)
        assertEquals(1, listAfterDelete.size)
        assertEquals("sess-1", listAfterDelete[0].id)
        assertNull(ChatHistoryManager.loadSession(filesDir, "sess-2"))

        // 8. Clear all
        val cleared = ChatHistoryManager.clearAll(filesDir)
        assertTrue(cleared)
        val emptyList = ChatHistoryManager.listSessions(filesDir)
        assertTrue(emptyList.isEmpty())
        assertNull(ChatHistoryManager.loadSession(filesDir, "sess-1"))
    }

    @Test
    fun testLoadAndContinueConversation() {
        val filesDir = tempFolder.newFolder("continue_chat_files")

        // 1. First turn: user asks question, assistant responds
        val sessionId = "cont-session-001"
        val sessionTitle = "红黑树算法实现"
        val turn1Messages = listOf(
            ChatMessage("user", "请用 JavaScript 实现一个红黑树"),
            ChatMessage("assistant", "这是红黑树基础实现...")
        )
        ChatHistoryManager.saveSession(
            filesDir,
            ChatSession(
                id = sessionId,
                title = sessionTitle,
                createdAt = 1000L,
                updatedAt = 1000L,
                messages = turn1Messages
            )
        )

        // 2. Load previous chat session to continue chatting
        val loaded = ChatHistoryManager.loadSession(filesDir, sessionId)
        assertNotNull(loaded)
        assertEquals(sessionId, loaded?.id)
        assertEquals("红黑树算法实现", loaded?.title)
        assertEquals(2, loaded?.messages?.size)

        // 3. User continues chatting: adds turn 2
        val continuedMessages = loaded!!.messages + listOf(
            ChatMessage("user", "请给这棵红黑树添加删除节点的逻辑并自测"),
            ChatMessage("assistant", "这是包含删除节点的完整实现及单元测试...")
        )
        val updatedSession = loaded.copy(
            updatedAt = 2000L,
            messages = continuedMessages
        )
        ChatHistoryManager.saveSession(filesDir, updatedSession)

        // 4. Verify updated session in list and detail
        val summaries = ChatHistoryManager.listSessions(filesDir)
        assertEquals(1, summaries.size)
        assertEquals(4, summaries[0].messageCount)
        assertEquals(2000L, summaries[0].updatedAt)
        assertTrue(summaries[0].snippet.contains("包含删除节点的完整实现"))

        val reloaded = ChatHistoryManager.loadSession(filesDir, sessionId)
        assertNotNull(reloaded)
        assertEquals(4, reloaded?.messages?.size)
        assertEquals("user", reloaded?.messages?.get(0)?.role)
        assertEquals("请用 JavaScript 实现一个红黑树", reloaded?.messages?.get(0)?.content)
        assertEquals("assistant", reloaded?.messages?.get(1)?.role)
        assertEquals("user", reloaded?.messages?.get(2)?.role)
        assertEquals("请给这棵红黑树添加删除节点的逻辑并自测", reloaded?.messages?.get(2)?.content)
        assertEquals("assistant", reloaded?.messages?.get(3)?.role)
        assertEquals("这是包含删除节点的完整实现及单元测试...", reloaded?.messages?.get(3)?.content)
    }

    @Test
    fun testLfmGenerationProgressMetrics() {
        val progress = io.github.bombless.htp_playpen.qnn.LfmGenerationProgress(
            text = "正在推理结果",
            tokenCount = 50,
            tokensPerSecond = 18.5,
            elapsedMs = 2702L,
            isFinished = false
        )
        assertEquals(50, progress.tokenCount)
        assertEquals(18.5, progress.tokensPerSecond, 0.001)
        assertEquals(2702L, progress.elapsedMs)
        assertFalse(progress.isFinished)

        val finishedProgress = progress.copy(isFinished = true)
        assertTrue(finishedProgress.isFinished)
    }
}
