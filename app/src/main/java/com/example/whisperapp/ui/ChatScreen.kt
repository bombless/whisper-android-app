package com.example.whisperapp.ui

import android.app.Activity
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.whisperapp.qnn.QnnLfm2ChatRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current
    LaunchedEffect(Unit) { Log.i("CHAT_UI", "ChatScreen composed package=${context.packageName}") }

    val scope = rememberCoroutineScope()
    val messages = remember { mutableStateListOf<Pair<String, String>>() }
    val listState = rememberLazyListState()

    var input by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("正在准备 LFM3-1.7B…") }
    var busy by remember { mutableStateOf(false) }
    var currentJob by remember { mutableStateOf<Job?>(null) }

    // Session & History state
    var currentSessionId by remember { mutableStateOf<String?>(null) }
    var currentSessionTitle by remember { mutableStateOf("新对话") }
    var showHistorySheet by remember { mutableStateOf(false) }
    var historyList by remember { mutableStateOf<List<ChatSessionSummary>>(emptyList()) }

    // Configurable Max Tokens: default 4096 (expanded from 1024)
    val tokenOptions = remember { listOf(1024, 2048, 4096, 8192) }
    var maxTokens by remember { mutableIntStateOf(4096) }
    var tokenMenuExpanded by remember { mutableStateOf(false) }

    val refreshHistory: () -> Unit = {
        scope.launch(Dispatchers.IO) {
            val list = ChatHistoryManager.listSessions(context)
            withContext(Dispatchers.Main) {
                historyList = list
            }
        }
    }

    LaunchedEffect(Unit) {
        status = QnnLfm2ChatRunner.start(context).message
        refreshHistory()
    }

    val activity = context as? Activity
    DisposableEffect(busy) {
        if (busy) {
            activity?.window?.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            activity?.window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
        onDispose {
            activity?.window?.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            QnnLfm2ChatRunner.cancelGeneration()
            currentJob?.cancel()
            QnnLfm2ChatRunner.stop()
        }
    }

    val stopGeneration: () -> Unit = {
        QnnLfm2ChatRunner.cancelGeneration()
        currentJob?.cancel()
        busy = false
        Log.i("CHAT_UI", "USER_STOPPED_GENERATION")
        // Persist whatever was received up to cancellation
        currentSessionId?.let { sId ->
            val snapshot = messages.toList()
            scope.launch(Dispatchers.IO) {
                val chatMsgs = snapshot.map { ChatMessage(it.first, it.second) }
                ChatHistoryManager.saveSession(
                    context,
                    ChatSession(
                        id = sId,
                        title = currentSessionTitle,
                        updatedAt = System.currentTimeMillis(),
                        messages = chatMsgs
                    )
                )
                val updated = ChatHistoryManager.listSessions(context)
                withContext(Dispatchers.Main) { historyList = updated }
            }
        }
    }

    val startNewChat: () -> Unit = {
        if (busy) {
            stopGeneration()
        }
        currentSessionId = null
        currentSessionTitle = "新对话"
        messages.clear()
        input = ""
        Toast.makeText(context, "已开启新对话", Toast.LENGTH_SHORT).show()
    }

    val loadSessionById: (String) -> Unit = { targetId ->
        if (busy) {
            stopGeneration()
        }
        scope.launch(Dispatchers.IO) {
            val loaded = ChatHistoryManager.loadSession(context, targetId)
            withContext(Dispatchers.Main) {
                if (loaded != null) {
                    currentSessionId = loaded.id
                    currentSessionTitle = loaded.title
                    messages.clear()
                    for (m in loaded.messages) {
                        messages.add(m.role to m.content)
                    }
                    Toast.makeText(context, "已载入：${loaded.title}，可继续对话", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(context, "未能载入该对话", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    val deleteSessionById: (String) -> Unit = { targetId ->
        scope.launch(Dispatchers.IO) {
            ChatHistoryManager.deleteSession(context, targetId)
            val updated = ChatHistoryManager.listSessions(context)
            withContext(Dispatchers.Main) {
                historyList = updated
                if (currentSessionId == targetId) {
                    currentSessionId = null
                    currentSessionTitle = "新对话"
                    messages.clear()
                }
                Toast.makeText(context, "对话已删除", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val clearAllHistory: () -> Unit = {
        scope.launch(Dispatchers.IO) {
            ChatHistoryManager.clearAll(context)
            withContext(Dispatchers.Main) {
                historyList = emptyList()
                currentSessionId = null
                currentSessionTitle = "新对话"
                messages.clear()
                Toast.makeText(context, "所有历史记录已清空", Toast.LENGTH_SHORT).show()
            }
        }
    }

    val sendText: (String) -> Unit = { rawPrompt ->
        val text = rawPrompt.trim()
        if (!busy && text.isNotBlank() && status.contains("已就绪")) {
            input = ""
            val isNewSession = currentSessionId == null
            val sessionId = currentSessionId ?: UUID.randomUUID().toString().also {
                currentSessionId = it
            }
            val sessionTitle = if (isNewSession || currentSessionTitle == "新对话") {
                ChatHistoryManager.generateTitle(text).also { currentSessionTitle = it }
            } else {
                currentSessionTitle
            }

            messages += "user" to text
            val historySnapshot = messages.toList()
            messages += "assistant" to "正在思考…"
            val assistantIndex = messages.lastIndex
            busy = true

            // Immediately persist user prompt so it won't be lost
            scope.launch(Dispatchers.IO) {
                val chatMessages = historySnapshot.map { ChatMessage(it.first, it.second) }
                ChatHistoryManager.saveSession(
                    context,
                    ChatSession(
                        id = sessionId,
                        title = sessionTitle,
                        updatedAt = System.currentTimeMillis(),
                        messages = chatMessages
                    )
                )
                val updated = ChatHistoryManager.listSessions(context)
                withContext(Dispatchers.Main) { historyList = updated }
            }

            Log.i("CHAT_UI", "SEND chars=${text.length} maxTokens=$maxTokens history=${historySnapshot.size}")
            currentJob = scope.launch(Dispatchers.Default) {
                var lastUiUpdateNs = 0L
                val uiThrottleNs = 60_000_000L // 60ms throttle
                val answer = runCatching {
                    QnnLfm2ChatRunner.generate(
                        context = context,
                        history = historySnapshot,
                        maxTokens = maxTokens,
                        isCancelled = { !isActive }
                    ) { partialText ->
                        val now = System.nanoTime()
                        val isBoundary = partialText.endsWith('\n') || partialText.endsWith('。') || partialText.endsWith('！') || partialText.endsWith('？') || partialText.endsWith("```")
                        if (isBoundary || now - lastUiUpdateNs >= uiThrottleNs) {
                            lastUiUpdateNs = now
                            launch(Dispatchers.Main) {
                                if (assistantIndex in messages.indices) {
                                    messages[assistantIndex] = "assistant" to partialText
                                }
                            }
                        }
                    }
                }.onFailure {
                    Log.e("CHAT_UI", "GENERATE_FAILED type=${it::class.java.name} message=${it.message}", it)
                }.getOrElse { "生成失败：${it.message}" }

                launch(Dispatchers.Main) {
                    if (assistantIndex in messages.indices) {
                        messages[assistantIndex] = "assistant" to answer
                    }
                    busy = false
                }

                // Persist full conversation with assistant reply
                withContext(Dispatchers.IO) {
                    val updatedMsgs = historySnapshot.map { ChatMessage(it.first, it.second) } +
                        ChatMessage("assistant", answer)
                    ChatHistoryManager.saveSession(
                        context,
                        ChatSession(
                            id = sessionId,
                            title = sessionTitle,
                            updatedAt = System.currentTimeMillis(),
                            messages = updatedMsgs
                        )
                    )
                    val updated = ChatHistoryManager.listSessions(context)
                    withContext(Dispatchers.Main) { historyList = updated }
                }
            }
        }
    }

    // Auto-scroll to bottom as new tokens arrive
    LaunchedEffect(messages.size, messages.lastOrNull()?.second?.length) {
        if (messages.isNotEmpty()) {
            listState.scrollToItem(messages.size - 1, scrollOffset = 100000)
        }
    }

    Column(modifier.fillMaxSize().padding(horizontal = 14.dp, vertical = 10.dp)) {
        // Top Toolbar
        Surface(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                // Row 1: Model Title on the left, History & New Chat on the right
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(
                        modifier = Modifier.weight(1f, fill = false),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            "LFM3-1.7B · QNN HTP",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            maxLines = 1
                        )
                        if (busy) {
                            Spacer(Modifier.width(6.dp))
                            Text(
                                "● 生成中",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }

                    // Top-right action buttons: History & New Chat
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        // History button
                        FilterChip(
                            selected = showHistorySheet,
                            onClick = {
                                if (!busy) {
                                    refreshHistory()
                                    showHistorySheet = true
                                }
                            },
                            label = { Text("📜 历史 (${historyList.size})", fontSize = 12.sp) },
                            enabled = !busy,
                            modifier = Modifier.height(32.dp)
                        )

                        // New chat button
                        OutlinedButton(
                            onClick = startNewChat,
                            enabled = !busy && messages.isNotEmpty(),
                            modifier = Modifier.height(32.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp)
                        ) {
                            Text("➕ 新对话", fontSize = 12.sp)
                        }
                    }
                }

                Spacer(Modifier.height(6.dp))
                HorizontalDivider(
                    color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.4f),
                    thickness = 0.5.dp
                )
                Spacer(Modifier.height(6.dp))

                // Row 2: Current Session Title / Status on the left, Tokens & Clear on the right
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Session title or current status
                    Text(
                        text = if (currentSessionId != null) "会话：$currentSessionTitle" else status,
                        style = MaterialTheme.typography.labelSmall,
                        color = if (currentSessionId != null) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier
                            .weight(1f)
                            .padding(end = 8.dp)
                    )

                    // Right action group: Max Tokens & Clear Screen
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        // Max tokens selector dropdown
                        Box {
                            FilterChip(
                                selected = true,
                                onClick = { if (!busy) tokenMenuExpanded = true },
                                label = { Text("$maxTokens 词 ▾", fontSize = 12.sp) },
                                enabled = !busy,
                                modifier = Modifier.height(32.dp)
                            )
                            DropdownMenu(
                                expanded = tokenMenuExpanded,
                                onDismissRequest = { tokenMenuExpanded = false }
                            ) {
                                tokenOptions.forEach { opt ->
                                    DropdownMenuItem(
                                        text = {
                                            Text(
                                                "$opt 词元" + if (opt == maxTokens) " ✓" else "",
                                                fontWeight = if (opt == maxTokens) FontWeight.Bold else FontWeight.Normal
                                            )
                                        },
                                        onClick = {
                                            maxTokens = opt
                                            tokenMenuExpanded = false
                                        }
                                    )
                                }
                            }
                        }

                        // Clear screen button
                        TextButton(
                            onClick = {
                                messages.clear()
                                currentSessionId = null
                                currentSessionTitle = "新对话"
                            },
                            enabled = !busy && messages.isNotEmpty(),
                            modifier = Modifier.height(32.dp),
                            contentPadding = PaddingValues(horizontal = 6.dp)
                        ) {
                            Text("清屏", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        // Chat Messages Area
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (messages.isEmpty()) {
                item {
                    EmptyChatGreeting(
                        ready = status.contains("已就绪"),
                        onSuggestionClick = { sendText(it) }
                    )
                }
            }

            itemsIndexed(messages) { index, (role, text) ->
                val isAssistant = role == "assistant"
                val isLastAssistant = isAssistant && (index == messages.lastIndex) && busy

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = if (isAssistant) Arrangement.Start else Arrangement.End
                ) {
                    if (isAssistant) {
                        // Assistant message bubble (full markdown formatted)
                        Surface(
                            modifier = Modifier.fillMaxWidth(0.96f),
                            shape = RoundedCornerShape(topStart = 4.dp, topEnd = 16.dp, bottomEnd = 16.dp, bottomStart = 16.dp),
                            color = MaterialTheme.colorScheme.surface,
                            tonalElevation = 2.dp,
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
                        ) {
                            Column(Modifier.padding(12.dp)) {
                                // Header: model badge & copy full answer button
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text(
                                            "🤖 LFM3",
                                            style = MaterialTheme.typography.labelSmall,
                                            fontWeight = FontWeight.Bold,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        if (isLastAssistant) {
                                            Spacer(Modifier.width(6.dp))
                                            Text(
                                                "正在输出…",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = MaterialTheme.colorScheme.tertiary
                                            )
                                        }
                                    }
                                    TextButton(
                                        onClick = {
                                            val textToCopy = if (text.contains("</think>")) {
                                                text.substringAfter("</think>").trim()
                                            } else {
                                                text
                                            }
                                            clipboardManager.setText(AnnotatedString(textToCopy))
                                            Toast.makeText(context, "已复制回复内容", Toast.LENGTH_SHORT).show()
                                        },
                                        contentPadding = PaddingValues(horizontal = 6.dp, vertical = 2.dp),
                                        modifier = Modifier.height(26.dp)
                                    ) {
                                        Text("复制全文", style = MaterialTheme.typography.labelSmall)
                                    }
                                }

                                Spacer(Modifier.height(4.dp))

                                // Render rich markdown content
                                MarkdownMessageView(
                                    text = text,
                                    isStreaming = isLastAssistant
                                )
                            }
                        }
                    } else {
                        // User message bubble
                        Surface(
                            shape = RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp, bottomEnd = 4.dp, bottomStart = 16.dp),
                            color = MaterialTheme.colorScheme.primaryContainer,
                            tonalElevation = 1.dp
                        ) {
                            Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                                Text(
                                    "你",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(Modifier.height(2.dp))
                                Text(
                                    text = text,
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onPrimaryContainer
                                )
                            }
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        // Input & Actions Row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                enabled = !busy,
                modifier = Modifier.weight(1f),
                maxLines = 4,
                placeholder = { Text("输入消息… (上限 $maxTokens 词元)") },
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
                    imeAction = androidx.compose.ui.text.input.ImeAction.Send
                ),
                keyboardActions = androidx.compose.foundation.text.KeyboardActions(
                    onSend = { sendText(input) }
                )
            )

            if (busy) {
                Button(
                    onClick = stopGeneration,
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                    modifier = Modifier.height(56.dp)
                ) {
                    Text("停止")
                }
            } else {
                Button(
                    enabled = input.isNotBlank() && status.contains("已就绪"),
                    onClick = { sendText(input) },
                    modifier = Modifier.height(56.dp)
                ) {
                    Text("发送")
                }
            }
        }
    }

    // Chat History Modal Bottom Sheet
    ChatHistorySheet(
        visible = showHistorySheet,
        onDismiss = { showHistorySheet = false },
        currentSessionId = currentSessionId,
        sessions = historyList,
        onSelectSession = loadSessionById,
        onNewChat = startNewChat,
        onDeleteSession = deleteSessionById,
        onClearAll = clearAllHistory
    )
}

@Composable
private fun EmptyChatGreeting(
    ready: Boolean,
    onSuggestionClick: (String) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f)),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("💬 LFM3 本地对话助手", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(
                "现已支持：\n• 🛠️ JavaScript 运行时与自主工具调用：模型可调用工具自测红黑树等算法\n• ⚡ 交互式代码测试：在任何 JavaScript 代码块上一键「▶ 运行 JS」并查看实时输出\n• 💭 CoT 深度思考过程折叠与复制\n• 📜 会话历史存储：随时查看往期记录并载入继续聊\n• 📝 4096 / 8192 词元长文本生成与 Markdown 高亮",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(Modifier.height(4.dp))
            Text("快速体验：", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)

            val suggestions = listOf(
                "请用 JavaScript 实现一个红黑树，并调用 JavaScript 工具运行测试以确保正确性",
                "写一个双指针快速排序，并调用工具测试逆序和重复元素用例",
                "请用 JavaScript 编写矩阵快速幂计算斐波那契数列第50项并运行验证",
                "请解释红黑树与普通平衡二叉树的区别"
            )

            for (suggestion in suggestions) {
                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surface,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .clickable(enabled = ready) { onSuggestionClick(suggestion) }
                ) {
                    Text(
                        suggestion,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                        style = MaterialTheme.bodySmall(MaterialTheme.typography.bodySmall),
                        color = if (ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
                    )
                }
            }
        }
    }
}

// Helper to keep style clean
private fun MaterialTheme.bodySmall(base: androidx.compose.ui.text.TextStyle): androidx.compose.ui.text.TextStyle = base
