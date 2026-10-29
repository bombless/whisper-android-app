package com.example.whisperapp.ui

import android.util.Log
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.whisperapp.qnn.QnnLfm2ChatRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

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

    // Configurable Max Tokens: default 4096 (expanded from 1024)
    val tokenOptions = remember { listOf(1024, 2048, 4096, 8192) }
    var maxTokens by remember { mutableIntStateOf(4096) }
    var tokenMenuExpanded by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        status = QnnLfm2ChatRunner.start(context).message
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
    }

    val sendText: (String) -> Unit = { rawPrompt ->
        val text = rawPrompt.trim()
        if (!busy && text.isNotBlank() && status.contains("已就绪")) {
            input = ""
            messages += "user" to text
            val historySnapshot = messages.toList()
            messages += "assistant" to "正在思考…"
            val assistantIndex = messages.lastIndex
            busy = true

            Log.i("CHAT_UI", "SEND chars=${text.length} maxTokens=$maxTokens history=${historySnapshot.size}")
            currentJob = scope.launch(Dispatchers.Default) {
                val answer = runCatching {
                    QnnLfm2ChatRunner.generate(
                        context = context,
                        history = historySnapshot,
                        maxTokens = maxTokens,
                        isCancelled = { !isActive }
                    ) { partialText ->
                        launch(Dispatchers.Main) {
                            if (assistantIndex in messages.indices) {
                                messages[assistantIndex] = "assistant" to partialText
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
            }
        }
    }

    // Auto-scroll to bottom as new tokens arrive
    LaunchedEffect(messages.size, messages.lastOrNull()?.second?.length) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1, scrollOffset = 100000)
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
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        "LFM3-1.7B · QNN HTP",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1
                    )
                    Text(
                        status,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1
                    )
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
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

                    Spacer(Modifier.width(6.dp))

                    // Clear button
                    TextButton(
                        onClick = { messages.clear() },
                        enabled = !busy && messages.isNotEmpty(),
                        modifier = Modifier.height(32.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp)
                    ) {
                        Text("清空", style = MaterialTheme.typography.labelSmall)
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
                "现已支持：\n• CoT 深度思考引导：以「让我把完整的方案想清楚再输出」进行长链路规划\n• 最大 4096 / 8192 词元长文本生成（解决红黑树等复杂代码截断）\n• 代码块高亮、横向滚动与一键复制代码\n• 折叠式思考过程与 Markdown 渲染",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(Modifier.height(4.dp))
            Text("快速体验：", style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)

            val suggestions = listOf(
                "write a python red black tree",
                "用 Python 实现一个完整的红黑树",
                "写一个带有协程取消处理的 Kotlin 示例",
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
                        style = MaterialTheme.typography.bodySmall,
                        color = if (ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
                    )
                }
            }
        }
    }
}
