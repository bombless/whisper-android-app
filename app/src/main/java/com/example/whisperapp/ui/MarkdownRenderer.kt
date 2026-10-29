package com.example.whisperapp.ui

import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.whisperapp.js.JsExecutionResult
import com.example.whisperapp.js.JsRuntime
import com.example.whisperapp.js.LfmToolCallParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

/**
 * Structural representations of Markdown blocks.
 */
sealed interface ChatMessageBlock {
    data class CodeBlock(val language: String, val code: String) : ChatMessageBlock
    data class ToolExecutionBlock(
        val toolName: String,
        val code: String,
        val response: String?,
        val isRunning: Boolean
    ) : ChatMessageBlock
    data class HeadingBlock(val level: Int, val text: String) : ChatMessageBlock
    data class BulletItemBlock(val text: String) : ChatMessageBlock
    data class NumberedItemBlock(val number: String, val text: String) : ChatMessageBlock
    data class QuoteBlock(val text: String) : ChatMessageBlock
    data class TextBlock(val content: String) : ChatMessageBlock
}

/**
 * Parses markdown text into structured blocks (Code blocks, tool executions, headings, lists, quotes, paragraphs).
 * Supports partial/streaming markdown safely.
 */
private val STRAY_XML_REGEX = Regex(
    """</?(?:tool_call(?::[a-zA-Z0-9_-]+)?|run_javascript|function(?:=[a-zA-Z0-9_-]+)?|parameter(?:=[a-zA-Z0-9_-]+)?|tool_response|tools|additional_tools)(?:\s*[^>]*)?>""",
    RegexOption.IGNORE_CASE
)

private val TOOL_CALL_BLOCK_PATTERN = Regex(
    """(?:<tool_call(?::([a-zA-Z0-9_-]+))?>([\s\S]*?)(?:</tool_call(?::\1)?>|$))""" +
    """|(?:<run_javascript>([\s\S]*?)(?:</run_javascript>|$))""" +
    """|(?:<function(?:=([a-zA-Z0-9_-]+))?>([\s\S]*?)(?:</function>|$))""",
    RegexOption.IGNORE_CASE
)

private val TOOL_RESPONSE_BLOCK_PATTERN = Regex(
    """\s*<tool_response>([\s\S]*?)(?:</tool_response>|$)""",
    RegexOption.IGNORE_CASE
)

/**
 * Parses markdown text into structured blocks (Code blocks, tool executions, headings, lists, quotes, paragraphs).
 * Supports partial/streaming markdown safely with zero XML tag leakage.
 */
fun parseMarkdownBlocks(rawText: String): List<ChatMessageBlock> {
    if (!rawText.contains("<tool_call", ignoreCase = true) &&
        !rawText.contains("<run_javascript", ignoreCase = true) &&
        !rawText.contains("<function", ignoreCase = true)) {
        return parseStandardMarkdownBlocks(rawText)
    }

    val blocks = mutableListOf<ChatMessageBlock>()
    var cursor = 0
    val matches = TOOL_CALL_BLOCK_PATTERN.findAll(rawText).toList()

    for (match in matches) {
        val start = match.range.first
        if (start < cursor) continue

        if (start > cursor) {
            val prefix = rawText.substring(cursor, start).trim()
            if (prefix.isNotEmpty()) {
                blocks.addAll(parseStandardMarkdownBlocks(prefix))
            }
        }

        val g1 = match.groupValues.getOrNull(1)?.takeIf { it.isNotBlank() }
        val g2 = match.groupValues.getOrNull(2)
        val g3 = match.groupValues.getOrNull(3)
        val g4 = match.groupValues.getOrNull(4)?.takeIf { it.isNotBlank() }
        val g5 = match.groupValues.getOrNull(5)

        val rawCode: String
        val toolName: String
        val isClosed: Boolean

        when {
            g2 != null -> {
                // <tool_call>...</tool_call>
                rawCode = g2
                val extractedName = Regex("""<function(?:\s*=\s*([a-zA-Z0-9_-]+)|\s+name="?([a-zA-Z0-9_-]+)"?)""", RegexOption.IGNORE_CASE)
                    .find(rawCode)?.let { it.groupValues[1].ifEmpty { it.groupValues[2] } }
                toolName = g1 ?: extractedName ?: "JavaScript 运行时"
                isClosed = match.value.contains("</tool_call", ignoreCase = true)
            }
            g3 != null -> {
                // <run_javascript>...</run_javascript>
                rawCode = g3
                toolName = "JavaScript 运行时"
                isClosed = match.value.contains("</run_javascript", ignoreCase = true)
            }
            else -> {
                // <function=...>...</function>
                rawCode = g5 ?: ""
                toolName = g4 ?: "JavaScript 运行时"
                isClosed = match.value.contains("</function", ignoreCase = true)
            }
        }

        val code = LfmToolCallParser.extractCode(rawCode)

        val afterToolCall = rawText.substring(match.range.last + 1)
        val responseMatch = TOOL_RESPONSE_BLOCK_PATTERN.find(afterToolCall)
        val response = if (responseMatch != null && responseMatch.range.first == 0) {
            cursor = match.range.last + 1 + responseMatch.range.last + 1
            responseMatch.groupValues[1].replace(STRAY_XML_REGEX, "").trim()
        } else {
            cursor = match.range.last + 1
            null
        }

        val isRunning = response == null && !isClosed
        val displayName = when {
            toolName.equals("generate_image", ignoreCase = true) || toolName.contains("image", ignoreCase = true) || toolName.contains("生图") -> "本地 SDXL 生图"
            toolName.equals("run_javascript", ignoreCase = true) -> "JavaScript 运行时"
            else -> toolName
        }
        blocks.add(
            ChatMessageBlock.ToolExecutionBlock(
                toolName = displayName,
                code = code,
                response = response,
                isRunning = isRunning
            )
        )
    }

    if (cursor < rawText.length) {
        val remaining = rawText.substring(cursor).trim()
        if (remaining.isNotEmpty()) {
            blocks.addAll(parseStandardMarkdownBlocks(remaining))
        }
    }

    return blocks
}

private fun parseStandardMarkdownBlocks(rawText: String): List<ChatMessageBlock> {
    val sanitized = rawText.replace(STRAY_XML_REGEX, "").trim()
    if (sanitized.isEmpty()) return emptyList()

    val blocks = mutableListOf<ChatMessageBlock>()
    val lines = sanitized.lines()
    var i = 0
    val textAccumulator = StringBuilder()

    fun flushText() {
        if (textAccumulator.isNotEmpty()) {
            val content = textAccumulator.toString().trim()
            if (content.isNotEmpty()) {
                blocks.add(ChatMessageBlock.TextBlock(content))
            }
            textAccumulator.clear()
        }
    }

    while (i < lines.size) {
        val line = lines[i]
        val trimmed = line.trim()

        // 1. Fenced Code Block (e.g. ```javascript)
        if (trimmed.startsWith("```")) {
            flushText()
            val lang = trimmed.removePrefix("```").trim()
            val codeLines = mutableListOf<String>()
            i++
            while (i < lines.size) {
                if (lines[i].trim().startsWith("```")) {
                    i++
                    break
                } else {
                    codeLines.add(lines[i])
                    i++
                }
            }
            blocks.add(ChatMessageBlock.CodeBlock(lang, codeLines.joinToString("\n")))
            continue
        }

        // 2. Headings (#, ##, ###)
        val headingMatch = Regex("^(#{1,6})\\s+(.*)$").matchEntire(trimmed)
        if (headingMatch != null) {
            flushText()
            val level = headingMatch.groupValues[1].length
            val title = headingMatch.groupValues[2]
            blocks.add(ChatMessageBlock.HeadingBlock(level, title))
            i++
            continue
        }

        // 3. Blockquotes (> quote)
        if (trimmed.startsWith(">")) {
            flushText()
            val quoteContent = trimmed.removePrefix(">").trim()
            blocks.add(ChatMessageBlock.QuoteBlock(quoteContent))
            i++
            continue
        }

        // 4. Bullet lists (- item or * item)
        val bulletMatch = Regex("^[-*]\\s+(.*)$").matchEntire(trimmed)
        if (bulletMatch != null) {
            flushText()
            blocks.add(ChatMessageBlock.BulletItemBlock(bulletMatch.groupValues[1]))
            i++
            continue
        }

        // 5. Numbered lists (1. item)
        val numMatch = Regex("^(\\d+\\.)\\s+(.*)$").matchEntire(trimmed)
        if (numMatch != null) {
            flushText()
            blocks.add(ChatMessageBlock.NumberedItemBlock(numMatch.groupValues[1], numMatch.groupValues[2]))
            i++
            continue
        }

        // 6. Blank lines separate paragraphs
        if (trimmed.isEmpty()) {
            flushText()
            i++
            continue
        }

        // 7. Regular paragraph text line
        if (textAccumulator.isNotEmpty()) textAccumulator.append("\n")
        textAccumulator.append(line)
        i++
    }

    flushText()
    return blocks
}

private val INLINE_REGEX = Regex(
    "(`[^`]+`)|(\\*\\*\\*(?:[^*]|\\*(?!\\*\\*))+\\*\\*\\*)|(\\*\\*(?:[^*]|\\*(?!\\*))+\\*\\*)|(__.+?__)|(\\*(?:[^*])+\\*)|(_.+?_)"
)

/**
 * Builds an AnnotatedString that formats bold, italic, and inline code with appropriate styles.
 */
fun buildAnnotatedMarkdown(
    text: String,
    codeBg: Color = Color(0x1F888888),
    codeColor: Color = Color(0xFFE57373)
): AnnotatedString {
    return buildAnnotatedString {
        var cursor = 0
        val matches = INLINE_REGEX.findAll(text).toList()

        for (match in matches) {
            val range = match.range
            if (range.first > cursor) {
                append(text.substring(cursor, range.first))
            }

            val value = match.value
            when {
                value.startsWith("`") && value.endsWith("`") -> {
                    withStyle(
                        SpanStyle(
                            background = codeBg,
                            color = codeColor,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 13.sp
                        )
                    ) {
                        append(" ${value.removeSurrounding("`")} ")
                    }
                }
                value.startsWith("***") && value.endsWith("***") -> {
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontStyle = FontStyle.Italic)) {
                        append(value.removeSurrounding("***"))
                    }
                }
                (value.startsWith("**") && value.endsWith("**")) ||
                (value.startsWith("__") && value.endsWith("__")) -> {
                    val raw = if (value.startsWith("**")) value.removeSurrounding("**") else value.removeSurrounding("__")
                    withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                        append(raw)
                    }
                }
                (value.startsWith("*") && value.endsWith("*")) ||
                (value.startsWith("_") && value.endsWith("_")) -> {
                    val raw = if (value.startsWith("*")) value.removeSurrounding("*") else value.removeSurrounding("_")
                    withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                        append(raw)
                    }
                }
                else -> append(value)
            }
            cursor = range.last + 1
        }

        if (cursor < text.length) {
            append(text.substring(cursor))
        }
    }
}

/**
 * Beautiful, dark styled code block view with syntax highlighting aesthetics,
 * horizontal scrolling, one-click copy, and interactive JavaScript run button.
 */
@Composable
fun FormattedCodeBlock(
    language: String,
    code: String,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current
    val scrollState = rememberScrollState()
    val scope = rememberCoroutineScope()
    var copied by remember { mutableStateOf(false) }
    var isRunning by remember { mutableStateOf(false) }
    var runResult by remember { mutableStateOf<JsExecutionResult?>(null) }

    val isJs = language.lowercase() in listOf("javascript", "js")

    LaunchedEffect(copied) {
        if (copied) {
            delay(2000)
            copied = false
        }
    }

    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color(0xFF1E1E24),
        border = BorderStroke(1.dp, Color(0xFF383844))
    ) {
        Column {
            // Header bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF282832))
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = language.ifBlank { "code" }.lowercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = Color(0xFFB0B0C4),
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold
                )

                Row(verticalAlignment = Alignment.CenterVertically) {
                    // Interactive Run JS button
                    if (isJs) {
                        TextButton(
                            onClick = {
                                if (!isRunning) {
                                    scope.launch(Dispatchers.Default) {
                                        isRunning = true
                                        val res = JsRuntime.execute(code)
                                        runResult = res
                                        isRunning = false
                                    }
                                }
                            },
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                            modifier = Modifier.height(26.dp)
                        ) {
                            Text(
                                text = if (isRunning) "▶ 运行中…" else "▶ 运行 JS",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (isRunning) Color(0xFFFFB74D) else Color(0xFF81C784),
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }

                    // Copy button
                    TextButton(
                        onClick = {
                            clipboardManager.setText(AnnotatedString(code))
                            copied = true
                        },
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                        modifier = Modifier.height(26.dp)
                    ) {
                        Text(
                            text = if (copied) "✓ 已复制" else "复制",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (copied) Color(0xFF4CAF50) else Color(0xFF90CAF9)
                        )
                    }
                }
            }

            // Code content with horizontal scroll
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(scrollState)
            ) {
                Text(
                    text = code,
                    modifier = Modifier.padding(12.dp),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 12.5.sp,
                    lineHeight = 18.sp,
                    color = Color(0xFFF2F2F2),
                    softWrap = false
                )
            }

            // Subtle scrollbar track when code exceeds viewport width
            if (scrollState.maxValue > 0) {
                LinearProgressIndicator(
                    progress = {
                        if (scrollState.maxValue > 0) {
                            (scrollState.value.toFloat() / scrollState.maxValue.toFloat()).coerceIn(0f, 1f)
                        } else 0f
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(2.5.dp),
                    color = Color(0xFF64B5F6),
                    trackColor = Color(0xFF282832)
                )
            }

            // Interactive execution output panel
            if (runResult != null) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    shape = RoundedCornerShape(6.dp),
                    color = Color(0xFF141419),
                    border = BorderStroke(1.dp, Color(0xFF2D2D3A))
                ) {
                    Column(Modifier.padding(10.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (runResult!!.success) "⚡ 运行结果 (${runResult!!.executionTimeMs}ms)" else "❌ 运行失败 (${runResult!!.executionTimeMs}ms)",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (runResult!!.success) Color(0xFF81C784) else Color(0xFFE57373),
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "✕ 关闭",
                                style = MaterialTheme.typography.labelSmall,
                                color = Color.Gray,
                                modifier = Modifier
                                    .clip(RoundedCornerShape(4.dp))
                                    .clickable { runResult = null }
                                    .padding(horizontal = 4.dp, vertical = 2.dp)
                            )
                        }
                        val out = runResult!!.toOutputString()
                        Text(
                            text = out.ifBlank { "[状态] 执行成功，无日志输出" },
                            fontFamily = FontFamily.Monospace,
                            fontSize = 11.5.sp,
                            lineHeight = 16.sp,
                            color = Color(0xFFCCCCCC),
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Dedicated Card for displaying autonomous model tool calls and responses.
 */
@Composable
fun ToolExecutionCard(
    block: ChatMessageBlock.ToolExecutionBlock,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current
    var expanded by remember { mutableStateOf(true) }

    val isImageTool = block.toolName.contains("生图") || block.toolName.contains("image", ignoreCase = true)
    val toolIcon = if (isImageTool) "🎨" else "🛠️"
    val runningText = if (isImageTool) "🎨 正在生成图像…" else "⚙️ 执行中…"
    val successText = if (isImageTool) "✅ 图像生成完成" else "✅ 验证通过"
    val failedText = if (isImageTool) "❌ 生成失败" else "❌ 未通过"
    val codeHeader = if (isImageTool) "🎨 画面提示词 (Prompt):" else "💻 执行代码:"

    val imagePath = remember(block.response) {
        if (!isImageTool || block.response == null) null
        else {
            val regex = Regex("""\[(?:图像路径|文件路径)\]\s*([^\r\n]+)""")
            regex.find(block.response)?.groupValues?.getOrNull(1)?.trim()
        }
    }
    val bitmap = remember(imagePath) {
        imagePath?.let { path ->
            val f = File(path)
            if (f.isFile) BitmapFactory.decodeFile(f.absolutePath) else null
        }
    }

    Surface(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp)),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.45f))
    ) {
        Column(Modifier.padding(10.dp)) {
            // Header bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { expanded = !expanded },
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "$toolIcon ${block.toolName}",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(Modifier.width(8.dp))
                    if (block.isRunning) {
                        Text(
                            text = runningText,
                            style = MaterialTheme.typography.labelSmall,
                            color = Color(0xFFFFB74D),
                            fontWeight = FontWeight.SemiBold
                        )
                    } else if (block.response != null) {
                        val isFailed = block.response.contains("[执行失败]")
                        Text(
                            text = if (isFailed) failedText else successText,
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isFailed) Color(0xFFE57373) else Color(0xFF81C784),
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }

                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (block.response != null) {
                        Text(
                            text = "复制结果",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.tertiary,
                            modifier = Modifier
                                .padding(end = 8.dp)
                                .clickable {
                                    clipboardManager.setText(AnnotatedString(block.response))
                                    Toast.makeText(context, "已复制工具执行输出", Toast.LENGTH_SHORT).show()
                                }
                        )
                    }
                    Text(
                        text = if (expanded) "▲ 收起" else "▼ 展开",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            AnimatedVisibility(visible = expanded) {
                Column(Modifier.padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        text = codeHeader,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    FormattedCodeBlock(
                        language = if (isImageTool) "markdown" else "javascript",
                        code = block.code
                    )

                    if (bitmap != null) {
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp)),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.5f))
                        ) {
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "SDXL 生成图像",
                                modifier = Modifier.fillMaxWidth(),
                                contentScale = ContentScale.FillWidth
                            )
                        }
                    }

                    if (block.response != null) {
                        Text(
                            text = if (isImageTool) "📋 生成详情与输出:" else "📋 运行时输出:",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Surface(
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(6.dp),
                            color = Color(0xFF141419),
                            border = BorderStroke(1.dp, Color(0xFF2E2E3C))
                        ) {
                            Text(
                                text = block.response,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.5.sp,
                                lineHeight = 16.sp,
                                color = Color(0xFFDDDDDD),
                                modifier = Modifier.padding(10.dp)
                            )
                        }
                    } else if (block.isRunning) {
                        LinearProgressIndicator(
                            modifier = Modifier.fillMaxWidth().height(3.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            }
        }
    }
}

/**
 * Renders a full Markdown formatted message.
 */
@Composable
fun MarkdownMessageView(
    text: String,
    modifier: Modifier = Modifier,
    isStreaming: Boolean = false
) {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current

    // Check for thinking blocks: <think> ... </think>
    val thinkPattern = Regex("<think>([\\s\\S]*?)(</think>|$)", RegexOption.IGNORE_CASE)
    val thinkMatch = thinkPattern.find(text)
    val hasThink = thinkMatch != null
    val thinkContent = thinkMatch?.groupValues?.getOrNull(1)?.trim() ?: ""
    val isThinkingInProgress = hasThink && !text.contains("</think>", ignoreCase = true)

    val cleanText = if (hasThink) {
        text.replace(thinkPattern, "").trim()
    } else {
        text
    }

    val blocks = remember(cleanText) { parseMarkdownBlocks(cleanText) }
    var manuallyToggled by remember { mutableStateOf(false) }
    var thinkExpanded by remember { mutableStateOf(isThinkingInProgress) }

    LaunchedEffect(isThinkingInProgress) {
        if (!manuallyToggled) {
            thinkExpanded = isThinkingInProgress
        }
    }

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        // Thinking process collapsible section
        if (hasThink && thinkContent.isNotBlank()) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .clickable {
                        manuallyToggled = true
                        thinkExpanded = !thinkExpanded
                    },
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant)
            ) {
                Column(Modifier.padding(10.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = if (isThinkingInProgress) "💭 正在思考…" else "💭 思考过程",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.SemiBold
                        )
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            if (!isThinkingInProgress && thinkContent.isNotBlank()) {
                                Text(
                                    text = "复制思考",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.tertiary,
                                    modifier = Modifier
                                        .padding(end = 10.dp)
                                        .clickable {
                                            clipboardManager.setText(AnnotatedString(thinkContent))
                                            Toast.makeText(context, "已复制思考过程", Toast.LENGTH_SHORT).show()
                                        }
                                )
                            }
                            Text(
                                text = if (thinkExpanded) "▲ 收起" else "▼ 展开",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    AnimatedVisibility(visible = thinkExpanded) {
                        Column(Modifier.padding(top = 8.dp)) {
                            Text(
                                text = thinkContent,
                                style = MaterialTheme.typography.bodySmall,
                                fontStyle = FontStyle.Italic,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }

        // Main Markdown blocks
        for (block in blocks) {
            when (block) {
                is ChatMessageBlock.ToolExecutionBlock -> {
                    ToolExecutionCard(block)
                }
                is ChatMessageBlock.CodeBlock -> {
                    FormattedCodeBlock(language = block.language, code = block.code)
                }
                is ChatMessageBlock.HeadingBlock -> {
                    val style = when (block.level) {
                        1 -> MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold)
                        2 -> MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        else -> MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.Bold)
                    }
                    Text(
                        text = buildAnnotatedMarkdown(block.text),
                        style = style,
                        modifier = Modifier.padding(top = 4.dp, bottom = 2.dp)
                    )
                }
                is ChatMessageBlock.BulletItemBlock -> {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 1.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Text(
                            text = "• ",
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = buildAnnotatedMarkdown(block.text),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
                is ChatMessageBlock.NumberedItemBlock -> {
                    Row(
                        modifier = Modifier.fillMaxWidth().padding(vertical = 1.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Text(
                            text = "${block.number} ",
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = buildAnnotatedMarkdown(block.text),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
                is ChatMessageBlock.QuoteBlock -> {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 2.dp)
                    ) {
                        Box(
                            Modifier
                                .width(3.dp)
                                .height(22.dp)
                                .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(2.dp))
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = buildAnnotatedMarkdown(block.text),
                            style = MaterialTheme.typography.bodyMedium,
                            fontStyle = FontStyle.Italic,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                is ChatMessageBlock.TextBlock -> {
                    Text(
                        text = buildAnnotatedMarkdown(block.content),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }

        // Streaming blinking cursor indicator
        if (isStreaming) {
            Text(
                text = "▊",
                color = MaterialTheme.colorScheme.primary,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}
