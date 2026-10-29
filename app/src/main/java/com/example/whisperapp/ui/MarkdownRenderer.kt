package com.example.whisperapp.ui

import android.widget.Toast
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
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
import kotlinx.coroutines.delay

/**
 * Structural representations of Markdown blocks.
 */
sealed interface ChatMessageBlock {
    data class CodeBlock(val language: String, val code: String) : ChatMessageBlock
    data class HeadingBlock(val level: Int, val text: String) : ChatMessageBlock
    data class BulletItemBlock(val text: String) : ChatMessageBlock
    data class NumberedItemBlock(val number: String, val text: String) : ChatMessageBlock
    data class QuoteBlock(val text: String) : ChatMessageBlock
    data class TextBlock(val content: String) : ChatMessageBlock
}

/**
 * Parses markdown text into structured blocks (Code blocks, headings, lists, quotes, paragraphs).
 * Supports partial/streaming markdown safely.
 */
fun parseMarkdownBlocks(rawText: String): List<ChatMessageBlock> {
    val blocks = mutableListOf<ChatMessageBlock>()
    val lines = rawText.lines()
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

        // 1. Fenced Code Block (e.g. ```python)
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
    codeColor: Color = Color.Unspecified
): AnnotatedString = buildAnnotatedString {
    var lastIndex = 0
    for (match in INLINE_REGEX.findAll(text)) {
        if (match.range.first > lastIndex) {
            append(text.substring(lastIndex, match.range.first))
        }
        val raw = match.value
        when {
            raw.startsWith("`") && raw.endsWith("`") && raw.length >= 2 -> {
                val code = raw.substring(1, raw.length - 1)
                withStyle(
                    SpanStyle(
                        fontFamily = FontFamily.Monospace,
                        background = codeBg,
                        color = codeColor,
                        fontWeight = FontWeight.SemiBold
                    )
                ) {
                    append(" $code ")
                }
            }
            raw.startsWith("***") && raw.endsWith("***") && raw.length >= 6 -> {
                val inner = raw.substring(3, raw.length - 3)
                withStyle(SpanStyle(fontWeight = FontWeight.Bold, fontStyle = FontStyle.Italic)) {
                    append(inner)
                }
            }
            (raw.startsWith("**") && raw.endsWith("**") && raw.length >= 4) ||
            (raw.startsWith("__") && raw.endsWith("__") && raw.length >= 4) -> {
                val inner = raw.substring(2, raw.length - 2)
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                    append(inner)
                }
            }
            (raw.startsWith("*") && raw.endsWith("*") && raw.length >= 2) ||
            (raw.startsWith("_") && raw.endsWith("_") && raw.length >= 2) -> {
                val inner = raw.substring(1, raw.length - 1)
                withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                    append(inner)
                }
            }
            else -> {
                append(raw)
            }
        }
        lastIndex = match.range.last + 1
    }
    if (lastIndex < text.length) {
        append(text.substring(lastIndex))
    }
}

/**
 * Renders a code block in a dark terminal style container with language badge,
 * copy button, and horizontal scrolling for monospace code.
 */
@Composable
fun FormattedCodeBlock(
    language: String,
    code: String,
    modifier: Modifier = Modifier
) {
    val clipboardManager = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

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

            // Code content with horizontal scroll to prevent wrap distortion
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

    // Auto-expand while generating thoughts, auto-collapse when thought finishes and code arrives
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
