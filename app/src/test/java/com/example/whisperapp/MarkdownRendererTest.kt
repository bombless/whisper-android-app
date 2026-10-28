package com.example.whisperapp

import com.example.whisperapp.ui.ChatMessageBlock
import com.example.whisperapp.ui.buildAnnotatedMarkdown
import com.example.whisperapp.ui.parseMarkdownBlocks
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MarkdownRendererTest {

    @Test
    fun testParseCodeBlocksAndHeadings() {
        val markdown = """
            # 红黑树实现
            
            这是一个 **红黑树** 的实现。
            
            ```python
            class Node:
                def __init__(self, key):
                    self.key = key
            ```
            
            - 颜色只能是红色或黑色
            - 根节点为黑色
            
            > 平衡二叉搜索树
        """.trimIndent()

        val blocks = parseMarkdownBlocks(markdown)
        assertTrue("Blocks should not be empty", blocks.isNotEmpty())

        val heading = blocks.filterIsInstance<ChatMessageBlock.HeadingBlock>().firstOrNull()
        assertEquals("红黑树实现", heading?.text)
        assertEquals(1, heading?.level)

        val codeBlock = blocks.filterIsInstance<ChatMessageBlock.CodeBlock>().firstOrNull()
        assertEquals("python", codeBlock?.language)
        assertTrue(codeBlock?.code?.contains("class Node:") == true)

        val bullets = blocks.filterIsInstance<ChatMessageBlock.BulletItemBlock>()
        assertEquals(2, bullets.size)
        assertEquals("颜色只能是红色或黑色", bullets[0].text)

        val quote = blocks.filterIsInstance<ChatMessageBlock.QuoteBlock>().firstOrNull()
        assertEquals("平衡二叉搜索树", quote?.text)
    }

    @Test
    fun testStreamingIncompleteCodeBlock() {
        val incompleteMarkdown = """
            正在输出代码：
            ```python
            class Node:
                def __init__(self):
                    pass
        """.trimIndent()

        val blocks = parseMarkdownBlocks(incompleteMarkdown)
        val codeBlock = blocks.filterIsInstance<ChatMessageBlock.CodeBlock>().firstOrNull()
        assertTrue("Streaming code block should be parsed", codeBlock != null)
        assertEquals("python", codeBlock?.language)
        assertTrue(codeBlock?.code?.contains("class Node:") == true)
    }

    @Test
    fun testInlineMarkdownStyling() {
        val text = "这是 **粗体** 和 `行内代码` 以及 *斜体*"
        val annotated = buildAnnotatedMarkdown(text)
        assertEquals("这是 粗体 和  行内代码  以及 斜体", annotated.text)
        assertTrue("Should have span styles", annotated.spanStyles.isNotEmpty())
    }
}
