package io.github.bombless.htp_playpen

import io.github.bombless.htp_playpen.ui.ChatMessageBlock
import io.github.bombless.htp_playpen.ui.buildAnnotatedMarkdown
import io.github.bombless.htp_playpen.ui.parseMarkdownBlocks
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
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

    @Test
    fun testThinkingProcessExtraction() {
        val messageWithThinking = """
            <think>让我把完整的方案想清楚再输出
            首先分析红黑树的性质：
            1. 根节点是黑色
            2. 红节点的孩子必须是黑节点
            </think>
            下面是完整的红黑树 Python 实现：
            ```python
            class RBNode:
                pass
            ```
        """.trimIndent()

        val thinkPattern = Regex("<think>([\\s\\S]*?)(</think>|$)", RegexOption.IGNORE_CASE)
        val match = thinkPattern.find(messageWithThinking)
        assertTrue("Should find thinking block", match != null)
        val thinkingContent = match?.groupValues?.getOrNull(1)?.trim() ?: ""
        assertTrue("Thinking content should start with user prompt", thinkingContent.startsWith("让我把完整的方案想清楚再输出"))
        assertTrue("Thinking content should contain analysis", thinkingContent.contains("首先分析红黑树的性质"))

        val cleanText = messageWithThinking.replace(thinkPattern, "").trim()
        assertTrue("Clean text should contain python code", cleanText.contains("```python"))
        assertTrue("Clean text should not contain think tag", !cleanText.contains("<think>"))

        val blocks = parseMarkdownBlocks(cleanText)
        val codeBlock = blocks.filterIsInstance<ChatMessageBlock.CodeBlock>().firstOrNull()
        assertTrue("Code block should be parsed", codeBlock != null)
        assertEquals("python", codeBlock?.language)
    }

    @Test
    fun testToolCallBlockParsing() {
        val message = """
            让我用 JavaScript 编写并测试红黑树：
            <tool_call>
            ```javascript
            const tree = new RedBlackTree();
            tree.insert(10);
            console.log("Root:", tree.root.key);
            ```
            </tool_call>
            <tool_response>
            [输出]
            Root: 10
            [返回值] undefined
            </tool_response>
            测试表明红黑树插入成功，黑色高度平衡！
        """.trimIndent()

        val blocks = parseMarkdownBlocks(message)
        assertEquals(3, blocks.size)
        assertTrue(blocks[0] is ChatMessageBlock.TextBlock)
        assertTrue((blocks[0] as ChatMessageBlock.TextBlock).content.contains("让我用 JavaScript 编写并测试红黑树"))

        assertTrue(blocks[1] is ChatMessageBlock.ToolExecutionBlock)
        val toolBlock = blocks[1] as ChatMessageBlock.ToolExecutionBlock
        assertEquals("JavaScript 运行时", toolBlock.toolName)
        assertTrue("Tool block code should be extracted", toolBlock.code.contains("const tree = new RedBlackTree();"))
        assertTrue("Tool block code should not contain fences", !toolBlock.code.contains("```"))
        assertNotNull(toolBlock.response)
        assertTrue("Tool response should contain Root: 10", toolBlock.response?.contains("Root: 10") == true)
        assertEquals(false, toolBlock.isRunning)

        assertTrue(blocks[2] is ChatMessageBlock.TextBlock)
        assertTrue((blocks[2] as ChatMessageBlock.TextBlock).content.contains("测试表明红黑树插入成功"))
    }

    @Test
    fun testNativeLfm3XmlToolCallRenderingWithoutXmlLeakage() {
        val message = """
            让我用 JavaScript 编写并测试红黑树：
            <tool_call>
            <function=run_javascript>
            <parameter=code>
            const tree = new RedBlackTree();
            tree.insert(10);
            console.log("Root:", tree.root.key);
            </parameter>
            </function>
            </tool_call>
            <tool_response>
            [输出]
            Root: 10
            [返回值] undefined
            </tool_response>
            测试表明红黑树插入成功，黑色高度平衡！
        """.trimIndent()

        val blocks = parseMarkdownBlocks(message)
        assertEquals(3, blocks.size)
        assertTrue(blocks[0] is ChatMessageBlock.TextBlock)
        assertTrue((blocks[0] as ChatMessageBlock.TextBlock).content.contains("让我用 JavaScript 编写并测试红黑树"))

        assertTrue(blocks[1] is ChatMessageBlock.ToolExecutionBlock)
        val toolBlock = blocks[1] as ChatMessageBlock.ToolExecutionBlock
        assertEquals("JavaScript 运行时", toolBlock.toolName)
        assertTrue("Tool block code should contain tree insert", toolBlock.code.contains("tree.insert(10);"))
        assertTrue("Tool block code MUST NOT contain <function>", !toolBlock.code.contains("<function"))
        assertTrue("Tool block code MUST NOT contain <parameter>", !toolBlock.code.contains("<parameter"))
        assertTrue("Tool block code MUST NOT contain </function>", !toolBlock.code.contains("</function"))
        assertTrue("Tool block code MUST NOT contain </parameter>", !toolBlock.code.contains("</parameter"))
        assertNotNull(toolBlock.response)
        assertTrue("Tool response should contain Root: 10", toolBlock.response?.contains("Root: 10") == true)
        assertTrue("Tool response MUST NOT contain <tool_response>", toolBlock.response?.contains("<tool_response>") == false)
        assertEquals(false, toolBlock.isRunning)

        assertTrue(blocks[2] is ChatMessageBlock.TextBlock)
        val textBlock = blocks[2] as ChatMessageBlock.TextBlock
        assertTrue(textBlock.content.contains("测试表明红黑树插入成功"))
        assertTrue("TextBlock MUST NOT contain </tool_call>", !textBlock.content.contains("</tool_call>"))
        assertTrue("TextBlock MUST NOT contain <tool_response>", !textBlock.content.contains("<tool_response>"))
        assertTrue("TextBlock MUST NOT contain </tool_response>", !textBlock.content.contains("</tool_response>"))
    }

    @Test
    fun testStreamingNativeXmlToolCallNoLeakage() {
        val streamingMessage = "让我写个算法：\n<tool_call>\n<function=run_javascript>\n<parameter=code>\nconst x = 123;"
        val blocks = parseMarkdownBlocks(streamingMessage)
        assertEquals(2, blocks.size)
        assertTrue(blocks[0] is ChatMessageBlock.TextBlock)
        assertTrue(blocks[1] is ChatMessageBlock.ToolExecutionBlock)

        val toolBlock = blocks[1] as ChatMessageBlock.ToolExecutionBlock
        assertEquals("const x = 123;", toolBlock.code)
        assertTrue("Must not contain <function>", !toolBlock.code.contains("<function"))
        assertTrue("Must not contain <parameter>", !toolBlock.code.contains("<parameter"))
        assertEquals(true, toolBlock.isRunning)
        assertEquals(null, toolBlock.response)
    }

    @Test
    fun testStrayXmlTagSanitizationInText() {
        val textWithStrayTags = "执行完成，包含 </function>\n</tool_call>\n<tool_response> 标签，应该被过滤"
        val blocks = parseMarkdownBlocks(textWithStrayTags)
        val text = blocks.filterIsInstance<ChatMessageBlock.TextBlock>().joinToString("\n") { it.content }
        assertTrue("Stray tag </function> should be sanitized", !text.contains("</function>"))
        assertTrue("Stray tag </tool_call> should be sanitized", !text.contains("</tool_call>"))
        assertTrue("Stray tag <tool_response> should be sanitized", !text.contains("<tool_response>"))
        assertTrue(text.contains("执行完成"))
        assertTrue(text.contains("应该被过滤"))
    }

    @Test
    fun testGenerateImageToolBlockRendering() {
        val message = """
            让我为您生成一张赛博朋克猫咪插图：
            <tool_call>
            <function=generate_image>
            <parameter=prompt>
            a cute cyberpunk cat with glowing neon glasses, 8k
            </parameter>
            </function>
            </tool_call>
            <tool_response>
            [状态] 图像生成成功 (1024 × 1024)
            [图像路径] /data/user/0/io.github.bombless.htp_playpen/files/generated_images/sdxl_test.png
            </tool_response>
            图像已为您生成完毕！
        """.trimIndent()

        val blocks = parseMarkdownBlocks(message)
        assertEquals(3, blocks.size)
        assertTrue(blocks[1] is ChatMessageBlock.ToolExecutionBlock)
        val toolBlock = blocks[1] as ChatMessageBlock.ToolExecutionBlock
        assertEquals("本地 SDXL 生图", toolBlock.toolName)
        assertEquals("a cute cyberpunk cat with glowing neon glasses, 8k", toolBlock.code)
        assertTrue(toolBlock.response?.contains("[图像路径]") == true)
        assertEquals(false, toolBlock.isRunning)
    }
}
