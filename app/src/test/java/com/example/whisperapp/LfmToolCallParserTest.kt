package com.example.whisperapp

import com.example.whisperapp.js.LfmToolCallParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LfmToolCallParserTest {

    @Test
    fun testToolCallWithMarkdownCodeFence() {
        val text = """
            让我用 JavaScript 测试一下红黑树：
            <tool_call>
            ```javascript
            class RBTree {
                constructor() { this.root = null; }
            }
            console.log("OK");
            ```
            </tool_call>
            稍等片刻。
        """.trimIndent()

        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("run_javascript", call?.name)
        assertTrue(call?.code?.contains("class RBTree") == true)
        assertTrue(call?.code?.contains("console.log(\"OK\");") == true)
        assertTrue(call?.code?.startsWith("class RBTree") == true)
    }

    @Test
    fun testNamedToolCallWithCodeTag() {
        val text = """
            <tool_call:run_javascript>
            <code>
            const x = 42;
            console.log(x);
            </code>
            </tool_call:run_javascript>
        """.trimIndent()

        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("run_javascript", call?.name)
        assertEquals("const x = 42;\nconsole.log(x);", call?.code)
    }

    @Test
    fun testRunJavascriptTag() {
        val text = "<run_javascript>console.log('hi');</run_javascript>"
        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("console.log('hi');", call?.code)
    }

    @Test
    fun testSequentialToolCallsWithFromIndex() {
        val text = """
            第一步：
            <tool_call>
            ```javascript
            console.log(1);
            ```
            </tool_call>
            第二步：
            <tool_call>
            ```javascript
            console.log(2);
            ```
            </tool_call>
        """.trimIndent()

        val call1 = LfmToolCallParser.findNextToolCall(text, fromIndex = 0)
        assertNotNull(call1)
        assertEquals("console.log(1);", call1?.code)

        val call2 = LfmToolCallParser.findNextToolCall(text, fromIndex = call1!!.endIndex)
        assertNotNull(call2)
        assertEquals("console.log(2);", call2?.code)

        val call3 = LfmToolCallParser.findNextToolCall(text, fromIndex = call2!!.endIndex)
        assertNull(call3)
    }
}
