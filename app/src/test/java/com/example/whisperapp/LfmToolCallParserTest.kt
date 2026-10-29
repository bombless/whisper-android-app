package com.example.whisperapp

import com.example.whisperapp.js.JsRuntime
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

    @Test
    fun testToolCallWithFunctionAndParameterTags() {
        val text = """
            <tool_call>
            <function=tool>
            <parameter=code>
            console.log("1 + 1 =", 1 + 1);
            </parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("console.log(\"1 + 1 =\", 1 + 1);", call?.code)

        val result = JsRuntime.execute(call!!.code)
        assertTrue("Execution should succeed without ReferenceError: ${result.error}", result.success)
        assertTrue(result.stdout.contains("1 + 1 = 2"))
    }

    @Test
    fun testToolCallWithParameterNameAttributeAndFence() {
        val text = """
            <tool_call:run_javascript>
            <function name="run_javascript">
            <parameter name="code">
            ```javascript
            assert(1 + 1 === 2, "OK");
            console.log("all good");
            ```
            </parameter>
            </function>
            </tool_call:run_javascript>
        """.trimIndent()

        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("assert(1 + 1 === 2, \"OK\");\nconsole.log(\"all good\");", call?.code)

        val result = JsRuntime.execute(call!!.code)
        assertTrue(result.success)
        assertTrue(result.stdout.contains("all good"))
    }

    @Test
    fun testStandaloneFunctionTag() {
        val text = """
            <function=tool>
            <parameter=code>
            console.log("hello from standalone");
            </parameter>
            </function>
        """.trimIndent()

        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("console.log(\"hello from standalone\");", call?.code)

        val result = JsRuntime.execute(call!!.code)
        assertTrue(result.success)
        assertTrue(result.stdout.contains("hello from standalone"))
    }

    @Test
    fun testNativeLfm3XmlToolCallParsing() {
        val text = """
            <tool_call>
            <function=run_javascript>
            <parameter=code>
            const a = 10;
            const b = 20;
            console.log(a + b);
            </parameter>
            </function>
            </tool_call>
        """.trimIndent()
        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("run_javascript", call?.name)
        assertEquals("const a = 10;\nconst b = 20;\nconsole.log(a + b);", call?.code)
    }

    @Test
    fun testStreamingUnclosedXmlCodeExtraction() {
        val streamingText = "<function=run_javascript>\n<parameter=code>\nconst x = 42;"
        val extracted = LfmToolCallParser.extractCode(streamingText)
        assertEquals("const x = 42;", extracted)

        val streamingWithClosing = "<function=run_javascript>\n<parameter=code>\nconst x = 42;\n</par"
        val extracted2 = LfmToolCallParser.extractCode(streamingWithClosing)
        assertEquals("const x = 42;", extracted2)
    }

    @Test
    fun testGenerateImageToolCallParsing() {
        val text = """
            <tool_call>
            <function=generate_image>
            <parameter=prompt>
            a cute cyberpunk cat wearing neon sunglasses, 8k
            </parameter>
            <parameter=negative_prompt>
            blurry, low quality
            </parameter>
            <parameter=steps>
            25
            </parameter>
            </function>
            </tool_call>
        """.trimIndent()

        val call = LfmToolCallParser.findNextToolCall(text)
        assertNotNull(call)
        assertEquals("generate_image", call?.name)
        assertEquals("a cute cyberpunk cat wearing neon sunglasses, 8k", call?.arguments?.get("prompt"))
        assertEquals("blurry, low quality", call?.arguments?.get("negative_prompt"))
        assertEquals("25", call?.arguments?.get("steps"))
        assertEquals("a cute cyberpunk cat wearing neon sunglasses, 8k", call?.code)
    }

    @Test
    fun testJsRuntimeFetchAndHttpGetDefined() {
        val js = """
            assert(typeof fetch === 'function', "fetch should be defined");
            assert(typeof httpGet === 'function', "httpGet should be defined");
            console.log("Fetch and httpGet are ready");
        """.trimIndent()
        val res = JsRuntime.execute(js)
        assertTrue("JS execution failed: ${res.error}", res.success)
        assertTrue(res.stdout.contains("Fetch and httpGet are ready"))
    }
}
