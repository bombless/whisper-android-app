package com.example.whisperapp.tools

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class JavaScriptInterpreterTest {
    @Test
    fun evaluatesCodeAndCapturesPrintOutput() {
        val result = JavaScriptInterpreter.execute(
            "function isRedBlackTreeValid() { return true; } print('valid=' + isRedBlackTreeValid());"
        )

        assertTrue(result.success, result.output)
        assertTrue(result.output.contains("valid=true"), result.output)
    }

    @Test
    fun reportsSyntaxErrors() {
        val result = JavaScriptInterpreter.execute("function ( {")
        assertFalse(result.success)
        assertTrue(result.output.isNotBlank())
    }

    @Test
    fun rejectsJavaInterop() {
        val result = JavaScriptInterpreter.execute("print(java.lang.System.getProperty('user.home'))")
        assertFalse(result.success)
    }

    @Test
    fun rejectsOversizedPrograms() {
        val result = JavaScriptInterpreter.execute(" ".repeat(20_001))
        assertFalse(result.success)
        assertTrue(result.output.contains("exceeds"))
    }
}
