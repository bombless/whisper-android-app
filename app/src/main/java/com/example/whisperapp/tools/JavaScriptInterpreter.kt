package com.example.whisperapp.tools

import org.mozilla.javascript.BaseFunction
import org.mozilla.javascript.ClassShutter
import org.mozilla.javascript.Context
import org.mozilla.javascript.ContextFactory
import org.mozilla.javascript.Scriptable
import org.mozilla.javascript.ScriptableObject
import org.mozilla.javascript.Undefined
import java.util.concurrent.TimeUnit

/**
 * Executes small JavaScript snippets for the chat tool loop.
 *
 * Rhino runs interpreted JavaScript with Java class access disabled and instruction-count
 * timeouts. This is a convenience sandbox, not a security boundary for hostile multi-tenant code.
 */
object JavaScriptInterpreter {
    private const val MAX_CODE_CHARS = 20_000
    private const val MAX_OUTPUT_CHARS = 8_000
    private const val TIME_LIMIT_MS = 2_000L
    private const val INSTRUCTION_LIMIT = 2_000_000

    data class Result(val success: Boolean, val output: String)

    private val factory = object : ContextFactory() {
        private val deadlineNs = ThreadLocal<Long>()

        override fun makeContext(): Context = super.makeContext().apply {
            optimizationLevel = -1 // interpreted mode enables instruction observation
            instructionObserverThreshold = 10_000
            classShutter = ClassShutter { false } // no Packages/java.lang/Java.type access
        }

        override fun observeInstructionCount(cx: Context, instructionCount: Int) {
            val deadline = deadlineNs.get() ?: return
            if (System.nanoTime() > deadline) {
                throw IllegalStateException("JavaScript execution timed out after ${TIME_LIMIT_MS} ms")
            }
            if (instructionCount > INSTRUCTION_LIMIT) {
                throw IllegalStateException("JavaScript instruction limit exceeded")
            }
        }

        fun <T> runWithDeadline(block: () -> T): T {
            deadlineNs.set(System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIME_LIMIT_MS))
            return try { block() } finally { deadlineNs.remove() }
        }
    }

    fun execute(code: String): Result {
        if (code.isBlank()) return Result(false, "No JavaScript code supplied.")
        if (code.length > MAX_CODE_CHARS) return Result(false, "Code exceeds $MAX_CODE_CHARS characters.")

        val output = StringBuilder()
        return try {
            val value = factory.runWithDeadline {
                factory.call { cx ->
                    val scope = cx.initSafeStandardObjects()
                    ScriptableObject.putProperty(scope, "print", object : BaseFunction() {
                        override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any>): Any {
                            if (output.length < MAX_OUTPUT_CHARS) {
                                if (output.isNotEmpty()) output.append('\\n')
                                output.append(args.joinToString(" ") { Context.toString(it) }.take(MAX_OUTPUT_CHARS - output.length))
                            }
                            return Undefined.instance
                        }
                    })
                    cx.evaluateString(scope, code, "chat-tool.js", 1, null)
                }
            }
            val resultText = if (value == null || value === Undefined.instance) "" else Context.toString(value)
            if (resultText.isNotBlank() && resultText != "undefined") {
                if (output.isNotEmpty()) output.append('\\n')
                if (output.length < MAX_OUTPUT_CHARS) output.append(resultText.take(MAX_OUTPUT_CHARS - output.length))
            }
            Result(true, output.toString().ifBlank { "(executed successfully; no output)" })
        } catch (t: Throwable) {
            Result(false, buildString {
                if (output.isNotEmpty()) append(output).append('\\n')
                append(t.message ?: t::class.java.simpleName)
            }.take(MAX_OUTPUT_CHARS))
        }
    }
}
