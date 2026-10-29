package com.example.whisperapp.js

import org.mozilla.javascript.BaseFunction
import org.mozilla.javascript.Context
import org.mozilla.javascript.ContextFactory
import org.mozilla.javascript.Function
import org.mozilla.javascript.Scriptable
import org.mozilla.javascript.ScriptableObject
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/**
 * Result of a JavaScript execution.
 */
data class JsExecutionResult(
    val success: Boolean,
    val result: String?,
    val stdout: String,
    val error: String?,
    val executionTimeMs: Long
) {
    /**
     * Formats output for tool injection or UI display.
     */
    fun toOutputString(): String = buildString {
        if (stdout.isNotBlank()) {
            append("[输出]\n").append(stdout.trimEnd()).append("\n")
        }
        if (success) {
            if (result != null && result != "undefined") {
                append("[返回值] ").append(result).append("\n")
            }
            if (stdout.isBlank() && (result == null || result == "undefined")) {
                append("[状态] 执行成功 (无输出)\n")
            }
        } else {
            append("[执行失败] ").append(error ?: "未知错误").append("\n")
        }
    }.trim()
}

/**
 * Custom ContextFactory to enforce instruction limits and interruption checks.
 */
class SandboxedContextFactory(
    private val maxInstructions: Int = 1_000_000
) : ContextFactory() {
    override fun makeContext(): Context {
        val cx = super.makeContext()
        cx.languageVersion = Context.VERSION_ES6
        cx.instructionObserverThreshold = 10_000
        cx.setClassShutter { false } // Sandbox: Disallow all Java reflection/classes
        return cx
    }

    override fun observeInstructionCount(cx: Context, instructionCount: Int) {
        if (instructionCount > maxInstructions) {
            throw Error("JavaScript 执行超时：超出最大指令配额 ($maxInstructions 步)")
        }
        if (Thread.currentThread().isInterrupted) {
            throw Error("JavaScript 执行已取消")
        }
    }
}

/**
 * Sandboxed, isolated JavaScript runtime powered by Mozilla Rhino with ES6 class transpilation.
 */
object JsRuntime {
    private val contextFactory = SandboxedContextFactory()
    private val executor = Executors.newCachedThreadPool { runnable ->
        Thread(runnable, "JsRuntimeWorker").apply { isDaemon = true }
    }

    /**
     * Executes the given JavaScript code string within an isolated sandbox.
     *
     * @param code JavaScript source code to execute.
     * @param timeoutMs Timeout in milliseconds (default 5000ms).
     */
    fun execute(code: String, timeoutMs: Long = 5000L): JsExecutionResult {
        val startNs = System.nanoTime()
        val processedCode = JsTranspiler.transpile(code)

        val task = executor.submit(Callable {
            executeInternal(processedCode)
        })

        return try {
            val (res, stdout) = task.get(timeoutMs, TimeUnit.MILLISECONDS)
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000
            JsExecutionResult(
                success = true,
                result = res,
                stdout = stdout,
                error = null,
                executionTimeMs = elapsedMs
            )
        } catch (te: TimeoutException) {
            task.cancel(true)
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000
            JsExecutionResult(
                success = false,
                result = null,
                stdout = "",
                error = "执行超时 (超过 ${timeoutMs}ms)",
                executionTimeMs = elapsedMs
            )
        } catch (t: Throwable) {
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000
            val rootCause = generateSequence(t) { it.cause }.last()
            JsExecutionResult(
                success = false,
                result = null,
                stdout = "",
                error = rootCause.message ?: rootCause::class.java.simpleName,
                executionTimeMs = elapsedMs
            )
        }
    }

    private fun executeInternal(code: String): Pair<String?, String> {
        val cx = contextFactory.enterContext()
        val stdoutList = mutableListOf<String>()
        try {
            val scope = cx.initSafeStandardObjects()

            // Define console.log, console.info, console.warn, console.error
            val consoleObj = cx.newObject(scope)
            val logFn = object : BaseFunction() {
                override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                    val line = args.joinToString(" ") { formatJsValue(cx, scope, it) }
                    stdoutList.add(line)
                    return Context.getUndefinedValue()
                }
            }
            ScriptableObject.putProperty(consoleObj, "log", logFn)
            ScriptableObject.putProperty(consoleObj, "info", logFn)
            ScriptableObject.putProperty(consoleObj, "warn", logFn)
            ScriptableObject.putProperty(consoleObj, "error", logFn)

            // Define assert(condition, message) and console.assert
            val assertFn = object : BaseFunction() {
                override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                    val condition = if (args.isNotEmpty()) Context.toBoolean(args[0]) else false
                    if (!condition) {
                        val message = if (args.size > 1) formatJsValue(cx, scope, args[1]) else "Assertion failed"
                        throw RuntimeException("AssertionError: $message")
                    }
                    return Context.getUndefinedValue()
                }
            }
            ScriptableObject.putProperty(consoleObj, "assert", assertFn)
            ScriptableObject.putProperty(scope, "console", consoleObj)
            ScriptableObject.putProperty(scope, "assert", assertFn)

            val rawResult = cx.evaluateString(scope, code, "tool_script.js", 1, null)
            val resultStr = formatJsValue(cx, scope, rawResult)
            return resultStr to stdoutList.joinToString("\n")
        } finally {
            Context.exit()
        }
    }

    private fun formatJsValue(cx: Context, scope: Scriptable, value: Any?): String {
        if (value == null) return "null"
        if (value == Context.getUndefinedValue()) return "undefined"
        if (value is Scriptable) {
            return runCatching {
                val json = ScriptableObject.getProperty(scope, "JSON") as? Scriptable
                val stringify = json?.let { ScriptableObject.getProperty(it, "stringify") as? Function }
                if (stringify != null) {
                    val jsonStr = stringify.call(cx, scope, json, arrayOf(value))
                    jsonStr?.toString() ?: Context.toString(value)
                } else {
                    Context.toString(value)
                }
            }.getOrElse { Context.toString(value) }
        }
        return Context.toString(value)
    }
}
