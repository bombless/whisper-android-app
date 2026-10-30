package io.github.bombless.htp_playpen.js

import org.mozilla.javascript.BaseFunction
import org.mozilla.javascript.Context
import org.mozilla.javascript.ContextFactory
import org.mozilla.javascript.Function
import org.mozilla.javascript.Scriptable
import org.mozilla.javascript.ScriptableObject
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
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

            // Define fetch(url, options)
            val fetchFn = object : BaseFunction() {
                override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                    if (args.isEmpty() || args[0] == null) {
                        throw RuntimeException("fetch() 缺少 URL 参数")
                    }
                    val urlStr = Context.toString(args[0])
                    val options = args.getOrNull(1) as? Scriptable

                    var method = "GET"
                    val headersMap = mutableMapOf<String, String>()
                    var bodyBytes: ByteArray? = null
                    var timeoutMs = 8000

                    if (options != null) {
                        val m = ScriptableObject.getProperty(options, "method")
                        if (m != null && m != Context.getUndefinedValue()) {
                            method = Context.toString(m).uppercase()
                        }
                        val h = ScriptableObject.getProperty(options, "headers")
                        if (h is Scriptable) {
                            for (id in h.ids) {
                                val key = id.toString()
                                val v = ScriptableObject.getProperty(h, key)
                                if (v != null && v != Context.getUndefinedValue()) {
                                    headersMap[key] = Context.toString(v)
                                }
                            }
                        }
                        val b = ScriptableObject.getProperty(options, "body")
                        if (b != null && b != Context.getUndefinedValue()) {
                            bodyBytes = Context.toString(b).toByteArray(Charsets.UTF_8)
                        }
                        val t = ScriptableObject.getProperty(options, "timeout")
                        if (t is Number) {
                            timeoutMs = t.toInt().coerceIn(500, 30000)
                        }
                    }

                    val httpResult = performHttpRequest(urlStr, method, headersMap, bodyBytes, timeoutMs)

                    val respObj = cx.newObject(scope)
                    ScriptableObject.putProperty(respObj, "status", httpResult.statusCode)
                    ScriptableObject.putProperty(respObj, "statusText", httpResult.statusMessage)
                    ScriptableObject.putProperty(respObj, "ok", httpResult.statusCode in 200..299)
                    ScriptableObject.putProperty(respObj, "url", urlStr)

                    val textFn = object : BaseFunction() {
                        override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                            return httpResult.body
                        }
                    }
                    ScriptableObject.putProperty(respObj, "text", textFn)

                    val jsonFn = object : BaseFunction() {
                        override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                            val jsonObj = ScriptableObject.getProperty(scope, "JSON") as? Scriptable
                            val parseFn = jsonObj?.let { ScriptableObject.getProperty(it, "parse") as? Function }
                            return parseFn?.call(cx, scope, jsonObj, arrayOf(httpResult.body)) ?: Context.getUndefinedValue()
                        }
                    }
                    ScriptableObject.putProperty(respObj, "json", jsonFn)

                    val toStringFn = object : BaseFunction() {
                        override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                            return httpResult.body
                        }
                    }
                    ScriptableObject.putProperty(respObj, "toString", toStringFn)

                    return respObj
                }
            }

            // Define httpGet(url) convenience wrapper
            val httpGetFn = object : BaseFunction() {
                override fun call(cx: Context, scope: Scriptable, thisObj: Scriptable, args: Array<out Any?>): Any {
                    if (args.isEmpty() || args[0] == null) {
                        throw RuntimeException("httpGet() 缺少 URL 参数")
                    }
                    val urlStr = Context.toString(args[0])
                    val httpResult = performHttpRequest(urlStr, "GET", emptyMap(), null, 8000)
                    return httpResult.body
                }
            }

            ScriptableObject.putProperty(scope, "fetch", fetchFn)
            ScriptableObject.putProperty(scope, "httpGet", httpGetFn)

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

    private data class HttpResponse(
        val statusCode: Int,
        val statusMessage: String,
        val body: String
    )

    private fun performHttpRequest(
        urlStr: String,
        method: String,
        headers: Map<String, String>,
        bodyBytes: ByteArray?,
        timeoutMs: Int
    ): HttpResponse {
        val url = URL(urlStr)
        val conn = url.openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.connectTimeout = timeoutMs
            conn.readTimeout = timeoutMs
            conn.instanceFollowRedirects = true
            conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Android; WhisperApp)")

            for ((k, v) in headers) {
                conn.setRequestProperty(k, v)
            }

            if (bodyBytes != null && (method == "POST" || method == "PUT" || method == "PATCH")) {
                conn.doOutput = true
                conn.outputStream.use { it.write(bodyBytes) }
            }

            val statusCode = conn.responseCode
            val statusMessage = conn.responseMessage ?: ""
            val stream: InputStream? = if (statusCode in 200..399) {
                conn.inputStream
            } else {
                conn.errorStream ?: runCatching { conn.inputStream }.getOrNull()
            }
            val body = stream?.bufferedReader()?.use { it.readText() } ?: ""
            return HttpResponse(
                statusCode = statusCode,
                statusMessage = statusMessage,
                body = body
            )
        } finally {
            runCatching { conn.disconnect() }
        }
    }
}
