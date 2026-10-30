package com.example.whisperapp.js

/**
 * Parsed representation of a tool call found in model output.
 */
data class ParsedToolCall(
    val name: String,
    val code: String,
    val arguments: Map<String, String> = emptyMap(),
    val rawText: String,
    val startIndex: Int,
    val endIndex: Int
)

object LfmToolCallParser {

    private val TOOL_CALL_REGEX = Regex(
        """<tool_call(?::([a-zA-Z0-9_-]+))?>([\s\S]*?)</tool_call(?::\1)?>""",
        RegexOption.IGNORE_CASE
    )

    private val RUN_JS_REGEX = Regex(
        """<run_javascript>([\s\S]*?)</run_javascript>""",
        RegexOption.IGNORE_CASE
    )

    private val FUNCTION_CALL_REGEX = Regex(
        """<function(?:=([a-zA-Z0-9_-]+))?>([\s\S]*?)</function>""",
        RegexOption.IGNORE_CASE
    )

    /**
     * Finds the first unexecuted tool call in [text] starting from [fromIndex].
     */
    fun findNextToolCall(text: String, fromIndex: Int = 0): ParsedToolCall? {
        if (fromIndex >= text.length) return null
        val sub = text.substring(fromIndex)

        // 1. Try standard <tool_call>...</tool_call>
        val match = TOOL_CALL_REGEX.find(sub)
        if (match != null) {
            val content = match.groupValues.getOrNull(2) ?: ""
            val extractedName = Regex(
                """<function(?:\s*=\s*([a-zA-Z0-9_-]+)|\s+name="?([a-zA-Z0-9_-]+)"?)""",
                RegexOption.IGNORE_CASE
            ).find(content)?.let { m ->
                m.groupValues[1].takeIf { it.isNotBlank() } ?: m.groupValues[2].takeIf { it.isNotBlank() }
            }
            val toolName = match.groupValues.getOrNull(1)?.takeIf { it.isNotBlank() }
                ?: extractedName
                ?: "run_javascript"
            val args = extractParameters(content)
            val code = args["code"] ?: args["prompt"] ?: extractCode(content)
            return ParsedToolCall(
                name = toolName,
                code = code,
                arguments = args,
                rawText = match.value,
                startIndex = fromIndex + match.range.first,
                endIndex = fromIndex + match.range.last + 1
            )
        }

        // 2. Fallback for <run_javascript>...</run_javascript>
        val matchRunJs = RUN_JS_REGEX.find(sub)
        if (matchRunJs != null) {
            val content = matchRunJs.groupValues.getOrNull(1) ?: ""
            val code = extractCode(content)
            return ParsedToolCall(
                name = "run_javascript",
                code = code,
                arguments = mapOf("code" to code),
                rawText = matchRunJs.value,
                startIndex = fromIndex + matchRunJs.range.first,
                endIndex = fromIndex + matchRunJs.range.last + 1
            )
        }

        // 3. Fallback for standalone <function=...>...</function>
        val matchFunc = FUNCTION_CALL_REGEX.find(sub)
        if (matchFunc != null) {
            val toolName = matchFunc.groupValues.getOrNull(1)?.takeIf { it.isNotBlank() } ?: "run_javascript"
            val content = matchFunc.groupValues.getOrNull(2) ?: ""
            val args = extractParameters(content)
            val code = args["code"] ?: args["prompt"] ?: extractCode(content)
            return ParsedToolCall(
                name = toolName,
                code = code,
                arguments = args,
                rawText = matchFunc.value,
                startIndex = fromIndex + matchFunc.range.first,
                endIndex = fromIndex + matchFunc.range.last + 1
            )
        }

        return null
    }

    /**
     * Extracts pure JavaScript code from tool call contents,
     * robustly stripping pseudo-XML wrappers (<function>, <parameter>, <code>, <script>),
     * markdown code fences (```javascript ... ```), CDATA blocks, and JSON formats.
     * Supports both complete blocks and in-progress streaming tokens with zero XML leakage.
     */
    fun extractCode(content: String): String {
        var text = content.trim()

        // 1. JSON format: {"name": "run_javascript", "arguments": {"code": "..."}} or {"code": "..."}
        if (text.startsWith("{") && text.endsWith("}")) {
            val jsonCode = runCatching {
                val json = org.json.JSONObject(text)
                when {
                    json.has("code") -> json.getString("code")
                    json.has("script") -> json.getString("script")
                    json.has("arguments") -> {
                        val args = json.get("arguments")
                        if (args is org.json.JSONObject) {
                            args.optString("code", args.optString("script", ""))
                        } else args.toString()
                    }
                    json.has("parameters") -> {
                        val params = json.get("parameters")
                        if (params is org.json.JSONObject) {
                            params.optString("code", params.optString("script", ""))
                        } else params.toString()
                    }
                    else -> null
                }
            }.getOrNull()
            if (!jsonCode.isNullOrBlank()) {
                text = jsonCode.trim()
            }
        }

        // 2. Unwrap <function...> ... </function> (or strip leading unclosed <function...>)
        val functionRegex = Regex("""<function(?:\s*=[^>]+|\s+[^>]+)?>([\s\S]*?)</function>""", RegexOption.IGNORE_CASE)
        val funcMatch = functionRegex.find(text)
        if (funcMatch != null) {
            text = funcMatch.groupValues[1].trim()
        } else {
            text = text.replace(Regex("""^\s*<function(?:\s*=[^>]*|\s+[^>]*)?>?""", RegexOption.IGNORE_CASE), "").trim()
        }

        // 3. Extract from <parameter...> ... </parameter> (or strip leading unclosed <parameter...>)
        val paramRegex = Regex("""<(?:parameter|param|arg|argument)(?:\s*=[^>]+|\s+[^>]+)?>([\s\S]*?)</(?:parameter|param|arg|argument)>""", RegexOption.IGNORE_CASE)
        val paramMatch = paramRegex.find(text)
        if (paramMatch != null) {
            text = paramMatch.groupValues[1].trim()
        } else {
            text = text.replace(Regex("""^\s*<(?:parameter|param|arg|argument)(?:\s*=[^>]*|\s+[^>]*)?>?""", RegexOption.IGNORE_CASE), "").trim()
        }

        // 4. Extract from markdown code fence ```javascript ... ``` (or strip opening fence if streaming)
        val codeFenceRegex = Regex("""```(?:javascript|js)?\s*([\s\S]*?)```""", RegexOption.IGNORE_CASE)
        val fenceMatch = codeFenceRegex.find(text)
        if (fenceMatch != null) {
            text = fenceMatch.groupValues[1].trim()
        } else {
            text = text.replace(Regex("""^\s*```(?:javascript|js)?\s*""", RegexOption.IGNORE_CASE), "").trim()
        }

        // 5. Extract from <code>...</code> or <script>...</script> or <javascript>...</javascript>
        val codeTagRegex = Regex("""<(?:code|script|javascript)>([\s\S]*?)</(?:code|script|javascript)>""", RegexOption.IGNORE_CASE)
        val tagMatch = codeTagRegex.find(text)
        if (tagMatch != null) {
            text = tagMatch.groupValues[1].trim()
        } else {
            text = text.replace(Regex("""^\s*<(?:code|script|javascript)>""", RegexOption.IGNORE_CASE), "").trim()
        }

        // 6. Strip CDATA: <![CDATA[ ... ]]>
        text = text.replace(Regex("""<!\[CDATA\[""", RegexOption.IGNORE_CASE), "")
        text = text.replace(Regex("""\]\]>""", RegexOption.IGNORE_CASE), "")

        // 7. Strip trailing closing tags or partial closing tags during streaming
        while (true) {
            val before = text
            text = text.replace(Regex("""</(?:parameter|param|arg|argument|function|tool_call|run_javascript|code|script)>\s*$""", RegexOption.IGNORE_CASE), "").trim()
            text = text.replace(Regex("""</[a-zA-Z0-9_-]*$""", RegexOption.IGNORE_CASE), "").trim()
            text = text.replace(Regex("""```$"""), "").trim()
            if (text == before) break
        }

        // 8. If text is still wrapped in stray XML tags (e.g. <tool>...</tool> or <xml>...</xml>)
        val outerTagRegex = Regex("""^<([a-zA-Z0-9_-]+)(?:=[^>]+|\s+[^>]+)?>([\s\S]*?)</\1>$""", RegexOption.IGNORE_CASE)
        val outerMatch = outerTagRegex.find(text)
        if (outerMatch != null) {
            text = outerMatch.groupValues[2].trim()
        }

        return text
    }

    /**
     * Extracts named parameter dictionary from tool call content.
     */
    fun extractParameters(content: String): Map<String, String> {
        val map = LinkedHashMap<String, String>()
        val paramRegex = Regex(
            """<(?:parameter|param)(?:\s*=\s*([a-zA-Z0-9_-]+)|\s+name="?([a-zA-Z0-9_-]+)"?)(?:\s*[^>]*)?>([\s\S]*?)</(?:parameter|param)>""",
            RegexOption.IGNORE_CASE
        )
        for (m in paramRegex.findAll(content)) {
            val key = m.groupValues[1].takeIf { it.isNotBlank() } ?: m.groupValues[2]
            val value = extractCode(m.groupValues[3])
            if (key.isNotBlank()) {
                map[key] = value
            }
        }
        if (map.isEmpty() && content.trim().startsWith("{")) {
            runCatching {
                val json = org.json.JSONObject(content.trim())
                val args = when {
                    json.has("arguments") && json.get("arguments") is org.json.JSONObject -> json.getJSONObject("arguments")
                    json.has("parameters") && json.get("parameters") is org.json.JSONObject -> json.getJSONObject("parameters")
                    else -> json
                }
                for (k in args.keys()) {
                    map[k] = args.optString(k, "")
                }
            }
        }
        return map
    }

    /**
     * Formats a tool response XML block.
     */
    fun formatToolResponse(result: JsExecutionResult): String = buildString {
        append("\n<tool_response>\n")
        append(result.toOutputString())
        append("\n</tool_response>\n")
    }

    /**
     * Formats an image generation tool response XML block.
     */
    fun formatImageToolResponse(
        success: Boolean,
        filePath: String?,
        error: String? = null
    ): String = buildString {
        append("\n<tool_response>\n")
        if (success && filePath != null) {
            append("[状态] 图像生成成功 (1024 × 1024)\n")
            append("[图像路径] ").append(filePath).append("\n")
        } else {
            append("[执行失败] ").append(error ?: "未知错误").append("\n")
        }
        append("</tool_response>\n")
    }
}
