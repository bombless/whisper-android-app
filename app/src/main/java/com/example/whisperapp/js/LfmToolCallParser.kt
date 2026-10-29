package com.example.whisperapp.js

/**
 * Parsed representation of a tool call found in model output.
 */
data class ParsedToolCall(
    val name: String,
    val code: String,
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
            val toolName = match.groupValues.getOrNull(1)?.takeIf { it.isNotBlank() } ?: "run_javascript"
            val content = match.groupValues.getOrNull(2) ?: ""
            val code = extractCode(content)
            return ParsedToolCall(
                name = toolName,
                code = code,
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
            val code = extractCode(content)
            return ParsedToolCall(
                name = toolName,
                code = code,
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

        // 2. Unwrap outer <function...> ... </function> if present
        val functionRegex = Regex("""<function(?:\s*=[^>]+|\s+[^>]+)?>([\s\S]*?)</function>""", RegexOption.IGNORE_CASE)
        val funcMatch = functionRegex.find(text)
        if (funcMatch != null) {
            text = funcMatch.groupValues[1].trim()
        }

        // 3. Extract from <parameter...> ... </parameter> or <param...> ... </param> or <arg...> ... </arg>
        val paramRegex = Regex("""<(?:parameter|param|arg|argument)(?:\s*=[^>]+|\s+[^>]+)?>([\s\S]*?)</(?:parameter|param|arg|argument)>""", RegexOption.IGNORE_CASE)
        val paramMatch = paramRegex.find(text)
        if (paramMatch != null) {
            text = paramMatch.groupValues[1].trim()
        }

        // 4. Extract from markdown code fence ```javascript ... ``` or ```js ... ``` or ``` ... ```
        val codeFenceRegex = Regex("""```(?:javascript|js)?\s*([\s\S]*?)```""", RegexOption.IGNORE_CASE)
        val fenceMatch = codeFenceRegex.find(text)
        if (fenceMatch != null) {
            text = fenceMatch.groupValues[1].trim()
        }

        // 5. Extract from <code>...</code> or <script>...</script> or <javascript>...</javascript>
        val codeTagRegex = Regex("""<(?:code|script|javascript)>([\s\S]*?)</(?:code|script|javascript)>""", RegexOption.IGNORE_CASE)
        val tagMatch = codeTagRegex.find(text)
        if (tagMatch != null) {
            text = tagMatch.groupValues[1].trim()
        }

        // 6. Strip CDATA: <![CDATA[ ... ]]>
        val cdataRegex = Regex("""<!\[CDATA\[([\s\S]*?)\]\]>""", RegexOption.IGNORE_CASE)
        val cdataMatch = cdataRegex.find(text)
        if (cdataMatch != null) {
            text = cdataMatch.groupValues[1].trim()
        }

        // 7. If text is still wrapped in stray XML tags (e.g. <tool>...</tool> or <xml>...</xml>)
        val outerTagRegex = Regex("""^<([a-zA-Z0-9_-]+)(?:=[^>]+|\s+[^>]+)?>([\s\S]*?)</\1>$""", RegexOption.IGNORE_CASE)
        val outerMatch = outerTagRegex.find(text)
        if (outerMatch != null) {
            text = outerMatch.groupValues[2].trim()
        }

        return text
    }

    /**
     * Formats a tool response XML block.
     */
    fun formatToolResponse(result: JsExecutionResult): String = buildString {
        append("\n<tool_response>\n")
        append(result.toOutputString())
        append("\n</tool_response>\n")
    }
}
