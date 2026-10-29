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

    /**
     * Finds the first unexecuted tool call in [text] starting from [fromIndex].
     */
    fun findNextToolCall(text: String, fromIndex: Int = 0): ParsedToolCall? {
        if (fromIndex >= text.length) return null
        val sub = text.substring(fromIndex)

        // Try standard <tool_call>...</tool_call>
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

        // Fallback for <run_javascript>...</run_javascript>
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

        return null
    }

    /**
     * Extracts pure JavaScript code from tool call contents,
     * stripping markdown code fences or XML tags if present.
     */
    fun extractCode(content: String): String {
        val trimmed = content.trim()

        // Markdown code fence ```javascript ... ``` or ```js ... ``` or ``` ... ```
        val codeFenceRegex = Regex("""```(?:javascript|js)?\s*([\s\S]*?)```""", RegexOption.IGNORE_CASE)
        val fenceMatch = codeFenceRegex.find(trimmed)
        if (fenceMatch != null) {
            return fenceMatch.groupValues[1].trim()
        }

        // <code> ... </code>
        val codeTagRegex = Regex("""<code>([\s\S]*?)</code>""", RegexOption.IGNORE_CASE)
        val tagMatch = codeTagRegex.find(trimmed)
        if (tagMatch != null) {
            return tagMatch.groupValues[1].trim()
        }

        // JSON format: {"name": "run_javascript", "arguments": {"code": "..."}} or {"code": "..."}
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            runCatching {
                val json = org.json.JSONObject(trimmed)
                if (json.has("code")) return json.getString("code")
                if (json.has("arguments")) {
                    val args = json.getJSONObject("arguments")
                    if (args.has("code")) return args.getString("code")
                }
            }
        }

        return trimmed
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
