package com.example.whisperapp.ui

import android.util.Log
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import com.example.whisperapp.qnn.QnnLfm2ChatRunner
import com.example.whisperapp.tools.JavaScriptInterpreter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject

private const val TOOL_SYSTEM_PROMPT = """
你是一个可以执行 JavaScript 并验证结果的编程助手。
当任务需要运行代码、测试算法或检查结果时，调用唯一可用工具 run_javascript。
工具调用必须是单独且严格合法的 JSON，放在下面标签中，不要用 Markdown 代码围栏：
<tool_call>{"name":"run_javascript","arguments":{"code":"print(1 + 1)"}}</tool_call>
JSON 的 code 字段必须正确转义换行和引号。收到 tool 角色的执行结果后，分析结果；若有错误，可修复代码并再次调用工具。最多进行 3 次工具调用。只有在看到真实执行结果后，才能声称代码测试通过。普通问答无需调用工具。
"""

private data class JavaScriptToolCall(val code: String)

private fun parseJavaScriptToolCall(text: String): JavaScriptToolCall? {
    val match = Regex("<tool_call>\\s*(\\{.*?})\\s*</tool_call>", setOf(RegexOption.DOT_MATCHES_ALL))
        .find(text) ?: return null
    return runCatching {
        val json = JSONObject(match.groupValues[1])
        if (json.optString("name") != "run_javascript") return@runCatching null
        val args = json.optJSONObject("arguments") ?: return@runCatching null
        val code = args.optString("code")
        if (code.isBlank()) null else JavaScriptToolCall(code)
    }.getOrNull()
}

@Composable
fun ChatScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    LaunchedEffect(Unit) { Log.i("CHAT_UI", "ChatScreen composed package=${context.packageName}") }
    val scope = rememberCoroutineScope()
    val messages = remember { mutableStateListOf<Pair<String, String>>() }
    var input by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("正在准备 LFM2.5-230M…") }
    var busy by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        status = QnnLfm2ChatRunner.start(context).message
    }

    DisposableEffect(Unit) { onDispose { QnnLfm2ChatRunner.stop() } }

    Column(modifier.fillMaxSize().padding(14.dp)) {
        Text(status, style = MaterialTheme.typography.labelMedium)
        Spacer(Modifier.height(8.dp))
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(messages) { (role, text) ->
                Surface(shape = MaterialTheme.shapes.medium, tonalElevation = 1.dp) {
                    Text(if (role == "user") "你：$text" else "LFM：$text", Modifier.padding(12.dp))
                }
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(value = input, onValueChange = { input = it }, enabled = !busy, modifier = Modifier.weight(1f), placeholder = { Text("输入消息…") })
            Button(enabled = !busy && input.isNotBlank() && status.contains("已就绪"), onClick = {
                val text = input.trim()
                input = ""
                messages += "user" to text
                busy = true
                val historySnapshot = messages.toList()
                Log.i("CHAT_UI", "SEND_CLICK chars=${text.length} history=${historySnapshot.size}")
                scope.launch(Dispatchers.Default) {
                    val answer = runCatching {
                        val workingHistory = mutableListOf<Pair<String, String>>()
                        workingHistory += "system" to TOOL_SYSTEM_PROMPT.trim()
                        workingHistory += historySnapshot
                        var response = QnnLfm2ChatRunner.generate(context, workingHistory)
                        var toolCalls = 0
                        while (toolCalls < 3) {
                            val call = parseJavaScriptToolCall(response) ?: break
                            workingHistory += "assistant" to response
                            val result = JavaScriptInterpreter.execute(call.code)
                            val toolOutput = JSONObject()
                                .put("success", result.success)
                                .put("output", result.output)
                                .toString()
                            workingHistory += "tool" to toolOutput
                            toolCalls++
                            Log.i("CHAT_TOOL", "javascript call=$toolCalls success=${result.success} codeChars=${call.code.length} outputChars=${result.output.length}")
                            response = QnnLfm2ChatRunner.generate(context, workingHistory)
                        }
                        if (parseJavaScriptToolCall(response) != null) {
                            "工具调用次数已达上限（3 次）。当前结果：\n$response"
                        } else response
                    }.onFailure { Log.e("CHAT_UI", "GENERATE_FAILED type=${it::class.java.name} message=${it.message}", it) }
                        .getOrElse { "生成失败：${it.message}" }
                    Log.i("CHAT_UI", "GENERATE_RETURN chars=${answer.length} answer=${answer.take(500)}")
                    launch(Dispatchers.Main) { messages += "assistant" to answer; busy = false }
                }
            }) { Text("发送") }
        }
    }
}
