package com.example.whisperapp.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import com.example.whisperapp.qnn.QnnLfm2ChatRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

@Composable
fun ChatScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
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
                val text = input.trim(); input = ""; messages += "user" to text; busy = true
                scope.launch(Dispatchers.Default) {
                    val answer = runCatching { QnnLfm2ChatRunner.generate(context, messages.toList()) }.getOrElse { "生成失败：${it.message}" }
                    launch(Dispatchers.Main) { messages += "assistant" to answer; busy = false }
                }
            }) { Text("发送") }
        }
    }
}
