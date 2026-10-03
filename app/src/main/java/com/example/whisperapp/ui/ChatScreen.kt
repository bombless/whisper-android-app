package com.example.whisperapp.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import android.util.Log
import com.example.whisperapp.qnn.QnnLfm2ChatRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

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
                val text = input.trim(); input = ""; messages += "user" to text; busy = true
                val historySnapshot = messages.toList()
                Log.i("CHAT_UI", "SEND_CLICK chars=${text.length} text=${text.take(200)} history=${historySnapshot.size}")
                scope.launch(Dispatchers.Default) {
                    Log.i("CHAT_UI", "GENERATE_DISPATCH thread=${Thread.currentThread().name}")
                    val answer = runCatching {
                        QnnLfm2ChatRunner.generate(context, historySnapshot)
                    }.onFailure { Log.e("CHAT_UI", "GENERATE_FAILED type=${it::class.java.name} message=${it.message}", it) }
                        .getOrElse { "生成失败：${it.message}" }
                    Log.i("CHAT_UI", "GENERATE_RETURN chars=${answer.length} answer=${answer.take(500)}")
                    launch(Dispatchers.Main) { messages += "assistant" to answer; busy = false }
                }
            }) { Text("发送") }
        }
    }
}
