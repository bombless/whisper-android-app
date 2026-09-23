package com.example.whisperapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

private data class ConfigDownloadState(
    val model: ManagedModel,
    val file: String,
    val current: Long,
    val total: Long,
)

@Composable
fun ConfigScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()
    var statuses by remember {
        mutableStateOf(ManagedModel.entries.associateWith { ModelManager.status(context, it) })
    }
    var download by remember { mutableStateOf<ConfigDownloadState?>(null) }
    var message by remember { mutableStateOf("") }

    fun refresh() {
        statuses = ManagedModel.entries.associateWith { ModelManager.status(context, it) }
    }

    fun startDownload(model: ManagedModel) {
        if (download != null) return
        message = ""
        scope.launch(Dispatchers.IO) {
            try {
                ModelManager.downloadModel(context, model) { file, current, total ->
                    download = ConfigDownloadState(model, file, current, total)
                }
                withContext(Dispatchers.Main) {
                    refresh()
                    message = model.title + " 下载完成并已重新检查"
                }
            } catch (t: Throwable) {
                withContext(Dispatchers.Main) {
                    message = "下载失败：" + (t.message ?: t::class.java.simpleName)
                }
            } finally {
                withContext(Dispatchers.Main) { download = null }
            }
        }
    }

    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .verticalScroll(scrollState)
                .padding(horizontal = 20.dp, vertical = 12.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                TextButton(onClick = onBack) { Text("返回") }
                Text("配置", style = MaterialTheme.typography.headlineMedium)
            }
            Spacer(Modifier.height(8.dp))
            Text("模型", style = MaterialTheme.typography.titleLarge)
            Text(
                "管理本地运行所需的 LFM2.5 230M 和 Whisper Large v3 模型文件。",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(12.dp))

            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                ManagedModel.entries.forEach { model ->
                    val status = statuses.getValue(model)
                    val active = download?.model == model
                    val progress = download?.takeIf { it.model == model }?.let {
                        if (it.total > 0) (it.current.toFloat() / it.total).coerceIn(0f, 1f) else 0f
                    }

                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(model.title, style = MaterialTheme.typography.titleMedium)
                            Text(
                                "本地模型 · " + status.existingFiles + " / " + status.files.size + " 文件",
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Text(
                                if (status.ready) "✓ 模型完整" else "⚠ 文件不完整",
                                color = if (status.ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                            )

                            if (!status.ready) {
                                Text(
                                    "缺少：" + status.missingFiles.take(4).joinToString("、") +
                                        if (status.missingFiles.size > 4) " 等 " + status.missingFiles.size + " 个文件" else "",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }

                            if (active) {
                                LinearProgressIndicator(
                                    progress = { progress ?: 0f },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                                Text((download?.file ?: "") + " · " + (((progress ?: 0f) * 100).roundToInt()) + "%")
                            }

                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                OutlinedButton(
                                    enabled = download == null,
                                    onClick = {
                                        scope.launch {
                                            val checked = withContext(Dispatchers.IO) {
                                                ModelManager.status(context, model)
                                            }
                                            statuses = statuses + (model to checked)
                                            message = model.title + " 检查完成"
                                        }
                                    },
                                ) { Text("检查") }
                                Button(
                                    enabled = download == null && !status.ready,
                                    onClick = { startDownload(model) },
                                ) { Text(if (status.existingFiles == 0) "下载" else "继续下载") }
                                if (status.ready) {
                                    OutlinedButton(
                                        enabled = download == null,
                                        onClick = { message = model.title + " 已完整，无需下载" },
                                    ) { Text("更新") }
                                }
                            }
                        }
                    }
                }
            }

            if (message.isNotBlank()) {
                Spacer(Modifier.height(8.dp))
                Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Spacer(Modifier.height(8.dp))
            OutlinedButton(
                enabled = download == null,
                onClick = { refresh(); message = "已重新检查全部模型" },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("重新检查全部") }
        }
    }
}