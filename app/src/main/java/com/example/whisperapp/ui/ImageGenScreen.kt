package com.example.whisperapp.ui

import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
internal fun ImageGenScreen(
    bitmap: Bitmap?,
    status: String,
    busy: Boolean,
    onGenerate: (String, String) -> Unit,
    onCancel: () -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var prompt by remember { mutableStateOf("a red apple on a wooden table, natural light, photograph") }
    var negativePrompt by remember { mutableStateOf("blurry, distorted") }
    var saveStatus by remember { mutableStateOf("") }

    // History state
    var showHistoryDialog by remember { mutableStateOf(false) }
    var historyList by remember { mutableStateOf<List<ImageGenHistoryItem>>(emptyList()) }
    var selectedDetailItem by remember { mutableStateOf<ImageGenHistoryItem?>(null) }
    var detailBitmap by remember { mutableStateOf<Bitmap?>(null) }

    fun refreshHistory() {
        scope.launch(Dispatchers.IO) {
            val list = ImageGenHistoryManager.loadHistory(context)
            withContext(Dispatchers.Main) {
                historyList = list
            }
        }
    }

    LaunchedEffect(Unit) {
        refreshHistory()
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("本地文生图", style = MaterialTheme.typography.headlineSmall)
            OutlinedButton(
                onClick = {
                    refreshHistory()
                    showHistoryDialog = true
                }
            ) {
                Text("历史记录 (${historyList.size})")
            }
        }

        Text(status, style = MaterialTheme.typography.bodyMedium)
        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("提示词（英文效果较好）") },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth().heightIn(min = 100.dp),
        )
        OutlinedTextField(
            value = negativePrompt,
            onValueChange = { negativePrompt = it },
            label = { Text("负面提示词") },
            enabled = !busy,
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                enabled = !busy && prompt.isNotBlank(),
                onClick = {
                    saveStatus = ""
                    onGenerate(prompt, negativePrompt)
                }
            ) {
                Text("生成")
            }
            if (busy) OutlinedButton(onClick = onCancel) { Text("取消") }
            if (bitmap != null) {
                OutlinedButton(enabled = !busy, onClick = {
                    scope.launch {
                        saveStatus = withContext(Dispatchers.IO) {
                            runCatching {
                                val values = ContentValues().apply {
                                    put(MediaStore.Images.Media.DISPLAY_NAME, "SDXL_${System.currentTimeMillis()}.png")
                                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                                    if (Build.VERSION.SDK_INT >= 29) {
                                        put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/WhisperApp")
                                        put(MediaStore.Images.Media.IS_PENDING, 1)
                                    }
                                }
                                val resolver = context.contentResolver
                                val uri = checkNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values))
                                try {
                                    checkNotNull(resolver.openOutputStream(uri)).use { check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) }
                                    if (Build.VERSION.SDK_INT >= 29) {
                                        values.clear()
                                        values.put(MediaStore.Images.Media.IS_PENDING, 0)
                                        resolver.update(uri, values, null, null)
                                    }
                                } catch (failure: Throwable) {
                                    resolver.delete(uri, null, null)
                                    throw failure
                                }
                                "已保存到相册"
                            }.getOrElse { "保存失败: ${it.message}" }
                        }
                    }
                }) { Text("保存到相册") }
            }
        }
        if (saveStatus.isNotEmpty()) Text(saveStatus)
        if (bitmap != null) Image(bitmap.asImageBitmap(), contentDescription = "生成的图像", modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp))
    }

    // History List Dialog / Sheet
    if (showHistoryDialog) {
        AlertDialog(
            onDismissRequest = { showHistoryDialog = false },
            title = {
                Text("历史生成记录", fontWeight = FontWeight.Bold)
            },
            text = {
                if (historyList.isEmpty()) {
                    Box(modifier = Modifier.fillMaxWidth().height(150.dp), contentAlignment = Alignment.Center) {
                        Text("暂无生成历史", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().heightIn(max = 420.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        items(historyList, key = { it.id }) { item ->
                            HistoryItemCard(
                                item = item,
                                onClick = {
                                    selectedDetailItem = item
                                },
                                onDelete = {
                                    scope.launch(Dispatchers.IO) {
                                        ImageGenHistoryManager.deleteItem(context, item.id)
                                        val updated = ImageGenHistoryManager.loadHistory(context)
                                        withContext(Dispatchers.Main) {
                                            historyList = updated
                                        }
                                    }
                                }
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showHistoryDialog = false }) {
                    Text("关闭")
                }
            }
        )
    }

    // Detail Dialog (Inspect & Modify Prompt)
    selectedDetailItem?.let { item ->
        LaunchedEffect(item) {
            detailBitmap = withContext(Dispatchers.IO) {
                ImageGenHistoryManager.loadBitmap(context, item.imageFileName)
            }
        }

        var detailPrompt by remember(item) { mutableStateOf(item.prompt) }
        var detailNegativePrompt by remember(item) { mutableStateOf(item.negativePrompt) }

        AlertDialog(
            onDismissRequest = {
                selectedDetailItem = null
                detailBitmap = null
            },
            title = {
                Text("图片详情与复用", fontWeight = FontWeight.Bold)
            },
            text = {
                Column(
                    modifier = Modifier.fillMaxWidth().verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    if (detailBitmap != null) {
                        Image(
                            bitmap = detailBitmap!!.asImageBitmap(),
                            contentDescription = "历史生成的图像",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(220.dp)
                                .clip(RoundedCornerShape(8.dp)),
                            contentScale = ContentScale.Crop
                        )
                    } else {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(160.dp)
                                .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(8.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text("正在加载图片或原图已删除", style = MaterialTheme.typography.bodySmall)
                        }
                    }

                    val dateStr = remember(item.timestamp) {
                        SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(item.timestamp))
                    }
                    Text("生成时间: $dateStr", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)

                    OutlinedTextField(
                        value = detailPrompt,
                        onValueChange = { detailPrompt = it },
                        label = { Text("修改提示词") },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 90.dp)
                    )

                    OutlinedTextField(
                        value = detailNegativePrompt,
                        onValueChange = { detailNegativePrompt = it },
                        label = { Text("修改负面提示词") },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    enabled = !busy && detailPrompt.isNotBlank(),
                    onClick = {
                        // Apply modified prompts and trigger generation
                        prompt = detailPrompt
                        negativePrompt = detailNegativePrompt
                        selectedDetailItem = null
                        detailBitmap = null
                        showHistoryDialog = false
                        saveStatus = ""
                        onGenerate(detailPrompt, detailNegativePrompt)
                    }
                ) {
                    Text("以修改后的提示词生成")
                }
            },
            dismissButton = {
                TextButton(
                    onClick = {
                        selectedDetailItem = null
                        detailBitmap = null
                    }
                ) {
                    Text("返回")
                }
            }
        )
    }
}

@Composable
private fun HistoryItemCard(
    item: ImageGenHistoryItem,
    onClick: () -> Unit,
    onDelete: () -> Unit,
) {
    val context = LocalContext.current
    var thumbBitmap by remember { mutableStateOf<Bitmap?>(null) }
    LaunchedEffect(item.imageFileName) {
        thumbBitmap = withContext(Dispatchers.IO) {
            ImageGenHistoryManager.loadBitmap(context, item.imageFileName)
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f))
    ) {
        Row(
            modifier = Modifier.padding(8.dp).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            if (thumbBitmap != null) {
                Image(
                    bitmap = thumbBitmap!!.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.size(64.dp).clip(RoundedCornerShape(8.dp)),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(64.dp)
                        .background(MaterialTheme.colorScheme.onSurface.copy(alpha = 0.1f), RoundedCornerShape(8.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("无图", fontSize = 10.sp)
                }
            }

            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = item.prompt,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
                val dateStr = remember(item.timestamp) {
                    SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()).format(Date(item.timestamp))
                }
                Text(
                    text = dateStr,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            TextButton(
                onClick = onDelete,
                colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)
            ) {
                Text("删除", fontSize = 12.sp)
            }
        }
    }
}
