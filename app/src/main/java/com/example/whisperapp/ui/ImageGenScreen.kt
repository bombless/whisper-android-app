package com.example.whisperapp.ui

import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

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
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("本地文生图", style = MaterialTheme.typography.headlineSmall)
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
            Button(enabled = !busy && prompt.isNotBlank(), onClick = { saveStatus = ""; onGenerate(prompt, negativePrompt) }) {
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
}
