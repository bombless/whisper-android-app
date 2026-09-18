package com.example.whisperapp.ui

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.whisperapp.audio.RollingPcmBuffer
import com.example.whisperapp.qnn.QnnWhisperRealAudioRunner
import com.example.whisperapp.tts.ChineseTextProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun RollingMainScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var recording by remember { mutableStateOf(false) }
    var processing by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("点击开始，说话即可实时转录") }
    val transcript = remember { mutableStateListOf<String>() }
    var recordingJob by remember { mutableStateOf<Job?>(null) }

    fun stop() {
        recording = false
        recordingJob?.cancel()
        recordingJob = null
        processing = false
        status = "已停止"
    }

    fun start() {
        transcript.clear()
        recording = true
        status = "正在监听…"
        recordingJob = scope.launch(Dispatchers.IO) {
            val rate = 16_000
            val windowSamples = rate * 5
            val readSamples = rate / 10
            val minBuffer = AudioRecord.getMinBufferSize(
                rate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            if (minBuffer <= 0) {
                withContext(Dispatchers.Main) { recording = false; status = "麦克风不可用" }
                return@launch
            }

            // Fixed five-second cache. It never grows with the recording duration.
            val rollingAudio = RollingPcmBuffer(windowSamples)
            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                rate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuffer, readSamples * 4),
            )
            val readBuffer = ShortArray(readSamples)
            var totalSamples = 0L
            var nextInferenceAt = windowSamples.toLong()
            var inferenceJob: Job? = null

            try {
                recorder.startRecording()
                while (currentCoroutineContext().isActive) {
                    val n = recorder.read(readBuffer, 0, readBuffer.size)
                    if (n <= 0) continue
                    rollingAudio.append(readBuffer, 0, n)
                    totalSamples += n

                    if (totalSamples >= nextInferenceAt) {
                        nextInferenceAt += rate
                        if (inferenceJob?.isActive != true && rollingAudio.size() == windowSamples) {
                            val snapshot = rollingAudio.snapshot()
                            val endSeconds = totalSamples.toDouble() / rate
                            inferenceJob = launch(Dispatchers.IO) {
                                withContext(Dispatchers.Main) {
                                    processing = true
                                    status = "正在转录（最近 5 秒）…"
                                }
                                val text = runCatching {
                                    QnnWhisperRealAudioRunner.run(
                                        context,
                                        snapshot,
                                        rate,
                                        requestedSteps = 12,
                                        autoregressive = true,
                                    ).also { check(it.passed) { it.report } }.text.trim()
                                }.getOrElse { "转录失败：${it.message ?: it::class.java.simpleName}" }

                                withContext(Dispatchers.Main) {
                                    val normalized = ChineseTextProcessor.normalizeChineseText(text)
                                    if (normalized.isNotBlank()) transcript.add(normalized)
                                    processing = false
                                    status = if (recording) {
                                        "正在监听…（最近 5 秒，结束于 ${"%.1f".format(java.util.Locale.US, endSeconds)}s）"
                                    } else "已停止"
                                }
                            }
                        }
                    }
                }
            } finally {
                inferenceJob?.cancel()
                runCatching { recorder.stop() }
                recorder.release()
                rollingAudio.clear()
            }
        }
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) start() else status = "需要麦克风权限才能转录"
    }

    DisposableEffect(Unit) {
        onDispose { recordingJob?.cancel() }
    }

    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier.fillMaxSize().padding(20.dp),
            verticalArrangement = Arrangement.SpaceBetween,
        ) {
            Column(Modifier.fillMaxWidth()) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        Modifier.size(12.dp).background(
                            if (recording) MaterialTheme.colorScheme.primary
                            else MaterialTheme.colorScheme.outline,
                            CircleShape,
                        ),
                    )
                    Spacer(Modifier.width(10.dp))
                    Text("实时转录", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                }
                Spacer(Modifier.height(8.dp))
                Text(status, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(24.dp))
                Surface(
                    Modifier.fillMaxWidth().height(420.dp),
                    shape = RoundedCornerShape(20.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                ) {
                    if (transcript.isEmpty()) {
                        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Text("说满 5 秒后开始每秒刷新", color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    } else {
                        LazyColumn(
                            Modifier.fillMaxSize().padding(20.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                        ) {
                            items(transcript) { line -> Text(line, style = MaterialTheme.typography.bodyLarge) }
                        }
                    }
                }
            }

            Column(Modifier.fillMaxWidth()) {
                Button(
                    onClick = {
                        if (recording) stop()
                        else if (
                            ContextCompat.checkSelfPermission(
                                context,
                                Manifest.permission.RECORD_AUDIO,
                            ) == PackageManager.PERMISSION_GRANTED
                        ) start()
                        else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    },
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    shape = RoundedCornerShape(18.dp),
                ) {
                    Text(if (recording) "停止转录" else "开始说话")
                }
                Spacer(Modifier.height(8.dp))
                Text(
                    "Whisper Tiny · 16 kHz · 每秒重新计算最近 5 秒",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.labelMedium,
                )
            }
        }
    }
}
