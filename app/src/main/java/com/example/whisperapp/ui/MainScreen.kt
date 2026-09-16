package com.example.whisperapp.ui

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.whisperapp.audio.AudioPlayer
import com.example.whisperapp.qnn.QnnWhisperRealAudioRunner
import com.example.whisperapp.tts.ChineseTextProcessor
import com.example.whisperapp.tts.VoiceAiTtsEngine
import com.example.whisperapp.tts.TtsQueue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.launch
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var selectedTab by remember { mutableIntStateOf(0) }
    var ttsText by remember { mutableStateOf("") }
    var ttsStatus by remember { mutableStateOf("输入文字后点击播放") }
    var recording by remember { mutableStateOf(false) }
    var processing by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("点击开始，说话即可实时转录") }
    val transcript = remember { mutableStateListOf<String>() }
    var recordingJob by remember { mutableStateOf<Job?>(null) }
    val recordingLifecycleMutex = remember { Mutex() }
    var previousWhisperText by remember { mutableStateOf("") }
    val audioPlayer = remember { AudioPlayer() }
    val ttsEngine = remember { VoiceAiTtsEngine(context) }
    val ttsQueue = remember {
        TtsQueue(scope, ttsEngine, audioPlayer) { error ->
            scope.launch(Dispatchers.Main) {
                if (recording) status = "朗读失败，继续监听…"
            }
        }
    }

    fun stop() {
        Log.i("WHISPER_DIAG", "RECORD_STOP recordingJob=${recordingJob != null}")
        recording = false
        recordingJob?.cancel()
        processing = false
        previousWhisperText = ""
        ttsQueue.clearAndStop()
        status = "已停止"
    }

    fun start() {
        Log.i("WHISPER_DIAG", "RECORD_START")
        transcript.clear()
        previousWhisperText = ""
        ttsQueue.clearAndStop()
        recording = true
        status = "正在监听…"
        recordingJob = scope.launch(Dispatchers.IO) {
            recordingLifecycleMutex.withLock {
            val rate = 16_000
            val samplesPerChunk = rate * 5
            val minBuffer = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (minBuffer <= 0) {
                withContext(Dispatchers.Main) { recording = false; status = "麦克风不可用" }
                return@launch
            }
            val recorder = AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, samplesPerChunk))
            val chunk = ShortArray(samplesPerChunk)
            var offset = 0
            try {
                QnnWhisperRealAudioRunner.start(context)
                Log.i("WHISPER_DIAG", "QNN_SESSION_READY")
                recorder.startRecording()
                Log.i("WHISPER_DIAG", "AUDIO_RECORD_STARTED state=${recorder.recordingState} minBuffer=$minBuffer chunkSamples=$samplesPerChunk")
                while (kotlinx.coroutines.currentCoroutineContext().isActive) {
                    val n = recorder.read(chunk, offset, samplesPerChunk - offset)
                    if (n <= 0) {
                        Log.w("WHISPER_DIAG", "AUDIO_READ_ERROR n=$n offset=$offset state=${recorder.recordingState}")
                        continue
                    }
                    offset += n
                    if (offset == samplesPerChunk) {
                        offset = 0
                        Log.i("WHISPER_DIAG", "CHUNK_READY samples=$samplesPerChunk rms=${kotlin.math.sqrt(chunk.map { it.toDouble() * it }.average())}")
                        withContext(Dispatchers.Main) { processing = true; status = "正在转录…" }
                        Log.i("WHISPER_DIAG", "TRANSCRIBE_START samples=${chunk.size} rate=$rate threads=2")
                        val text = runCatching {
                            QnnWhisperRealAudioRunner.transcribeChunk(context, chunk.copyOf(), rate, requestedSteps = 12, autoregressive = true).also { result -> check(result.passed) { result.report } }.text.trim()
                        }.onSuccess {
                            Log.i("WHISPER_DIAG", "NATIVE_RESULT chars=${it.length} text=${it.take(160)}")
                        }.onFailure {
                            Log.e("WHISPER_DIAG", "NATIVE_ERROR type=${it::class.java.name} message=${it.message}", it)
                        }.getOrElse { "转录失败：${it.message ?: it::class.java.simpleName}" }
                        withContext(Dispatchers.Main) {
                            val normalized = ChineseTextProcessor.normalizeChineseText(text)
                            Log.i("WHISPER_DIAG", "UI_UPDATE rawChars=${text.length} normalizedChars=${normalized.length} blank=${normalized.isBlank()}")
                            if (normalized.isNotBlank()) {
                                transcript.add(normalized)
                                previousWhisperText = normalized
                            }
                            processing = false
                            status = if (recording) "正在监听…" else "已停止"
                        }
                    }
                }
            } finally {
                runCatching { recorder.stop() }
                recorder.release()
                runCatching { QnnWhisperRealAudioRunner.stop() }
                Log.i("WHISPER_DIAG", "QNN_SESSION_STOPPED")
            }
        }
    }
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) start() else status = "需要麦克风权限才能转录"
    }

    DisposableEffect(Unit) {
        onDispose {
            recordingJob?.cancel()
            ttsQueue.close()
            audioPlayer.release()
        }
    }

    Surface(Modifier.fillMaxSize()) {
        Column(Modifier.fillMaxSize().padding(20.dp)) {
            TabRow(selectedTabIndex = selectedTab) {
                Tab(selected = selectedTab == 0, onClick = { selectedTab = 0 }, text = { Text("文字转语音") })
                Tab(selected = selectedTab == 1, onClick = { selectedTab = 1 }, text = { Text("语音转文字") })
            }
            Spacer(Modifier.height(20.dp))
            if (selectedTab == 0) {
                Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.SpaceBetween) {
                    Column(Modifier.fillMaxWidth()) {
                        Text("文字转语音", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                        Spacer(Modifier.height(8.dp))
                        Text(ttsStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(20.dp))
                        OutlinedTextField(
                            value = ttsText,
                            onValueChange = { ttsText = it },
                            modifier = Modifier.fillMaxWidth().height(280.dp),
                            placeholder = { Text("输入要朗读的中文文字") },
                            label = { Text("朗读文本") },
                        )
                    }
                    Button(
                        onClick = {
                            val text = ttsText.trim()
                            if (text.isBlank()) ttsStatus = "请先输入文字"
                            else {
                                ttsStatus = "已加入播放"
                                ttsQueue.offer(text)
                            }
                        },
                        modifier = Modifier.fillMaxWidth().height(56.dp),
                        shape = RoundedCornerShape(18.dp),
                    ) { Text("播放语音") }
                }
            } else {
                Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.SpaceBetween) {
                    Column(Modifier.fillMaxWidth()) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(12.dp).background(if (recording) Color(0xFF42A85F) else MaterialTheme.colorScheme.outline, CircleShape))
                            Spacer(Modifier.width(10.dp))
                            Text("语音转文字", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                        }
                        Spacer(Modifier.height(8.dp))
                        Text(status, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(24.dp))
                        Surface(Modifier.fillMaxWidth().height(420.dp), shape = RoundedCornerShape(20.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
                            if (transcript.isEmpty()) {
                                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("你的语音会出现在这里", color = MaterialTheme.colorScheme.onSurfaceVariant) }
                            } else {
                                LazyColumn(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                                    items(transcript) { line -> Text(line, style = MaterialTheme.typography.bodyLarge) }
                                }
                            }
                        }
                    }
                    Column(Modifier.fillMaxWidth()) {
                        Button(
                            onClick = {
                                if (recording) stop()
                                else if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) start()
                                else permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            },
                            modifier = Modifier.fillMaxWidth().height(56.dp),
                            shape = RoundedCornerShape(18.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = if (recording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary)
                        ) { Text(if (recording) "停止转录" else "开始说话") }
                        Spacer(Modifier.height(8.dp))
                        Text("Whisper Tiny · 16 kHz · 每 5 秒更新一次", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.labelMedium)
                    }
                }
            }
        }
    }
}



