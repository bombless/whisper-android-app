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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
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
import com.example.whisperapp.tts.SherpaOnnxTtsEngine
import com.example.whisperapp.tts.TtsQueue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import androidx.compose.runtime.LaunchedEffect

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var selectedTab by remember { mutableIntStateOf(0) }
    var ttsText by remember { mutableStateOf("") }
    var ttsStatus by remember { mutableStateOf("TTS 就绪") }
    var selectedTtsVariant by remember { mutableStateOf(SherpaOnnxTtsEngine.Variant.PIPER_XIAO_YA_INT8) }
    var ttsMenuExpanded by remember { mutableStateOf(false) }
    var recording by remember { mutableStateOf(false) }
    var processing by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("") }
    val transcript = remember { mutableStateListOf<String>() }
    var recordingJob by remember { mutableStateOf<Job?>(null) }
    val recordingLifecycleMutex = remember { Mutex() }
    var previousWhisperText by remember { mutableStateOf("") }
    val audioPlayer = remember { AudioPlayer() }
    fun createTtsQueue(variant: SherpaOnnxTtsEngine.Variant): TtsQueue = TtsQueue(scope, SherpaOnnxTtsEngine(context, variant), audioPlayer) { error ->
        Log.e("WHISPER_TTS", "TTS failed variant=$variant: ${error.message}", error)
        scope.launch(Dispatchers.Main) { ttsStatus = "TTS 失败: ${error.message ?: error::class.java.simpleName}" }
    }
    var ttsQueue by remember { mutableStateOf<TtsQueue?>(null) }
    var ttsReady by remember { mutableStateOf(false) }
    var sttReady by remember { mutableStateOf(false) }
    var nativeSwitching by remember { mutableStateOf(true) }
    var switchJob by remember { mutableStateOf<Job?>(null) }

    fun switchToTab(target: Int) {
        if (target == selectedTab || nativeSwitching) return
        switchJob?.cancel()
        switchJob = scope.launch(Dispatchers.IO) {
            nativeSwitching = true
            try {
                if (recording) {
                    recording = false
                    recordingJob?.cancelAndJoin()
                    recordingJob = null
                    processing = false
                    QnnWhisperRealAudioRunner.stop()
                }
                ttsQueue?.clearAndStop()
                ttsQueue?.closeAndWait()
                ttsQueue = null
                ttsReady = false
                sttReady = false
                withContext(Dispatchers.Main) { selectedTab = target }
                if (target == 1) {
                    status = "STT native 正在加载…"
                    QnnWhisperRealAudioRunner.start(context)
                    sttReady = true
                    status = "STT 已就绪"
                    Log.i("WHISPER_DIAG", "TAB_NATIVE_READY target=STT")
                } else {
                    ttsStatus = "TTS native 正在加载…"
                    val queue = createTtsQueue(selectedTtsVariant)
                    ttsQueue = queue
                    queue.prepare()
                    ttsReady = true
                    ttsStatus = "TTS 已就绪: ${selectedTtsVariant.displayName}"
                    Log.i("WHISPER_TTS", "TAB_NATIVE_READY target=TTS variant=$selectedTtsVariant")
                }
            } catch (t: Throwable) {
                Log.e("WHISPER_NATIVE", "TAB_NATIVE_LOAD_FAILED target=$target", t)
                withContext(Dispatchers.Main) {
                    if (target == 1) status = "STT 加载失败: ${t.message ?: t::class.java.simpleName}"
                    else ttsStatus = "TTS 加载失败: ${t.message ?: t::class.java.simpleName}"
                }
            } finally {
                nativeSwitching = false
            }
        }
    }

    LaunchedEffect(selectedTtsVariant) {
        if (selectedTab != 0 || nativeSwitching) return@LaunchedEffect
        nativeSwitching = true
        ttsReady = false
        try {
            ttsQueue?.closeAndWait()
            val queue = createTtsQueue(selectedTtsVariant)
            ttsQueue = queue
            ttsStatus = "TTS native 正在加载…"
            queue.prepare()
            ttsReady = true
            ttsStatus = "TTS 已就绪: ${selectedTtsVariant.displayName}"
            Log.i("WHISPER_TTS", "TTS_NATIVE_READY variant=$selectedTtsVariant")
        } catch (t: Throwable) {
            Log.e("WHISPER_TTS", "TTS_NATIVE_LOAD_FAILED variant=$selectedTtsVariant", t)
            ttsStatus = "TTS 加载失败: ${t.message ?: t::class.java.simpleName}"
        } finally {
            nativeSwitching = false
        }
    }

    LaunchedEffect(Unit) {
        nativeSwitching = true
        try {
            val queue = createTtsQueue(selectedTtsVariant)
            ttsQueue = queue
            ttsStatus = "TTS native 正在加载…"
            queue.prepare()
            ttsReady = true
            ttsStatus = "TTS 已就绪: ${selectedTtsVariant.displayName}"
            Log.i("WHISPER_TTS", "TTS_NATIVE_READY_INITIAL variant=$selectedTtsVariant")
        } catch (t: Throwable) {
            Log.e("WHISPER_TTS", "TTS_NATIVE_LOAD_FAILED_INITIAL", t)
            ttsStatus = "TTS 加载失败: ${t.message ?: t::class.java.simpleName}"
        } finally {
            nativeSwitching = false
        }
    }

    fun stop() {
        Log.i("WHISPER_DIAG", "RECORD_STOP recordingJob=${recordingJob != null}")
        recording = false
        recordingJob?.cancel()
        processing = false
        previousWhisperText = ""
        ttsQueue?.clearAndStop()
        status = "已停止"
    }

    fun start() {
        Log.i("WHISPER_DIAG", "RECORD_START")
        transcript.clear()
        previousWhisperText = ""
        ttsQueue?.clearAndStop()
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
            var recorder: AudioRecord? = null
            val chunk = ShortArray(samplesPerChunk)
            var offset = 0
            try {
                QnnWhisperRealAudioRunner.start(context)
                recorder = AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minBuffer, samplesPerChunk))
                check(recorder?.state == AudioRecord.STATE_INITIALIZED) { "麦克风初始化失败" }
                val activeRecorder = recorder ?: error("麦克风录音器未初始化")
                Log.i("WHISPER_DIAG", "QNN_SESSION_READY")
                activeRecorder.startRecording()
                check(activeRecorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) { "麦克风未进入录音状态" }
                Log.i("WHISPER_DIAG", "AUDIO_RECORD_STARTED state=${activeRecorder.recordingState} minBuffer=$minBuffer chunkSamples=$samplesPerChunk")
                while (kotlinx.coroutines.currentCoroutineContext().isActive) {
                    val n = activeRecorder.read(chunk, offset, samplesPerChunk - offset)
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
            } catch (t: Throwable) {
                Log.e("WHISPER_DIAG", "RECORD_START_FAILED type=${t::class.java.name} message=${t.message}", t)
                withContext(Dispatchers.Main) {
                    recording = false
                    processing = false
                    status = "启动失败：${t.message ?: t::class.java.simpleName}"
                }
            } finally {
                runCatching { recorder?.stop() }
                runCatching { recorder?.release() }
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
            scope.launch {
                ttsQueue?.closeAndWait()
                audioPlayer.release()
            }
        }
    }

    val contentScrollState = rememberScrollState()

    Surface(Modifier.fillMaxSize()) {
        Column(
            Modifier
                .fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(horizontal = 20.dp)
        ) {
            TabRow(selectedTabIndex = selectedTab) {
                Tab(selected = selectedTab == 0, onClick = { switchToTab(0) }, text = { Text("文字转语音") })
                Tab(selected = selectedTab == 1, onClick = { switchToTab(1) }, text = { Text("语音转文字") })
            }
            Spacer(Modifier.height(20.dp))
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(contentScrollState)
            ) {
            if (selectedTab == 0) {
                Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(20.dp)) {
                    Column(Modifier.fillMaxWidth()) {
                        Text("文字转语音", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
                        Spacer(Modifier.height(8.dp))
                        Text(ttsStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Spacer(Modifier.height(12.dp))
                        Box(Modifier.fillMaxWidth()) {
                            OutlinedButton(enabled = ttsReady && !nativeSwitching, onClick = { ttsMenuExpanded = true }, modifier = Modifier.fillMaxWidth()) {
                                Text("TTS 引擎：${selectedTtsVariant.displayName}")
                            }
                            DropdownMenu(expanded = ttsMenuExpanded && ttsReady && !nativeSwitching, onDismissRequest = { ttsMenuExpanded = false }) {
                                SherpaOnnxTtsEngine.Variant.entries.forEach { variant ->
                                    DropdownMenuItem(text = { Text(variant.displayName) }, onClick = {
                                        if (variant != selectedTtsVariant) { ttsQueue?.clearAndStop(); selectedTtsVariant = variant }
                                        ttsMenuExpanded = false
                                    })
                                }
                            }
                        }
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
                        enabled = ttsReady && !nativeSwitching,
                        onClick = {
                            val text = ttsText.trim()
                            Log.i("WHISPER_TTS", "Play clicked, textLength=${text.length}")
                            if (text.isBlank()) ttsStatus = "请输入要朗读的文字"
                            else {
                                ttsStatus = "已加入播放"
                                ttsQueue?.offer(text)
                            }
                        },
                        modifier = Modifier.fillMaxWidth().height(56.dp),
                        shape = RoundedCornerShape(18.dp),
                    ) { Text("播放语音") }
                }
            } else {
                Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(20.dp)) {
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
                            enabled = sttReady && !nativeSwitching,
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
}







