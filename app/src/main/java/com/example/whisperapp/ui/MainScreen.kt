package com.example.whisperapp.ui

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.whisperapp.audio.WavPcmReader
import com.example.whisperapp.qnn.WhisperReferenceRunner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var text by remember { mutableStateOf("") }
    var recording by remember { mutableStateOf(false) }
    var transcribing by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Top
    ) {
        Text("Snapdragon Voice Lab", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(8.dp))
        Text("Backend: HTP / QNN")
        Text("ASR: Whisper-Tiny")
        Text("TTS: MeloTTS-ZH")
        Spacer(Modifier.height(20.dp))
        Button(
            onClick = { context.startActivity(Intent(context, com.example.whisperapp.qnn.LongAudioVerificationActivity::class.java)) },
            modifier = Modifier.fillMaxWidth()
        ) { Text("Run Long Audio Verification") }
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = { context.startActivity(Intent(context, com.example.whisperapp.qnn.WhisperReferenceActivity::class.java)) },
            modifier = Modifier.fillMaxWidth()
        ) { Text("Run Whisper 5-Step Comparison") }
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = {
                transcribing = true
                scope.launch(Dispatchers.IO) {
                    val result = runCatching {
                        val wav = java.io.File(context.getExternalFilesDir(null), "long_audio_input.wav")
                        require(wav.isFile) { "Missing ${wav.absolutePath}" }
                        val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                        require(pcm.sampleRate == 16_000) { "Expected 16000 Hz, got ${pcm.sampleRate}" }
                        require(pcm.samples.size >= 80_000) { "Need at least 80000 samples, got ${pcm.samples.size}" }
                        WhisperReferenceRunner.transcribeFirst5s(context, pcm.samples.copyOf(80_000), 16_000, threads = 1)
                    }.getOrElse { "WHISPER_TRANSCRIBE_FAIL ${it.message ?: it::class.java.simpleName}" }
                    withContext(Dispatchers.Main) {
                        text = result.substringAfter("WHISPER_TRANSCRIBE_TEXT ", result).substringBefore("\nWHISPER_TRANSCRIBE_DONE")
                        transcribing = false
                    }
                }
            },
            enabled = !transcribing,
            modifier = Modifier.fillMaxWidth()
        ) { Text(if (transcribing) "Transcribing first 5 seconds..." else "Transcribe First 5 Seconds (whisper.cpp)") }
        Spacer(Modifier.height(12.dp))
        Button(onClick = { recording = !recording }, modifier = Modifier.fillMaxWidth()) {
            Text(if (recording) "Stop Recording" else "Start Recording")
        }
        Spacer(Modifier.height(16.dp))
        Text("Recognized Text")
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.fillMaxWidth().height(140.dp),
            placeholder = { Text("ASR output will appear here") }
        )
        Spacer(Modifier.height(12.dp))
        Button(onClick = { }, modifier = Modifier.fillMaxWidth()) { Text("Speak") }
        Spacer(Modifier.height(16.dp))
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("Inference: ?")
                Text("Audio: ?")
                Text("RTF: ?")
                Text("Backend: HTP (pending smoke test)")
                Text("Offline: YES")
            }
        }
    }
}