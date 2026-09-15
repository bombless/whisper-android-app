package com.example.whisperapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import android.content.Intent
import androidx.compose.ui.platform.LocalContext
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun MainScreen() {
    val context = LocalContext.current
    var text by remember { mutableStateOf("") }
    var recording by remember { mutableStateOf(false) }

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
            onClick = {
                context.startActivity(Intent(context, com.example.whisperapp.qnn.LongAudioVerificationActivity::class.java))
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Run Long Audio Verification")
        }
        Spacer(Modifier.height(8.dp))
        Button(
            onClick = {
                context.startActivity(Intent(context, com.example.whisperapp.qnn.WhisperReferenceActivity::class.java))
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Run Whisper 5-Step Comparison")
        }
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
                Text("Inference: —")
                Text("Audio: —")
                Text("RTF: —")
                Text("Backend: HTP (pending smoke test)")
                Text("Offline: YES")
            }
        }
    }
}
