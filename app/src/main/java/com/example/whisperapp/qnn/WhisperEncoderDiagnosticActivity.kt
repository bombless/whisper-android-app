package com.example.whisperapp.qnn

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.example.whisperapp.audio.WavPcmReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class WhisperEncoderDiagnosticActivity : ComponentActivity() {
    companion object { private const val TAG = "WhisperEncoderDiagnostic" }
    private var status by mutableStateOf("Idle")
    private var report by mutableStateOf("")
    private var lastLog: File? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val runEncoderTest = intent.getBooleanExtra("run_encoder_test", false)
        Log.i(TAG, "DIAG_ACTIVITY_ON_CREATE run_encoder_test=$runEncoderTest")
        setContent {
            val scroll = rememberScrollState()
            Column(Modifier.fillMaxSize().padding(16.dp)) {
                Text("Whisper Encoder Diagnostic", style = MaterialTheme.typography.headlineSmall)
                Spacer(Modifier.height(8.dp))
                Button(onClick = { Log.i(TAG, "DIAG_ACTIVITY_MANUAL_RUN"); runDiagnostic() }, enabled = status != "Running", modifier = Modifier.fillMaxWidth()) { Text("Run Encoder Diagnostic") }
                Spacer(Modifier.height(8.dp))
                Button(onClick = { exportLog() }, enabled = lastLog?.isFile == true, modifier = Modifier.fillMaxWidth()) { Text("Export Log") }
                Spacer(Modifier.height(8.dp))
                Text("Status: $status")
                Spacer(Modifier.height(8.dp))
                Text("Last Result: ${lastLog?.name ?: "None"}")
                Spacer(Modifier.height(8.dp))
                Column(Modifier.weight(1f).fillMaxWidth().verticalScroll(scroll)) { Text(report, style = MaterialTheme.typography.bodySmall) }
            }
        }
        if (runEncoderTest) { Log.i(TAG, "DIAG_ACTIVITY_AUTO_RUN"); runDiagnostic() }
    }
    private fun runDiagnostic() {
        Log.i(TAG, "DIAG_RUN_DIAGNOSTIC_START")
        status = "Running"; report = ""
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val wav = File(getExternalFilesDir(null), "long_audio_input.wav")
                require(wav.isFile) { "Missing ${wav.absolutePath}" }
                Log.i(TAG, "DIAG_WAV_LOAD_START")
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                Log.i(TAG, "DIAG_WAV_LOAD_DONE sampleRate=${pcm.sampleRate} sampleCount=${pcm.samples.size}")
                require(pcm.sampleRate == 16_000) { "Expected 16000 Hz, got ${pcm.sampleRate}" }
                require(pcm.samples.size >= 80_000) { "Need at least 80000 samples, got ${pcm.samples.size}" }
                Log.i(TAG, "DIAG_RUNNER_CALL_START samples=80000 sampleRate=16000 threads=1")
                val result = WhisperEncoderDiagnosticRunner.run(this@WhisperEncoderDiagnosticActivity, pcm.samples.copyOf(80_000), 16_000, 1)
                Log.i(TAG, "DIAG_RUNNER_CALL_DONE result=${result.result} logFile=${result.logFile.name}")
                withContext(Dispatchers.Main) { lastLog = result.logFile; report = result.report; status = result.result }
            } catch (t: Throwable) {
                Log.e(TAG, "DIAG_RUN_DIAGNOSTIC_EXCEPTION type=${t::class.java.name} message=${t.message}")
                withContext(Dispatchers.Main) { report = "ENCODER_TEST_ERROR type=kotlin message=${t.stackTraceToString()}"; status = "Failed" }
            }
        }
    }
    private fun exportLog() {
        val file = lastLog ?: return
        val uri: Uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        startActivity(Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) })
    }
}
