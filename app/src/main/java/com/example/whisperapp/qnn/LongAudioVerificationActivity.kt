package com.example.whisperapp.qnn

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.example.whisperapp.audio.WavPcmReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.sqrt

/** Device-only verification harness for the first Long Audio milestone. */
class LongAudioVerificationActivity : ComponentActivity() {
    private var status by mutableStateOf("Starting...")
    private var finalTranscript by mutableStateOf("")
    private var copied by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        LongAudioAppLogger.startTest(this)
        setContent {
            val lines by LongAudioAppLogger.lines.collectAsState()
            val scroll = rememberScrollState()
            Column(Modifier.fillMaxSize().padding(16.dp)) {
                Text("Long Audio Verification", style = MaterialTheme.typography.headlineSmall)
                Spacer(Modifier.height(8.dp))
                Button(onClick = { runVerification() }, enabled = status != "Running...", modifier = Modifier.fillMaxWidth()) { Text("Run Test") }
                Spacer(Modifier.height(8.dp))
                Text("Status: $status")
                Spacer(Modifier.height(8.dp))
                Text("Final Transcript")
                Text(finalTranscript.ifEmpty { "—" })
                Spacer(Modifier.height(12.dp))
                Text("Logs", style = MaterialTheme.typography.titleMedium)
                Column(Modifier.weight(1f).fillMaxWidth().verticalScroll(scroll)) {
                    Text(lines.joinToString("\n"), style = MaterialTheme.typography.bodySmall)
                }
                Spacer(Modifier.height(8.dp))
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    TextButton(onClick = { LongAudioAppLogger.clear(); copied = false }) { Text("Clear Logs") }
                    Button(onClick = { copyLogs() }) { Text(if (copied) "Logs copied" else "Copy Logs") }
                }
            }
            LaunchedEffect(lines.size) { if (lines.isNotEmpty()) scroll.scrollTo(scroll.maxValue) }
        }
        runVerification()
    }

    private fun runVerification() {
        status = "Running..."
        copied = false
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val wav = File(getExternalFilesDir(null), "long_audio_input.wav")
                LongAudioAppLogger.info("VERIFICATION_START wav=${wav.absolutePath} exists=${wav.isFile}")
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                LongAudioAppLogger.info("VERIFICATION_PCM samples=${pcm.samples.size} rate=${pcm.sampleRate} channels=${pcm.channels}")
                val stats = pcmStats(pcm.samples)
                LongAudioAppLogger.info("PCM_STATS samples=${pcm.samples.size} min=${stats.min} max=${stats.max} rms=${stats.rms} mean=${stats.mean}")
                val result = LongAudioSingleContextRunner.run(this@LongAudioVerificationActivity, pcm.samples, pcm.sampleRate)
                val report = buildString {
                    appendLine("LONG_AUDIO status=${if (result.passed) "PASS" else "FAIL"}")
                    appendLine("chunks=${result.chunks.size}")
                    result.chunks.forEach { chunk ->
                        appendLine("chunk=${chunk.chunkIndex} start=${chunk.startTimeSeconds} end=${chunk.endTimeSeconds} samples=${chunk.endSample - chunk.startSample} eos=${chunk.eosReached} text=${chunk.decodedText}")
                    }
                    appendLine("final_transcript=${result.finalTranscript}")
                }
                File(getExternalFilesDir(null), "long_audio_verification_result.txt").writeText(report)
                LongAudioAppLogger.info("RESULT_WRITTEN")
                withContext(Dispatchers.Main) {
                    status = "PASS (diagnostics complete)"
                    finalTranscript = result.finalTranscript
                }
            } catch (t: Throwable) {
                File(getExternalFilesDir(null), "long_audio_verification_result.txt").writeText("LONG_AUDIO FAIL\n${t.stackTraceToString()}")
                LongAudioAppLogger.error("VERIFICATION_EXCEPTION", t)
                LongAudioAppLogger.error("ERROR type=${t.javaClass.name} message=${t.message}", t)
                withContext(Dispatchers.Main) { status = "FAIL"; finalTranscript = "" }
            }
        }
    }

    private fun copyLogs() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("Long Audio Verification Logs", LongAudioAppLogger.snapshot()))
        copied = true
    }

    private data class PcmStats(val min: Int, val max: Int, val rms: Double, val mean: Double)

    private fun pcmStats(samples: ShortArray): PcmStats {
        if (samples.isEmpty()) return PcmStats(0, 0, 0.0, 0.0)
        var min = Int.MAX_VALUE
        var max = Int.MIN_VALUE
        var sum = 0.0
        var squares = 0.0
        for (sample in samples) {
            val value = sample.toInt()
            min = minOf(min, value)
            max = maxOf(max, value)
            sum += value
            squares += value.toDouble() * value.toDouble()
        }
        return PcmStats(min, max, sqrt(squares / samples.size), sum / samples.size)
    }
}
