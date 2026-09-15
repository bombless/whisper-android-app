package com.example.whisperapp.qnn

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.example.whisperapp.audio.WavPcmReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class WhisperReferenceActivity : ComponentActivity() {
    private var status by mutableStateOf("Ready")
    private var report by mutableStateOf("")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        LongAudioAppLogger.startTest(this)
        setContent {
            val lines by LongAudioAppLogger.lines.collectAsState()
            val scroll = rememberScrollState()
            Column(Modifier.fillMaxSize().padding(16.dp)) {
                Text("Whisper 5-Step Comparison", style = MaterialTheme.typography.headlineSmall)
                Spacer(Modifier.height(8.dp))
                Text("Reference: whisper.cpp CPU oracle / QNN: existing HTP runner")
                Spacer(Modifier.height(8.dp))
                Button(onClick = { runComparison() }, enabled = status != "Running...", modifier = Modifier.fillMaxWidth()) {
                    Text("Run 5-Step Comparison")
                }
                Spacer(Modifier.height(8.dp))
                Text("Status: $status")
                Spacer(Modifier.height(8.dp))
                Column(Modifier.weight(1f).fillMaxWidth().verticalScroll(scroll)) {
                    Text(report.ifEmpty { lines.joinToString("\n") }, style = MaterialTheme.typography.bodySmall)
                }
                LaunchedEffect(lines.size) { if (lines.isNotEmpty()) scroll.scrollTo(scroll.maxValue) }
            }
        }
    }

    private fun runComparison() {
        status = "Running..."
        report = ""
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val wav = File(getExternalFilesDir(null), "long_audio_input.wav")
                require(wav.isFile) { "Missing ${wav.absolutePath}" }
                val pcm = wav.inputStream().use { WavPcmReader.read(it) }
                require(pcm.sampleRate == 16_000) { "Expected 16000 Hz, got ${pcm.sampleRate}" }
                require(pcm.samples.size >= 80_000) { "Need at least 80000 samples, got ${pcm.samples.size}" }
                val first5s = pcm.samples.copyOf(80_000)
                LongAudioAppLogger.clear()
                LongAudioAppLogger.info("COMPARE_INPUT samples=80000 sampleRate=16000")

                val reference = WhisperReferenceRunner.run(this@WhisperReferenceActivity, first5s, 16_000, threads = 1)
                LongAudioAppLogger.info("REFERENCE_RUN_DONE sha256=${reference.pcmSha256}")

                val qnn = QnnWhisperRealAudioRunner.run(
                    this@WhisperReferenceActivity,
                    first5s,
                    16_000,
                    requestedSteps = 5,
                    autoregressive = true,
                )
                require(qnn.passed) { qnn.report }

                val comparison = compare(reference.report, LongAudioAppLogger.snapshot())
                val full = buildString {
                    append(reference.report)
                    appendLine("QNN_REPORT")
                    appendLine(qnn.report)
                    appendLine("COMPARISON")
                    append(comparison)
                    appendLine("COMPARE_DONE")
                }
                File(getExternalFilesDir(null), "whisper_5step_comparison.txt").writeText(full)
                withContext(Dispatchers.Main) {
                    report = full
                    status = if (comparison.contains("COMPARE_FIRST_DIVERGENCE step=", ignoreCase = false)) "DIVERGENCE FOUND" else "MATCH"
                }
            } catch (t: Throwable) {
                val failure = "WHISPER_COMPARE_FAIL ${t.stackTraceToString()}"
                File(getExternalFilesDir(null), "whisper_5step_comparison.txt").writeText(failure)
                withContext(Dispatchers.Main) {
                    report = failure
                    status = "FAIL"
                }
            }
        }
    }

    private data class Step(val input: Int, val position: Int, val top1: Int, val topK: List<Pair<Int, Float>>)

    private fun compare(reference: String, qnnLogs: String): String {
        val refSteps = parseReference(reference)
        val qnnSteps = parseQnn(qnnLogs)
        val out = StringBuilder()
        var first: Int? = null
        for (step in 0 until 5) {
            val r = refSteps[step]
            val q = qnnSteps[step]
            val match = r != null && q != null && r.input == q.input && r.position == q.position && r.top1 == q.top1 && topIds(r.topK) == topIds(q.topK)
            if (!match && first == null) first = step
            out.appendLine("COMPARE_STEP step=$step refTop1=${r?.top1 ?: "NA"} qnnTop1=${q?.top1 ?: "NA"} refInput=${r?.input ?: "NA"} qnnInput=${q?.input ?: "NA"} refPosition=${r?.position ?: "NA"} qnnPosition=${q?.position ?: "NA"} topKMatch=${r != null && q != null && topIds(r.topK) == topIds(q.topK)} match=$match")
            if (r != null) out.appendLine("REF_TOPK step=$step ${r.topK.joinToString(" ") { "${it.first}:${it.second}" }}")
            if (q != null) out.appendLine("QNN_TOPK step=$step ${q.topK.joinToString(" ") { "${it.first}:${it.second}" }}")
        }
        out.appendLine("COMPARE_FIRST_DIVERGENCE step=${first ?: "NONE"}")
        return out.toString()
    }

    private fun topIds(values: List<Pair<Int, Float>>): List<Int> = values.map { it.first }

    private fun parseReference(text: String): Map<Int, Step> {
        val steps = mutableMapOf<Int, Step>()
        val stepRegex = Regex("WHISPER_REF_STEP step=(\\d+) inputToken=(\\d+) position=(\\d+) top1=(\\d+)")
        val topRegex = Regex("WHISPER_REF_TOPK step=(\\d+)(.*)")
        val tops = mutableMapOf<Int, List<Pair<Int, Float>>>()
        for (line in text.lineSequence()) {
            stepRegex.find(line)?.let {
                val s = it.groupValues[1].toInt()
                steps[s] = Step(it.groupValues[2].toInt(), it.groupValues[3].toInt(), it.groupValues[4].toInt(), emptyList())
            }
            topRegex.find(line)?.let { tops[it.groupValues[1].toInt()] = parseTopK(it.groupValues[2]) }
        }
        return steps.mapValues { (s, step) -> step.copy(topK = tops[s].orEmpty()) }
    }

    private fun parseQnn(text: String): Map<Int, Step> {
        val steps = mutableMapOf<Int, Step>()
        val stepRegex = Regex("QNN_COMPARE_STEP step=(\\d+) inputToken=(\\d+) position=(\\d+) top1=(\\d+)")
        val topRegex = Regex("QNN_COMPARE_TOPK step=(\\d+)\\s*(.*)")
        val tops = mutableMapOf<Int, List<Pair<Int, Float>>>()
        for (line in text.lineSequence()) {
            stepRegex.find(line)?.let {
                val s = it.groupValues[1].toInt()
                steps[s] = Step(it.groupValues[2].toInt(), it.groupValues[3].toInt(), it.groupValues[4].toInt(), emptyList())
            }
            topRegex.find(line)?.let { tops[it.groupValues[1].toInt()] = parseTopK(it.groupValues[2]) }
        }
        return steps.mapValues { (s, step) -> step.copy(topK = tops[s].orEmpty()) }
    }

    private fun parseTopK(text: String): List<Pair<Int, Float>> = text.trim().split(Regex("\\s+"))
        .mapNotNull { token ->
            val parts = token.split(":", limit = 2)
            if (parts.size != 2) null else parts[0].toIntOrNull()?.let { id -> parts[1].toFloatOrNull()?.let { id to it } }
        }
}
