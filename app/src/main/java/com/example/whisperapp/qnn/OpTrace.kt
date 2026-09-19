package com.example.whisperapp.qnn

import android.util.Log
import java.io.File
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max
import kotlin.math.min

/**
 * Global op-trace recorder for the Whisper QNN pipeline.
 *
 * Two layers of detail:
 *  1. App-level timed spans (mel, encoder.run, decoder step, argmax, copies, tokenizer, ...).
 *     Recorded with [time]/[begin]; aggregated per-run and cumulatively.
 *  2. QNN HTP optrace CSV files (profiling_level=optrace) parsed into per-op aggregates
 *     so the HTTP dashboard can show exactly which device ops burn time.
 *
 * All state is global + thread-safe so QnnProfileServer can read it at any moment.
 */
object OpTrace {
    private const val TAG = "OPTRACE"
    private const val MAX_EVENTS = 8192
    private const val MAX_RUNS = 64

    data class SpanEvent(
        val runId: Long,
        val name: String,
        val category: String,
        val startMs: Double,   // ms since OpTrace epoch
        val durationMs: Double,
        val attrs: Map<String, String> = emptyMap(),
    )

    data class Stats(
        val name: String,
        val category: String,
        val count: Int,
        val totalMs: Double,
        val meanMs: Double,
        val minMs: Double,
        val p50Ms: Double,
        val p95Ms: Double,
        val maxMs: Double,
    )

    data class Run(
        val runId: Long,
        val label: String,
        val startedMs: Double,
        val wallMs: Double,
        val audioSeconds: Double,
        val spans: List<SpanEvent>,
        val attrs: Map<String, String>,
    ) {
        val totalMs: Double get() = wallMs
    }

    @PublishedApi internal val epoch = System.nanoTime()
    private fun nowMs() = (System.nanoTime() - epoch) / 1_000_000.0

    internal val lock = Any()
    private val events = ArrayDeque<SpanEvent>()
    private val runs = ArrayDeque<Run>()
    private val runCounter = AtomicLong(0)
    internal var currentRunId = -1L
    private var currentRunWallStartMs = 0.0

    @Volatile @PublishedApi internal var enabled = true

    private fun push(e: SpanEvent) = synchronized(lock) {
        events.addLast(e)
        while (events.size > MAX_EVENTS) events.removeFirst()
    }

    /** Start a new logical run (one transcribe chunk). Returns the run id. */
    fun beginRun(label: String, attrs: Map<String, String> = emptyMap()): Long {
        val id = runCounter.incrementAndGet()
        synchronized(lock) {
            currentRunId = id
            currentRunWallStartMs = nowMs()
        }
        Log.i(TAG, "RUN_BEGIN id=$id label=$label attrs=$attrs")
        return id
    }

    fun endRun(runId: Long, audioSeconds: Double, attrs: Map<String, String> = emptyMap()) = synchronized(lock) {
        val spans = events.filter { it.runId == runId }
        val merged = attrs.toMutableMap()
        if (audioSeconds > 0) merged["audio_seconds"] = "%.3f".format(audioSeconds)
        val wallMs = if (currentRunId == runId) (nowMs() - currentRunWallStartMs).coerceAtLeast(0.0) else spans.sumOf { it.durationMs }
        runs.addLast(Run(runId, "run_$runId", currentRunWallStartMs, wallMs, audioSeconds, spans, merged))
        while (runs.size > MAX_RUNS) runs.removeFirst()
        if (currentRunId == runId) currentRunId = -1
        Log.i(TAG, "RUN_END id=$runId spans=${spans.size} wallMs=%.2f audioSec=%.3f rtf=%.4f".format(wallMs, audioSeconds, if (audioSeconds > 0) wallMs / 1000.0 / audioSeconds else 0.0))
    }

    /** Record a completed span. */
    fun record(name: String, category: String, durationMs: Double, attrs: Map<String, String> = emptyMap(), runId: Long = synchronized(lock) { currentRunId }, startMs: Double = nowMs()): SpanEvent? {
        if (!enabled) return null
        val e = SpanEvent(runId, name, category, startMs, durationMs, attrs)
        push(e)
        return e
    }

    /** Time a block as a span. */
    inline fun <T> time(name: String, category: String, attrs: Map<String, String> = emptyMap(), block: () -> T): T {
        if (!enabled) return block()
        val start = System.nanoTime()
        val startMs = (start - epoch) / 1_000_000.0
        try {
            return block()
        } finally {
            record(name, category, (System.nanoTime() - start) / 1_000_000.0, attrs, startMs = startMs)
        }
    }

    fun clear() = synchronized(lock) { events.clear(); runs.clear() }

    // ------------------------------------------------------------------
    // Aggregation
    // ------------------------------------------------------------------

    private fun aggregate(spans: List<SpanEvent>): List<Stats> =
        spans.groupBy { it.category + "\u0000" + it.name }
            .map { (_, list) ->
                val sorted = list.map { it.durationMs }.sorted()
                Stats(
                    name = list.first().name,
                    category = list.first().category,
                    count = sorted.size,
                    totalMs = sorted.sum(),
                    meanMs = sorted.average(),
                    minMs = sorted.first(),
                    p50Ms = sorted[sorted.size / 2],
                    p95Ms = sorted[min(sorted.size - 1, (sorted.size * 0.95).toInt())],
                    maxMs = sorted.last(),
                )
            }
            .sortedByDescending { it.totalMs }

    fun snapshot(): Snapshot = synchronized(lock) {
        Snapshot(
            enabled = enabled,
            events = events.toList(),
            runs = runs.toList(),
            runStats = aggregate(runs.flatMap { it.spans }),
            eventStats = aggregate(events.toList()),
        )
    }

    data class Snapshot(
        val enabled: Boolean,
        val events: List<SpanEvent>,
        val runs: List<Run>,
        val runStats: List<Stats>,
        val eventStats: List<Stats>,
    )

    // ------------------------------------------------------------------
    // QNN HTP optrace CSV parsing
    // ------------------------------------------------------------------

    data class OpAggregate(
        val file: String,
        val opName: String,
        val count: Int,
        val totalUs: Double,
        val meanUs: Double,
        val minUs: Double,
        val maxUs: Double,
        val sharePct: Double,
    )

    /**
     * Parse QNN/QAIRT optrace CSV files into per-op aggregates.
     * Supports both layouts:
     *  - columnar per-op ("Op Name", "Time (us)", ...)
     *  - event stream ("Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier")
     * Unknown layouts return an empty list.
     */
    fun parseQnnOptraceCsv(file: File): List<OpAggregate> {
        if (!file.isFile || file.length() > 64L * 1024 * 1024) return emptyList()
        val lines = runCatching { file.readLines(Charsets.UTF_8) }.getOrElse { return emptyList() }
        if (lines.size < 2) return emptyList()
        val header = splitCsv(lines.first()).map { it.trim() }
        if (header.any { it.equals("Event Identifier", ignoreCase = true) }) {
            return parseEventStreamCsv(file.name, lines)
        }
        fun col(vararg names: String): Int? {
            for (n in names) {
                val idx = header.indexOfFirst { it.trim().equals(n, ignoreCase = true) }
                if (idx >= 0) return idx
            }
            return null
        }
        val nameIdx = col("Op Name", "Name", "op_name", "OpName") ?: return emptyList()
        val timeIdx = col("Time (us)", "Total Time (us)", "Time(us)", "Total Time(us)", "TotalTime(us)", "time_us", "Time")
            ?: col("Avg Time (us)", "Avg Time(us)", "AvgTime(us)", "avg_time_us")
            ?: return emptyList()
        val isTotal = header[timeIdx].contains("total", ignoreCase = true)
        val agg = LinkedHashMap<String, MutableList<Double>>()
        var fileTotalUs = 0.0
        for (line in lines.drop(1)) {
            if (line.isBlank()) continue
            val cells = splitCsv(line)
            val name = cells.getOrNull(nameIdx)?.trim().orEmpty()
            if (name.isEmpty() || name.startsWith("Total") || name.startsWith("#")) continue
            val value = cells.getOrNull(timeIdx)?.trim()?.toDoubleOrNull() ?: continue
            if (isTotal) {
                agg.getOrPut(name) { mutableListOf() }.add(value)
                fileTotalUs += value
            } else {
                agg.getOrPut(name) { mutableListOf() }
            }
        }
        if (agg.isEmpty()) return emptyList()
        val perOpTotal = if (isTotal) null else {
            // avg-time layout: weight by op count if a count column exists
            val countIdx = col("Count", "Num", "Calls", "count")
            val totals = LinkedHashMap<String, Double>()
            for (line in lines.drop(1)) {
                if (line.isBlank()) continue
                val cells = splitCsv(line)
                val name = cells.getOrNull(nameIdx)?.trim().orEmpty()
                if (name.isEmpty() || name.startsWith("Total") || name.startsWith("#")) continue
                val value = cells.getOrNull(timeIdx)?.trim()?.toDoubleOrNull() ?: continue
                val cnt = countIdx?.let { cells.getOrNull(it)?.trim()?.toDoubleOrNull() } ?: 1.0
                totals[name] = (totals[name] ?: 0.0) + value * cnt
            }
            totals
        }
        val totalAll = perOpTotal?.values?.sum() ?: fileTotalUs
        return agg.map { (name, values) ->
            val totalUs = perOpTotal?.get(name) ?: values.sum()
            OpAggregate(
                file = file.name,
                opName = name,
                count = values.size,
                totalUs = totalUs,
                meanUs = if (values.isEmpty()) 0.0 else totalUs / values.size,
                minUs = values.minOrNull() ?: 0.0,
                maxUs = values.maxOrNull() ?: 0.0,
                sharePct = if (totalAll > 0) totalUs / totalAll * 100.0 else 0.0,
            )
        }.sortedByDescending { it.totalUs }
    }

    /**
     * QAIRT event-stream layout:
     * Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier
     * NODE rows carry CYCLES per executed op; US rows carry wall-time events
     * (execute / RPC / power-on / VTCM). Cycles are converted to µs using the
     * "Accelerator (execute)" wall time as the clock calibration.
     */
    private fun parseEventStreamCsv(fileName: String, lines: List<String>): List<OpAggregate> {
        data class Row(val message: String, val time: Double, val unit: String, val level: String, val ident: String)
        val rows = lines.drop(1).mapNotNull { line ->
            if (line.isBlank()) return@mapNotNull null
            val c = splitCsv(line)
            if (c.size < 7) return@mapNotNull null
            val t = c[2].trim().toDoubleOrNull() ?: return@mapNotNull null
            Row(c[1].trim(), t, c[3].trim(), c[5].trim(), c[6].trim())
        }
        // calibrate cycles -> us from the accelerator execute event pair
        val accUs = rows.filter { it.unit == "US" && it.ident == "Accelerator (execute) time" }.sumOf { it.time }
        val accCycles = rows.filter { it.unit == "CYCLES" && it.ident == "Accelerator (execute) time (cycles)" }.sumOf { it.time }
        val cyclesPerUs = if (accUs > 0 && accCycles > 0) accCycles / accUs else 0.0
        val agg = LinkedHashMap<String, MutableList<Double>>()
        for (r in rows) {
            if (r.unit == "CYCLES") {
                if (r.level != "NODE") continue
                val name = r.ident.removeSuffix(" (cycles)")
                if (name.isEmpty() || cyclesPerUs <= 0) continue
                agg.getOrPut("node:$name") { mutableListOf() }.add(r.time / cyclesPerUs)
            } else if (r.unit == "US") {
                // keep the interesting wall-time events, drop counters like "Number of HVX threads"
                if (r.time <= 0) continue
                if (r.ident.startsWith("Number of")) continue
                agg.getOrPut(r.ident) { mutableListOf() }.add(r.time)
            }
        }
        val totalAll = agg.values.sumOf { it.sum() }
        return agg.map { (name, values) ->
            OpAggregate(
                file = fileName,
                opName = name,
                count = values.size,
                totalUs = values.sum(),
                meanUs = values.average(),
                minUs = values.minOrNull() ?: 0.0,
                maxUs = values.maxOrNull() ?: 0.0,
                sharePct = if (totalAll > 0) values.sum() / totalAll * 100.0 else 0.0,
            )
        }.sortedByDescending { it.totalUs }
    }

    private fun splitCsv(s: String): List<String> {
        val out = mutableListOf<String>()
        val cur = StringBuilder()
        var quoted = false
        var i = 0
        while (i < s.length) {
            val c = s[i]
            if (c == '"') {
                if (quoted && i + 1 < s.length && s[i + 1] == '"') { cur.append('"'); i++ } else quoted = !quoted
            } else if (c == ',' && !quoted) { out += cur.toString(); cur.setLength(0) } else cur.append(c)
            i++
        }
        out += cur.toString()
        return out
    }

    fun jsonEscape(s: String): String = buildString(s.length + 8) {
        for (c in s) {
            when (c) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(c)
            }
        }
    }

    fun attrsJson(attrs: Map<String, String>): String =
        attrs.entries.joinToString(",") { "\"${jsonEscape(it.key)}\":\"${jsonEscape(it.value)}\"" }

    fun statsJson(s: Stats): String = buildString {
        append("{\"name\":\"${jsonEscape(s.name)}\",\"category\":\"${jsonEscape(s.category)}\",\"count\":${s.count},")
        append("\"total_ms\":${fmt(s.totalMs)},\"mean_ms\":${fmt(s.meanMs)},\"min_ms\":${fmt(s.minMs)},\"p50_ms\":${fmt(s.p50Ms)},\"p95_ms\":${fmt(s.p95Ms)},\"max_ms\":${fmt(s.maxMs)}}")
    }

    fun opJson(o: OpAggregate): String = buildString {
        append("{\"file\":\"${jsonEscape(o.file)}\",\"op\":\"${jsonEscape(o.opName)}\",\"count\":${o.count},")
        append("\"total_us\":${fmt(o.totalUs)},\"mean_us\":${fmt(o.meanUs)},\"min_us\":${fmt(o.minUs)},\"max_us\":${fmt(o.maxUs)},\"share_pct\":${fmt(o.sharePct)}}")
    }

    fun eventJson(e: SpanEvent): String = buildString {
        append("{\"run_id\":${e.runId},\"name\":\"${jsonEscape(e.name)}\",\"category\":\"${jsonEscape(e.category)}\",")
        append("\"start_ms\":${fmt(e.startMs)},\"duration_ms\":${fmt(e.durationMs)},\"attrs\":{${attrsJson(e.attrs)}}}")
    }

    fun runJson(r: Run): String {
        val total = r.totalMs
        val rtf = if (r.audioSeconds > 0) total / 1000.0 / r.audioSeconds else null
        return buildString {
            append("{\"run_id\":${r.runId},\"label\":\"${jsonEscape(r.label)}\",\"audio_seconds\":${fmt(r.audioSeconds)},")
            append("\"total_ms\":${fmt(total)},\"spans\":[${r.spans.joinToString(",") { eventJson(it) }}],")
            append("\"attrs\":{${attrsJson(r.attrs)}}")
            if (rtf != null) append(",\"rtf\":${fmt(rtf)}")
            append("}")
        }
    }

    /** Format without scientific notation, 3 decimals, locale-independent (safe for JSON). */
    fun fmt(v: Double): String = if (v.isNaN() || v.isInfinite()) "0" else java.lang.String.format(java.util.Locale.US, "%.3f", v)
}
