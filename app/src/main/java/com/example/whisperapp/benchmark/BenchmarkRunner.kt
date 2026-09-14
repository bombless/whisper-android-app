package com.example.whisperapp.benchmark

import kotlin.math.roundToLong

object BenchmarkRunner {
    data class Sample(val latencyMs: Long)

    fun median(samples: List<Sample>): Long {
        require(samples.isNotEmpty())
        val values = samples.map { it.latencyMs }.sorted()
        return values[values.size / 2]
    }

    fun p90(samples: List<Sample>): Long {
        require(samples.isNotEmpty())
        val values = samples.map { it.latencyMs }.sorted()
        return values[((values.lastIndex * 0.90).roundToLong()).toInt()]
    }
}
