package com.example.whisperapp.audio

import java.io.BufferedInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class WavPcm16(val sampleRate: Int, val channels: Int, val samples: ShortArray)

object WavPcmReader {
    fun read(input: InputStream): WavPcm16 {
        val stream = BufferedInputStream(input)
        fun ascii(n: Int): String { val b = ByteArray(n); readFully(stream, b); return b.toString(Charsets.US_ASCII) }
        fun u16(): Int = readLe(stream, 2)
        fun u32(): Int = readLe(stream, 4)
        require(ascii(4) == "RIFF") { "Not RIFF" }
        u32(); require(ascii(4) == "WAVE") { "Not WAVE" }
        var format: Int? = null; var channels: Int? = null; var rate: Int? = null; var bits: Int? = null; var data: ByteArray? = null
        while (true) {
            val id = ByteArray(4); val first = stream.read(); if (first < 0) break; id[0] = first.toByte(); readFully(stream, id, 1, 3)
            val size = u32()
            when (id.toString(Charsets.US_ASCII)) {
                "fmt " -> { format = u16(); channels = u16(); rate = u32(); u32(); u16(); bits = u16(); if (size > 16) skipFully(stream, size - 16) }
                "data" -> { data = ByteArray(size); readFully(stream, data) }
                else -> skipFully(stream, size)
            }
            if ((size and 1) != 0) skipFully(stream, 1)
        }
        require(format == 1) { "Only PCM WAV is supported: format=$format" }
        require(channels == 1) { "Only mono WAV is supported: channels=$channels" }
        require(bits == 16) { "Only PCM16 WAV is supported: bits=$bits" }
        val bytes = data ?: error("WAV data chunk missing")
        require(bytes.size % 2 == 0) { "Odd PCM16 byte count" }
        val samples = ShortArray(bytes.size / 2)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(samples)
        return WavPcm16(rate ?: error("WAV fmt chunk missing"), 1, samples)
    }

    private fun readLe(input: InputStream, count: Int): Int {
        val b = ByteArray(count); readFully(input, b); var v = 0
        for (i in b.indices) v = v or ((b[i].toInt() and 0xff) shl (8 * i))
        return v
    }
    private fun readFully(input: InputStream, b: ByteArray, offset: Int = 0, length: Int = b.size - offset) {
        var p = offset
        while (p < offset + length) { val n = input.read(b, p, offset + length - p); require(n >= 0) { "Unexpected EOF" }; p += n }
    }
    private fun skipFully(input: InputStream, count: Int) { var left = count; while (left > 0) { val n = input.skip(left.toLong()).toInt(); if (n > 0) left -= n else require(input.read() >= 0) { "Unexpected EOF" } } }
}
