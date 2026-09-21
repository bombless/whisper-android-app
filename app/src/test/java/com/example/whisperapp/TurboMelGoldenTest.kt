package com.example.whisperapp

import com.example.whisperapp.audio.WhisperMelConfig
import com.example.whisperapp.audio.WhisperTurboFeatureExtractor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** P1 golden test: Turbo mel only; no QNN, decoder, or Tiny code path is involved. */
class TurboMelGoldenTest {
    @Test
    fun sampleWavProducesStableTurboMelGolden() {
        val wav = sequenceOf(File("sample.wav"), File("../sample.wav"))
            .firstOrNull { it.isFile }
            ?: error("missing sample.wav from ${File(".").absolutePath}")
        val pcm = readPcm16MonoWav(wav)

        val extractor = WhisperTurboFeatureExtractor(WhisperMelConfig.LargeV3Turbo)
        val first = extractor.extract(pcm, 16_000)
        val second = extractor.extract(pcm, 16_000)

        assertEquals(listOf(1L, 128L, 3000L), first.shape.toList())
        assertEquals(128 * 3000, first.data.size)
        assertTrue(first.data.all { it.isFinite() })

        var maxAbs = 0f
        var sumAbs = 0.0
        for (i in first.data.indices) {
            val d = abs(first.data[i] - second.data[i])
            maxAbs = maxOf(maxAbs, d)
            sumAbs += d
        }
        assertTrue("non-deterministic maxAbs=$maxAbs", maxAbs == 0f)

        val outDir = File("build/turbo-mel-golden").apply { mkdirs() }
        writeFloat32(File(outDir, "mel_turbo.bin"), first.data)
        File(outDir, "mel_turbo.txt").writeText(
            buildString {
                appendLine("shape=[1,128,3000]")
                appendLine("dtype=float32")
                appendLine("elements=${first.data.size}")
                appendLine("min=${first.data.minOrNull()}")
                appendLine("max=${first.data.maxOrNull()}")
                appendLine("mean=${first.data.average()}")
                appendLine("first16=${first.data.take(16).joinToString(",")}")
                appendLine("last16=${first.data.takeLast(16).joinToString(",")}")
                appendLine("repeat_max_abs_diff=$maxAbs")
                appendLine("repeat_sum_abs_diff=$sumAbs")
            }
        )

        val half = extractor.extractHalf(pcm, 16_000)
        assertEquals(128 * 3000, half.size)
        writeInt16(File(outDir, "mel_turbo_fp16.bin"), half)
    }

    private fun writeFloat32(file: File, values: FloatArray) {
        val bytes = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(bytes::putFloat)
        file.writeBytes(bytes.array())
    }

    private fun writeInt16(file: File, values: ShortArray) {
        val bytes = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(bytes::putShort)
        file.writeBytes(bytes.array())
    }

    private fun readPcm16MonoWav(file: File): ShortArray {
        val b = file.readBytes()
        require(b.size >= 44 && String(b, 0, 4, Charsets.US_ASCII) == "RIFF")
        require(String(b, 8, 4, Charsets.US_ASCII) == "WAVE")
        var pos = 12
        var channels = 0
        var sampleRate = 0
        var bits = 0
        var dataOffset = -1
        var dataSize = -1
        while (pos + 8 <= b.size) {
            val id = String(b, pos, 4, Charsets.US_ASCII)
            val size = ByteBuffer.wrap(b, pos + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
            val next = pos + 8 + size + (size and 1)
            if (id == "fmt ") {
                val audioFormat = ByteBuffer.wrap(b, pos + 8, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()
                channels = ByteBuffer.wrap(b, pos + 10, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()
                sampleRate = ByteBuffer.wrap(b, pos + 12, 4).order(ByteOrder.LITTLE_ENDIAN).int
                bits = ByteBuffer.wrap(b, pos + 22, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()
                require(audioFormat == 1) { "expected PCM WAV, format=$audioFormat" }
            } else if (id == "data") {
                dataOffset = pos + 8
                dataSize = size
                break
            }
            pos = next
        }
        require(channels == 1 && sampleRate == 16_000 && bits == 16) {
            "expected mono PCM16 16kHz, got channels=$channels rate=$sampleRate bits=$bits"
        }
        require(dataOffset >= 0 && dataSize >= 0 && dataOffset + dataSize <= b.size)
        val count = dataSize / 2
        val pcm = ShortArray(count)
        val bb = ByteBuffer.wrap(b, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until count) pcm[i] = bb.short
        return pcm
    }
}