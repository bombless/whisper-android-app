package com.example.whisperapp

import com.example.whisperapp.audio.WhisperFeatureExtractor
import org.junit.Test
import java.io.File

/**
 * TEMPORARY diagnostic: dumps the Java mel pipeline output for sample.wav so it can be
 * compared against the native mel_host binary output. Delete after verification.
 */
class MelDumpTest {
    private fun readWavSamples(path: String): ShortArray {
        val bytes = File(path).readBytes()
        check(bytes.size >= 12 && bytes.toString(Charsets.US_ASCII).substring(0, 4) == "RIFF") { "not wav" }
        var pos = 12
        while (pos + 8 <= bytes.size) {
            val id = bytes.copyOfRange(pos, pos + 4).toString(Charsets.US_ASCII)
            val sz = (bytes[pos + 4].toInt() and 0xff) or
                ((bytes[pos + 5].toInt() and 0xff) shl 8) or
                ((bytes[pos + 6].toInt() and 0xff) shl 16) or
                ((bytes[pos + 7].toInt() and 0xff) shl 24)
            if (id == "data") {
                val n = sz / 2
                val pcm = ShortArray(n)
                for (i in 0 until n) {
                    pcm[i] = ((bytes[pos + 8 + 2 * i].toInt() and 0xff) or
                        ((bytes[pos + 9 + 2 * i].toInt() and 0xff) shl 8)).toShort()
                }
                return pcm
            }
            pos += 8 + sz + (sz and 1)
        }
        error("no data chunk")
    }

    @Test
    fun dumpMel() {
        val pcm = readWavSamples("../sample.wav")
        val ex = WhisperFeatureExtractor()
        val floats = ex.extract(pcm, 16000).data
        val half = ex.extractHalf(pcm, 16000)

        File("../mel_java.bin").writeBytes(FloatArray(floats.size) { floats[it] }.let { fa ->
            val bb = java.nio.ByteBuffer.allocate(fa.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            fa.forEach { bb.putFloat(it) }
            bb.array()
        })
        File("../melh_java.bin").writeBytes(ShortArray(half.size) { half[it] }.let { sa ->
            val bb = java.nio.ByteBuffer.allocate(sa.size * 2).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            sa.forEach { bb.putShort(it) }
            bb.array()
        })
    }
}
