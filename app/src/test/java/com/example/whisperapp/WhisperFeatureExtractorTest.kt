package com.example.whisperapp

import com.example.whisperapp.audio.WhisperFeatureExtractor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/**
 * The optimized extractor switched the pipeline from double-precision dense mel
 * projection to single-precision sparse scatter. This test pins the two
 * implementations together on deterministic synthetic audio.
 */
class WhisperFeatureExtractorTest {

    /** Copy of the previous double-precision dense implementation, used as the reference. */
    private class ReferenceExtractor {
        private val window = FloatArray(N_FFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / N_FFT)).toFloat() }
        private val melFilters = buildMelFilters()
        private val bluesteinSize = 1024
        private val bluesteinChirpCos = DoubleArray(N_FFT) { k -> cos(PI * k.toDouble() * k / N_FFT) }
        private val bluesteinChirpSin = DoubleArray(N_FFT) { k -> sin(PI * k.toDouble() * k / N_FFT) }
        private val bluesteinKernelReal = DoubleArray(bluesteinSize)
        private val bluesteinKernelImag = DoubleArray(bluesteinSize)
        private val fftScratchReal = DoubleArray(bluesteinSize)
        private val fftScratchImag = DoubleArray(bluesteinSize)

        init {
            for (k in 0 until N_FFT) {
                val c = bluesteinChirpCos[k]
                val s = bluesteinChirpSin[k]
                bluesteinKernelReal[k] = c
                bluesteinKernelImag[k] = s
                if (k != 0) {
                    bluesteinKernelReal[bluesteinSize - k] = c
                    bluesteinKernelImag[bluesteinSize - k] = s
                }
            }
            radix2Fft(bluesteinKernelReal, bluesteinKernelImag)
        }

        fun extract(pcm16: ShortArray): FloatArray {
            val waveform = FloatArray(CHUNK_SAMPLES)
            val copy = minOf(pcm16.size, waveform.size)
            for (i in 0 until copy) waveform[i] = pcm16[i] / 32768f
            val padded = FloatArray(CHUNK_SAMPLES + N_FFT)
            val pad = N_FFT / 2
            for (i in padded.indices) padded[i] = waveform[reflectIndex(i - pad, waveform.size)]
            val real = DoubleArray(N_FFT)
            val imag = DoubleArray(N_FFT)
            val mel = FloatArray(N_MELS * N_FRAMES)
            for (frame in 0 until N_FRAMES) {
                val start = frame * HOP_LENGTH
                for (i in 0 until N_FFT) { real[i] = padded[start + i] * window[i].toDouble(); imag[i] = 0.0 }
                bluesteinFft(real, imag)
                for (m in 0 until N_MELS) {
                    var energy = 0.0
                    for (k in melFilters[m].indices) {
                        val w = melFilters[m][k]
                        if (w != 0f) energy += (real[k] * real[k] + imag[k] * imag[k]) * w
                    }
                    mel[m * N_FRAMES + frame] = maxOf(energy.toFloat(), 1e-10f)
                }
            }
            var maxLog = Float.NEGATIVE_INFINITY
            for (i in mel.indices) { mel[i] = kotlin.math.log10(mel[i]); maxLog = maxOf(maxLog, mel[i]) }
            val floor = maxLog - 8f
            for (i in mel.indices) mel[i] = (maxOf(mel[i], floor) + 4f) / 4f
            return mel
        }

        private fun buildMelFilters(): Array<FloatArray> {
            val filters = Array(N_MELS) { FloatArray(N_FFT / 2 + 1) }
            val melMin = hzToMel(0.0)
            val melMax = hzToMel(8000.0)
            val filterHz = DoubleArray(N_MELS + 2) { i -> melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1)) }
            val fftHz = DoubleArray(N_FFT / 2 + 1) { k -> k.toDouble() * SAMPLE_RATE / N_FFT }
            for (m in 0 until N_MELS) {
                val left = filterHz[m]; val center = filterHz[m + 1]; val right = filterHz[m + 2]
                for (k in fftHz.indices) {
                    val down = if (center > left) (fftHz[k] - left) / (center - left) else 0.0
                    val up = if (right > center) (right - fftHz[k]) / (right - center) else 0.0
                    filters[m][k] = maxOf(0.0, minOf(down, up)).toFloat()
                }
                val enorm = 2.0 / maxOf(right - left, 1e-12)
                for (k in filters[m].indices) filters[m][k] = (filters[m][k] * enorm).toFloat()
            }
            return filters
        }

        private fun hzToMel(hz: Double): Double {
            return if (hz < 1000.0) 3.0 * hz / 200.0 else 15.0 + kotlin.math.ln(hz / 1000.0) * (27.0 / kotlin.math.ln(6.4))
        }

        private fun melToHz(mel: Double): Double {
            return if (mel < 15.0) 200.0 * mel / 3.0 else 1000.0 * kotlin.math.exp(kotlin.math.ln(6.4) / 27.0 * (mel - 15.0))
        }

        private fun reflectIndex(index: Int, size: Int): Int {
            var i = index
            while (i < 0 || i >= size) i = if (i < 0) -i else 2 * size - 2 - i
            return i
        }

        private fun radix2Fft(real: DoubleArray, imag: DoubleArray) {
            val n = real.size
            var j = 0
            for (i in 1 until n) {
                var bit = n shr 1
                while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
                j = j xor bit
                if (i < j) {
                    val tr = real[i]; real[i] = real[j]; real[j] = tr
                    val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
                }
            }
            var len = 2
            while (len <= n) {
                val angle = -2.0 * PI / len
                val wr0 = cos(angle); val wi0 = sin(angle)
                var base = 0
                while (base < n) {
                    var wr = 1.0; var wi = 0.0
                    for (k in 0 until len / 2) {
                        val a = base + k; val b = a + len / 2
                        val vr = real[b] * wr - imag[b] * wi
                        val vi = real[b] * wi + imag[b] * wr
                        val ur = real[a]; val ui = imag[a]
                        real[a] = ur + vr; imag[a] = ui + vi
                        real[b] = ur - vr; imag[b] = ui - vi
                        val next = wr * wr0 - wi * wi0
                        wi = wr * wi0 + wi * wr0; wr = next
                    }
                    base += len
                }
                len = len shl 1
            }
        }

        private fun bluesteinFft(real: DoubleArray, imag: DoubleArray) {
            val n = real.size
            java.util.Arrays.fill(fftScratchReal, 0.0)
            java.util.Arrays.fill(fftScratchImag, 0.0)
            for (k in 0 until n) {
                val c = bluesteinChirpCos[k]; val s = bluesteinChirpSin[k]
                fftScratchReal[k] = real[k] * c + imag[k] * s
                fftScratchImag[k] = imag[k] * c - real[k] * s
            }
            radix2Fft(fftScratchReal, fftScratchImag)
            for (k in fftScratchReal.indices) {
                val r = fftScratchReal[k] * bluesteinKernelReal[k] - fftScratchImag[k] * bluesteinKernelImag[k]
                val i = fftScratchReal[k] * bluesteinKernelImag[k] + fftScratchImag[k] * bluesteinKernelReal[k]
                fftScratchReal[k] = r; fftScratchImag[k] = i
            }
            for (i in fftScratchReal.indices) fftScratchImag[i] = -fftScratchImag[i]
            radix2Fft(fftScratchReal, fftScratchImag)
            val scale = 1.0 / fftScratchReal.size
            for (k in 0 until n) {
                val c = bluesteinChirpCos[k]; val s = bluesteinChirpSin[k]
                val sr = fftScratchReal[k] * scale
                val si = -fftScratchImag[k] * scale
                real[k] = sr * c + si * s
                imag[k] = si * c - sr * s
            }
        }

        companion object {
            private const val SAMPLE_RATE = 16_000
            private const val N_FFT = 400
            private const val HOP_LENGTH = 160
            private const val N_MELS = 80
            private const val N_FRAMES = 3000
            private const val CHUNK_SAMPLES = SAMPLE_RATE * 30
        }
    }

    private fun syntheticPcm(seconds: Int, seed: Long): ShortArray {
        val n = 16000 * seconds
        val rng = java.util.Random(seed)
        val phase = rng.nextDouble() * PI
        return ShortArray(n) { i ->
            val t = i / 16000.0
            val s = 0.5 * sin(2 * PI * 220.0 * t + phase) +
                0.3 * sin(2 * PI * 440.0 * t) +
                0.2 * sin(2 * PI * 1320.0 * t) +
                0.05 * rng.nextGaussian()
            (s.coerceIn(-1.0, 1.0) * 30000).toInt().toShort()
        }
    }

    @Test
    fun optimizedMatchesReferenceWithinTolerance() {
        val pcm = syntheticPcm(5, seed = 42)
        val ref = ReferenceExtractor().extract(pcm)
        val opt = WhisperFeatureExtractor().extract(pcm, 16000).data
        assertEquals(ref.size, opt.size)
        var maxAbs = 0.0
        var sumAbs = 0.0
        var count = 0
        for (i in ref.indices) {
            val d = abs(ref[i] - opt[i]).toDouble()
            if (d > maxAbs) maxAbs = d
            sumAbs += d; count++
        }
        // Float-vs-double plus twiddle-table vs recurrence differences land around 1e-3;
        // the transcription-relevant dynamic range is ~2.0, so 0.02 is a generous bound.
        assertTrue("maxAbs=$maxAbs mean=${sumAbs / count}", maxAbs < 0.02)
    }

    @Test
    fun optimizedIsMuchFasterThanReference() {
        val pcm = syntheticPcm(5, seed = 7)
        val refMs = measure { ReferenceExtractor().extract(pcm) }
        val optMs = measure { WhisperFeatureExtractor().extract(pcm, 16000) }
        println("reference=${refMs}ms optimized=${optMs}ms")
        assertTrue("optimized=$optMs ref=$refMs", optMs < refMs * 0.85)
    }

    private fun measure(block: () -> Unit): Double {
        block() // warmup
        val t0 = System.nanoTime()
        block()
        return (System.nanoTime() - t0) / 1_000_000.0
    }
}
