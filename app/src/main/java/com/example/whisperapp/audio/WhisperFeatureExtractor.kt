package com.example.whisperapp.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

data class WhisperFeatures(val data: FloatArray, val shape: LongArray = longArrayOf(1, 80, 3000))

/** CPU Whisper preprocessing matching the model's Hugging Face Whisper-Tiny contract. */
class WhisperFeatureExtractor {
    companion object {
        const val SAMPLE_RATE = 16_000
        const val N_FFT = 400
        const val HOP_LENGTH = 160
        const val N_MELS = 80
        const val N_FRAMES = 3000
        const val CHUNK_SAMPLES = SAMPLE_RATE * 30
    }

    private val window = FloatArray(N_FFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / N_FFT)).toFloat() }
    private val melFilters = buildMelFilters()

    // n_fft=400 is fixed, so Bluestein's chirp and convolution-kernel FFT are also fixed.
    // Keeping these outside the frame loop removes 3000 repeated kernel FFTs and all hot-path
    // scratch allocation while preserving the same 400-point DFT definition.
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

    fun extract(pcm16: ShortArray, sampleRate: Int): WhisperFeatures {
        require(sampleRate == SAMPLE_RATE) { "Whisper expects 16000 Hz, got $sampleRate" }
        val waveform = FloatArray(CHUNK_SAMPLES)
        val copy = min(pcm16.size, waveform.size)
        for (i in 0 until copy) waveform[i] = pcm16[i] / 32768f

        // Whisper/Hugging Face center=True STFT uses reflect padding and yields 3001 frames;
        // the model drops the last frame.
        val padded = FloatArray(CHUNK_SAMPLES + N_FFT)
        val pad = N_FFT / 2
        for (i in padded.indices) {
            padded[i] = waveform[reflectIndex(i - pad, waveform.size)]
        }
        val real = DoubleArray(N_FFT)
        val imag = DoubleArray(N_FFT)
        val mel = FloatArray(N_MELS * N_FRAMES)

        for (frame in 0 until N_FRAMES) {
            val start = frame * HOP_LENGTH
            for (i in 0 until N_FFT) { real[i] = padded[start + i] * window[i].toDouble(); imag[i] = 0.0 }
            fft(real, imag)
            for (m in 0 until N_MELS) {
                var energy = 0.0
                for (k in melFilters[m].indices) {
                    val w = melFilters[m][k]
                    if (w != 0f) energy += (real[k] * real[k] + imag[k] * imag[k]) * w
                }
                mel[m * N_FRAMES + frame] = max(energy.toFloat(), 1e-10f)
            }
        }

        var maxLog = Float.NEGATIVE_INFINITY
        for (i in mel.indices) { mel[i] = log10(mel[i]); maxLog = max(maxLog, mel[i]) }
        val floor = maxLog - 8f
        for (i in mel.indices) mel[i] = (max(mel[i], floor) + 4f) / 4f
        require(mel.all { it.isFinite() }) { "Whisper features contain NaN/Inf" }
        require(mel.any { it != 0f }) { "Whisper features are all zero" }
        return WhisperFeatures(mel)
    }

    private fun buildMelFilters(): Array<FloatArray> {
        val filters = Array(N_MELS) { FloatArray(N_FFT / 2 + 1) }
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(8000.0)
        val filterHz = DoubleArray(N_MELS + 2) { i ->
            melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1))
        }
        val fftHz = DoubleArray(N_FFT / 2 + 1) { k -> k.toDouble() * SAMPLE_RATE / N_FFT }
        for (m in 0 until N_MELS) {
            val left = filterHz[m]
            val center = filterHz[m + 1]
            val right = filterHz[m + 2]
            for (k in fftHz.indices) {
                val down = if (center > left) (fftHz[k] - left) / (center - left) else 0.0
                val up = if (right > center) (right - fftHz[k]) / (right - center) else 0.0
                filters[m][k] = max(0.0, min(down, up)).toFloat()
            }
            val enorm = 2.0 / max(right - left, 1e-12)
            for (k in filters[m].indices) filters[m][k] = (filters[m][k] * enorm).toFloat()
        }
        return filters
    }

    // Slaney mel scale, matching Hugging Face's mel_filter_bank(..., norm="slaney", mel_scale="slaney").
    private fun hzToMel(hz: Double): Double {
        val minLogHertz = 1000.0
        val minLogMel = 15.0
        val logStep = 27.0 / ln(6.4)
        return if (hz < minLogHertz) 3.0 * hz / 200.0
        else minLogMel + ln(hz / minLogHertz) * logStep
    }

    private fun melToHz(mel: Double): Double {
        val minLogHertz = 1000.0
        val minLogMel = 15.0
        val logStep = ln(6.4) / 27.0
        return if (mel < minLogMel) 200.0 * mel / 3.0
        else minLogHertz * kotlin.math.exp(logStep * (mel - minLogMel))
    }

    private fun reflectIndex(index: Int, size: Int): Int {
        require(size > 1) { "Reflect padding requires at least two samples" }
        var i = index
        while (i < 0 || i >= size) {
            i = if (i < 0) -i else 2 * size - 2 - i
        }
        return i
    }

    private fun fft(real: DoubleArray, imag: DoubleArray) {
        require(real.size == imag.size)
        if (real.size and (real.size - 1) == 0) {
            radix2Fft(real, imag)
        } else {
            bluesteinFft(real, imag)
        }
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
            val wr0 = cos(angle)
            val wi0 = sin(angle)
            var base = 0
            while (base < n) {
                var wr = 1.0
                var wi = 0.0
                for (k in 0 until len / 2) {
                    val a = base + k
                    val b = a + len / 2
                    val vr = real[b] * wr - imag[b] * wi
                    val vi = real[b] * wi + imag[b] * wr
                    val ur = real[a]
                    val ui = imag[a]
                    real[a] = ur + vr
                    imag[a] = ui + vi
                    real[b] = ur - vr
                    imag[b] = ui - vi
                    val next = wr * wr0 - wi * wi0
                    wi = wr * wi0 + wi * wr0
                    wr = next
                }
                base += len
            }
            len = len shl 1
        }
    }

    // Whisper uses n_fft=400, which is not radix-2. Bluestein reduces the exact 400-point
    // DFT to a power-of-two convolution without changing the FFT bin frequencies.
    private fun bluesteinFft(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        require(n == N_FFT) { "Optimized Bluestein path only supports n_fft=$N_FFT" }
        java.util.Arrays.fill(fftScratchReal, 0.0)
        java.util.Arrays.fill(fftScratchImag, 0.0)
        for (k in 0 until n) {
            val c = bluesteinChirpCos[k]
            val s = bluesteinChirpSin[k]
            fftScratchReal[k] = real[k] * c + imag[k] * s
            fftScratchImag[k] = imag[k] * c - real[k] * s
        }
        radix2Fft(fftScratchReal, fftScratchImag)
        for (k in fftScratchReal.indices) {
            val r = fftScratchReal[k] * bluesteinKernelReal[k] - fftScratchImag[k] * bluesteinKernelImag[k]
            val i = fftScratchReal[k] * bluesteinKernelImag[k] + fftScratchImag[k] * bluesteinKernelReal[k]
            fftScratchReal[k] = r
            fftScratchImag[k] = i
        }
        inverseRadix2Fft(fftScratchReal, fftScratchImag)
        for (k in 0 until n) {
            val c = bluesteinChirpCos[k]
            val s = bluesteinChirpSin[k]
            real[k] = fftScratchReal[k] * c + fftScratchImag[k] * s
            imag[k] = fftScratchImag[k] * c - fftScratchReal[k] * s
        }
    }

    private fun inverseRadix2Fft(real: DoubleArray, imag: DoubleArray) {
        for (i in real.indices) imag[i] = -imag[i]
        radix2Fft(real, imag)
        val scale = 1.0 / real.size
        for (i in real.indices) {
            real[i] *= scale
            imag[i] = -imag[i] * scale
        }
    }
}
