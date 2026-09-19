package com.example.whisperapp.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

data class WhisperFeatures(val data: FloatArray, val shape: LongArray = longArrayOf(1, 80, 3000))

/**
 * CPU Whisper preprocessing matching the model's Hugging Face Whisper-Tiny contract.
 *
 * Performance notes (vs the previous double-precision dense version, ~610 ms/chunk):
 *  - All DSP runs in single precision with precomputed per-stage FFT twiddle tables
 *    (no recurrence drift, no per-butterfly twiddle advance).
 *  - Mel projection uses a sparse bin→filter scatter: each FFT bin feeds at most two
 *    triangular mel filters, so the per-frame cost drops from 80×201 dense multiply-adds
 *    to ~201 power computations plus ~400 scatter multiply-adds. Per-mel accumulation
 *    order stays ascending in k, matching the dense summation order.
 */
class WhisperFeatureExtractor {
    companion object {
        const val SAMPLE_RATE = 16_000
        const val N_FFT = 400
        const val HOP_LENGTH = 160
        const val N_MELS = 80
        const val N_FRAMES = 3000
        const val CHUNK_SAMPLES = SAMPLE_RATE * 30

        /** Max float-mel deviation vs the Java reference before we permanently fall back. */
        private const val NATIVE_TOLERANCE = 0.05f

        @Volatile private var nativeLibLoaded = false
        @Volatile private var nativeValidationDone = false
        @Volatile private var nativeUsable = false

        init {
            nativeLibLoaded = try {
                System.loadLibrary("mel_native")
                true
            } catch (t: Throwable) {
                // Deliberately silent on JVM unit tests (android.util.Log is not mocked there).
                System.err.println("mel_native unavailable, Java mel path will be used: $t")
                false
            }
        }

        private external fun nativeExtractFloat(pcm: ShortArray, sampleRate: Int): FloatArray
        private external fun nativeExtractHalf(pcm: ShortArray, sampleRate: Int): ShortArray
    }

    private val fftSize = 1024 // Bluestein convolution size for n_fft=400
    private val window = FloatArray(N_FFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / N_FFT)).toFloat() }

    // Sparse mel filterbank in CSR form: for FFT bin k, melBinIds[k] / melBinWeights[k]
    // list the (mel, weight) pairs with nonzero weight, sorted by mel index.
    private val melBinIds: Array<IntArray>
    private val melBinWeights: Array<FloatArray>

    // Fixed Bluestein chirps and pre-transformed convolution kernel (float).
    private val bluesteinChirpRe = FloatArray(N_FFT)
    private val bluesteinChirpIm = FloatArray(N_FFT)
    private val kernelRe = FloatArray(fftSize)
    private val kernelIm = FloatArray(fftSize)
    private val scratchRe = FloatArray(fftSize)
    private val scratchIm = FloatArray(fftSize)

    // Precomputed radix-2 twiddles for all stages of the 1024-point FFT.
    // Stage with span `len` starts at cumulative offset len/2 - 1.
    private val twiddleRe: FloatArray
    private val twiddleIm: FloatArray

    init {
        // Twiddle tables.
        val totalTwiddles = fftSize - 1
        twiddleRe = FloatArray(totalTwiddles)
        twiddleIm = FloatArray(totalTwiddles)
        var offset = 0
        var len = 2
        while (len <= fftSize) {
            for (k in 0 until len / 2) {
                val angle = -2.0 * PI * k / len
                twiddleRe[offset + k] = cos(angle).toFloat()
                twiddleIm[offset + k] = sin(angle).toFloat()
            }
            offset += len / 2
            len = len shl 1
        }

        // Bluestein chirp + kernel.
        val chirpRe = DoubleArray(N_FFT) { k -> cos(PI * k.toDouble() * k / N_FFT) }
        val chirpIm = DoubleArray(N_FFT) { k -> sin(PI * k.toDouble() * k / N_FFT) }
        for (k in 0 until N_FFT) {
            bluesteinChirpRe[k] = chirpRe[k].toFloat()
            bluesteinChirpIm[k] = chirpIm[k].toFloat()
            kernelRe[k] = chirpRe[k].toFloat()
            kernelIm[k] = chirpIm[k].toFloat()
            if (k != 0) {
                kernelRe[fftSize - k] = chirpRe[k].toFloat()
                kernelIm[fftSize - k] = chirpIm[k].toFloat()
            }
        }
        radix2Fft(kernelRe, kernelIm)

        // Dense mel filters (same Slaney construction as before), then sparsify per bin.
        val dense = buildMelFilters()
        melBinIds = Array(N_FFT / 2 + 1) { k ->
            (0 until N_MELS).filter { dense[it][k] != 0f }.toIntArray()
        }
        melBinWeights = Array(N_FFT / 2 + 1) { k ->
            FloatArray(melBinIds[k].size) { i -> dense[melBinIds[k][i]][k] }
        }
    }

    fun extract(pcm16: ShortArray, sampleRate: Int): WhisperFeatures {
        val mel = extractNormalizedMel(pcm16, sampleRate)
        require(mel.all { it.isFinite() }) { "Whisper features contain NaN/Inf" }
        require(mel.any { it != 0f }) { "Whisper features are all zero" }
        return WhisperFeatures(mel)
    }

    /** Same pipeline as [extract] but returns FP16 samples ready for the encoder input. */
    fun extractHalf(pcm16: ShortArray, sampleRate: Int): ShortArray {
        if (nativeReady(pcm16, sampleRate)) {
            val out = try {
                nativeExtractHalf(pcm16, sampleRate)
            } catch (t: Throwable) {
                android.util.Log.e("WhisperFeatureExtractor", "nativeExtractHalf failed, falling back", t)
                null
            }
            if (out != null && out.size == N_MELS * N_FRAMES && out.any { it.toInt() != 0 }) return out
            nativeUsable = false
            android.util.Log.e("WhisperFeatureExtractor", "native half output invalid, falling back to Java mel")
        }
        val mel = javaNormalizedMel(pcm16, sampleRate)
        val result = ShortArray(mel.size)
        for (i in mel.indices) result[i] = floatToHalf(mel[i])
        return result
    }

    /**
     * Returns log-mel normalized per whisper's (log10, max-8 floor, +4 /4) contract,
     * preferring the JNI native implementation once it has been validated against the
     * Java reference on real audio.
     */
    private fun extractNormalizedMel(pcm16: ShortArray, sampleRate: Int): FloatArray {
        if (nativeReady(pcm16, sampleRate)) {
            val out = try {
                nativeExtractFloat(pcm16, sampleRate)
            } catch (t: Throwable) {
                android.util.Log.e("WhisperFeatureExtractor", "nativeExtractFloat failed, falling back", t)
                null
            }
            if (out != null && out.size == N_MELS * N_FRAMES && out.all { it.isFinite() }) return out
            nativeUsable = false
            android.util.Log.e("WhisperFeatureExtractor", "native float output invalid, falling back to Java mel")
        }
        return javaNormalizedMel(pcm16, sampleRate)
    }

    private fun javaNormalizedMel(pcm16: ShortArray, sampleRate: Int): FloatArray {
        val mel = computeMel(pcm16, sampleRate)
        for (i in mel.indices) mel[i] = log10(mel[i])
        val floor = melMaxLog(mel) - 8f
        for (i in mel.indices) mel[i] = (max(mel[i], floor) + 4f) / 4f
        return mel
    }

    private fun melMaxLog(mel: FloatArray): Float {
        var maxLog = Float.NEGATIVE_INFINITY
        for (v in mel) maxLog = max(maxLog, v)
        return maxLog
    }

    /** True iff the native library is loaded and (on first use) validated vs the Java reference. */
    @Synchronized
    private fun nativeReady(pcm16: ShortArray, sampleRate: Int): Boolean {
        if (!nativeLibLoaded) return false
        if (nativeValidationDone) return nativeUsable
        nativeValidationDone = true
        return try {
            val ref = javaNormalizedMel(pcm16, sampleRate)
            val nat = nativeExtractFloat(pcm16, sampleRate)
            var maxDiff = 0f
            if (nat.size == ref.size) {
                for (i in ref.indices) maxDiff = max(maxDiff, abs(ref[i] - nat[i]))
            }
            if (nat.size == ref.size && maxDiff <= NATIVE_TOLERANCE) {
                nativeUsable = true
                android.util.Log.i("WhisperFeatureExtractor", "native mel validated, maxAbsDiff=$maxDiff")
            } else {
                nativeUsable = false
                android.util.Log.e(
                    "WhisperFeatureExtractor",
                    "native mel mismatch (size ${nat.size} vs ${ref.size}, maxAbsDiff=$maxDiff), falling back to Java"
                )
            }
            nativeUsable
        } catch (t: Throwable) {
            nativeUsable = false
            android.util.Log.e("WhisperFeatureExtractor", "native mel validation crashed, falling back", t)
            false
        }
    }

    private fun computeMel(pcm16: ShortArray, sampleRate: Int): FloatArray {
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
        val real = FloatArray(N_FFT)
        val imag = FloatArray(N_FFT)
        val mel = FloatArray(N_MELS * N_FRAMES)

        for (frame in 0 until N_FRAMES) {
            val start = frame * HOP_LENGTH
            for (i in 0 until N_FFT) { real[i] = padded[start + i] * window[i]; imag[i] = 0f }
            bluesteinFft(real, imag)
            // Sparse scatter: power per bin, then accumulate into the ≤2 mel bands per bin.
            for (k in 0..N_FFT / 2) {
                val p = real[k] * real[k] + imag[k] * imag[k]
                val ids = melBinIds[k]
                val ws = melBinWeights[k]
                for (j in ids.indices) {
                    val idx = ids[j] * N_FRAMES + frame
                    mel[idx] += p * ws[j]
                }
            }
        }
        for (i in mel.indices) if (mel[i] < 1e-10f) mel[i] = 1e-10f
        require(mel.all { it.isFinite() }) { "Whisper features contain NaN/Inf" }
        require(mel.any { it != 0f }) { "Whisper features are all zero" }
        return mel
    }

    /** Float -> IEEE-754 binary16 with round-to-nearest-even, matching the runner's converter. */
    private fun floatToHalf(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exponent = ((bits ushr 23) and 0xff) - 127 + 15
        val mantissa = bits and 0x7fffff
        return when {
            exponent <= 0 -> if (exponent < -10) sign.toShort() else (sign or ((mantissa or 0x800000) shr (1 - exponent + 13))).toShort()
            exponent >= 31 -> (sign or 0x7c00 or if (mantissa == 0) 0 else 0x0200).toShort()
            else -> {
                var halfExponent = exponent
                var halfMantissa = (mantissa + 0x1000) shr 13
                if (halfMantissa == 0x400) { halfMantissa = 0; halfExponent++ }
                if (halfExponent >= 31) (sign or 0x7c00).toShort()
                else (sign or (halfExponent shl 10) or halfMantissa).toShort()
            }
        }
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

    private fun radix2Fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        // Bit-reversal permutation.
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        // Iterative FFT with table twiddles.
        var len = 2
        var offset = 0
        while (len <= n) {
            val half = len shr 1
            for (base in 0 until n step len) {
                for (k in 0 until half) {
                    val tw = offset + k
                    val b = base + k + half
                    val vr = re[b] * twiddleRe[tw] - im[b] * twiddleIm[tw]
                    val vi = re[b] * twiddleIm[tw] + im[b] * twiddleRe[tw]
                    val a = base + k
                    val ur = re[a]
                    val ui = im[a]
                    re[a] = ur + vr
                    im[a] = ui + vi
                    re[b] = ur - vr
                    im[b] = ui - vi
                }
            }
            offset += half
            len = len shl 1
        }
    }

    // Whisper uses n_fft=400, which is not radix-2. Bluestein reduces the exact 400-point
    // DFT to a 1024-point power-of-two convolution without changing the FFT bin frequencies.
    private fun bluesteinFft(real: FloatArray, imag: FloatArray) {
        java.util.Arrays.fill(scratchRe, 0f)
        java.util.Arrays.fill(scratchIm, 0f)
        for (k in 0 until N_FFT) {
            val c = bluesteinChirpRe[k]
            val s = bluesteinChirpIm[k]
            scratchRe[k] = real[k] * c + imag[k] * s
            scratchIm[k] = imag[k] * c - real[k] * s
        }
        radix2Fft(scratchRe, scratchIm)
        for (k in 0 until fftSize) {
            val r = scratchRe[k] * kernelRe[k] - scratchIm[k] * kernelIm[k]
            val i = scratchRe[k] * kernelIm[k] + scratchIm[k] * kernelRe[k]
            scratchRe[k] = r
            scratchIm[k] = i
        }
        // Inverse via conjugation: negate imag, forward FFT, negate imag, scale.
        for (k in 0 until fftSize) scratchIm[k] = -scratchIm[k]
        radix2Fft(scratchRe, scratchIm)
        val scale = 1.0f / fftSize
        for (k in 0 until N_FFT) {
            val c = bluesteinChirpRe[k]
            val s = bluesteinChirpIm[k]
            val sr = scratchRe[k] * scale
            val si = -scratchIm[k] * scale
            real[k] = sr * c + si * s
            imag[k] = si * c - sr * s
        }
    }
}
