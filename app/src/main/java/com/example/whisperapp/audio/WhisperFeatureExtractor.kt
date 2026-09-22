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
 *  - All DSP runs in single precision with a direct mixed-radix 400-point DFT
 *    (N=400 = 2^4 * 5^2: four DIT radix-2 stages down to 16x DFT25, each DFT25 via
 *    a 5x5 Cooley-Tukey step). This replaces the former Bluestein 400-point DFT via
 *    2x 1024-point convolutions and is ~4x faster in host benchmarks at
 *    spectrum maxAbs ~4e-6.
 *  - Mel projection uses a sparse bin→filter scatter: each FFT bin feeds at most two
 *    triangular mel filters, so the per-frame cost drops from 80×201 dense multiply-adds
 *    to ~201 power computations plus ~400 scatter multiply-adds. Per-mel accumulation
 *    order stays ascending in k, matching the dense summation order.
 */
class WhisperFeatureExtractor : WhisperMelFrontend {
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

    private val window = FloatArray(N_FFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / N_FFT)).toFloat() }

    // Sparse mel filterbank in CSR form: for FFT bin k, melBinIds[k] / melBinWeights[k]
    // list the (mel, weight) pairs with nonzero weight, sorted by mel index.
    private val melBinIds: Array<IntArray>
    private val melBinWeights: Array<FloatArray>

    // Direct mixed-radix FFT400 tables (N=400 = 2^4 * 5^2).
    private val w5Re = FloatArray(25)
    private val w5Im = FloatArray(25)
    private val w25Re = FloatArray(25)
    private val w25Im = FloatArray(25)
    private val c50Re = FloatArray(25)
    private val c50Im = FloatArray(25)
    private val c100Re = FloatArray(50)
    private val c100Im = FloatArray(50)
    private val c200Re = FloatArray(100)
    private val c200Im = FloatArray(100)
    private val c400Re = FloatArray(200)
    private val c400Im = FloatArray(200)
    private val tmpARe = FloatArray(N_FFT)
    private val tmpAIm = FloatArray(N_FFT)
    private val tmpBRe = FloatArray(N_FFT)
    private val tmpBIm = FloatArray(N_FFT)
    private val incrementalWindow = FloatArray(N_FFT)
    private val outRe = FloatArray(N_FFT)
    private val outIm = FloatArray(N_FFT)

    init {
        for (k1 in 0 until 5) {
            for (n1 in 0 until 5) {
                val a = -2.0 * PI * k1 * n1 / 5.0
                w5Re[k1 * 5 + n1] = cos(a).toFloat()
                w5Im[k1 * 5 + n1] = sin(a).toFloat()
            }
        }
        for (k1 in 0 until 5) {
            for (n2 in 0 until 5) {
                val a = -2.0 * PI * k1 * n2 / 25.0
                w25Re[k1 * 5 + n2] = cos(a).toFloat()
                w25Im[k1 * 5 + n2] = sin(a).toFloat()
            }
        }
        fun fillCombine(re: FloatArray, im: FloatArray) {
            val half = re.size
            for (k in 0 until half) {
                val a = -2.0 * PI * k / (2 * half)
                re[k] = cos(a).toFloat()
                im[k] = sin(a).toFloat()
            }
        }
        fillCombine(c50Re, c50Im)
        fillCombine(c100Re, c100Im)
        fillCombine(c200Re, c200Im)
        fillCombine(c400Re, c400Im)

        // Dense mel filters (same Slaney construction as before), then sparsify per bin.
        val dense = buildMelFilters()
        melBinIds = Array(N_FFT / 2 + 1) { k ->
            (0 until N_MELS).filter { dense[it][k] != 0f }.toIntArray()
        }
        melBinWeights = Array(N_FFT / 2 + 1) { k ->
            FloatArray(melBinIds[k].size) { i -> dense[melBinIds[k][i]][k] }
        }
    }

    override fun extract(pcm16: ShortArray, sampleRate: Int): WhisperFeatures {
        val mel = extractNormalizedMel(pcm16, sampleRate)
        require(mel.all { it.isFinite() }) { "Whisper features contain NaN/Inf" }
        require(mel.any { it != 0f }) { "Whisper features are all zero" }
        return WhisperFeatures(mel)
    }

    /** Same pipeline as [extract] but returns FP16 samples ready for the encoder input. */
    override fun extractHalf(pcm16: ShortArray, sampleRate: Int): ShortArray {
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
     * Incremental feature cache for live transcription. Only FFT/mel frames
     * touched by newly appended PCM are recomputed; normalization is a cheap
     * linear pass over the cached 80x3000 raw-mel matrix.
     */
    inner class IncrementalCache : WhisperIncrementalMelCache {
        private val pcm = ShortArray(CHUNK_SAMPLES)
        private val rawMel = FloatArray(N_MELS * N_FRAMES) { 1e-10f }
        private var sampleCount = 0

        override fun append(samples: ShortArray, sampleRate: Int): ShortArray {
            require(sampleRate == SAMPLE_RATE) { "Whisper expects 16000 Hz, got $sampleRate" }
            val oldCount = sampleCount
            val copy = min(samples.size, CHUNK_SAMPLES - oldCount)
            if (copy > 0) samples.copyInto(pcm, oldCount, 0, copy)
            sampleCount = min(CHUNK_SAMPLES, oldCount + samples.size)
            if (sampleCount > oldCount) recomputeAffectedFrames(oldCount, sampleCount)

            var minFrame = N_FRAMES
            var maxFrame = -1
            if (sampleCount > oldCount) {
                val centerPad = N_FFT / 2
                minFrame = max(0, oldCount - (N_FFT - centerPad - 1) + HOP_LENGTH - 1) / HOP_LENGTH
                maxFrame = min(N_FRAMES - 1, (sampleCount - 1 + centerPad) / HOP_LENGTH)
            }

            val mel = rawMel.copyOf()
            var maxLog = Float.NEGATIVE_INFINITY
            for (i in mel.indices) {
                val v = log10(max(mel[i], 1e-10f))
                mel[i] = v
                if (v > maxLog) maxLog = v
            }
            val floor = maxLog - 8f
            val out = ShortArray(mel.size)
            for (i in mel.indices) {
                out[i] = floatToHalf((max(mel[i], floor) + 4f) / 4f)
            }
            android.util.Log.i("WHISPER_DEBUG", "CACHE old=$oldCount new=${samples.size} copied=$copy pcmTotal=$sampleCount frameRange=$minFrame..$maxFrame totalFrames=${if (maxFrame >= 0) maxFrame + 1 else 0} melChecksum=${checksum(out)}")
            return out
        }

        override fun reset() {
            pcm.fill(0)
            rawMel.fill(1e-10f)
            sampleCount = 0
        }

        private fun checksum(values: ShortArray): Long {
            var h = -375076303657289L
            for (v in values) h = (h xor (v.toInt() and 0xffff).toLong()) * 0x100000001b3L
            return h
        }

        private fun recomputeAffectedFrames(oldSamples: Int, newSamples: Int) {
            val centerPad = N_FFT / 2
            // A frame is affected whenever its 400-sample analysis window can overlap
            // newly appended PCM.  Include the frame immediately before the nominal
            // boundary as well; integer rounding in the old expression could leave the
            // boundary frame stale by one frame for some append sizes.
            val firstFrame = max(0, (oldSamples - N_FFT + HOP_LENGTH - 1) / HOP_LENGTH)
            val lastFrame = min(N_FRAMES - 1, (newSamples - 1 + centerPad) / HOP_LENGTH)
            if (firstFrame > lastFrame) return
            for (frame in firstFrame..lastFrame) {
                for (m in 0 until N_MELS) rawMel[m * N_FRAMES + frame] = 0f
                val start = frame * HOP_LENGTH
                for (i in 0 until N_FFT) {
                    val sourceIndex = reflectIndex(start + i - N_FFT / 2, CHUNK_SAMPLES)
                    incrementalWindow[i] = pcm[sourceIndex] / 32768f * window[i]
                }
                fft400(incrementalWindow, outRe, outIm, tmpARe, tmpAIm, tmpBRe, tmpBIm)
                for (k in 0..N_FFT / 2) {
                    val p = outRe[k] * outRe[k] + outIm[k] * outIm[k]
                    val ids = melBinIds[k]
                    val ws = melBinWeights[k]
                    for (j in ids.indices) rawMel[ids[j] * N_FRAMES + frame] += p * ws[j]
                }
            }
        }
    }

    fun newIncrementalCache(): IncrementalCache = IncrementalCache()

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
        val xw = FloatArray(N_FFT)
        val mel = FloatArray(N_MELS * N_FRAMES)

        for (frame in 0 until N_FRAMES) {
            val start = frame * HOP_LENGTH
            for (i in 0 until N_FFT) { xw[i] = padded[start + i] * window[i] }
            fft400(xw, outRe, outIm, tmpARe, tmpAIm, tmpBRe, tmpBIm)
            // Sparse scatter: power per bin, then accumulate into the ≤2 mel bands per bin.
            for (k in 0..N_FFT / 2) {
                val p = outRe[k] * outRe[k] + outIm[k] * outIm[k]
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

    // Whisper uses n_fft=400 = 2^4 * 5^2. Four DIT radix-2 stages reduce the exact
    // 400-point DFT to 16x 25-point DFTs; each DFT25 runs as a 5x5 Cooley-Tukey step.
    private fun dft5(inRe: FloatArray, inIm: FloatArray, inOff: Int, stride: Int,
                     outRe: FloatArray, outOff: Int, outIm: FloatArray) {
        for (k in 0 until 5) {
            var sr = 0f
            var si = 0f
            for (n in 0 until 5) {
                val wr = w5Re[k * 5 + n]
                val wi = w5Im[k * 5 + n]
                val xr = inRe[inOff + n * stride]
                val xi = inIm[inOff + n * stride]
                sr += xr * wr - xi * wi
                si += xr * wi + xi * wr
            }
            outRe[outOff + k] = sr
            outIm[outOff + k] = si
        }
    }

    private val dft25ColRe = FloatArray(5)
    private val dft25ColIm = FloatArray(5)
    private val dft25ColOutRe = FloatArray(5)
    private val dft25ColOutIm = FloatArray(5)
    private val dft25F1Re = FloatArray(25)
    private val dft25F1Im = FloatArray(25)
    private val dft25RowOutRe = FloatArray(5)
    private val dft25RowOutIm = FloatArray(5)

    // 25-point DFT: Xin[n1][n2] = in[(5*n1+n2)*stride]; Xout[k1+5*k2].
    private fun dft25(inRe: FloatArray, inIm: FloatArray, inOff: Int, stride: Int,
                      outRe: FloatArray, outOff: Int, outIm: FloatArray) {
        for (n2 in 0 until 5) {
            for (n1 in 0 until 5) {
                dft25ColRe[n1] = inRe[inOff + (5 * n1 + n2) * stride]
                dft25ColIm[n1] = inIm[inOff + (5 * n1 + n2) * stride]
            }
            dft5(dft25ColRe, dft25ColIm, 0, 1, dft25ColOutRe, 0, dft25ColOutIm)
            for (k1 in 0 until 5) {
                val wr = w25Re[k1 * 5 + n2]
                val wi = w25Im[k1 * 5 + n2]
                val xr = dft25ColOutRe[k1]
                val xi = dft25ColOutIm[k1]
                dft25F1Re[k1 * 5 + n2] = xr * wr - xi * wi
                dft25F1Im[k1 * 5 + n2] = xr * wi + xi * wr
            }
        }
        for (k1 in 0 until 5) {
            // Row k1 of F1 is contiguous at [k1*5 .. k1*5+4].
            for (n2 in 0 until 5) {
                dft25ColRe[n2] = dft25F1Re[k1 * 5 + n2]
                dft25ColIm[n2] = dft25F1Im[k1 * 5 + n2]
            }
            dft5(dft25ColRe, dft25ColIm, 0, 1, dft25RowOutRe, 0, dft25RowOutIm)
            for (k2 in 0 until 5) {
                outRe[outOff + k1 + 5 * k2] = dft25RowOutRe[k2]
                outIm[outOff + k1 + 5 * k2] = dft25RowOutIm[k2]
            }
        }
    }

    private fun combineHalves(tmpRe: FloatArray, tmpIm: FloatArray, tmpOff: Int, h: Int,
                              twRe: FloatArray, twIm: FloatArray,
                              outRe: FloatArray, outOff: Int, outIm: FloatArray) {
        for (k in 0 until h) {
            val er = tmpRe[tmpOff + k]
            val ei = tmpIm[tmpOff + k]
            val orv = tmpRe[tmpOff + h + k]
            val oi = tmpIm[tmpOff + h + k]
            val wr = twRe[k]
            val wi = twIm[k]
            val vr = orv * wr - oi * wi
            val vi = orv * wi + oi * wr
            outRe[outOff + k] = er + vr
            outIm[outOff + k] = ei + vi
            outRe[outOff + h + k] = er - vr
            outIm[outOff + h + k] = ei - vi
        }
    }

    private fun fftRec(inRe: FloatArray, inIm: FloatArray, inOff: Int, n: Int, stride: Int,
                       outRe: FloatArray, outOff: Int, outIm: FloatArray,
                       tmpRe: FloatArray, tmpOff: Int, tmpIm: FloatArray) {
        if (n == 25) {
            dft25(inRe, inIm, inOff, stride, outRe, outOff, outIm)
            return
        }
        val h = n shr 1
        fftRec(inRe, inIm, inOff, h, stride shl 1, tmpRe, tmpOff, tmpIm, outRe, outOff, outIm)
        fftRec(inRe, inIm, inOff + stride, h, stride shl 1,
            tmpRe, tmpOff + h, tmpIm, outRe, outOff, outIm)
        val (twRe, twIm) = when (n) {
            50 -> c50Re to c50Im
            100 -> c100Re to c100Im
            200 -> c200Re to c200Im
            else -> c400Re to c400Im
        }
        combineHalves(tmpRe, tmpIm, tmpOff, h, twRe, twIm, outRe, outOff, outIm)
    }

    private fun fft400(xWin: FloatArray,
                       outRe: FloatArray, outIm: FloatArray,
                       tmpARe: FloatArray, tmpAIm: FloatArray,
                       tmpBRe: FloatArray, tmpBIm: FloatArray) {
        for (i in 0 until N_FFT) {
            tmpARe[i] = xWin[i]
            tmpAIm[i] = 0f
        }
        fftRec(tmpARe, tmpAIm, 0, N_FFT, 1, outRe, 0, outIm, tmpBRe, 0, tmpBIm)
    }
}
