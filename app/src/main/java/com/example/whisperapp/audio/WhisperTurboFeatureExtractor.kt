package com.example.whisperapp.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Isolated Turbo mel frontend.
 *
 * Contract follows the official Whisper-Large-v3-Turbo processor:
 * 16 kHz, n_fft=400, hop=160, 128 Slaney-normalized filters from 0..8 kHz,
 * 30 s / 3000 frames, log10, max-8 floor, then (x+4)/4.
 *
 * This does not modify or delegate through WhisperFeatureExtractor, which remains
 * the 80-bin Tiny baseline.
 */
class WhisperTurboFeatureExtractor(
    private val config: WhisperMelConfig = WhisperMelConfig.LargeV3Turbo,
) : WhisperMelFrontend {
    init {
        require(config == WhisperMelConfig.LargeV3Turbo) {
            "Turbo extractor must use the verified Large-v3-Turbo mel contract"
        }
    }

    private val window = FloatArray(config.nFft) { i ->
        (0.5 - 0.5 * cos(2.0 * PI * i / config.nFft)).toFloat()
    }
    private val melFilters = buildMelFilters()
    private val fft = WhisperFft400()

    // Sparse bin -> filter tables for the incremental cache: each FFT bin feeds at most
    // two triangular filters, so scattering per bin is far cheaper than the dense
    // 128x201 sweep. Per-mel accumulation order stays ascending in k, matching [extract].
    private val melBinIds: Array<IntArray> = Array(config.nFft / 2 + 1) { k ->
        (0 until config.nMels).filter { melFilters[it][k] != 0f }.toIntArray()
    }
    private val melBinWeights: Array<FloatArray> = Array(config.nFft / 2 + 1) { k ->
        FloatArray(melBinIds[k].size) { i -> melFilters[melBinIds[k][i]][k] }
    }
    private val incrementalWindow = FloatArray(config.nFft)

    override fun extract(pcm16: ShortArray, sampleRate: Int): WhisperFeatures {
        val mel = computeRawMel(pcm16, sampleRate)
        val maxLog = logAndFloor(mel)
        val floor = maxLog - config.logDynamicRange
        for (i in mel.indices) mel[i] = (max(mel[i], floor) + 4f) / 4f
        return WhisperFeatures(mel, longArrayOf(1, config.nMels.toLong(), config.nFrames.toLong()))
    }

    override fun extractHalf(pcm16: ShortArray, sampleRate: Int): ShortArray {
        val mel = computeRawMel(pcm16, sampleRate)
        val maxLog = logAndFloor(mel)
        val floor = maxLog - config.logDynamicRange
        // Normalize and convert in the same pass: the float intermediate is never observed.
        val out = ShortArray(mel.size)
        for (i in mel.indices) out[i] = floatToHalf((max(mel[i], floor) + 4f) / 4f)
        return out
    }

    /**
     * Raw log-mel energies as [nMels, nFrames], before log/normalization.
     *
     * Two properties keep this cheap without changing a single bit of the result:
     *  - the mel projection is a sparse scatter (see below), not a dense nMels x (nFft/2+1) sweep;
     *  - frames whose entire n_fft window is past the supplied PCM are known to be exactly
     *    logFloor, so their FFT is skipped.
     */
    private fun computeRawMel(pcm16: ShortArray, sampleRate: Int): FloatArray {
        require(sampleRate == config.sampleRate) {
            "Turbo Whisper expects ${config.sampleRate} Hz, got $sampleRate"
        }
        val waveform = FloatArray(config.chunkSamples)
        val copy = min(pcm16.size, waveform.size)
        for (i in 0 until copy) waveform[i] = pcm16[i] / 32768f

        // torch.stft(center=True, pad_mode=reflect) produces 3001 frames here;
        // Whisper deliberately drops the final frame.
        // Only the first and last n_fft/2 entries need the reflect mapping; the whole middle is
        // the waveform itself, so it is copied in bulk instead of calling reflectIndex per sample.
        val padded = FloatArray(config.chunkSamples + config.nFft)
        val pad = config.nFft / 2
        System.arraycopy(waveform, 0, padded, pad, waveform.size)
        for (i in 0 until pad) padded[i] = waveform[reflectIndex(i - pad, waveform.size)]
        for (i in pad + waveform.size until padded.size) {
            padded[i] = waveform[reflectIndex(i - pad, waveform.size)]
        }

        val mel = FloatArray(config.nMels * config.nFrames)
        val frameInput = FloatArray(config.nFft)
        val firstZeroFrame = min(config.nFrames, (copy + pad + config.hopLength - 1) / config.hopLength)
        for (frame in 0 until firstZeroFrame) {
            val start = frame * config.hopLength
            for (i in 0 until config.nFft) {
                frameInput[i] = padded[start + i] * window[i]
            }
            val spectrum = fft.transform(frameInput)
            val frameRe = spectrum.first
            val frameIm = spectrum.second
            // Sparse scatter: each FFT bin feeds at most two filters, so this visits the same
            // (bin, filter, weight) terms in the same ascending-bin order as the dense 128x201
            // sweep it replaces and yields bit-identical energy sums for ~50x less work, because
            // one power value per bin now serves every filter instead of one per (bin, filter).
            for (k in 0..config.nFft / 2) {
                val power = frameRe[k] * frameRe[k] + frameIm[k] * frameIm[k]
                val ids = melBinIds[k]
                val weights = melBinWeights[k]
                for (j in ids.indices) mel[ids[j] * config.nFrames + frame] += power * weights[j]
            }
            for (m in 0 until config.nMels) {
                val index = m * config.nFrames + frame
                mel[index] = max(mel[index], config.logFloor)
            }
        }
        if (firstZeroFrame < config.nFrames) {
            for (m in 0 until config.nMels) {
                val base = m * config.nFrames
                for (frame in firstZeroFrame until config.nFrames) mel[base + frame] = config.logFloor
            }
        }
        return mel
    }

    /**
     * log10 in place, returning the maximum. The NaN/Inf validation that used to be a separate
     * full-array scan is folded into this pass: `v != v` is the single cheapest NaN test, and a
     * +Inf input always propagates into [maxLog].
     */
    private fun logAndFloor(mel: FloatArray): Float {
        var maxLog = Float.NEGATIVE_INFINITY
        var nan = 0
        for (i in mel.indices) {
            val v = kotlin.math.log10(max(mel[i], config.logFloor))
            mel[i] = v
            if (v > maxLog) maxLog = v
            if (v != v) nan++
        }
        require(nan == 0 && maxLog.isFinite()) { "Turbo mel contains NaN/Inf ($nan NaN)" }
        return maxLog
    }

    fun newIncrementalCache(): IncrementalCache = IncrementalCache()

    /**
     * Incremental mel cache for live transcription: only FFT/mel frames touched by
     * newly appended PCM are recomputed, and normalization is a cheap linear pass
     * over the cached [nMels, nFrames] raw-mel matrix. Mirrors the Tiny cache but
     * with the 128-bin Turbo filterbank.
     */
    inner class IncrementalCache : WhisperIncrementalMelCache {
        private val pcm = ShortArray(config.chunkSamples)
        private val rawMel = FloatArray(config.nMels * config.nFrames) { 1e-10f }

        /**
         * Scratch buffers, allocated once.
         *
         * The previous version called `rawMel.copyOf()` on every append, so a 10 Hz producer
         * allocated and discarded a 1.5 MB float array ten times a second on the capture
         * thread. Reusing [rawLogMel] removes that garbage.
         *
         * [out] is *not* reused across calls: the returned array must stay valid for the
         * caller until it is done with it (the live path hands it to a decoder that may still
         * be running when the next append arrives). Handing back a shared buffer corrupted the
         * previous window, which is what `incrementalCacheReturnsIndependentArrays` guards.
         */
        private val rawLogMel = FloatArray(config.nMels * config.nFrames)
        private var sampleCount = 0

        override fun append(samples: ShortArray, sampleRate: Int): ShortArray {
            require(sampleRate == config.sampleRate) {
                "Turbo Whisper expects ${config.sampleRate} Hz, got $sampleRate"
            }
            val oldCount = sampleCount
            val copy = min(samples.size, config.chunkSamples - oldCount)
            if (copy > 0) samples.copyInto(pcm, oldCount, 0, copy)
            sampleCount = min(config.chunkSamples, oldCount + samples.size)
            if (sampleCount > oldCount) recomputeAffectedFrames(oldCount, sampleCount)

            // Only the frames whose raw energy changed need re-normalizing. This used to run a
            // full 128x3000 pass (plus a `rawMel.copyOf()` allocation) on every append, i.e.
            // 384k elements ~10x/second, and it ran on the *capture* thread because the
            // producer feeds the mel cache (see LiveTranscriber.onAudioAppendedWithMel).
            // Measured on SM8650 that made the producer spend 27.3 s of a 30 s capture inside
            // this function. Frames outside [firstFrame, lastFrame] still hold exactly the
            // values computed by their own append, so leaving them untouched is bit-identical.
            var minFrame = config.nFrames
            var maxFrame = -1
            if (sampleCount > oldCount) {
                val centerPad = config.nFft / 2
                minFrame = max(0, oldCount - (config.nFft - centerPad - 1) + config.hopLength - 1) / config.hopLength
                maxFrame = min(config.nFrames - 1, (sampleCount - 1 + centerPad) / config.hopLength)
            }

            // Pass 1: log10 over the whole window, tracking the max. The floor is a function of
            // the max over *every* frame, so this pass cannot be restricted to the changed
            // frames — but it writes into a reusable buffer instead of the old
            // `rawMel.copyOf()` allocation, so it costs no garbage. This is the same
            // `log10(max(rawMel, logFloor))` the previous implementation computed.
            var maxLog = Float.NEGATIVE_INFINITY
            for (m in 0 until config.nMels) {
                val base = m * config.nFrames
                for (frame in 0 until config.nFrames) {
                    val v = kotlin.math.log10(max(rawMel[base + frame], config.logFloor))
                    rawLogMel[base + frame] = v
                    if (v > maxLog) maxLog = v
                }
            }
            val floor = maxLog - config.logDynamicRange

            // Pass 2: the floor changed, so *every* frame's normalized value must be revisited,
            // not just the ones whose raw energy changed — a frame written by an earlier append
            // may now need to be clamped down to a higher floor. Normalization is therefore a
            // full-window pass; that is inherent to the contract, and it is why the win here
            // comes from removing the redundant allocation and the duplicated log10 rather than
            // from skipping frames.
            val out = ShortArray(config.nMels * config.nFrames)
            for (m in 0 until config.nMels) {
                val base = m * config.nFrames
                for (frame in 0 until config.nFrames) {
                    out[base + frame] = floatToHalf((max(rawLogMel[base + frame], floor) + 4f) / 4f)
                }
            }
            android.util.Log.i(
                "WHISPER_DEBUG",
                "TURBO_CACHE old=$oldCount new=${samples.size} copied=$copy pcmTotal=$sampleCount " +
                    "frameRange=$minFrame..$maxFrame totalFrames=${if (maxFrame >= 0) maxFrame + 1 else 0}"
            )
            return out
        }

        override fun reset() {
            pcm.fill(0)
            rawMel.fill(1e-10f)
            rawLogMel.fill(0f)
            sampleCount = 0
        }

        private fun recomputeAffectedFrames(oldSamples: Int, newSamples: Int) {
            val centerPad = config.nFft / 2
            val firstFrame = max(0, oldSamples - (config.nFft - centerPad - 1) + config.hopLength - 1) / config.hopLength
            val lastFrame = min(config.nFrames - 1, (newSamples - 1 + centerPad) / config.hopLength)
            if (firstFrame > lastFrame) return
            for (frame in firstFrame..lastFrame) {
                for (m in 0 until config.nMels) rawMel[m * config.nFrames + frame] = 0f
                val start = frame * config.hopLength
                for (i in 0 until config.nFft) {
                    val sourceIndex = reflectIndex(start + i - config.nFft / 2, config.chunkSamples)
                    incrementalWindow[i] = pcm[sourceIndex] / 32768f * window[i]
                }
                val spectrum = fft.transform(incrementalWindow)
                val re = spectrum.first
                val im = spectrum.second
                for (k in 0..config.nFft / 2) {
                    val p = re[k] * re[k] + im[k] * im[k]
                    val ids = melBinIds[k]
                    val ws = melBinWeights[k]
                    for (j in ids.indices) rawMel[ids[j] * config.nFrames + frame] += p * ws[j]
                }
            }
        }
    }

    private fun buildMelFilters(): Array<FloatArray> {
        val filters = Array(config.nMels) { FloatArray(config.nFft / 2 + 1) }
        val melMin = hzToMel(config.minFrequencyHz)
        val melMax = hzToMel(config.maxFrequencyHz)
        val filterHz = DoubleArray(config.nMels + 2) { i ->
            melToHz(melMin + (melMax - melMin) * i / (config.nMels + 1))
        }
        val fftHz = DoubleArray(config.nFft / 2 + 1) { k ->
            k.toDouble() * config.sampleRate / config.nFft
        }
        for (m in 0 until config.nMels) {
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

    private fun hzToMel(hz: Double): Double =
        if (hz < 1000.0) 3.0 * hz / 200.0
        else 15.0 + ln(hz / 1000.0) * (27.0 / ln(6.4))

    private fun melToHz(mel: Double): Double =
        if (mel < 15.0) 200.0 * mel / 3.0
        else 1000.0 * kotlin.math.exp(ln(6.4) / 27.0 * (mel - 15.0))

    private fun reflectIndex(index: Int, size: Int): Int {
        require(size > 1)
        var i = index
        while (i < 0 || i >= size) i = if (i < 0) -i else 2 * size - 2 - i
        return i
    }


    private fun floatToHalf(value: Float): Short {
        val bits = value.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exponent = ((bits ushr 23) and 0xff) - 127 + 15
        val mantissa = bits and 0x7fffff
        return when {
            exponent <= 0 ->
                if (exponent < -10) sign.toShort()
                else (sign or ((mantissa or 0x800000) shr (1 - exponent + 13))).toShort()
            exponent >= 31 ->
                (sign or 0x7c00 or if (mantissa == 0) 0 else 0x0200).toShort()
            else -> {
                var halfExponent = exponent
                var halfMantissa = (mantissa + 0x1000) shr 13
                if (halfMantissa == 0x400) {
                    halfMantissa = 0
                    halfExponent++
                }
                if (halfExponent >= 31) (sign or 0x7c00).toShort()
                else (sign or (halfExponent shl 10) or halfMantissa).toShort()
            }
        }
    }
}