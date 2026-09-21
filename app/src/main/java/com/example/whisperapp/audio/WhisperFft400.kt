package com.example.whisperapp.audio

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Exact 400-point DFT using the established Tiny frontend's mixed-radix algorithm.
 * Kept as a standalone utility so Turbo does not depend on Tiny's implementation.
 */
internal class WhisperFft400 {
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
    private val tmpARe = FloatArray(400)
    private val tmpAIm = FloatArray(400)
    private val tmpBRe = FloatArray(400)
    private val tmpBIm = FloatArray(400)
    private val outRe = FloatArray(400)
    private val outIm = FloatArray(400)
    private val colRe = FloatArray(5)
    private val colIm = FloatArray(5)
    private val colOutRe = FloatArray(5)
    private val colOutIm = FloatArray(5)
    private val f1Re = FloatArray(25)
    private val f1Im = FloatArray(25)
    private val rowOutRe = FloatArray(5)
    private val rowOutIm = FloatArray(5)

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
        fillCombine(c50Re, c50Im)
        fillCombine(c100Re, c100Im)
        fillCombine(c200Re, c200Im)
        fillCombine(c400Re, c400Im)
    }

    fun transform(input: FloatArray): Pair<FloatArray, FloatArray> {
        require(input.size == 400)
        for (i in 0 until 400) {
            tmpARe[i] = input[i]
            tmpAIm[i] = 0f
        }
        fftRec(
            tmpARe, tmpAIm, 0, 400, 1,
            outRe, 0, outIm,
            tmpBRe, 0, tmpBIm,
        )
        return outRe to outIm
    }

    private fun fillCombine(re: FloatArray, im: FloatArray) {
        val half = re.size
        for (k in 0 until half) {
            val a = -2.0 * PI * k / (2 * half)
            re[k] = cos(a).toFloat()
            im[k] = sin(a).toFloat()
        }
    }

    private fun dft5(
        inRe: FloatArray, inIm: FloatArray, inOff: Int, stride: Int,
        outRe: FloatArray, outOff: Int, outIm: FloatArray,
    ) {
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

    private fun dft25(
        inRe: FloatArray, inIm: FloatArray, inOff: Int, stride: Int,
        outRe: FloatArray, outOff: Int, outIm: FloatArray,
    ) {
        for (n2 in 0 until 5) {
            for (n1 in 0 until 5) {
                colRe[n1] = inRe[inOff + (5 * n1 + n2) * stride]
                colIm[n1] = inIm[inOff + (5 * n1 + n2) * stride]
            }
            dft5(colRe, colIm, 0, 1, colOutRe, 0, colOutIm)
            for (k1 in 0 until 5) {
                val wr = w25Re[k1 * 5 + n2]
                val wi = w25Im[k1 * 5 + n2]
                val xr = colOutRe[k1]
                val xi = colOutIm[k1]
                f1Re[k1 * 5 + n2] = xr * wr - xi * wi
                f1Im[k1 * 5 + n2] = xr * wi + xi * wr
            }
        }
        for (k1 in 0 until 5) {
            for (n2 in 0 until 5) {
                colRe[n2] = f1Re[k1 * 5 + n2]
                colIm[n2] = f1Im[k1 * 5 + n2]
            }
            dft5(colRe, colIm, 0, 1, rowOutRe, 0, rowOutIm)
            for (k2 in 0 until 5) {
                outRe[outOff + k1 + 5 * k2] = rowOutRe[k2]
                outIm[outOff + k1 + 5 * k2] = rowOutIm[k2]
            }
        }
    }

    private fun combineHalves(
        tmpRe: FloatArray, tmpIm: FloatArray, tmpOff: Int, h: Int,
        twRe: FloatArray, twIm: FloatArray,
        outRe: FloatArray, outOff: Int, outIm: FloatArray,
    ) {
        for (k in 0 until h) {
            val er = tmpRe[tmpOff + k]
            val ei = tmpIm[tmpOff + k]
            val or = tmpRe[tmpOff + h + k]
            val oi = tmpIm[tmpOff + h + k]
            val wr = twRe[k]
            val wi = twIm[k]
            val vr = or * wr - oi * wi
            val vi = or * wi + oi * wr
            outRe[outOff + k] = er + vr
            outIm[outOff + k] = ei + vi
            outRe[outOff + h + k] = er - vr
            outIm[outOff + h + k] = ei - vi
        }
    }

    private fun fftRec(
        inRe: FloatArray, inIm: FloatArray, inOff: Int, n: Int, stride: Int,
        outRe: FloatArray, outOff: Int, outIm: FloatArray,
        tmpRe: FloatArray, tmpOff: Int, tmpIm: FloatArray,
    ) {
        if (n == 25) {
            dft25(inRe, inIm, inOff, stride, outRe, outOff, outIm)
            return
        }
        val h = n shr 1
        fftRec(inRe, inIm, inOff, h, stride shl 1, tmpRe, tmpOff, tmpIm, outRe, outOff, outIm)
        fftRec(inRe, inIm, inOff + stride, h, stride shl 1, tmpRe, tmpOff + h, tmpIm, outRe, outOff, outIm)
        val (twRe, twIm) = when (n) {
            50 -> c50Re to c50Im
            100 -> c100Re to c100Im
            200 -> c200Re to c200Im
            else -> c400Re to c400Im
        }
        combineHalves(tmpRe, tmpIm, tmpOff, h, twRe, twIm, outRe, outOff, outIm)
    }
}