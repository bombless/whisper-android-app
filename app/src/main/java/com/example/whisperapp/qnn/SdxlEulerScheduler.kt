package com.example.whisperapp.qnn

import kotlin.math.sqrt

internal class SdxlEulerScheduler(val steps: Int) {
    val timesteps: IntArray
    val sigmas: FloatArray
    val initialNoiseSigma: Float

    init {
        require(steps in 1..50)
        val trainingSigmas = FloatArray(1000)
        var alphaProduct = 1.0
        val start = sqrt(0.00085)
        val end = sqrt(0.012)
        for (index in trainingSigmas.indices) {
            val rootBeta = start + (end - start) * index / 999.0
            alphaProduct *= 1.0 - rootBeta * rootBeta
            trainingSigmas[index] = sqrt((1.0 - alphaProduct) / alphaProduct).toFloat()
        }
        val ratio = 1000 / steps
        timesteps = IntArray(steps) { (steps - 1 - it) * ratio + 1 }
        sigmas = FloatArray(steps + 1) { index ->
            if (index == steps) 0f else trainingSigmas[timesteps[index]]
        }
        initialNoiseSigma = sqrt(sigmas[0] * sigmas[0] + 1f)
    }

    fun scaleInput(latents: FloatArray, step: Int, destination: FloatArray) {
        require(latents.size == destination.size)
        val sigma = sigmas[step]
        val scale = 1f / sqrt(sigma * sigma + 1f)
        for (index in latents.indices) destination[index] = latents[index] * scale
    }

    fun advance(latents: FloatArray, unconditional: FloatArray, conditional: FloatArray, step: Int, guidance: Float) {
        require(latents.size == unconditional.size && latents.size == conditional.size)
        val delta = sigmas[step + 1] - sigmas[step]
        for (index in latents.indices) {
            val prediction = unconditional[index] + guidance * (conditional[index] - unconditional[index])
            latents[index] += prediction * delta
        }
    }
}
