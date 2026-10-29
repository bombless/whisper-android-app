package com.example.whisperapp

import com.example.whisperapp.qnn.SdxlEulerScheduler
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class SdxlEulerSchedulerTest {
    @Test
    fun usesSdxlBaseLeadingSchedule() {
        val scheduler = SdxlEulerScheduler(20)
        assertArrayEquals(intArrayOf(951, 901, 851), scheduler.timesteps.copyOfRange(0, 3))
        assertEquals(1, scheduler.timesteps.last())
        assertEquals(11.028331f, scheduler.sigmas[0], 0.0001f)
        assertEquals(8.390685f, scheduler.sigmas[1], 0.0001f)
        assertEquals(0.0413144f, scheduler.sigmas[19], 0.0001f)
        assertEquals(0f, scheduler.sigmas[20], 0f)
        assertEquals(11.073576f, scheduler.initialNoiseSigma, 0.0001f)
    }

    @Test
    fun scalesAndAppliesClassifierFreeGuidance() {
        val scheduler = SdxlEulerScheduler(20)
        val latents = floatArrayOf(2f, -1f)
        val scaled = FloatArray(2)
        scheduler.scaleInput(latents, 0, scaled)
        assertEquals(2f / scheduler.initialNoiseSigma, scaled[0], 0.00001f)
        val delta = scheduler.sigmas[1] - scheduler.sigmas[0]
        scheduler.advance(latents, floatArrayOf(1f, 2f), floatArrayOf(3f, 4f), 0, 2f)
        assertArrayEquals(floatArrayOf(2f + 5f * delta, -1f + 6f * delta), latents, 0.00001f)
    }
}
