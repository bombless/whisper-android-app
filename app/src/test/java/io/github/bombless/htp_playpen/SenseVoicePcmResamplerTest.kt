package io.github.bombless.htp_playpen

import io.github.bombless.htp_playpen.asr.Pcm16Resampler
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class SenseVoicePcmResamplerTest {
    @Test
    fun sameRateReturnsAnIndependentCopy() {
        val input = shortArrayOf(-32768, -1, 0, 32767)
        val output = Pcm16Resampler.resample(input, 16_000, 16_000)
        assertArrayEquals(input, output)
        output[0] = 0
        assertEquals(Short.MIN_VALUE.toInt(), input[0].toInt())
    }

    @Test
    fun upsamplesWithLinearInterpolation() {
        val output = Pcm16Resampler.resample(shortArrayOf(0, 1000, 2000, 3000), 8_000, 16_000)
        assertArrayEquals(shortArrayOf(0, 500, 1000, 1500, 2000, 2500, 3000, 3000), output)
    }

    @Test
    fun downsamplesToTheExpectedDuration() {
        val input = ShortArray(48_000) { (it % 100).toShort() }
        assertEquals(16_000, Pcm16Resampler.resample(input, 48_000, 16_000).size)
    }

    @Test
    fun rejectsInvalidSampleRates() {
        assertThrows(IllegalArgumentException::class.java) {
            Pcm16Resampler.resample(shortArrayOf(1), 0, 16_000)
        }
    }
}