package com.example.whisperapp.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

class AudioPlayer {
    @Volatile private var currentTrack: AudioTrack? = null

    fun play(pcm16: ShortArray, sampleRate: Int = 16_000) {
        require(sampleRate > 0) { "Invalid sample rate: $sampleRate" }
        if (pcm16.isEmpty()) return

        stop()
        val min = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(min > 0) { "AudioTrack is unavailable for sample rate $sampleRate" }

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(min, pcm16.size * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        currentTrack = track
        try {
            track.write(pcm16, 0, pcm16.size)
            track.play()
            val durationMs = pcm16.size * 1000L / sampleRate
            val deadline = android.os.SystemClock.elapsedRealtime() + durationMs + 100L
            while (currentTrack === track && android.os.SystemClock.elapsedRealtime() < deadline) {
                Thread.sleep(20L)
            }
        } finally {
            if (currentTrack === track) currentTrack = null
            runCatching { track.stop() }
            track.release()
        }
    }

    @Synchronized
    fun stop() {
        currentTrack?.let { track ->
            currentTrack = null
            runCatching { track.pause() }
            runCatching { track.flush() }
            runCatching { track.stop() }
        }
    }

    @Synchronized
    fun release() = stop()
}
