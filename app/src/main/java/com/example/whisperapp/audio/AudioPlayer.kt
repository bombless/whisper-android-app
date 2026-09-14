package com.example.whisperapp.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

class AudioPlayer {
    fun play(pcm16: ShortArray, sampleRate: Int = 16_000) {
        val bytes = pcm16.size * 2
        val min = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(sampleRate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
            .setBufferSizeInBytes(maxOf(min, bytes))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        try {
            track.write(pcm16, 0, pcm16.size)
            track.play()
            Thread.sleep((pcm16.size * 1000L / sampleRate).coerceAtLeast(1))
        } finally { track.stop(); track.release() }
    }
}
