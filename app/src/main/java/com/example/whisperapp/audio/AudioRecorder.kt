package com.example.whisperapp.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder

class AudioRecorder {
    companion object { const val SAMPLE_RATE = 16_000 }

    fun record(durationMs: Long): ShortArray {
        val min = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        require(min > 0) { "AudioRecord is unavailable" }
        val record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(min, SAMPLE_RATE / 2)
        )
        val out = ShortArray((SAMPLE_RATE * durationMs / 1000).toInt())
        var offset = 0
        record.startRecording()
        try {
            while (offset < out.size) {
                val n = record.read(out, offset, out.size - offset)
                if (n <= 0) break
                offset += n
            }
        } finally {
            record.stop(); record.release()
        }
        return if (offset == out.size) out else out.copyOf(offset)
    }
}
