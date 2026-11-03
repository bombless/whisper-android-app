package io.github.bombless.htp_playpen.asr

import android.content.Context
import io.github.bombless.htp_playpen.qnn.QnnWhisperRealAudioRunner

/** Centralizes ASR engine construction and native-session lifecycle for the STT screen. */
object AsrEngineFactory {
    fun create(context: Context, choice: SttEngineChoice): AsrEngine? = when (choice) {
        SttEngineChoice.SENSEVOICE -> SenseVoiceAsrEngine(context.applicationContext)
        // Whisper is a live QNN pipeline driven by LiveTranscriber rather than an Offline AsrEngine.
        SttEngineChoice.WHISPER_QNN -> null
    }

    suspend fun prepare(context: Context, choice: SttEngineChoice, whisperVariant: WhisperVariant): AsrEngine? {
        val engine = create(context, choice)
        if (engine == null) {
            QnnWhisperRealAudioRunner.start(context, whisperVariant)
        } else {
            try {
                (engine as? SenseVoiceAsrEngine)?.prepare()
            } catch (failure: Throwable) {
                engine.close()
                throw failure
            }
        }
        return engine
    }

    fun close(choice: SttEngineChoice, engine: AsrEngine?) {
        engine?.close()
        if (choice == SttEngineChoice.WHISPER_QNN) QnnWhisperRealAudioRunner.stop()
    }
}