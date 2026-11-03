package io.github.bombless.htp_playpen.tts

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class KokoroOnnxTtsSmokeTest {
    @Test
    fun chineseHanTextBecomesNumberedPinyin() {
        val pinyin = KokoroTextNormalizer.chinese("你好，世界。")
        Log.i(TAG, "Chinese frontend: $pinyin")
        assertTrue("Expected numbered Pinyin, got: $pinyin", pinyin.contains("ni3"))
        assertTrue("Expected numbered Pinyin, got: $pinyin", pinyin.contains("hao3"))
        assertFalse(pinyin.any { it in '\u4e00'..'\u9fff' })
    }

    @Test
    fun japaneseKanjiTextBecomesKana() {
        val kana = KokoroTextNormalizer.japanese("今日はいい天気です。")
        Log.i(TAG, "Japanese frontend: $kana")
        assertTrue(kana.any { it in '\u3040'..'\u309f' })
        assertFalse(kana.any { it in '\u4e00'..'\u9fff' })
    }

    @Test
    fun nativeKokoroSynthesizesChineseAndJapanese() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val engine = KokoroOnnxTtsEngine(context) { progress ->
            Log.i(
                TAG,
                "Model ${progress.completed}/${progress.totalFiles} " +
                    "${progress.totalBytesDownloaded}/${progress.totalBytes} bytes (${progress.file})",
            )
        }
        try {
            engine.prepare()
            engine.setLanguagePreference(KokoroOnnxTtsEngine.LANGUAGE_CHINESE)
            val chinese = engine.synthesize("你好，世界。")
            assertEquals(24_000, chinese.sampleRate)
            assertTrue("Chinese PCM should be non-empty", chinese.pcm16.size >= 2_400)
            assertTrue("Chinese PCM should not be silent", chinese.pcm16.any { it != 0.toShort() })

            engine.setLanguagePreference(KokoroOnnxTtsEngine.LANGUAGE_JAPANESE)
            val japanese = engine.synthesize("今日はいい天気です。")
            assertEquals(24_000, japanese.sampleRate)
            assertTrue("Japanese PCM should be non-empty", japanese.pcm16.size >= 2_400)
            assertTrue("Japanese PCM should not be silent", japanese.pcm16.any { it != 0.toShort() })

            // Exercise both language frontends repeatedly on the same native session.
            repeat(20) { index ->
                val isJapanese = index % 2 == 1
                engine.setLanguagePreference(
                    if (isJapanese) KokoroOnnxTtsEngine.LANGUAGE_JAPANESE
                    else KokoroOnnxTtsEngine.LANGUAGE_CHINESE,
                )
                val result = engine.synthesize(if (isJapanese) "今日はいい天気です。" else "你好，世界。")
                assertTrue("Segment $index returned empty PCM", result.pcm16.isNotEmpty())
                assertTrue("Segment $index returned invalid sample rate", result.sampleRate == 24_000)
            }
        } finally {
            engine.close()
        }

        // Closing and recreating the adapter must release and reload native resources cleanly.
        val recreated = KokoroOnnxTtsEngine(context)
        try {
            recreated.prepare()
            recreated.setLanguagePreference(KokoroOnnxTtsEngine.LANGUAGE_CHINESE)
            assertTrue(recreated.synthesize("你好。").pcm16.isNotEmpty())
        } finally {
            recreated.close()
        }
    }

    companion object {
        private const val TAG = "KOKORO_SMOKE"
    }
}