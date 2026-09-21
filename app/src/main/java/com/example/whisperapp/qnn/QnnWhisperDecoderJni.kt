package com.example.whisperapp.qnn

import android.util.Log

/**
 * JNI facade over the native decoder-step loop (M2 / P1).
 *
 * The native side owns an `OrtSession` + `OrtIoBinding` pair for the Whisper decoder and
 * keeps the 8 cross-KV tensors and the attention mask bound to device memory for the whole
 * encoder window. Only `input_ids`, `position_ids` and the self-KV ping-pong buffers
 * change per step.
 *
 * The Kotlin caller stays responsible for prompt construction and argmax, so the numeric
 * contract is unchanged: this class moves tensors, it does not compute logits.
 *
 * All methods are called under the runner's lifecycle lock, so the handle is only touched
 * from one thread at a time. [close] must be called exactly once per successful [create].
 */
object QnnWhisperDecoderJni {
    private const val TAG = "WHISPER_DEC_JNI"

    /** False when `libwhisper_decoder.so` is absent; callers fall back to the Java loop. */
    val available: Boolean = try {
        System.loadLibrary("whisper_decoder")
        true
    } catch (t: Throwable) {
        Log.w(TAG, "native decoder unavailable, Java decode loop will be used: ${t.message}")
        false
    }

    private external fun nativeCreate(
        modelPath: String,
        backendPath: String,
        socModel: String,
        htpArch: String,
        perfMode: String,
        pluginPath: String,
        nHeads: Int,
        selfSlots: Int,
        vocabSize: Int,
    ): Long

    private external fun nativeBindWindow(handle: Long, crossKvFlat: ShortArray, mask: ShortArray, position: Int): Boolean
    private external fun nativeStep(handle: Long, inputToken: Int, position: Int, logitsOut: ShortArray): Boolean
    private external fun nativeLastStepMs(handle: Long): Double
    private external fun nativeLastBindMs(handle: Long): Double
    private external fun nativeDescribe(handle: Long): String
    private external fun nativeClose(handle: Long)
    private external fun nativeSetStageLog(path: String?)
    private external fun nativeSetSharedEnv(handle: Long)

    /**
     * Mirrors the native side's failure reasons into the same file [OpTrace.stage] writes,
     * because this device's ROM suppresses the app's logcat entirely.
     */
    fun setStageLog(path: String?) {
        if (!available) return
        runCatching { nativeSetStageLog(path) }
    }

    /**
     * Native pointer of the process-wide Java [ai.onnxruntime.OrtEnvironment].
     *
     * ONNX Runtime permits the QNN EP plugin to be registered only once per environment, and
     * the Java runner already owns that registration. Rather than fighting for it, the native
     * decoder adopts the same `OrtEnv` through this handle. `getNativeHandle()` is
     * package-private, so it is read reflectively; a failure here is non-fatal because
     * [create] simply reports the native path as unavailable and the Java loop runs instead.
     */
    fun javaEnvHandle(env: ai.onnxruntime.OrtEnvironment): Long = try {
        val method = ai.onnxruntime.OrtEnvironment::class.java.getDeclaredMethod("getNativeHandle")
        method.isAccessible = true
        (method.invoke(env) as? Long) ?: 0L
    } catch (t: Throwable) {
        Log.w(TAG, "cannot read OrtEnvironment native handle: ${t.message}")
        0L
    }

    /** Publishes [handle] so subsequent [create] calls attach to the shared environment. */
    fun setSharedEnvHandle(handle: Long) {
        if (!available) return
        runCatching { nativeSetSharedEnv(handle) }
    }

    /** An open native decoder session. Not thread-safe; the runner serialises access. */
    class Handle internal constructor(internal val pointer: Long) : AutoCloseable {
        private var closed = false

        val describe: String get() = nativeDescribe(pointer)
        val lastStepMs: Double get() = nativeLastStepMs(pointer)
        val lastBindMs: Double get() = nativeLastBindMs(pointer)

        /** Binds cross-KV + mask for one encoder window and resets the self-KV cache. */
        fun bindWindow(crossKvFlat: ShortArray, mask: ShortArray, position: Int = 0): Boolean =
            nativeBindWindow(pointer, crossKvFlat, mask, position)

        /** Runs one step; [logitsOut] receives the full FP16 logits row. */
        fun step(inputToken: Int, position: Int, logitsOut: ShortArray): Boolean =
            nativeStep(pointer, inputToken, position, logitsOut)

        override fun close() {
            if (closed) return
            closed = true
            nativeClose(pointer)
        }
    }

    /**
     * Creates a decoder session bound to the QNN EP.
     *
     * @param modelPath  absolute path to the `decoder_ctx.onnx` EPContext wrapper
     * @param backendPath absolute path to `libQnnHtp.so`
     * @return a handle, or null when native creation failed (caller falls back to Java)
     */
    fun create(
        modelPath: String,
        backendPath: String,
        socModel: String,
        htpArch: String,
        perfMode: String,
        pluginPath: String,
        nHeads: Int,
        selfSlots: Int,
        vocabSize: Int,
    ): Handle? {
        if (!available) {
            Log.e(TAG, "native library not loaded")
            return null
        }
        val pointer = try {
            nativeCreate(modelPath, backendPath, socModel, htpArch, perfMode, pluginPath, nHeads, selfSlots, vocabSize)
        } catch (t: Throwable) {
            OpTrace.stage("native.create.threw ${t.javaClass.name}: ${t.message}")
            Log.e(TAG, "nativeCreate threw: ${t.message}", t)
            return null
        }
        if (pointer == 0L) {
            OpTrace.stage("native.create.failed handle=0 (see WHISPER_DEC_JNI CREATE_FAILED)")
            Log.e(TAG, "nativeCreate returned null handle")
            return null
        }
        val handle = Handle(pointer)
        OpTrace.stage("native.create.ok ${handle.describe}")
        Log.i(TAG, "NATIVE_DECODER_READY ${handle.describe}")
        return handle
    }
}
