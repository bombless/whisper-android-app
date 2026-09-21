package com.example.whisperapp.qnn

import ai.onnxruntime.OrtEnvironment
import android.util.Log

/**
 * Single process-wide registration point for the ONNX Runtime QNN execution provider.
 *
 * [OrtEnvironment] is a singleton shared by every runner in the app (Whisper ASR, LFM2 chat and
 * the smoke-test activities), but ONNX Runtime accepts only one registration of a provider
 * library per environment. Registering it twice aborts with
 * `ORT_FAIL: library is already registered under QNNExecutionProvider`, which is exactly what
 * made the 聊天 tab fail whenever the ASR runner had already run in the same process
 * (`QNN_LIFECYCLE STOP_DONE ENV_EP_RETAINED` deliberately keeps the EP alive after recording).
 *
 * Every runner must therefore ask this gate instead of calling
 * [OrtEnvironment.registerExecutionProviderLibrary] on its own.
 */
internal object QnnEpRegistration {
    const val EP_NAME = "QNNExecutionProvider"

    private const val TAG = "QNN_EP"
    private const val NATIVE_LIBRARY = "onnxruntime_providers_qnn"

    /**
     * File name of the QNN EP plugin, resolvable inside `nativeLibraryDir`.
     *
     * Also used by the native decoder session (P1), which must register the same plugin into
     * its own `Ort::Env` — the QNN EP is not linked into `libonnxruntime.so`.
     */
    const val PLUGIN_LIBRARY = "libonnxruntime_providers_qnn.so"

    /**
     * Makes sure the QNN EP is registered in [env].
     *
     * @return true when this call performed the registration, false when an existing
     *   registration in the shared environment was reused.
     */
    @Synchronized
    fun ensureRegistered(env: OrtEnvironment): Boolean {
        // The shared environment is the only reliable source of truth: another runner may have
        // registered the EP, and a smoke-test activity may have unregistered it again.
        if (isDeviceExposed(env)) {
            Log.i(TAG, "QNN EP already registered in this process; reusing the existing registration")
            return false
        }
        System.loadLibrary(NATIVE_LIBRARY)
        return try {
            env.registerExecutionProviderLibrary(EP_NAME, PLUGIN_LIBRARY)
            Log.i(TAG, "QNN EP registered")
            true
        } catch (t: Throwable) {
            // Only tolerate a lost race: if the device is not there afterwards the failure is real.
            if (!isDeviceExposed(env)) throw t
            Log.i(TAG, "QNN EP registration raced with another runner; reusing the existing registration")
            false
        }
    }

    /** True when the shared environment already exposes a QNN hardware device. */
    fun isDeviceExposed(env: OrtEnvironment): Boolean =
        runCatching { env.epDevices.any { it.epName == EP_NAME } }.getOrDefault(false)
}
