package com.example.whisperapp.qnn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtEpDevice
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.max

object QnnSmokeTest {
    private const val TAG = "QNN_SMOKE"
    private const val MODEL_ASSET = "models/tiny_cnn.onnx"
    private const val EP_NAME = "QNNExecutionProvider"
    private val EXPECTED = floatArrayOf(-0.22386418282985687f, 0.3315545916557312f)
    private val INPUT = FloatArray(192) { i -> i / 64.0f - 1.5f }
    private val SHAPE = longArrayOf(1, 3, 8, 8)

    data class Result(
        val passed: Boolean,
        val message: String,
        val medianMs: Double? = null,
        val p95Ms: Double? = null,
        val maxMs: Double? = null,
        val maxAbsError: Float? = null,
    )

    fun run(context: Context): Result {
        Log.i(TAG, "loading model")
        val model = File(context.cacheDir, "tiny_cnn.onnx")
        context.assets.open(MODEL_ASSET).use { input -> model.outputStream().use { input.copyTo(it) } }
        Log.i(TAG, "model=${model.name} size=${model.length()}")

        var env: OrtEnvironment? = null
        var session: OrtSession? = null
        var options: OrtSession.SessionOptions? = null
        var inputTensor: OnnxTensor? = null
        try {
            env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO, TAG)
            Log.i(TAG, "ORT version=${env.version}")

            val nativeQnn = "libonnxruntime_providers_qnn.so"
            Log.i(TAG, "QNN EP library=$nativeQnn")
            System.loadLibrary("onnxruntime_providers_qnn")
            env.registerExecutionProviderLibrary(EP_NAME, nativeQnn)
            Log.i(TAG, "QNN EP registered name=$EP_NAME")

            val devices = env.epDevices
            val qnnDevices = devices.filter { it.epName == EP_NAME }
            check(qnnDevices.isNotEmpty()) { "QNN EP registered but no QNN EP device was exposed" }
            qnnDevices.forEach { logEpDevice(it) }
            val device = qnnDevices.first()
            Log.i(TAG, "selected EP=${device.epName} vendor=${device.epVendor}")

            // The QNN EP derives ADSP_LIBRARY_PATH from the backend's directory.
            // Do not prepend standalone Hexagon libc++ libraries: they can override
            // the device's DSP runtime and crash aligned allocation on SM8650.
            val backend = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")
            check(backend.isFile) { "HTP backend missing: ${backend.absolutePath}" }
            Log.i(TAG, "backend_path=${backend.absolutePath}")
            options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.disable_cpu_ep_fallback", "1")
                setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)
                addExecutionProvider(listOf(device), mapOf(
                    "backend_path" to backend.absolutePath,
                    "offload_graph_io_quantization" to "0",
                ))
            }
            Log.i(TAG, "backend=HTP")
            Log.i(TAG, "CPU fallback=DISABLED")
            Log.i(TAG, "creating session")
            val t0 = System.nanoTime()
            session = env.createSession(model.absolutePath, options)
            val sessionMs = (System.nanoTime() - t0) / 1_000_000.0
            Log.i(TAG, "session created ms=$sessionMs")
            Log.i(TAG, "inputs=${session.inputNames} outputs=${session.outputNames}")

            inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(INPUT), SHAPE)
            val feeds = mapOf("input" to inputTensor)
            Log.i(TAG, "warmup")
            repeat(5) { session.run(feeds).use { } }
            Log.i(TAG, "warmup OK")

            val latencies = DoubleArray(20)
            var output = FloatArray(0)
            Log.i(TAG, "inference iterations=20")
            repeat(20) { index ->
                val start = System.nanoTime()
                session.run(feeds).use { result ->
                    val value = result[0].value
                    output = when (value) {
                        is FloatArray -> value.copyOf()
                        is Array<*> -> (value as Array<*>).flatMap { (it as FloatArray).asList() }.toFloatArray()
                        else -> error("unexpected output type=${value?.javaClass?.name}")
                    }
                }
                latencies[index] = (System.nanoTime() - start) / 1_000_000.0
            }
            Log.i(TAG, "inference OK")

            check(output.size == 2) { "output size=${output.size}, expected=2" }
            check(output.all { it.isFinite() }) { "output contains NaN/Inf" }
            var maxError = 0f
            for (i in output.indices) maxError = max(maxError, abs(output[i] - EXPECTED[i]))
            val sorted = latencies.sorted()
            val median = (sorted[9] + sorted[10]) / 2.0
            val p95 = sorted[18]
            val max = sorted[19]
            val top1 = if (output[0] >= output[1]) 0 else 1
            val expectedTop1 = if (EXPECTED[0] >= EXPECTED[1]) 0 else 1
            check(top1 == expectedTop1) { "top1=$top1 expected=$expectedTop1" }
            check(maxError < 0.02f) { "maxAbsError=$maxError" }

            Log.i(TAG, "output=[${output.joinToString()}]")
            Log.i(TAG, "maxAbsError=$maxError")
            Log.i(TAG, "top1=$top1")
            Log.i(TAG, "medianMs=$median p95Ms=$p95 maxMs=$max")
            Log.i(TAG, "EP=$EP_NAME")
            Log.i(TAG, "Backend=HTP")
            Log.i(TAG, "CPU fallback=NO")
            Log.i(TAG, "PASS")
            return Result(true, "PASS", median, p95, max, maxError)
        } catch (t: Throwable) {
            Log.e(TAG, "FAIL: ${t.javaClass.name}: ${t.message}", t)
            Log.e(TAG, "EP=$EP_NAME")
            Log.e(TAG, "Backend=HTP")
            Log.e(TAG, "CPU fallback=NO (session config requested; session/inference failure is FAIL)")
            return Result(false, "FAIL: ${t.message}")
        } finally {
            try { inputTensor?.close() } catch (_: Throwable) {}
            try { session?.close() } catch (_: Throwable) {}
            try { options?.close() } catch (_: Throwable) {}
            try { env?.unregisterExecutionProviderLibrary(EP_NAME) } catch (_: Throwable) {}
        }
    }

    private fun logEpDevice(device: OrtEpDevice) {
        Log.i(TAG, "EP_DEVICE name=${device.epName} vendor=${device.epVendor} metadata=${device.epMetadata} options=${device.epOptions}")
    }
}
