// Native Whisper decoder step loop (M2 / P1).
//
// Why this exists
// ---------------
// The Java decode loop re-submitted eight cross-KV tensors (30.7 MB of FP16) to
// `OrtSession.run` on *every* step, and `encoder.host_copy` spent ~40 ms pulling the
// encoder outputs back to host memory only to hand them straight back to the device.
// Measured on SM8650/HTP v75: `decoder.run` = 12.6 ms/step while the QNN optrace showed
// the DSP itself (Accelerator execute excluding wait) at only 8.4-10.8 ms. The gap is
// host<->device traffic per step.
//
// With an OrtIoBinding the cross-KV and attention mask are bound once per window and stay
// device-resident; each step only re-binds the two tiny inputs (`input_ids`,
// `position_ids`) plus the self-KV feedback.
//
// Numerical contract
// ------------------
// This is a *transport* change, not a math change: the same session, the same graph, the
// same input values, same dtype, same shapes. The Kotlin caller keeps prompt construction
// and argmax, and the JNI layer reports the logits for the host to reduce, so a token
// mismatch against the Java path is a bug in this file, not a rounding difference.
//
// FP16 values are passed as raw `uint16_t` bit patterns in both directions. Nothing here
// converts or reorders them.

#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "WHISPER_DEC_JNI", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "WHISPER_DEC_JNI", __VA_ARGS__)

namespace {

/**
 * Mirror of the Kotlin-side stage log.
 *
 * Some OEM ROMs drop this app's logcat entirely, which makes a native session-creation
 * failure indistinguishable from "the native path was never tried". The path is set from
 * Kotlin via `nativeSetStageLog`; every failure reason is appended there.
 */
std::string g_stageLogPath;

/**
 * Native pointer of the process-wide Java `OrtEnvironment`.
 *
 * ORT allows the QNN EP plugin to be registered only once per environment, and the Java
 * runner registers it first. The native decoder therefore adopts that same `OrtEnv` instead
 * of creating a private one; creating a second env makes
 * `RegisterExecutionProviderLibrary` fail with "library is already registered".
 */
OrtEnv* g_sharedEnv = nullptr;

void stageLog(const std::string& message) {
    if (g_stageLogPath.empty()) return;
    if (FILE* f = std::fopen(g_stageLogPath.c_str(), "a")) {
        std::fprintf(f, "native: %s\n", message.c_str());
        std::fclose(f);
    }
}

constexpr int kNumLayers = 4;
constexpr int kMaskWidth = 200;

std::string elementTypeName(ONNXTensorElementDataType t) {
    switch (t) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        default: return "type_" + std::to_string(static_cast<int>(t));
    }
}

std::string shapeToString(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

/**
 * One decoder session plus the IoBinding that keeps cross-KV resident on the device.
 *
 * Lifetime: created by `nativeCreate`, released by `nativeClose`. All entry points are
 * called from the single Kotlin decode thread that owns the runner lock, so no internal
 * synchronisation is needed; the Kotlin side guarantees it (see
 * `QnnWhisperDecoderJni` usage inside the runner's lifecycle lock).
 */
struct DecoderBinding {
    /**
     * Adopts the shared Java environment when a handle was published; otherwise owns a
     * private one. `ownsEnv` records which, so an adopted env is never destroyed here.
     */
    Ort::Env env{nullptr};
    bool ownsEnv = false;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::IoBinding> binding;

    // Inputs re-bound every step (tiny).
    std::vector<int32_t> inputIds{0};
    std::vector<int32_t> positionIds{0};
    // Self-KV ping-pong: the graph reads one buffer while writing the other.
    std::vector<uint16_t> selfKv[2];
    // Cross-KV + mask, bound once per encoder window.
    std::vector<uint16_t> crossKv[kNumLayers * 2];
    std::vector<uint16_t> mask;
    // Logits pong buffer, bound as an output so no per-step tensor object is created.
    std::vector<uint16_t> logits;

    std::vector<std::string> outputNames;
    std::vector<const char*> outputNamePtrs;

    int selfSlots = 199;
    int nHeads = 20;
    int vocabSize = 51866;

    /** Run-level HTP profile (`qnn.perf_mode`), mirrored from the Java path. */
    std::string perfMode_ = "sustained_high_performance";

    bool bound = false;
    bool shapesLogged = false;
    int pingPong = 0;

    // Bookkeeping surfaced to the Kotlin side for logging.
    double lastBindMs = 0.0;
    double lastStepMs = 0.0;
    int steps = 0;
};

std::string kCacheSelfIn[kNumLayers * 2];
std::string kCacheSelfOut[kNumLayers * 2];
std::string kCacheCross[kNumLayers * 2];

void initNames() {
    static bool done = false;
    if (done) return;
    for (int i = 0; i < kNumLayers; ++i) {
        kCacheSelfIn[i * 2 + 0] = "k_cache_self_" + std::to_string(i) + "_in";
        kCacheSelfIn[i * 2 + 1] = "v_cache_self_" + std::to_string(i) + "_in";
        kCacheSelfOut[i * 2 + 0] = "k_cache_self_" + std::to_string(i) + "_out";
        kCacheSelfOut[i * 2 + 1] = "v_cache_self_" + std::to_string(i) + "_out";
        kCacheCross[i * 2 + 0] = "k_cache_cross_" + std::to_string(i);
        kCacheCross[i * 2 + 1] = "v_cache_cross_" + std::to_string(i);
    }
    done = true;
}

size_t selfKvElements(bool isKey, int nHeads, int slots) {
    // k_cache_self_* is [heads,1,64,slots]; v_cache_self_* is [heads,1,slots,64].
    return static_cast<size_t>(nHeads) * 64 * static_cast<size_t>(slots);
}

size_t crossKvElements(bool isKey, int nHeads, int slots) {
    return static_cast<size_t>(nHeads) * 64 * static_cast<size_t>(slots);
}

}  // namespace

extern "C" {

/** Points the native stage log at the same file the Kotlin side uses. */
JNIEXPORT void JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeSetStageLog(JNIEnv* env, jclass,
                                                                      jstring path) {
    if (path == nullptr) {
        g_stageLogPath.clear();
        return;
    }
    const char* c = env->GetStringUTFChars(path, nullptr);
    g_stageLogPath = c;
    env->ReleaseStringUTFChars(path, c);
    stageLog("stage_log_attached");
}

/** Records the shared Java OrtEnvironment this session must attach to. */
JNIEXPORT void JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeSetSharedEnv(JNIEnv*, jclass,
                                                                       jlong handle) {
    g_sharedEnv = reinterpret_cast<OrtEnv*>(handle);
    stageLog("shared_env_set handle=" + std::to_string(static_cast<long long>(handle)));
}

/**
 * Creates a decoder session from an EPContext wrapper and prepares the binding.
 *
 * The QNN execution provider is NOT compiled into `libonnxruntime.so`; it ships as the
 * separate plugin `libonnxruntime_providers_qnn.so`, which must be registered into the
 * environment before it can be used. The Java path does this in `QnnEpRegistration`
 * (`OrtEnvironment.registerExecutionProviderLibrary`), and this mirrors it exactly: load the
 * plugin into *this* env, look up the resulting QNN `OrtEpDevice`, and attach the provider
 * to the session through the device. Skipping the registration is what produces
 * "QNN execution provider is not supported in this build".
 *
 * Returns 0 on failure (with the reason written to the stage log) so the caller can fall
 * back to the Java decode loop.
 */
JNIEXPORT jlong JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeCreate(
    JNIEnv* env, jclass, jstring modelPath, jstring backendPath, jstring socModel,
    jstring htpArch, jstring perfMode, jstring pluginPath, jint nHeads, jint selfSlots,
    jint vocabSize) {
    initNames();
    auto state = std::make_unique<DecoderBinding>();
    state->nHeads = nHeads;
    state->selfSlots = selfSlots;
    state->vocabSize = vocabSize;
    try {
        const char* modelPathC = env->GetStringUTFChars(modelPath, nullptr);
        const char* backendPathC = env->GetStringUTFChars(backendPath, nullptr);
        const char* socModelC = env->GetStringUTFChars(socModel, nullptr);
        const char* htpArchC = env->GetStringUTFChars(htpArch, nullptr);
        const char* perfModeC = env->GetStringUTFChars(perfMode, nullptr);
        const char* pluginPathC = env->GetStringUTFChars(pluginPath, nullptr);
        stageLog(std::string("create.begin model=") + modelPathC + " backend=" + backendPathC +
                 " plugin=" + pluginPathC + " perf=" + perfModeC);

        if (g_sharedEnv != nullptr) {
            // Adopted: the Java runner already created and registered this environment.
            state->env = Ort::Env(g_sharedEnv);
            state->ownsEnv = false;
            stageLog("create.env=shared");
        } else {
            // No shared env published (e.g. a standalone native A/B run): build a private one
            // and do the plugin registration ourselves.
            state->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "whisper_decoder_jni");
            state->ownsEnv = true;
            state->env.RegisterExecutionProviderLibrary("QNNExecutionProvider", pluginPathC);
            stageLog("create.env=private_registered");
        }

        // Resolve the QNN device the environment exposes.
        std::vector<Ort::ConstEpDevice> qnnDevices;
        for (const auto& device : state->env.GetEpDevices()) {
            if (std::string(device.EpName()) == "QNNExecutionProvider") qnnDevices.push_back(device);
        }
        if (qnnDevices.empty()) {
            stageLog("create.failed no QNN EP device in environment");
            return 0;
        }
        stageLog("create.ep_devices=" + std::to_string(qnnDevices.size()));
        state->perfMode_ = perfModeC;

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_INFO);
        // Provider options mirror the Java path exactly. Passing them through the V2 API is
        // what ties them to the registered QNN device; the older name-based
        // AppendExecutionProvider("QNN", ...) fails because the EP is not built in.
        std::unordered_map<std::string, std::string> providerOptions = {
            {"backend_path", backendPathC},
            {"soc_model", socModelC},
            {"htp_arch", htpArchC},
            // EPContext wrappers carry a small CPU-side IO adapter; the Java path leaves CPU
            // fallback enabled and so must this one.
            {"offload_graph_io_quantization", "0"},
            {"htp_performance_mode", perfModeC},
        };
        options.AppendExecutionProvider_V2(state->env, qnnDevices, providerOptions);
        stageLog("create.provider_appended");

        state->session = std::make_unique<Ort::Session>(state->env, modelPathC, options);
        state->binding = std::make_unique<Ort::IoBinding>(*state->session);

        env->ReleaseStringUTFChars(modelPath, modelPathC);
        env->ReleaseStringUTFChars(backendPath, backendPathC);
        env->ReleaseStringUTFChars(socModel, socModelC);
        env->ReleaseStringUTFChars(htpArch, htpArchC);
        env->ReleaseStringUTFChars(perfMode, perfModeC);
        env->ReleaseStringUTFChars(pluginPath, pluginPathC);

        // Resolve "logits" by name; its index is not guaranteed across exports.
        Ort::AllocatorWithDefaultOptions allocator;
        size_t outCount = state->session->GetOutputCount();
        for (size_t i = 0; i < outCount; ++i) {
            auto name = state->session->GetOutputNameAllocated(i, allocator);
            state->outputNames.emplace_back(name.get());
        }
        for (auto& n : state->outputNames) state->outputNamePtrs.push_back(n.c_str());

        LOGE("CREATE_OK outputs=%zu selfSlots=%d heads=%d", outCount, selfSlots,
             static_cast<int>(nHeads));
        stageLog("create.ok outputs=" + std::to_string(outCount));
        return reinterpret_cast<jlong>(state.release());
    } catch (const std::exception& e) {
        LOGE("CREATE_FAILED %s", e.what());
        stageLog(std::string("create.failed ") + e.what());
        return 0;
    }
}

JNIEXPORT jboolean JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeBindWindow(
    JNIEnv* env, jclass, jlong handle, jshortArray crossKvFlat, jshortArray mask, jint position) {
    auto* state = reinterpret_cast<DecoderBinding*>(handle);
    if (!state || !state->binding) return JNI_FALSE;
    try {
        const jsize crossLen = env->GetArrayLength(crossKvFlat);
        const size_t perTensor = crossKvElements(true, state->nHeads, 1500);
        if (static_cast<size_t>(crossLen) != perTensor * kNumLayers * 2) {
            LOGE("BIND_WINDOW bad cross length=%d expected=%zu", crossLen,
                 perTensor * kNumLayers * 2);
            stageLog("bind.failed bad cross length=" + std::to_string(crossLen) + " expected=" +
                     std::to_string(perTensor * kNumLayers * 2));
            return JNI_FALSE;
        }
        auto started = std::chrono::steady_clock::now();

        // Cross-KV is bound once and then stays bound for the whole window: it is the
        // 30.7 MB that used to cross the bus on every step.
        jshort* cross = env->GetShortArrayElements(crossKvFlat, nullptr);
        for (int t = 0; t < kNumLayers * 2; ++t) {
            state->crossKv[t].assign(cross + perTensor * t, cross + perTensor * (t + 1));
        }
        env->ReleaseShortArrayElements(crossKvFlat, cross, JNI_ABORT);

        const jsize maskLen = env->GetArrayLength(mask);
        if (maskLen != kMaskWidth) {
            LOGE("BIND_WINDOW bad mask length=%d expected=%d", maskLen, kMaskWidth);
            stageLog("bind.failed bad mask length=" + std::to_string(maskLen));
            return JNI_FALSE;
        }
        // Only the buffer is allocated here; `nativeStep` rewrites its contents each step
        // because the causal mask grows with position.
        state->mask.assign(kMaskWidth, 0);

        auto selfShapeKey = std::vector<int64_t>{state->nHeads, 1, 64, state->selfSlots};
        auto selfShapeVal = std::vector<int64_t>{state->nHeads, 1, state->selfSlots, 64};
        auto crossShapeKey = std::vector<int64_t>{state->nHeads, 1, 64, 1500};
        auto crossShapeVal = std::vector<int64_t>{state->nHeads, 1, 1500, 64};
        auto maskShape = std::vector<int64_t>{1, 1, 1, kMaskWidth};

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Log the graph's own declared cross-KV shapes once: a mismatch with the shapes built
        // here is the difference between "the host buffer is too small" and "the graph wants a
        // different layout", and only the session knows which.
        if (!state->shapesLogged) {
            Ort::AllocatorWithDefaultOptions allocator;
            std::unordered_map<std::string, size_t> inputIndex;
            for (size_t i = 0; i < state->session->GetInputCount(); ++i) {
                auto name = state->session->GetInputNameAllocated(i, allocator);
                inputIndex[std::string(name.get())] = i;
            }
            for (int t = 0; t < kNumLayers * 2; ++t) {
                auto it = inputIndex.find(kCacheCross[t]);
                if (it == inputIndex.end()) {
                    stageLog("bind.graph_shape missing input " + kCacheCross[t]);
                    continue;
                }
                auto info = state->session->GetInputTypeInfo(it->second).GetTensorTypeAndShapeInfo();
                stageLog("bind.graph_shape " + kCacheCross[t] + "=" +
                         shapeToString(info.GetShape()) + " host=" +
                         std::to_string(state->crossKv[t].size()));
            }
            // Output shapes matter just as much: an IoBinding output buffer must match the
            // graph's declared shape exactly, and logits is not always [1,1,1,vocab].
            for (size_t i = 0; i < state->session->GetOutputCount(); ++i) {
                auto name = state->session->GetOutputNameAllocated(i, allocator);
                auto info = state->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
                stageLog("bind.graph_output " + std::string(name.get()) + "=" +
                         shapeToString(info.GetShape()) + " dtype=" +
                         elementTypeName(info.GetElementType()));
            }
            state->shapesLogged = true;
        }

        for (int t = 0; t < kNumLayers * 2; ++t) {
            const bool isKey = (t % 2) == 0;
            auto shape = isKey ? crossShapeKey : crossShapeVal;
            // NOTE: this CreateTensor overload takes a BYTE count, not an element count.
            Ort::Value v = Ort::Value::CreateTensor(memInfo, state->crossKv[t].data(),
                                                    state->crossKv[t].size() * sizeof(uint16_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            state->binding->BindInput(kCacheCross[t].c_str(), v);
        }

        // Self-KV ping-pong buffers, zero-filled at window start.
        // `position` is accepted for symmetry with the Java path; the mask already encodes it.
        (void)position;
        for (int slot = 0; slot < 2; ++slot) {
            state->selfKv[slot].assign(selfKvElements(false, state->nHeads, state->selfSlots) *
                                           kNumLayers * 2,
                                       0);
        }
        state->logits.assign(static_cast<size_t>(state->vocabSize), 0);
        state->bound = true;
        state->pingPong = 0;
        state->steps = 0;
        state->lastBindMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        LOGI("BIND_WINDOW ok crossTensors=%d maskWidth=%d bindMs=%.2f", kNumLayers * 2,
             kMaskWidth, state->lastBindMs);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("BIND_WINDOW failed %s", e.what());
        stageLog(std::string("bind.failed ") + e.what());
        return JNI_FALSE;
    }
}

/**
 * Runs one decode step and copies the resulting logits into [logitsOut].
 *
 * Self-KV output is kept device-side and becomes the next step's input buffer, so the
 * 199-slot KV feedback never round-trips through the host.
 *
 * The attention mask is *not* constant across a window: it is a right-aligned causal mask
 * whose valid-KV region grows by one slot per step (`WhisperDecoderMask.forPosition`). It is
 * rebuilt in place here rather than re-bound from the host, which keeps the per-step host
 * traffic to the two int32 scalars while preserving the exact Java semantics.
 */
JNIEXPORT jboolean JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeStep(
    JNIEnv* env, jclass, jlong handle, jint inputToken, jint position, jshortArray logitsOut) {
    auto* state = reinterpret_cast<DecoderBinding*>(handle);
    if (!state || !state->bound) return JNI_FALSE;
    if (position < 0 || position >= kMaskWidth) {
        LOGE("STEP bad position=%d (mask width %d)", position, kMaskWidth);
        return JNI_FALSE;
    }
    try {
        auto started = std::chrono::steady_clock::now();
        state->inputIds[0] = inputToken;
        state->positionIds[0] = position;

        // Right-aligned causal mask: slots >= width-position-1 are valid (0), the rest are
        // masked to FP16 -100.0. Bit pattern 0xd640, matching WhisperDecoderMask.maskedHalf.
        const uint16_t kMaskedHalf = 0xd640;
        const int firstValidIndex = kMaskWidth - position - 1;
        for (int i = 0; i < kMaskWidth; ++i) {
            state->mask[i] = (i >= firstValidIndex) ? 0 : kMaskedHalf;
        }

        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // All CreateTensor calls below pass BYTE counts (see the overload's contract).
        {
            auto shape = std::vector<int64_t>{1, 1};
            Ort::Value v = Ort::Value::CreateTensor(memInfo, state->inputIds.data(),
                                                    state->inputIds.size() * sizeof(int32_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            state->binding->BindInput("input_ids", v);
        }
        {
            auto shape = std::vector<int64_t>{1};
            Ort::Value v = Ort::Value::CreateTensor(memInfo, state->positionIds.data(),
                                                    state->positionIds.size() * sizeof(int32_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            state->binding->BindInput("position_ids", v);
        }
        {
            auto shape = std::vector<int64_t>{1, 1, 1, kMaskWidth};
            Ort::Value v = Ort::Value::CreateTensor(memInfo, state->mask.data(),
                                                    state->mask.size() * sizeof(uint16_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            state->binding->BindInput("attention_mask", v);
        }

        const int readSlot = state->pingPong;
        const int writeSlot = 1 - state->pingPong;
        const size_t selfPerTensor = selfKvElements(false, state->nHeads, state->selfSlots);
        auto selfShapeKey = std::vector<int64_t>{state->nHeads, 1, 64, state->selfSlots};
        auto selfShapeVal = std::vector<int64_t>{state->nHeads, 1, state->selfSlots, 64};

        for (int t = 0; t < kNumLayers * 2; ++t) {
            const bool isKey = (t % 2) == 0;
            auto shape = isKey ? selfShapeKey : selfShapeVal;
            uint16_t* ptr = state->selfKv[readSlot].data() + selfPerTensor * t;
            Ort::Value v = Ort::Value::CreateTensor(memInfo, ptr, selfPerTensor * sizeof(uint16_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            state->binding->BindInput(kCacheSelfIn[t].c_str(), v);
        }
        for (int t = 0; t < kNumLayers * 2; ++t) {
            const bool isKey = (t % 2) == 0;
            auto shape = isKey ? selfShapeKey : selfShapeVal;
            uint16_t* ptr = state->selfKv[writeSlot].data() + selfPerTensor * t;
            Ort::Value v = Ort::Value::CreateTensor(memInfo, ptr, selfPerTensor * sizeof(uint16_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            state->binding->BindOutput(kCacheSelfOut[t].c_str(), v);
        }
        {
            // The graph declares logits as [1, vocabSize, 1, 1] (confirmed from the session's
            // own output type info), not the [1,1,1,vocab] a reader might assume. The element
            // count is the same, but the IoBinding output buffer must match the declared shape
            // exactly or ORT rejects the run with "invalid dimensions for output".
            auto shape = std::vector<int64_t>{1, static_cast<int64_t>(state->vocabSize), 1, 1};
            Ort::Value v = Ort::Value::CreateTensor(memInfo, state->logits.data(),
                                                    state->logits.size() * sizeof(uint16_t),
                                                    shape.data(), shape.size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            state->binding->BindOutput("logits", v);
        }

        // RunWithBinding is the only overload that takes the pre-bound buffers; it also
        // requires a RunOptions, which we use to vote the same HTP performance profile the
        // Java path does (ORT gates its DSP queue polling on this run option).
        Ort::RunOptions runOptions;
        runOptions.AddConfigEntry("qnn.perf_mode", state->perfMode_.c_str());
        state->session->Run(runOptions, *state->binding);
        state->binding->SynchronizeOutputs();

        env->SetShortArrayRegion(logitsOut, 0, static_cast<jsize>(state->logits.size()),
                                 reinterpret_cast<const jshort*>(state->logits.data()));
        state->pingPong = writeSlot;
        state->steps++;
        state->lastStepMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        return JNI_TRUE;
    } catch (const std::exception& e) {
        LOGE("STEP failed token=%d position=%d: %s", inputToken, position, e.what());
        stageLog(std::string("step.failed token=") + std::to_string(inputToken) + " position=" +
                 std::to_string(position) + " " + e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jdouble JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeLastStepMs(JNIEnv*, jclass, jlong handle) {
    auto* state = reinterpret_cast<DecoderBinding*>(handle);
    return state ? state->lastStepMs : 0.0;
}

JNIEXPORT jdouble JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeLastBindMs(JNIEnv*, jclass, jlong handle) {
    auto* state = reinterpret_cast<DecoderBinding*>(handle);
    return state ? state->lastBindMs : 0.0;
}

JNIEXPORT jstring JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeDescribe(JNIEnv* env, jclass,
                                                                   jlong handle) {
    auto* state = reinterpret_cast<DecoderBinding*>(handle);
    if (!state || !state->session) return env->NewStringUTF("uninitialized");
    std::string s = "outputs=" + std::to_string(state->outputNames.size()) +
                    " selfSlots=" + std::to_string(state->selfSlots) +
                    " heads=" + std::to_string(state->nHeads) +
                    " vocab=" + std::to_string(state->vocabSize);
    return env->NewStringUTF(s.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_whisperapp_qnn_QnnWhisperDecoderJni_nativeClose(JNIEnv*, jclass, jlong handle) {
    auto* state = reinterpret_cast<DecoderBinding*>(handle);
    if (!state) return;
    // Order matters: the binding references the session's allocators.
    state->binding.reset();
    state->session.reset();
    delete state;
}

}  // extern "C"
