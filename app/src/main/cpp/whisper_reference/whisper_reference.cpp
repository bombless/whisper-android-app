#include "whisper_reference.h"
#include "whisper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <android/log.h>

#define REF_LOG(...) __android_log_print(ANDROID_LOG_INFO, "WHISPER_REF_NATIVE", __VA_ARGS__)
#define DIAG_LOG(...) __android_log_print(ANDROID_LOG_INFO, "WHISPER_ENCODER_DIAG", __VA_ARGS__)

namespace {
constexpr int kSampleRate = 16000;
constexpr int kReferenceSamples = 80000;
constexpr int kModelSamples = 480000;
constexpr int kSteps = 5;
constexpr int kTopK = 5;
constexpr whisper_token kEos = 50257;
constexpr whisper_token kPrompt[] = {50258, 50259, 50359, 50363};

struct Pair { int id; float value; };

std::vector<Pair> top_k(const float * logits, int vocab) {
    std::vector<Pair> out;
    out.reserve(vocab);
    for (int i = 0; i < vocab; ++i) {
        if (std::isfinite(logits[i])) out.push_back({i, logits[i]});
    }
    const int take = std::min(kTopK, static_cast<int>(out.size()));
    std::partial_sort(out.begin(), out.begin() + take, out.end(), [](const Pair & a, const Pair & b) {
        return a.value > b.value;
    });
    out.resize(take);
    return out;
}

extern "C" const ggml_tensor * whisper_get_encoder_output(const whisper_context * ctx);

std::string inspect_encoder_output(const whisper_context * ctx) {
    const ggml_tensor * tensor = whisper_get_encoder_output(ctx);
    std::ostringstream report;
    report << "ENCODER_OUTPUT_START\\n";
    DIAG_LOG("ENCODER_OUTPUT_START");
    if (!tensor) {
        report << "ENCODER_OUTPUT_UNAVAILABLE\\nENCODER_OUTPUT_END\\n";
        DIAG_LOG("ENCODER_OUTPUT_UNAVAILABLE");
        DIAG_LOG("ENCODER_OUTPUT_END");
        return report.str();
    }

    const size_t elements = ggml_nelements(tensor);
    report << "ENCODER_OUTPUT_SHAPE ne0=" << tensor->ne[0] << " ne1=" << tensor->ne[1]
           << " ne2=" << tensor->ne[2] << " ne3=" << tensor->ne[3] << "\\n";
    report << "ENCODER_OUTPUT_TYPE " << ggml_type_name(tensor->type) << "\\n";
    report << "ENCODER_OUTPUT_ELEMENTS " << elements << "\\n";
    report << "ENCODER_OUTPUT_CONTIGUOUS " << (ggml_is_contiguous(tensor) ? "true" : "false") << "\\n";
    DIAG_LOG("ENCODER_OUTPUT_SHAPE ne0=%lld ne1=%lld ne2=%lld ne3=%lld",
             static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
             static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]));
    DIAG_LOG("ENCODER_OUTPUT_TYPE %s", ggml_type_name(tensor->type));
    DIAG_LOG("ENCODER_OUTPUT_ELEMENTS %zu", elements);
    DIAG_LOG("ENCODER_OUTPUT_CONTIGUOUS %s", ggml_is_contiguous(tensor) ? "true" : "false");

    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        report << "ENCODER_OUTPUT_TYPE_UNSUPPORTED " << ggml_type_name(tensor->type) << "\\nENCODER_OUTPUT_END\\n";
        DIAG_LOG("ENCODER_OUTPUT_TYPE_UNSUPPORTED %s", ggml_type_name(tensor->type));
        DIAG_LOG("ENCODER_OUTPUT_END");
        return report.str();
    }

    const size_t tensor_bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> cpu_buffer(tensor_bytes);
    if (tensor_bytes != 0) {
        ggml_backend_tensor_get(tensor, cpu_buffer.data(), 0, tensor_bytes);
    }

    size_t finite = 0;
    size_t nan = 0;
    size_t inf = 0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t i = 0; i < elements; ++i) {
        float value;
        if (tensor->type == GGML_TYPE_F32) {
            value = reinterpret_cast<const float *>(cpu_buffer.data())[i];
        } else {
            value = ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(cpu_buffer.data())[i]);
        }

        if (std::isnan(value)) {
            ++nan;
        } else if (std::isinf(value)) {
            ++inf;
        } else {
            ++finite;
            const double v = static_cast<double>(value);
            min_value = std::min(min_value, v);
            max_value = std::max(max_value, v);
            sum += v;
            sum_sq += v * v;
        }
    }

    const double mean = finite > 0 ? sum / static_cast<double>(finite) : std::numeric_limits<double>::quiet_NaN();
    const double rms = finite > 0 ? std::sqrt(sum_sq / static_cast<double>(finite)) : std::numeric_limits<double>::quiet_NaN();
    if (finite == 0) {
        min_value = std::numeric_limits<double>::quiet_NaN();
        max_value = std::numeric_limits<double>::quiet_NaN();
    }

    report << std::setprecision(9)
           << "ENCODER_OUTPUT_MIN " << min_value << "\\n"
           << "ENCODER_OUTPUT_MAX " << max_value << "\\n"
           << "ENCODER_OUTPUT_MEAN " << mean << "\\n"
           << "ENCODER_OUTPUT_RMS " << rms << "\\n"
           << "ENCODER_OUTPUT_FINITE " << finite << "\\n"
           << "ENCODER_OUTPUT_NAN " << nan << "\\n"
           << "ENCODER_OUTPUT_INF " << inf << "\\n"
           << "ENCODER_OUTPUT_END\\n";
    DIAG_LOG("ENCODER_OUTPUT_MIN %.9g", min_value);
    DIAG_LOG("ENCODER_OUTPUT_MAX %.9g", max_value);
    DIAG_LOG("ENCODER_OUTPUT_MEAN %.9g", mean);
    DIAG_LOG("ENCODER_OUTPUT_RMS %.9g", rms);
    DIAG_LOG("ENCODER_OUTPUT_FINITE %zu", finite);
    DIAG_LOG("ENCODER_OUTPUT_NAN %zu", nan);
    DIAG_LOG("ENCODER_OUTPUT_INF %zu", inf);
    DIAG_LOG("ENCODER_OUTPUT_END");
    return report.str();
}

std::string run_reference(const int16_t * samples, size_t sample_count, const std::string & model_path, int threads) {
    if (sample_count != kReferenceSamples) {
        throw std::runtime_error("reference requires exactly 80000 PCM16 samples");
    }
    if (threads < 1) threads = 1;

    std::vector<float> pcm(kModelSamples, 0.0f);
    for (size_t i = 0; i < sample_count; ++i) pcm[i] = static_cast<float>(samples[i]) / 32768.0f;

    double sum = 0.0;
    double squares = 0.0;
    int min_sample = std::numeric_limits<int>::max();
    int max_sample = std::numeric_limits<int>::min();
    for (size_t i = 0; i < sample_count; ++i) {
        const int v = samples[i];
        min_sample = std::min(min_sample, v);
        max_sample = std::max(max_sample, v);
        sum += v;
        squares += static_cast<double>(v) * static_cast<double>(v);
    }

    std::ostringstream out;
    out << std::setprecision(9);
    out << "WHISPER_REF_START\n";
    out << "WHISPER_REF_INPUT samples=80000 paddedSamples=480000 sampleRate=16000 mono=true\n";
    out << "WHISPER_REF_PCM_STATS min=" << min_sample
        << " max=" << max_sample
        << " mean=" << (sum / sample_count)
        << " rms=" << std::sqrt(squares / sample_count) << "\n";
    out << "WHISPER_REF_MODEL path=\"" << model_path << "\"\n";

    REF_LOG("INIT_START model=%s threads=%d", model_path.c_str(), threads);
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), params);
    REF_LOG("INIT_DONE ok=%d", ctx != nullptr ? 1 : 0);
    if (!ctx) throw std::runtime_error("whisper_init_from_file failed");

    try {
        out << "WHISPER_REF_MODEL_INFO version=" << whisper_version()
            << " vocab=" << whisper_n_vocab(ctx)
            << " textCtx=" << whisper_n_text_ctx(ctx)
            << " audioCtx=" << whisper_n_audio_ctx(ctx)
            << " eot=" << whisper_token_eot(ctx)
            << " sot=" << whisper_token_sot(ctx) << "\n";

        REF_LOG("MEL_START samples=%d", kModelSamples);
        if (whisper_pcm_to_mel(ctx, pcm.data(), static_cast<int>(pcm.size()), threads) != 0) {
            throw std::runtime_error("whisper_pcm_to_mel failed");
        }
        REF_LOG("MEL_DONE");
        out << "WHISPER_REF_MEL_READY\n";
        REF_LOG("ENCODER_START");
        if (whisper_encode(ctx, 0, threads) != 0) throw std::runtime_error("whisper_encode failed");
        REF_LOG("ENCODER_DONE");
        out << "WHISPER_REF_ENCODER_DONE\n";

        int next_token = -1;
        const int vocab = whisper_n_vocab(ctx);
        for (int step = 0; step < kSteps; ++step) {
            const whisper_token input = step < 4 ? kPrompt[step] : static_cast<whisper_token>(next_token);
            REF_LOG("DECODE_START step=%d input=%d", step, static_cast<int>(input));
            if (whisper_decode(ctx, &input, 1, step, threads) != 0) {
                throw std::runtime_error("whisper_decode failed at step " + std::to_string(step));
            }
            const float * logits = whisper_get_logits(ctx);
            if (!logits) throw std::runtime_error("whisper_get_logits returned null");

            int top1 = 0;
            float max_logit = -std::numeric_limits<float>::infinity();
            float min_logit = std::numeric_limits<float>::infinity();
            int finite = 0;
            for (int i = 0; i < vocab; ++i) {
                if (!std::isfinite(logits[i])) continue;
                ++finite;
                if (logits[i] > max_logit) { max_logit = logits[i]; top1 = i; }
                min_logit = std::min(min_logit, logits[i]);
            }
            const auto top = top_k(logits, vocab);
            next_token = top1;
            REF_LOG("DECODE_DONE step=%d top1=%d max=%f eos=%f", step, top1, max_logit, logits[kEos]);

            out << "WHISPER_REF_STEP step=" << step
                << " inputToken=" << input
                << " position=" << step
                << " top1=" << top1
                << " eosLogit=" << logits[kEos]
                << " maxLogit=" << max_logit
                << " minLogit=" << min_logit
                << " finite=" << finite << "\n";
            out << "WHISPER_REF_TOPK step=" << step;
            for (const auto & p : top) out << " " << p.id << ":" << p.value;
            out << "\n";
        }
        out << "WHISPER_REF_DONE\n";
        whisper_free(ctx);
        return out.str();
    } catch (...) {
        whisper_free(ctx);
        throw;
    }
}
std::string run_encoder_diagnostic(const int16_t * samples, size_t sample_count, const std::string & model_path, int threads) {
    if (sample_count != kReferenceSamples) throw std::runtime_error("diagnostic requires exactly 80000 PCM16 samples");
    if (threads < 1) threads = 1;
    using clock = std::chrono::steady_clock;
    const auto test_start = clock::now();
    auto elapsed_ms = [&]() -> long long { return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - test_start).count(); };
    std::vector<float> pcm(kModelSamples, 0.0f);
    for (size_t i = 0; i < sample_count; ++i) pcm[i] = static_cast<float>(samples[i]) / 32768.0f;
    DIAG_LOG("ENCODER_TEST_START");
    DIAG_LOG("ENCODER_MODEL_LOAD_START");
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), params);
    if (!ctx) throw std::runtime_error("whisper_init_from_file failed");
    DIAG_LOG("ENCODER_MODEL_LOAD_DONE elapsed_ms=%lld", elapsed_ms());
    try {
        DIAG_LOG("ENCODER_CONTEXT_INIT_START");
        DIAG_LOG("ENCODER_CONTEXT_INIT_DONE elapsed_ms=%lld vocab=%d audioCtx=%d", elapsed_ms(), whisper_n_vocab(ctx), whisper_n_audio_ctx(ctx));
        DIAG_LOG("ENCODER_PCM_START samples=80000 paddedSamples=480000 sampleRate=16000 threads=%d", threads);
        DIAG_LOG("ENCODER_PCM_DONE elapsed_ms=%lld", elapsed_ms());
        DIAG_LOG("ENCODER_MEL_START samples=480000");
        const auto mel_start = clock::now();
        const int mel_rc = whisper_pcm_to_mel(ctx, pcm.data(), static_cast<int>(pcm.size()), threads);
        const auto mel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - mel_start).count();
        if (mel_rc != 0) throw std::runtime_error("whisper_pcm_to_mel failed");
        DIAG_LOG("ENCODER_MEL_DONE elapsed_ms=%lld stage_ms=%lld", elapsed_ms(), static_cast<long long>(mel_ms));
        DIAG_LOG("ENCODER_START");
        std::atomic<bool> encoder_running{true};
        std::thread watchdog([&]() {
            while (encoder_running.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                if (!encoder_running.load(std::memory_order_relaxed)) break;
                DIAG_LOG("ENCODER_HEARTBEAT elapsed_ms=%lld", elapsed_ms());
            }
        });
        const auto encoder_start = clock::now();
        int rc = whisper_encode(ctx, 0, threads);
        encoder_running.store(false, std::memory_order_relaxed);
        watchdog.join();
        const auto encoder_ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - encoder_start).count();
        DIAG_LOG("ENCODER_DONE rc=%d elapsed_ms=%lld", rc, static_cast<long long>(encoder_ms));
        if (rc != 0) { DIAG_LOG("ENCODER_TEST_END result=FAILED"); whisper_free(ctx); return "ENCODER_TEST_END result=FAILED\\n"; }
        const std::string output_report = inspect_encoder_output(ctx);
        DIAG_LOG("ENCODER_TEST_END result=SUCCESS");
        whisper_free(ctx);
        return output_report + "ENCODER_TEST_END result=SUCCESS\\n";
    } catch (...) {
        whisper_free(ctx);
        DIAG_LOG("ENCODER_TEST_END result=FAILED");
        throw;
    }
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_whisperapp_qnn_WhisperReferenceRunner_runNative(
        JNIEnv * env,
        jobject,
        jshortArray pcm,
        jstring model_path,
        jint threads) {
    try {
        if (!pcm || !model_path) throw std::runtime_error("pcm/model_path is null");
        const jsize count = env->GetArrayLength(pcm);
        if (count != kReferenceSamples) throw std::runtime_error("PCM length must be 80000");
        const jshort * data = env->GetShortArrayElements(pcm, nullptr);
        if (!data) throw std::runtime_error("GetShortArrayElements failed");
        const char * path = env->GetStringUTFChars(model_path, nullptr);
        if (!path) {
            env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT);
            throw std::runtime_error("GetStringUTFChars failed");
        }
        std::string result;
        try {
            result = run_reference(reinterpret_cast<const int16_t *>(data), static_cast<size_t>(count), path, threads);
        } catch (...) {
            env->ReleaseStringUTFChars(model_path, path);
            env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT);
            throw;
        }
        env->ReleaseStringUTFChars(model_path, path);
        env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception & e) {
        return env->NewStringUTF((std::string("WHISPER_REF_FAIL ") + e.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_whisperapp_qnn_WhisperEncoderDiagnosticRunner_runNative(
        JNIEnv * env, jobject, jshortArray pcm, jstring model_path, jint threads) {
    try {
        if (!pcm || !model_path) throw std::runtime_error("pcm/model_path is null");
        const jsize count = env->GetArrayLength(pcm);
        if (count != kReferenceSamples) throw std::runtime_error("PCM length must be 80000");
        const jshort * data = env->GetShortArrayElements(pcm, nullptr);
        if (!data) throw std::runtime_error("GetShortArrayElements failed");
        const char * path = env->GetStringUTFChars(model_path, nullptr);
        if (!path) { env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT); throw std::runtime_error("GetStringUTFChars failed"); }
        std::string result;
        try { result = run_encoder_diagnostic(reinterpret_cast<const int16_t *>(data), static_cast<size_t>(count), path, threads); }
        catch (...) { env->ReleaseStringUTFChars(model_path, path); env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT); throw; }
        env->ReleaseStringUTFChars(model_path, path);
        env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception & e) {
        DIAG_LOG("ENCODER_TEST_ERROR type=exception message=%s", e.what());
        return env->NewStringUTF((std::string("ENCODER_TEST_END result=FAILED error=") + e.what() + "\n").c_str());
    }
}
