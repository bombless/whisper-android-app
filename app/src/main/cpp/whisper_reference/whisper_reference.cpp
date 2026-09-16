#include "whisper_reference.h"
#include "whisper.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ctime>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
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
extern "C" int64_t whisper_get_encoder_build_time_us(const whisper_context * ctx);
extern "C" int64_t whisper_get_encoder_compute_time_us(const whisper_context * ctx);
extern "C" const float * whisper_get_mel_data(const whisper_context * ctx, int64_t * elements, int * n_len, int * n_mel);
extern "C" const float * whisper_get_encoder_input_data(const whisper_context * ctx, int64_t * elements);

extern "C" const ggml_tensor * whisper_get_cross_k_cache(const whisper_context * ctx);
extern "C" const ggml_tensor * whisper_get_cross_v_cache(const whisper_context * ctx);

std::string dump_reference_cross_kv(const whisper_context * ctx, const std::string & dump_dir) {
    constexpr int n_layer = 4;
    constexpr int n_state = 384;
    constexpr int n_head = 6;
    constexpr int n_audio_ctx = 1500;
    constexpr int head_dim = 64;
    const ggml_tensor * cross_k = whisper_get_cross_k_cache(ctx);
    const ggml_tensor * cross_v = whisper_get_cross_v_cache(ctx);
    if (!cross_k || !cross_v) throw std::runtime_error("reference cross-K/V cache is unavailable");

    if (cross_k->type != GGML_TYPE_F16 || cross_v->type != GGML_TYPE_F16) throw std::runtime_error("reference cross-K/V cache is not FP16");
    const size_t logical_elements = static_cast<size_t>(n_state) * n_audio_ctx;
    const size_t padded_elements_per_layer = ggml_nelements(cross_k) / n_layer;
    const int padded_audio_ctx = static_cast<int>(padded_elements_per_layer / n_state);
    if (padded_audio_ctx < n_audio_ctx) throw std::runtime_error("reference cross-K/V cache is smaller than logical audio context");
    const size_t raw_layer_bytes = padded_elements_per_layer * ggml_element_size(cross_k);
    std::vector<uint8_t> raw_k(raw_layer_bytes), raw_v(raw_layer_bytes);
    std::filesystem::create_directories(dump_dir);
    std::ostringstream report;
    report << "REFERENCE_CROSS_KV_DUMP_START\\n"
           << "layers=4 state=384 heads=6 headDim=64 audioCtx=1500\\n"
           << "reference_K_logical_shape=[64,1500,6] axis=head_dim,sequence,head dtype=FP32\\n"
           << "reference_V_logical_shape=[1500,64,6] axis=sequence,head_dim,head dtype=FP32\\n"
           << "source_cache_layout=per_layer_flat; decoder_view_K=[64,1500,6]; decoder_view_V=[1500,64,6]\\n";

    for (int il = 0; il < n_layer; ++il) {
        const size_t layer_offset = static_cast<size_t>(il) * raw_layer_bytes;
        ggml_backend_tensor_get(cross_k, raw_k.data(), layer_offset, raw_layer_bytes);
        ggml_backend_tensor_get(cross_v, raw_v.data(), layer_offset, raw_layer_bytes);

        const std::string k_path = dump_dir + "/reference_cross_k_" + std::to_string(il) + ".bin";
        const std::string v_path = dump_dir + "/reference_cross_v_" + std::to_string(il) + ".bin";
        const std::string k_meta = dump_dir + "/reference_cross_k_" + std::to_string(il) + ".metadata";
        const std::string v_meta = dump_dir + "/reference_cross_v_" + std::to_string(il) + ".metadata";
        std::ofstream kout(k_path, std::ios::binary), vout(v_path, std::ios::binary);
        if (!kout || !vout) throw std::runtime_error("failed to create reference cross-K/V dump");

        std::vector<float> k_canonical(logical_elements), v_canonical(logical_elements);
        const ggml_fp16_t * k16 = reinterpret_cast<const ggml_fp16_t *>(raw_k.data());
        const ggml_fp16_t * v16 = reinterpret_cast<const ggml_fp16_t *>(raw_v.data());
        for (int h = 0; h < n_head; ++h) {
            for (int s = 0; s < n_audio_ctx; ++s) {
                for (int d = 0; d < head_dim; ++d) {
                    const size_t k_raw_idx = static_cast<size_t>(s) * n_state + static_cast<size_t>(h) * head_dim + d;
                    const size_t v_raw_idx = static_cast<size_t>(h) * n_audio_ctx * head_dim + static_cast<size_t>(d) * n_audio_ctx + s;
                    const size_t k_idx = static_cast<size_t>(d) * n_audio_ctx * n_head + static_cast<size_t>(s) * n_head + h;
                    const size_t v_idx = static_cast<size_t>(s) * head_dim * n_head + static_cast<size_t>(d) * n_head + h;
                    k_canonical[k_idx] = ggml_fp16_to_fp32(k16[k_raw_idx]);
                    v_canonical[v_idx] = ggml_fp16_to_fp32(v16[v_raw_idx]);
                }
            }
        }
        kout.write(reinterpret_cast<const char *>(k_canonical.data()), static_cast<std::streamsize>(k_canonical.size() * sizeof(float)));
        vout.write(reinterpret_cast<const char *>(v_canonical.data()), static_cast<std::streamsize>(v_canonical.size() * sizeof(float)));
        if (!kout || !vout) throw std::runtime_error("failed writing reference cross-K/V dump");

        std::ofstream km(k_meta), vm(v_meta);
        km << "layer=" << il << "\\ndtype=FP32\\nshape=[64,1500,6]\\nlogical_axis_order=head_dim,sequence,head\\nelements=" << logical_elements << "\\n";
        vm << "layer=" << il << "\\ndtype=FP32\\nshape=[1500,64,6]\\nlogical_axis_order=sequence,head_dim,head\\nelements=" << logical_elements << "\\n";
        if (!km || !vm) throw std::runtime_error("failed writing reference cross-K/V metadata");
        report << "layer=" << il << " k=" << k_path << " v=" << v_path << " elements=" << logical_elements << "\\n";
    }
    report << "REFERENCE_CROSS_KV_DUMP_END\\n";
    return report.str();
}

std::string inspect_float_checkpoint(const char * name, const float * data, size_t elements, int64_t ne0, int64_t ne1) {
    std::ostringstream report;
    size_t finite = 0, nan = 0, inf = 0;
    double min_value = std::numeric_limits<double>::infinity();
    double max_value = -std::numeric_limits<double>::infinity();
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < elements; ++i) {
        const float value = data[i];
        if (std::isnan(value)) ++nan;
        else if (std::isinf(value)) ++inf;
        else { ++finite; const double v = value; min_value = std::min(min_value, v); max_value = std::max(max_value, v); sum += v; sum_sq += v * v; }
    }
    const double mean = finite ? sum / static_cast<double>(finite) : std::numeric_limits<double>::quiet_NaN();
    const double rms = finite ? std::sqrt(sum_sq / static_cast<double>(finite)) : std::numeric_limits<double>::quiet_NaN();
    report << std::setprecision(9)
           << name << "_SHAPE ne0=" << ne0 << " ne1=" << ne1 << " elements=" << elements << "\\n"
           << name << "_MIN " << (finite ? min_value : std::numeric_limits<double>::quiet_NaN()) << "\\n"
           << name << "_MAX " << (finite ? max_value : std::numeric_limits<double>::quiet_NaN()) << "\\n"
           << name << "_MEAN " << mean << "\\n"
           << name << "_RMS " << rms << "\\n"
           << name << "_FINITE " << finite << "\\n"
           << name << "_NAN " << nan << "\\n"
           << name << "_INF " << inf << "\\n";
    return report.str();
}

std::string inspect_input_checkpoints(const whisper_context * ctx) {
    std::ostringstream report;
    int64_t mel_elements = 0; int mel_len = 0, mel_count = 0;
    const float * mel = whisper_get_mel_data(ctx, &mel_elements, &mel_len, &mel_count);
    report << "CHECKPOINT_MEL_START\\n";
    if (mel) report << inspect_float_checkpoint("CHECKPOINT_MEL", mel, static_cast<size_t>(mel_elements), mel_len, mel_count);
    else report << "CHECKPOINT_MEL_UNAVAILABLE\\n";
    report << "CHECKPOINT_MEL_END\\n";
    int64_t input_elements = 0;
    const float * input = whisper_get_encoder_input_data(ctx, &input_elements);
    report << "CHECKPOINT_ENCODER_INPUT_START\\n";
    if (input) report << inspect_float_checkpoint("CHECKPOINT_ENCODER_INPUT", input, static_cast<size_t>(input_elements), input_elements, 1);
    else report << "CHECKPOINT_ENCODER_INPUT_UNAVAILABLE\\n";
    report << "CHECKPOINT_ENCODER_INPUT_END\\n";
    return report.str();
}

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
    DIAG_LOG("ENCODER_NATIVE_DIAGNOSTIC_ENTER samples=%zu threads=%d", sample_count, threads);
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
        const std::string checkpoint_report = inspect_input_checkpoints(ctx);
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
        const int64_t build_us = whisper_get_encoder_build_time_us(ctx);
        const int64_t compute_us = whisper_get_encoder_compute_time_us(ctx);
        const double build_ms = static_cast<double>(build_us) / 1000.0;
        const double compute_ms = static_cast<double>(compute_us) / 1000.0;
        DIAG_LOG("ENCODER_BUILD_DONE elapsed_ms=%.3f", build_ms);
        DIAG_LOG("ENCODER_COMPUTE_DONE elapsed_ms=%.3f", compute_ms);
        DIAG_LOG("ENCODER_DONE rc=%d elapsed_ms=%lld", rc, static_cast<long long>(encoder_ms));
        std::ostringstream timing_report;
        timing_report << std::fixed << std::setprecision(3);
        timing_report << "ENCODER_BUILD_DONE elapsed_ms=" << build_ms << "\\n";
        timing_report << "ENCODER_COMPUTE_DONE elapsed_ms=" << compute_ms << "\\n";
        timing_report << "ENCODER_DONE rc=" << rc << " elapsed_ms=" << encoder_ms << "\\n";
        if (rc != 0) { DIAG_LOG("ENCODER_TEST_END result=FAILED"); whisper_free(ctx); return timing_report.str() + "ENCODER_TEST_END result=FAILED\\n"; }
        const std::string output_report = inspect_encoder_output(ctx);
        const std::time_t now = std::time(nullptr);
        std::tm tm_now{};
#if defined(_WIN32)
        localtime_s(&tm_now, &now);
#else
        localtime_r(&now, &tm_now);
#endif
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_now);
        const std::string dump_dir = model_path + "_reference_cross_kv_" + stamp;
        const std::string cross_report = dump_reference_cross_kv(ctx, dump_dir);
        DIAG_LOG("REFERENCE_CROSS_KV_DUMP_DONE dir=%s", dump_dir.c_str());
        DIAG_LOG("ENCODER_TEST_END result=SUCCESS");
        whisper_free(ctx);
        return checkpoint_report + timing_report.str() + output_report + cross_report + "ENCODER_TEST_END result=SUCCESS\\n";
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
Java_com_example_whisperapp_qnn_WhisperReferenceRunner_transcribeNative(JNIEnv * env, jobject, jshortArray pcm, jstring model_path, jint threads) {
    try {
        if (!pcm || !model_path) throw std::runtime_error("pcm/model_path is null");
        const jsize count = env->GetArrayLength(pcm);
        if (count != kReferenceSamples) throw std::runtime_error("PCM length must be 80000");
        const jshort * data = env->GetShortArrayElements(pcm, nullptr);
        if (!data) throw std::runtime_error("GetShortArrayElements failed");
        const char * path = env->GetStringUTFChars(model_path, nullptr);
        if (!path) throw std::runtime_error("GetStringUTFChars failed");
        std::vector<float> samples(kReferenceSamples);
        for (int i = 0; i < kReferenceSamples; ++i) samples[i] = static_cast<float>(data[i]) / 32768.0f;
        whisper_context_params ctx_params = whisper_context_default_params(); ctx_params.use_gpu = false;
        whisper_context * ctx = whisper_init_from_file_with_params(path, ctx_params);
        if (!ctx) throw std::runtime_error("whisper_init_from_file failed");
        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.print_progress = false; params.print_special = false; params.print_realtime = false; params.print_timestamps = false;
        params.translate = false; params.no_context = true; params.language = "en"; params.n_threads = std::max(1, static_cast<int>(threads));
        if (whisper_full(ctx, params, samples.data(), static_cast<int>(samples.size())) != 0) throw std::runtime_error("whisper_full failed");
        std::ostringstream out; out << "WHISPER_TRANSCRIBE_TEXT ";
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) { const char * text = whisper_full_get_segment_text(ctx, i); if (text) out << text; }
        out << "\nWHISPER_TRANSCRIBE_DONE segments=" << n_segments << "\n";
        whisper_free(ctx); env->ReleaseStringUTFChars(model_path, path); env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT);
        return env->NewStringUTF(out.str().c_str());
    } catch (const std::exception & e) { return env->NewStringUTF((std::string("WHISPER_TRANSCRIBE_FAIL ") + e.what()).c_str()); }
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_whisperapp_qnn_WhisperEncoderDiagnosticRunner_runNative(
        JNIEnv * env, jobject, jshortArray pcm, jstring model_path, jint threads) {
    DIAG_LOG("ENCODER_JNI_START threads=%d", threads);
    try {
        if (!pcm || !model_path) throw std::runtime_error("pcm/model_path is null");
        const jsize count = env->GetArrayLength(pcm);
        DIAG_LOG("ENCODER_JNI_PCM count=%d", static_cast<int>(count));
        if (count != kReferenceSamples) throw std::runtime_error("PCM length must be 80000");
        const jshort * data = env->GetShortArrayElements(pcm, nullptr);
        if (!data) throw std::runtime_error("GetShortArrayElements failed");
        DIAG_LOG("ENCODER_JNI_PCM_ACQUIRE_DONE");
        const char * path = env->GetStringUTFChars(model_path, nullptr);
        if (!path) { env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT); throw std::runtime_error("GetStringUTFChars failed"); }
        DIAG_LOG("ENCODER_JNI_MODEL_PATH_READY");
        std::string result;
        try {
            DIAG_LOG("ENCODER_JNI_NATIVE_CALL_START");
            result = run_encoder_diagnostic(reinterpret_cast<const int16_t *>(data), static_cast<size_t>(count), path, threads);
            DIAG_LOG("ENCODER_JNI_NATIVE_CALL_DONE reportChars=%zu", result.size());
        }
        catch (...) { env->ReleaseStringUTFChars(model_path, path); env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT); throw; }
        env->ReleaseStringUTFChars(model_path, path);
        env->ReleaseShortArrayElements(pcm, const_cast<jshort *>(data), JNI_ABORT);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception & e) {
        DIAG_LOG("ENCODER_JNI_EXCEPTION");
        DIAG_LOG("ENCODER_TEST_ERROR type=exception message=%s", e.what());
        return env->NewStringUTF((std::string("ENCODER_TEST_END result=FAILED error=") + e.what() + "\n").c_str());
    }
}
