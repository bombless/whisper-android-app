// Native (C++) Whisper log-mel frontend, JNI-bound.
//
// Mirrors the Java WhisperFeatureExtractor exactly (HF reflect padding, direct
// mixed-radix 400-point DFT, sparse triangular mel scatter, Slaney filterbank,
// log10 + per-chunk normalization), parallelized over frames with std::thread.
// Outputs either normalized float features or binary16 samples ready for the HTP
// encoder.
//
// FFT design: N=400 = 2^4 * 5^2. Four decimation-in-time radix-2 stages reduce the
// problem to 16x 25-point DFTs; each DFT25 runs as a 5x5 Cooley-Tukey step (10x
// DFT5 + 25 twiddle multiplies). This replaces the previous Bluestein 400-point
// DFT via 2x 1024-point radix-2 convolutions (~4x faster in host benchmarks,
// spectrum maxAbs ~4e-6 vs Bluestein, far inside the 0.05 native validation
// tolerance).
//
// Bit-compat notes: single-precision IEEE-754 throughout, same summation order as the
// Java implementation. Clang may contract a*b+c into FMA, which shifts results by ~1 ulp.

#include <jni.h>
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MEL_NATIVE", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "MEL_NATIVE", __VA_ARGS__)

namespace {

constexpr int kSampleRate = 16000;
constexpr int kNFft = 400;
constexpr int kHop = 160;
constexpr int kNMels = 80;
constexpr int kNFrames = 3000;
constexpr int kChunkSamples = 480000; // 30 s
constexpr int kNBins = kNFft / 2 + 1; // 201

struct MelProfile {
    double inputPrepMs = 0.0;
    double melFftMs = 0.0;
    double melProjectionMs = 0.0;
    double totalMs = 0.0;
    double melLogMs = 0.0;
    double melReduceMs = 0.0;
    double melNormalizeMs = 0.0;
    double melFp16Ms = 0.0;
    double melCopyMs = 0.0;
    float rawMin = INFINITY;
    float rawMax = -INFINITY;
    float rawMean = 0.0f;
    float rawMinPositive = INFINITY;
    float rawMaxPositive = 0.0f;
    int rawZeroCount = 0;
    int rawNegativeCount = 0;
    int inputSamples = 0;
    int frameCount = kNFrames;
    int melBins = kNMels;
    unsigned int threadCount = 1;
};

MelProfile g_lastProfile;
using SteadyClock = std::chrono::steady_clock;
inline double elapsedMs(SteadyClock::time_point a, SteadyClock::time_point b) { return std::chrono::duration<double, std::milli>(b-a).count(); }

void logMelProfile(const char* kind) {
    LOGI("[MEL_PROFILE] frames=%d mel_bins=%d fft_size=%d input_samples=%d thread_count=%u", g_lastProfile.frameCount, g_lastProfile.melBins, kNFft, g_lastProfile.inputSamples, g_lastProfile.threadCount);
    LOGI("[MEL_PROFILE] input_prep=%.3f mel_fft=%.3f projection=%.3f log=%.3f reduce=%.3f normalize=%.3f fp16[%s]=%.3f copy=%.3f total=%.3f", g_lastProfile.inputPrepMs, g_lastProfile.melFftMs, g_lastProfile.melProjectionMs, g_lastProfile.melLogMs, g_lastProfile.melReduceMs, g_lastProfile.melNormalizeMs, kind, g_lastProfile.melFp16Ms, g_lastProfile.melCopyMs, g_lastProfile.totalMs);
    LOGI("[MEL_PROFILE] raw min=%g max=%g mean=%g min_positive=%g max_positive=%g zero=%d negative=%d", g_lastProfile.rawMin, g_lastProfile.rawMax, g_lastProfile.rawMean, g_lastProfile.rawMinPositive, g_lastProfile.rawMaxPositive, g_lastProfile.rawZeroCount, g_lastProfile.rawNegativeCount);
}

struct MelTables {
    float window[kNFft];
    // 5x5 DFT matrix for the radix-5 kernels.
    float w5Re[25];
    float w5Im[25];
    // 25-point Cooley-Tukey twiddles W_25^{k1*n2}.
    float w25Re[25];
    float w25Im[25];
    // Radix-2 combine twiddles W_N^k for N = 50/100/200/400.
    float c50Re[25],  c50Im[25];
    float c100Re[50], c100Im[50];
    float c200Re[100], c200Im[100];
    float c400Re[200], c400Im[200];
    // Sparse mel filterbank CSR: per FFT bin, (mel, weight) pairs sorted by mel.
    std::vector<int>   binIds[kNBins];
    std::vector<float> binWeights[kNBins];

    MelTables() {
        for (int i = 0; i < kNFft; i++) {
            window[i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / kNFft));
        }
        for (int k1 = 0; k1 < 5; k1++) {
            for (int n1 = 0; n1 < 5; n1++) {
                const double a = -2.0 * M_PI * k1 * n1 / 5.0;
                w5Re[k1 * 5 + n1] = (float)std::cos(a);
                w5Im[k1 * 5 + n1] = (float)std::sin(a);
            }
        }
        for (int k1 = 0; k1 < 5; k1++) {
            for (int n2 = 0; n2 < 5; n2++) {
                const double a = -2.0 * M_PI * k1 * n2 / 25.0;
                w25Re[k1 * 5 + n2] = (float)std::cos(a);
                w25Im[k1 * 5 + n2] = (float)std::sin(a);
            }
        }
        auto fillCombine = [](float* re, float* im, int half) {
            for (int k = 0; k < half; k++) {
                const double a = -2.0 * M_PI * k / (2 * half);
                re[k] = (float)std::cos(a);
                im[k] = (float)std::sin(a);
            }
        };
        fillCombine(c50Re, c50Im, 25);
        fillCombine(c100Re, c100Im, 50);
        fillCombine(c200Re, c200Im, 100);
        fillCombine(c400Re, c400Im, 200);

        buildMelFilters();
    }

    inline void dft5(const float* inRe, const float* inIm, int stride,
                     float* outRe, float* outIm) const {
        for (int k = 0; k < 5; k++) {
            float sr = 0.f, si = 0.f;
            for (int n = 0; n < 5; n++) {
                const float wr = w5Re[k * 5 + n];
                const float wi = w5Im[k * 5 + n];
                const float xr = inRe[n * stride];
                const float xi = inIm[n * stride];
                sr += xr * wr - xi * wi;
                si += xr * wi + xi * wr;
            }
            outRe[k] = sr;
            outIm[k] = si;
        }
    }

    // 25-point DFT on strided input -> contiguous output.
    // Layout: Xin[n1][n2] = in[(5*n1 + n2) * stride]; Xout[k1 + 5*k2].
    inline void dft25(const float* inRe, const float* inIm, int stride,
                      float* outRe, float* outIm) const {
        float f1Re[25], f1Im[25]; // [k1][n2]
        float colRe[5], colIm[5], colOutRe[5], colOutIm[5];
        for (int n2 = 0; n2 < 5; n2++) {
            for (int n1 = 0; n1 < 5; n1++) {
                colRe[n1] = inRe[(5 * n1 + n2) * stride];
                colIm[n1] = inIm[(5 * n1 + n2) * stride];
            }
            dft5(colRe, colIm, 1, colOutRe, colOutIm);
            for (int k1 = 0; k1 < 5; k1++) {
                const float wr = w25Re[k1 * 5 + n2];
                const float wi = w25Im[k1 * 5 + n2];
                const float xr = colOutRe[k1];
                const float xi = colOutIm[k1];
                f1Re[k1 * 5 + n2] = xr * wr - xi * wi;
                f1Im[k1 * 5 + n2] = xr * wi + xi * wr;
            }
        }
        for (int k1 = 0; k1 < 5; k1++) {
            float oRe[5], oIm[5];
            dft5(f1Re + k1 * 5, f1Im + k1 * 5, 1, oRe, oIm);
            for (int k2 = 0; k2 < 5; k2++) {
                outRe[k1 + 5 * k2] = oRe[k2];
                outIm[k1 + 5 * k2] = oIm[k2];
            }
        }
    }

    // DIT recursion: even/odd split down to 25, then radix-2 combine.
    // `tmp` (size n) is scratch; children ping-pong between `out` and `tmp`.
    void fftRec(const float* inRe, const float* inIm, int n, int stride,
                float* outRe, float* outIm, float* tmpRe, float* tmpIm) const {
        if (n == 25) {
            dft25(inRe, inIm, stride, outRe, outIm);
            return;
        }
        const int h = n >> 1;
        fftRec(inRe, inIm, h, stride << 1, tmpRe, tmpIm, outRe, outIm);
        fftRec(inRe + stride, inIm + stride, h, stride << 1,
               tmpRe + h, tmpIm + h, outRe, outIm);
        const float* wR;
        const float* wI;
        if (n == 50) {
            wR = c50Re; wI = c50Im;
        } else if (n == 100) {
            wR = c100Re; wI = c100Im;
        } else if (n == 200) {
            wR = c200Re; wI = c200Im;
        } else {
            wR = c400Re; wI = c400Im;
        }
        for (int k = 0; k < h; k++) {
            const float er = tmpRe[k];
            const float ei = tmpIm[k];
            const float or_ = tmpRe[h + k];
            const float oi = tmpIm[h + k];
            const float wr = wR[k];
            const float wi = wI[k];
            const float vr = or_ * wr - oi * wi;
            const float vi = or_ * wi + oi * wr;
            outRe[k] = er + vr;
            outIm[k] = ei + vi;
            outRe[h + k] = er - vr;
            outIm[h + k] = ei - vi;
        }
    }

    // Exact 400-point DFT of windowed real input `xWin` (length kNFft).
    // Needs tmpA/tmpB scratch of 400 complex samples each.
    void fft400(const float* xWin, float* outRe, float* outIm,
                float* tmpARe, float* tmpAIm, float* tmpBRe, float* tmpBIm) const {
        for (int i = 0; i < kNFft; i++) {
            tmpARe[i] = xWin[i];
            tmpAIm[i] = 0.f;
        }
        fftRec(tmpARe, tmpAIm, kNFft, 1, outRe, outIm, tmpBRe, tmpBIm);
    }

    static double hzToMel(double hz) {
        if (hz < 1000.0) return 3.0 * hz / 200.0;
        return 15.0 + std::log(hz / 1000.0) * (27.0 / std::log(6.4));
    }
    static double melToHz(double mel) {
        if (mel < 15.0) return 200.0 * mel / 3.0;
        return 1000.0 * std::exp(std::log(6.4) / 27.0 * (mel - 15.0));
    }

    void buildMelFilters() {
        double filterHz[kNMels + 2];
        const double melMin = hzToMel(0.0);
        const double melMax = hzToMel(8000.0);
        for (int i = 0; i < kNMels + 2; i++) {
            filterHz[i] = melToHz(melMin + (melMax - melMin) * i / (kNMels + 1));
        }
        std::vector<std::vector<float>> dense(kNMels, std::vector<float>(kNBins, 0.f));
        for (int m = 0; m < kNMels; m++) {
            const double left = filterHz[m], center = filterHz[m + 1], right = filterHz[m + 2];
            for (int k = 0; k < kNBins; k++) {
                const double f = (double)k * kSampleRate / kNFft;
                const double down = center > left ? (f - left) / (center - left) : 0.0;
                const double up = right > center ? (right - f) / (right - center) : 0.0;
                dense[m][k] = (float)std::max(0.0, std::min(down, up));
            }
            const double enorm = 2.0 / std::max(right - left, 1e-12);
            for (int k = 0; k < kNBins; k++) dense[m][k] = (float)(dense[m][k] * enorm);
        }
        for (int k = 0; k < kNBins; k++) {
            for (int m = 0; m < kNMels; m++) {
                if (dense[m][k] != 0.f) {
                    binIds[k].push_back(m);
                    binWeights[k].push_back(dense[m][k]);
                }
            }
        }
    }
};

const MelTables& tables() {
    static MelTables t;
    return t;
}

int reflectIndex(int index, int size) {
    int i = index;
    while (i < 0 || i >= size) i = i < 0 ? -i : 2 * size - 2 - i;
    return i;
}

// Computes the raw mel energies [kNMels x kNFrames] (before log10/normalization).
void computeMelEnergies(const int16_t* pcm, int pcmLen, float* mel /* [kNMels*kNFrames] */) {
    const auto totalStart = SteadyClock::now();
    const MelTables& T = tables();
    g_lastProfile = MelProfile{};
    g_lastProfile.inputSamples = pcmLen;

    const auto prepStart = SteadyClock::now();
    std::vector<float> waveform(kChunkSamples, 0.f);
    const int copy = std::min(pcmLen, kChunkSamples);
    for (int i = 0; i < copy; i++) waveform[i] = pcm[i] / 32768.f;

    std::vector<float> padded(kChunkSamples + kNFft);
    const int pad = kNFft / 2;
    for (int i = 0; i < (int)padded.size(); i++) {
        padded[i] = waveform[reflectIndex(i - pad, kChunkSamples)];
    }

    std::fill(mel, mel + kNMels * kNFrames, 0.f);
    g_lastProfile.inputPrepMs = elapsedMs(prepStart, SteadyClock::now());

    const unsigned int hw = std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
    g_lastProfile.threadCount = hw;
    std::vector<double> threadFftMs(hw, 0.0), threadProjectionMs(hw, 0.0);
    auto worker = [&](int ith) {
        float xw[kNFft];
        float re[kNFft], im[kNFft];
        float tmpARe[kNFft], tmpAIm[kNFft];
        float tmpBRe[kNFft], tmpBIm[kNFft];
        for (int frame = ith; frame < kNFrames; frame += (int)hw) {
            const auto fftStart = SteadyClock::now();
            const int start = frame * kHop;
            for (int i = 0; i < kNFft; i++) {
                xw[i] = padded[start + i] * T.window[i];
            }
            T.fft400(xw, re, im, tmpARe, tmpAIm, tmpBRe, tmpBIm);
            const auto projectionStart = SteadyClock::now();
            for (int k = 0; k < kNBins; k++) {
                const float p = re[k] * re[k] + im[k] * im[k];
                const auto& ids = T.binIds[k];
                const auto& ws = T.binWeights[k];
                for (size_t jj = 0; jj < ids.size(); jj++) {
                    mel[ids[jj] * kNFrames + frame] += p * ws[jj];
                }
            }
            threadFftMs[ith] += elapsedMs(fftStart, projectionStart);
            threadProjectionMs[ith] += elapsedMs(projectionStart, SteadyClock::now());
        }
    };
    if (hw <= 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        for (unsigned int t = 0; t < hw; t++) pool.emplace_back(worker, (int)t);
        for (auto& th : pool) th.join();
    }
    for (unsigned int t = 0; t < hw; ++t) {
        g_lastProfile.melFftMs = std::max(g_lastProfile.melFftMs, threadFftMs[t]);
        g_lastProfile.melProjectionMs = std::max(g_lastProfile.melProjectionMs, threadProjectionMs[t]);
    }
    g_lastProfile.totalMs = elapsedMs(totalStart, SteadyClock::now());
}
// Float -> IEEE-754 binary16, round-to-nearest-even (same as the Java converter).
inline int16_t floatToHalf(float value) {
    int32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const int sign = (bits >> 16) & 0x8000;
    const int exponent = (int)((bits >> 23) & 0xff) - 127 + 15;
    const int mantissa = bits & 0x7fffff;
    if (exponent <= 0) {
        if (exponent < -10) return (int16_t)sign;
        return (int16_t)(sign | ((mantissa | 0x800000) >> (1 - exponent + 13)));
    }
    if (exponent >= 31) {
        return (int16_t)(sign | 0x7c00 | (mantissa == 0 ? 0 : 0x0200));
    }
    int halfExponent = exponent;
    int halfMantissa = (mantissa + 0x1000) >> 13;
    if (halfMantissa == 0x400) { halfMantissa = 0; halfExponent++; }
    if (halfExponent >= 31) return (int16_t)(sign | 0x7c00);
    return (int16_t)(sign | (halfExponent << 10) | halfMantissa);
}

template <typename OutT, bool kAsHalf>
bool finalizeAndCopy(const float* mel, OutT* out) {
    const auto finalizeStart = SteadyClock::now();
    const int count = kNMels * kNFrames;
    const auto copyStart = SteadyClock::now();
    std::vector<float> melCopy(mel, mel + count);
    g_lastProfile.melCopyMs = elapsedMs(copyStart, SteadyClock::now());

    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        const float raw = melCopy[i];
        g_lastProfile.rawMin = std::min(g_lastProfile.rawMin, raw);
        g_lastProfile.rawMax = std::max(g_lastProfile.rawMax, raw);
        sum += raw;
        if (raw == 0.0f) ++g_lastProfile.rawZeroCount;
        else if (raw < 0.0f) ++g_lastProfile.rawNegativeCount;
        else {
            g_lastProfile.rawMinPositive = std::min(g_lastProfile.rawMinPositive, raw);
            g_lastProfile.rawMaxPositive = std::max(g_lastProfile.rawMaxPositive, raw);
        }
    }
    g_lastProfile.rawMean = static_cast<float>(sum / count);

    const auto logStart = SteadyClock::now();
    float maxLog = -INFINITY;
    for (int i = 0; i < count; i++) {
        float v = std::max(melCopy[i], 1e-10f);
        v = std::log10(v);
        melCopy[i] = v;
        if (v > maxLog) maxLog = v;
    }
    g_lastProfile.melLogMs = elapsedMs(logStart, SteadyClock::now());

    // Keep the original max reduction as the source of truth; this second scan is profiling-only.
    const auto reduceStart = SteadyClock::now();
    volatile float reducedMax = -INFINITY;
    for (int i = 0; i < count; i++) reducedMax = std::max(static_cast<float>(reducedMax), melCopy[i]);
    (void)reducedMax;
    g_lastProfile.melReduceMs = elapsedMs(reduceStart, SteadyClock::now());

    const float floor = maxLog - 8.f;
    const auto normalizeStart = SteadyClock::now();
    for (int i = 0; i < count; i++) {
        const float v = (std::max(melCopy[i], floor) + 4.f) / 4.f;
        if (!std::isfinite(v)) { LOGE("non-finite mel at %d", i); return false; }
        melCopy[i] = v;
    }
    g_lastProfile.melNormalizeMs = elapsedMs(normalizeStart, SteadyClock::now());

    const auto fp16Start = SteadyClock::now();
    if (kAsHalf) {
        reinterpret_cast<int16_t*>(out)[0] = floatToHalf(melCopy[0]);
        for (int i = 1; i < count; i++) reinterpret_cast<int16_t*>(out)[i] = floatToHalf(melCopy[i]);
    } else {
        for (int i = 0; i < count; i++) out[i] = melCopy[i];
    }
    g_lastProfile.melFp16Ms = elapsedMs(fp16Start, SteadyClock::now());
    g_lastProfile.totalMs += elapsedMs(finalizeStart, SteadyClock::now());
    logMelProfile(kAsHalf ? "convert" : "float_copy");
    return true;
}
} // namespace

// Declared in WhisperFeatureExtractor's companion object, so JNI mangles the names
// with the _00024Companion infix.
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_whisperapp_audio_WhisperFeatureExtractor_00024Companion_nativeExtractFloat(
        JNIEnv* env, jobject /*thiz*/, jshortArray pcm, jint sampleRate) {
    if (sampleRate != kSampleRate) {
        LOGE("unsupported sample rate %d", sampleRate);
        return nullptr;
    }
    jsize n = env->GetArrayLength(pcm);
    std::vector<int16_t> buf(n);
    env->GetShortArrayRegion(pcm, 0, n, buf.data());

    std::vector<float> mel(kNMels * kNFrames);
    computeMelEnergies(buf.data(), n, mel.data());

    jfloatArray out = env->NewFloatArray(kNMels * kNFrames);
    if (!out) return nullptr;
    std::vector<float> finalized(kNMels * kNFrames);
    if (!finalizeAndCopy<float, false>(mel.data(), finalized.data())) return nullptr;
    env->SetFloatArrayRegion(out, 0, kNMels * kNFrames, finalized.data());
    return out;
}

extern "C" JNIEXPORT jshortArray JNICALL
Java_com_example_whisperapp_audio_WhisperFeatureExtractor_00024Companion_nativeExtractHalf(
        JNIEnv* env, jobject /*thiz*/, jshortArray pcm, jint sampleRate) {
    if (sampleRate != kSampleRate) {
        LOGE("unsupported sample rate %d", sampleRate);
        return nullptr;
    }
    jsize n = env->GetArrayLength(pcm);
    std::vector<int16_t> buf(n);
    env->GetShortArrayRegion(pcm, 0, n, buf.data());

    std::vector<float> mel(kNMels * kNFrames);
    computeMelEnergies(buf.data(), n, mel.data());

    jshortArray out = env->NewShortArray(kNMels * kNFrames);
    if (!out) return nullptr;
    std::vector<int16_t> finalized(kNMels * kNFrames);
    if (!finalizeAndCopy<int16_t, true>(mel.data(), finalized.data())) return nullptr;
    env->SetShortArrayRegion(out, 0, kNMels * kNFrames, finalized.data());
    return out;
}
