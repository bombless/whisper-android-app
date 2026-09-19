// Native (C++) Whisper log-mel frontend, JNI-bound.
//
// Mirrors the Java WhisperFeatureExtractor exactly (HF reflect padding, Bluestein
// 400-point DFT via 1024-point radix-2 convolution, sparse triangular mel scatter,
// Slaney filterbank, log10 + per-chunk normalization), parallelized over frames with
// std::thread. Outputs either normalized float features or binary16 samples ready
// for the HTP encoder.
//
// Bit-compat notes: single-precision IEEE-754 throughout, same summation order as the
// Java implementation. Clang may contract a*b+c into FMA, which shifts results by ~1 ulp.

#include <jni.h>
#include <android/log.h>
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
constexpr int kFftSize = 1024;
constexpr int kNBins = kNFft / 2 + 1; // 201

struct MelTables {
    float window[kNFft];
    float chirpRe[kNFft];
    float chirpIm[kNFft];
    float kernelRe[kFftSize];
    float kernelIm[kFftSize];
    // Twiddles for all stages of the 1024-point FFT; stage span `len` starts at len/2 - 1.
    float twRe[kFftSize - 1];
    float twIm[kFftSize - 1];
    // Sparse mel filterbank CSR: per FFT bin, (mel, weight) pairs sorted by mel.
    std::vector<int>   binIds[kNBins];
    std::vector<float> binWeights[kNBins];

    MelTables() {
        for (int i = 0; i < kNFft; i++) {
            window[i] = (float)(0.5 - 0.5 * std::cos(2.0 * M_PI * i / kNFft));
        }
        for (int k = 0; k < kNFft; k++) {
            double a = M_PI * (double)k * (double)k / kNFft;
            chirpRe[k] = (float)std::cos(a);
            chirpIm[k] = (float)std::sin(a);
        }
        std::vector<float> kre(kFftSize, 0.f), kim(kFftSize, 0.f);
        for (int k = 0; k < kNFft; k++) {
            kre[k] = chirpRe[k];
            kim[k] = chirpIm[k];
            if (k != 0) {
                kre[kFftSize - k] = chirpRe[k];
                kim[kFftSize - k] = chirpIm[k];
            }
        }
        fft1024(kre.data(), kim.data());
        std::memcpy(kernelRe, kre.data(), sizeof(kernelRe));
        std::memcpy(kernelIm, kim.data(), sizeof(kernelIm));

        int offset = 0;
        for (int len = 2; len <= kFftSize; len <<= 1) {
            for (int k = 0; k < len / 2; k++) {
                double angle = -2.0 * M_PI * k / len;
                twRe[offset + k] = (float)std::cos(angle);
                twIm[offset + k] = (float)std::sin(angle);
            }
            offset += len / 2;
        }

        buildMelFilters();
    }

    void fft1024(float* re, float* im) const {
        // Bit-reversal permutation.
        int j = 0;
        for (int i = 1; i < kFftSize; i++) {
            int bit = kFftSize >> 1;
            while (j & bit) { j ^= bit; bit >>= 1; }
            j ^= bit;
            if (i < j) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }
        int offset = 0;
        for (int len = 2; len <= kFftSize; len <<= 1) {
            const int half = len >> 1;
            for (int base = 0; base < kFftSize; base += len) {
                for (int k = 0; k < half; k++) {
                    const int tw = offset + k;
                    const int b = base + k + half;
                    const float vr = re[b] * twRe[tw] - im[b] * twIm[tw];
                    const float vi = re[b] * twIm[tw] + im[b] * twRe[tw];
                    const int a = base + k;
                    const float ur = re[a];
                    const float ui = im[a];
                    re[a] = ur + vr;
                    im[a] = ui + vi;
                    re[b] = ur - vr;
                    im[b] = ui - vi;
                }
            }
            offset += half;
        }
    }

    // Bluestein: exact 400-point DFT via 1024-point convolution. scratch must be 1024.
    void bluestein(float* re400, float* im400, float* scratchRe, float* scratchIm) const {
        std::fill(scratchRe, scratchRe + kFftSize, 0.f);
        std::fill(scratchIm, scratchIm + kFftSize, 0.f);
        for (int k = 0; k < kNFft; k++) {
            const float c = chirpRe[k];
            const float s = chirpIm[k];
            scratchRe[k] = re400[k] * c + im400[k] * s;
            scratchIm[k] = im400[k] * c - re400[k] * s;
        }
        fft1024(scratchRe, scratchIm);
        for (int k = 0; k < kFftSize; k++) {
            const float r = scratchRe[k] * kernelRe[k] - scratchIm[k] * kernelIm[k];
            const float i = scratchRe[k] * kernelIm[k] + scratchIm[k] * kernelRe[k];
            scratchRe[k] = r;
            scratchIm[k] = i;
        }
        for (int k = 0; k < kFftSize; k++) scratchIm[k] = -scratchIm[k];
        fft1024(scratchRe, scratchIm);
        const float scale = 1.0f / kFftSize;
        for (int k = 0; k < kNFft; k++) {
            const float c = chirpRe[k];
            const float s = chirpIm[k];
            const float sr = scratchRe[k] * scale;
            const float si = -scratchIm[k] * scale;
            re400[k] = sr * c + si * s;
            im400[k] = si * c - sr * s;
        }
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
    const MelTables& T = tables();
    std::vector<float> waveform(kChunkSamples, 0.f);
    const int copy = std::min(pcmLen, kChunkSamples);
    for (int i = 0; i < copy; i++) waveform[i] = pcm[i] / 32768.f;

    std::vector<float> padded(kChunkSamples + kNFft);
    const int pad = kNFft / 2;
    for (int i = 0; i < (int)padded.size(); i++) {
        padded[i] = waveform[reflectIndex(i - pad, kChunkSamples)];
    }

    std::fill(mel, mel + kNMels * kNFrames, 0.f);

    const unsigned int hw = std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
    auto worker = [&](int ith) {
        float re[kNFft], im[kNFft];
        float sre[kFftSize], sim[kFftSize];
        for (int frame = ith; frame < kNFrames; frame += (int)hw) {
            const int start = frame * kHop;
            for (int i = 0; i < kNFft; i++) {
                re[i] = padded[start + i] * T.window[i];
                im[i] = 0.f;
            }
            T.bluestein(re, im, sre, sim);
            for (int k = 0; k < kNBins; k++) {
                const float p = re[k] * re[k] + im[k] * im[k];
                const auto& ids = T.binIds[k];
                const auto& ws = T.binWeights[k];
                for (size_t jj = 0; jj < ids.size(); jj++) {
                    mel[ids[jj] * kNFrames + frame] += p * ws[jj];
                }
            }
        }
    };
    if (hw <= 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        for (unsigned int t = 0; t < hw; t++) pool.emplace_back(worker, (int)t);
        for (auto& th : pool) th.join();
    }
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
    std::vector<float> melCopy(mel, mel + kNMels * kNFrames);
    float maxLog = -INFINITY;
    for (int i = 0; i < kNMels * kNFrames; i++) {
        float v = std::max(melCopy[i], 1e-10f);
        v = std::log10(v);
        melCopy[i] = v;
        if (v > maxLog) maxLog = v;
    }
    const float floor = maxLog - 8.f;
    for (int i = 0; i < kNMels * kNFrames; i++) {
        const float v = (std::max(melCopy[i], floor) + 4.f) / 4.f;
        if (!std::isfinite(v)) { LOGE("non-finite mel at %d", i); return false; }
        if (kAsHalf) {
            reinterpret_cast<int16_t*>(out)[i] = floatToHalf(v);
        } else {
            out[i] = v;
        }
    }
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
