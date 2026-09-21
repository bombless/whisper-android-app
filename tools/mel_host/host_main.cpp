// Host-side numeric verification for mel_native.cpp: reads a 16-bit mono PCM WAV,
// runs the exact native mel pipeline, and dumps normalized float + FP16 binaries
// for comparison against the Java WhisperFeatureExtractor reference.
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <vector>

#include "../../app/src/main/cpp/mel/mel_native.cpp"

static std::vector<int16_t> readWavData(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); exit(2); }
    fclose(f);
    if (sz < 12 || memcmp(buf.data(), "RIFF", 4) || memcmp(buf.data() + 8, "WAVE", 4)) {
        fprintf(stderr, "not a RIFF/WAVE file\n"); exit(2);
    }
    size_t pos = 12;
    while (pos + 8 <= (size_t)sz) {
        uint32_t chunkSz = (uint32_t)buf[pos + 4] | ((uint32_t)buf[pos + 5] << 8) |
                           ((uint32_t)buf[pos + 6] << 16) | ((uint32_t)buf[pos + 7] << 24);
        if (!memcmp(buf.data() + pos, "data", 4)) {
            const uint8_t* d = buf.data() + pos + 8;
            size_t n = chunkSz / 2;
            std::vector<int16_t> pcm(n);
            for (size_t i = 0; i < n; i++) {
                pcm[i] = (int16_t)((uint16_t)d[2 * i] | ((uint16_t)d[2 * i + 1] << 8));
            }
            return pcm;
        }
        pos += 8 + chunkSz + (chunkSz & 1);
    }
    fprintf(stderr, "no data chunk\n"); exit(2);
}

static void writeBin(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s in.wav out_float.bin out_half.bin\n", argv[0]);
        return 2;
    }
    std::vector<int16_t> pcm = readWavData(argv[1]);
    fprintf(stderr, "pcm samples: %zu\n", pcm.size());

    std::vector<float> mel(kNMels * kNFrames);
    computeMelEnergies(pcm.data(), (int)pcm.size(), mel.data());

    std::vector<float> fin(kNMels * kNFrames);
    if (!finalizeAndCopy<float, false>(mel.data(), fin.data())) { fprintf(stderr, "finalize float failed\n"); return 1; }
    writeBin(argv[2], fin.data(), fin.size() * sizeof(float));

    std::vector<int16_t> half(kNMels * kNFrames);
    if (!finalizeAndCopy<int16_t, true>(mel.data(), half.data())) { fprintf(stderr, "finalize half failed\n"); return 1; }
    writeBin(argv[3], half.data(), half.size() * sizeof(int16_t));

    fprintf(stderr, "ok\n");
    return 0;
}

// Stub for the android/log.h stub (never actually logs on host).
extern "C" int __android_log_print(int, const char*, const char* fmt, ...) { va_list ap; va_start(ap, fmt); int rc = vfprintf(stderr, fmt, ap); va_end(ap); fputc('\n', stderr); return rc; }
