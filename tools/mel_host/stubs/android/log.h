#pragma once
// Minimal android/log.h stub for host-side compilation of mel_native.cpp.
#define ANDROID_LOG_ERROR 6
#define ANDROID_LOG_INFO 4
#ifdef __cplusplus
extern "C" {
#endif
int __android_log_print(int prio, const char* tag, const char* fmt, ...);
#ifdef __cplusplus
}
#endif
