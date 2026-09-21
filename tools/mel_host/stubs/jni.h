#pragma once
// Minimal jni.h stub for host-side compilation of mel_native.cpp (verification only).
#include <cstddef>
#include <cstdint>

typedef int32_t jint;
typedef int16_t jshort;
typedef uint16_t jchar;
typedef int64_t jlong;
typedef float jfloat;
typedef double jdouble;
typedef uint8_t jboolean;
typedef int32_t jsize;
typedef void* jobject;
typedef jobject jclass;
typedef jobject jstring;
typedef jobject jarray;
typedef jarray jshortArray;
typedef jarray jfloatArray;
typedef jobject jthrowable;

struct _jNIEnv;
typedef _jNIEnv JNIEnv;

#define JNIEXPORT
#define JNICALL

struct _jNIEnv {
    jsize GetArrayLength(jarray) { return 0; }
    void GetShortArrayRegion(jshortArray, jsize, jsize, jshort*) {}
    void SetShortArrayRegion(jshortArray, jsize, jsize, const jshort*) {}
    jfloatArray NewFloatArray(jsize) { return nullptr; }
    jshortArray NewShortArray(jsize) { return nullptr; }
    void SetFloatArrayRegion(jfloatArray, jsize, jsize, const jfloat*) {}
};
