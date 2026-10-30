#pragma once

#include <jni.h>

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_bombless_htp_playpen_qnn_WhisperReferenceRunner_runNative(
        JNIEnv * env,
        jobject thiz,
        jshortArray pcm,
        jstring model_path,
        jint threads);

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_bombless_htp_playpen_qnn_WhisperEncoderDiagnosticRunner_runNative(
        JNIEnv * env,
        jobject thiz,
        jshortArray pcm,
        jstring model_path,
        jint threads);
