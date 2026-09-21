cmake_minimum_required(VERSION 3.22.1)
project(whisper_reference_android LANGUAGES C CXX)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_IS_DEV OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

# Match whisper.cpp's upstream Android arm64 v8fp16 build, while keeping the
# reference runner CPU-only and independent of the production QNN path.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+fp16")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+fp16")

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../../tools/whisper.cpp ${CMAKE_BINARY_DIR}/whisper.cpp EXCLUDE_FROM_ALL)

add_library(whisper_reference SHARED
    whisper_reference/whisper_reference.cpp
)

target_compile_features(whisper_reference PRIVATE cxx_std_17)
target_compile_definitions(whisper_reference PRIVATE WHISPER_BUILD GGML_USE_CPU)
target_include_directories(whisper_reference PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../../../tools/whisper.cpp/include
)
target_link_libraries(whisper_reference PRIVATE whisper log android)

target_compile_options(whisper_reference PRIVATE -fvisibility=hidden)

# Native mel spectrogram (production hot path; mirrors Java WhisperFeatureExtractor).
add_library(mel_native SHARED
    mel/mel_native.cpp
)
target_compile_features(mel_native PRIVATE cxx_std_17)
target_compile_options(mel_native PRIVATE -O3 -fvisibility=hidden)
target_link_libraries(mel_native PRIVATE log)

# Native decoder step loop (P1): OrtSession + OrtIoBinding so the cross-KV tensors and the
# attention mask stay in one bound buffer set instead of being rebuilt per decode step.
# The ORT C++ headers and the link stub live in third_party/onnxruntime; regenerate them from
# the onnxruntime-android AAR with `pwsh -File tools/fetch_ort_headers.ps1` (keep the version
# in step with the Gradle dependency). libonnxruntime.so is already packaged into the APK by
# that same AAR, so this links against it rather than vendoring a second runtime.
add_library(whisper_decoder SHARED
    whisper_decoder/whisper_decoder_jni.cpp
)
target_compile_features(whisper_decoder PRIVATE cxx_std_17)
target_compile_options(whisper_decoder PRIVATE -O3 -fvisibility=hidden)

set(ORT_ROOT ${CMAKE_CURRENT_LIST_DIR}/third_party/onnxruntime)
if (NOT EXISTS ${ORT_ROOT}/onnxruntime_cxx_api.h)
    message(FATAL_ERROR
        "ONNX Runtime headers missing at ${ORT_ROOT}. "
        "Run: pwsh -File tools/fetch_ort_headers.ps1")
endif()
target_include_directories(whisper_decoder PRIVATE ${ORT_ROOT})
# Link against the onnxruntime shared library extracted from the onnxruntime-android AAR.
# It is the exact build that AAR already packages into the APK, so linking against it adds no
# second runtime and no version skew; the JNI lib only needs its symbols resolved at link
# time and gets the real .so from the APK at run time.
set(ORT_LIB ${ORT_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so)
if (NOT EXISTS ${ORT_LIB})
    message(FATAL_ERROR
        "ONNX Runtime link stub missing at ${ORT_LIB}. "
        "Run: pwsh -File tools/fetch_ort_headers.ps1 -Abi ${ANDROID_ABI}")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ORT_LIB})
target_link_libraries(whisper_decoder PRIVATE log onnxruntime)
