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
