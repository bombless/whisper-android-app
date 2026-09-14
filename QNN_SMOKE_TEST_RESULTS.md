# QNN Smoke Test Results

Date: 2026-09-15
Status: **PASS on RMX3800 / SM8650 (HTP, CPU fallback disabled)**

The fixed debug APK completed two consecutive app cold starts, each with five warmup runs and twenty measured inference runs. The tiny CNN returned finite outputs, the expected top-1 class, and a maximum absolute error below the 0.02 acceptance threshold. This validates the QNN HTP smoke test only; it does not establish Whisper or TTS model compatibility or performance.

## Root cause

The reported `QNN_GRAPH_ERROR_INVALID_HANDLE` (6001) followed a DSP process crash. The original implementation copied standalone Hexagon `libc++.so.1` and `libc++abi.so.1` from `assets/qnn-dsp` into the app's files directory and prepended that directory to `ADSP_LIBRARY_PATH`. This caused those copies to be selected by the DSP loader.

The original device log (`qnn_latest_log.txt`, lines 1874 onward) records:

```text
Process on cDSP0 CRASHED!!!!!!!
due to TLBMISS X (execution)
Fault PC : 0x0
Bad VA : 0x0
_ZnwjSt11align_val_t+0x28: (./libc++abi.so.1)
... (./libQnnHtpV75Skel.so)
SSR successfully recovered CDSP sessions!
QNN_GRAPH_ERROR_INVALID_HANDLE: Invalid graph handle, Code: 6001
```

The stack symbol is `operator new(unsigned int, std::align_val_t)`. ELF inspection identified the injected ABI library as Hexagon v75 built with Clang 19.0.04; it also imports `aligned_alloc`. Together with successful execution after removing the override, the evidence points to the injected DSP C++ runtime being incompatible with this device's runtime environment. The precise null call target was not established by disassembly, so this report does not claim that a specific firmware symbol is absent.

The resulting DSP crash/recovery invalidated the graph state. The graph-handle error was a downstream symptom; changing Java graph lifetime or retrying that handle would not address the crash.

There was also a library-discovery problem when relying on `backend_type=htp` after removing the manual search path: the EP could initialize an empty `ADSP_LIBRARY_PATH` and fail to locate the v75 skeleton. Supplying the absolute installed path of `libQnnHtp.so` lets the EP derive the correct directory for the packaged DSP libraries.

Earlier reports treated DSP C++ library lookup warnings as a blocker. Similar lookup warnings, FastRPC `Permission denied` messages, and `0x80000414` also appeared during successful runs. Those messages alone do not establish failure; the DSP crash trace and completed inference result are the relevant evidence.

## Fix

- `QnnSmokeTest.kt` no longer copies or injects standalone DSP C++ libraries or sets `ADSP_LIBRARY_PATH` manually.
- The two problematic assets were moved to `build/incompatible-dsp-runtime/` for diagnosis and are absent from the final APK.
- The session receives `backend_path = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so").absolutePath`, with an explicit existence check.
- `session.disable_cpu_ep_fallback=1` and `offload_graph_io_quantization=0` remain enabled; session creation or inference failure is reported as FAIL.
- ORT logging uses INFO to reduce vendor-log noise.
- `QnnSmokeActivity.kt` displays real newlines in a scrollable report and persists that report to `files/qnn-smoke-result.txt`. Logcat did not reliably retain the final result on this device, so the saved report is also used for verification.

## Versions and packaging

| Component | Version |
|---|---|
| ONNX Runtime Android | 1.27.0 |
| Qualcomm QNN EP | 2.6.0 |
| QNN runtime | 2.50.0 |
| Gradle / Android Gradle Plugin | 9.1.0 / 8.7.3 |
| Kotlin / compileSdk | 2.1.21 / 35 |
| Device / SoC / DSP generation | RMX3800 / SM8650 / Hexagon v75 |

The three runtime dependency versions in `app/build.gradle.kts` are unchanged. This combination is documented by the official QNN EP v2.6.0 Android instructions:

https://github.com/onnxruntime/onnxruntime-qnn/blob/v2.6.0/docs/execution_providers/QNN-ExecutionProvider.md

JNI legacy packaging extracts the native libraries, and the manifest exposes `libcdsprpc.so` using `uses-native-library`. Final APK inspection confirmed the HTP backend and v75 skeleton are packaged, with no manually injected `qnn-dsp` C++ assets.

## Model and correctness

- Asset: `app/src/main/assets/models/tiny_cnn.onnx` (911 bytes; unchanged).
- SHA256: `0C8EC8A422CD454DD9612129D27D5CD47C94A7537A390103D5198F855734DC21`.
- Input: float32 `[1,3,8,8]`, values `i / 64 - 1.5`, i = 0..191.
- Model output: float32 `[1,2]`.
- Operators: Conv -> Relu -> GlobalAveragePool -> Flatten -> Gemm.
- CPU reference: `[-0.22386418282985687, 0.3315545916557312]`.
- Successful QNN diagnostic output: `[-0.22387697, 0.331543]`; top-1 = 1.
- Maximum absolute error: `1.2785196E-5`, below the `0.02` threshold.

## Final APK verification

`gradle.bat :app:installDebug` succeeded. The installed final APK was then force-stopped and launched twice on the connected device. These are app process cold starts, not device reboots. Latencies measure twenty inference calls after five warmup runs and exclude session creation.

| Final APK run | Result | Median | P95 | Max | Maximum absolute error |
|---|---|---:|---:|---:|---:|
| Cold start 1 | PASS | 1.148 ms | 1.373 ms | 1.774 ms | 0.00001279 |
| Cold start 2 | PASS | 1.257 ms | 1.347 ms | 2.824 ms | 0.00001279 |

Both results were read from the app's persisted report. The second is saved locally as `build/qnn-final-cold-start-2.txt`.

An earlier successful diagnostic run in `build/qnn-live.log` also records `session created`, `warmup OK`, `inference iterations=20`, `inference OK`, the output values, `EP=QNNExecutionProvider`, `Backend=HTP`, `CPU fallback=NO`, and `ACTIVITY_RESULT=PASS`. Its latency values differ from the final two runs; the evidence excerpt keeps those runs labeled separately.

## Reproduce

From the project directory in PowerShell:

```powershell
gradle.bat :app:installDebug
adb shell am force-stop com.example.whisperapp
adb shell am start -W -n com.example.whisperapp/.qnn.QnnSmokeActivity
Start-Sleep -Seconds 3
adb shell run-as com.example.whisperapp cat files/qnn-smoke-result.txt
```

Run the launch commands after a successful build. `am start -W` waits for Activity launch, while inference runs on a worker thread. The report initially contains `RUNNING`; if it still does, read it again after the worker finishes. A normal completed report contains PASS or FAIL, backend and fallback settings, and metrics on success. `run-as` is available for this debug build.

APK: `app/build/outputs/apk/debug/app-debug.apk`.

## Evidence

- `qnn_latest_log.txt`: original DSP crash and subsequent invalid graph handle.
- `qnn_smoke_logcat_excerpt.txt`: labeled original failure, successful diagnostic log, and final second cold-start report.
- `build/qnn-live.log`: successful diagnostic execution log.
- `build/qnn-final-cold-start-2.txt`: final APK persisted PASS report.
- `build/qnn-smoke-results-before-fix.md`: original report preserved for history; its FAIL conclusion and runtime-lookup diagnosis are superseded by this report.
- `build/qnn-smoke-logcat-excerpt-before-fix.txt`: previous log excerpt.

The project directory is not a Git repository, so there is no commit or Git diff for these local changes.
