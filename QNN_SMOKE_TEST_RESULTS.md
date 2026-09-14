# QNN Smoke Test Results

Date: 2026-09-14
Status: **FAIL / NOT PASS**

## Acceptance result

The strict smoke-test success criteria are **not satisfied**. The test reached the Qualcomm QNN EP, selected HTP, initialized the HTP device/context, and validated the tiny CNN nodes, but it did **not** produce a completed inference result with proof of HTP execution. Therefore this run must not be reported as PASS.

## Project / source-control state

- Requested project: `D:\mcp-agent-workspace\whisper-android-app`
- `git status --short`: unavailable because this directory and its parents are not a Git repository (`fatal: not a git repository`)
- No project rebuild/recreation was performed.
- Whisper/TTS/Audio implementations were not modified.
- Smoke-test code is isolated under `app/src/main/java/com/example/whisperapp/qnn/`.

## Toolchain

- Gradle: 9.1.0
- Android Gradle Plugin: 8.7.3
- Kotlin: 2.1.21
- compileSdk: 35

## Android target

- Device model: RMX3800
- SoC: SM8650 (QNN log explicitly detected `Snapdragon SOC SM8650`)
- ABI: arm64-v8a
- API: 35
- Build fingerprint: `realme/RMX3800/RE5C4FL1:16/UKQ1.231108.001/U.23f9ba7_b5cf2_5ece0:user/release-keys`
- Vendor FastRPC library present: `/vendor/lib64/libcdsprpc.so`

## Runtime artifacts

| Artifact | Version | Size | SHA256 |
|---|---:|---:|---|
| `com.microsoft.onnxruntime:onnxruntime-android` | 1.27.0 | 44,532,227 | `077DEC5E2D821234C7DC0ABA584BEC8F999854B546C754CAB93A90741C56FBEB` |
| `com.qualcomm.qti:onnxruntime-android-qnn` | 2.6.0 | 1,595,551 | `169ADD91932E27196E0AB704F53FE8CD517AF3ECEE0D4A73FBF14513AB4551A6` |
| `com.qualcomm.qti:qnn-runtime` | 2.50.0 | 71,270,746 | `B507656E4031D8AA6A7AB0FECF725B941096737AD7440AD7EA288F7C9CC9D743` |

The three dependencies are explicitly declared in `app/build.gradle.kts`.

## Model

Path: `app/src/main/assets/models/tiny_cnn.onnx`

- Size: 911 bytes
- SHA256: `0C8EC8A422CD454DD9612129D27D5CD47C94A7537A390103D5198F855734DC21`
- Input: `float32 [1,3,8,8]`
- Output: `float32 [1,2]`
- Operators: Conv -> Relu -> GlobalAveragePool -> Flatten -> Gemm
- Deterministic input: values `i / 64 - 1.5`, i=0..191
- PC ORT CPU expected output: `[-0.22386418282985687, 0.3315545916557312]`
- Expected output shape: `[1,2]`
- Model validated with `onnx.checker.check_model` and PC ONNX Runtime CPU.

## Strict QNN configuration

The smoke runner uses:

- ORT Android 1.27.0
- QNN EP plugin 2.6.0
- QNN runtime 2.50.0
- EP: `QNNExecutionProvider`
- Backend: `htp`
- Session config: `session.disable_cpu_ep_fallback=1`
- No CPU EP was explicitly added as a fallback provider.
- The smoke runner treats session creation, graph preparation, or inference failure as FAIL.
- Android manifest declares `<uses-native-library android:name="libcdsprpc.so" android:required="false" />`.
- JNI legacy packaging is enabled in Gradle so QNN native libraries are extracted.

## Evidence captured from real device

The real-device log showed:

- `ORT version=1.27.0`
- `QNN EP registered name=QNNExecutionProvider`
- `EP_DEVICE name=QNNExecutionProvider vendor=Qualcomm`
- `backend=HTP`
- `CPU fallback=DISABLED`
- ORT session options contained `session.disable_cpu_ep_fallback: 1`
- QNN EP selected backend path: `libQnnHtp.so`
- QNN backend build: `v2.50.0.260828221209`
- QNN detected `Snapdragon SOC SM8650`
- `First connection to QNN stub established!`
- QNN transport session was found and transport calls returned status 0
- `QnnDevice_create done. device = 0x1. status 0x0`
- `QnnContext_create done successfully`
- ORT logged `QNN SetupBackend succeed`
- QNN EP logged node validation success for Conv, Relu, GlobalAveragePool, Flatten, and Gemm.

These are strong HTP initialization/graph-support signals, but they are **not sufficient for PASS** because completed inference was not observed.

## Blocking evidence

The device log also contained:

- `remote_handle_control_domain failed for request ID 2 on domain 3`
- `open_shell failed for domain 3 ... /dsp/ /vendor/dsp/ /vendor/dsp/xdsp/ (errno Permission denied)`
- DSP-side lookup failures for `/vendor/dsp/cdsp/./libc++.so.1` and `libc++abi.so.1`
- `remote_handle64_invoke failed ... domain 3`

The run subsequently remained in QNN/HTP preparation without producing the smoke runner's `session created`, `warmup OK`, `inference OK`, or final `PASS` evidence. The process remained alive during the observation window.

The device does expose `/vendor/lib64/libcdsprpc.so`, so this is no longer the initial linker-visibility failure. The remaining blocker is in the CDSP/HTP runtime path during graph/device preparation, with the DSP-side C++ runtime lookup being a concrete observed error.

## Conclusion

**Do not mark QNN HTP smoke test PASS.**

What is proven:

1. ONNX Runtime Android is running.
2. Qualcomm QNN Execution Provider is loaded and registered.
3. QNN EP device is Qualcomm.
4. Backend selection is explicitly HTP (`libQnnHtp.so`).
5. CPU fallback is explicitly disabled and ORT logged that setting.
6. The SM8650 HTP device/context and QNN graph capability checks were reached.

What is **not** proven:

7. A completed model inference returned from HTP.
8. The tiny CNN actually executed on HTP end-to-end.
9. A final inference output was obtained under the strict no-fallback configuration.

Therefore the Phase-1 acceptance gate remains **FAIL** pending resolution of the target device's CDSP/HTP runtime preparation issue.

## Official reference

ONNX Runtime's QNN EP documentation specifies Android ARM64 support, HTP via `backend_type=htp`, and the QNN Android Maven composition. It also documents HTP-specific backend configuration and CPU fallback behavior.
