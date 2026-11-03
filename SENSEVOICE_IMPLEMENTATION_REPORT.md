# SenseVoiceSmall INT8 implementation report

Date: 2026-10-30

## Delivered

- Added `SenseVoiceAsrEngine`, backed by the existing sherpa-onnx 1.13.8 AAR and `OfflineRecognizer`/`OfflineSenseVoiceModelConfig`.
- Added `AsrEngineFactory`; the existing Whisper QNN live path remains available and was not replaced.
- Added the STT provider selector in `MainScreen`: Whisper QNN or SenseVoice INT8 CPU.
- Added 16 kHz mono PCM conversion, linear-interpolation resampling, PCM16-to-Float32 normalization, serialized offline decode, and native resource cleanup.
- Added a bounded microphone producer queue and energy-based endpointing: 100 ms frames, 700 ms silence endpoint, 15 s maximum utterance. This is VAD-segmented offline recognition, not token-level streaming.
- Added JVM resampler tests, an Android instrumentation smoke test for standard test devices, and a debug-only sample-audio activity. QA WAVs are outside release assets; the debug activity/WAVs are debug-only.
- Kept the model's original `LICENSE` pointer and added `MODEL_ATTRIBUTION.md` plus `model-manifest.json`.

## Model and integrity

- Model: `SenseVoiceSmall`, sherpa-onnx Offline SenseVoice, INT8.
- Languages: `zh`, `yue`, `en`, `ja`, `ko`; app default is `auto`.
- Sample rate: 16,000 Hz mono.
- Model: `lib/assets/models/asr/sense_voice_small_int8/model.int8.onnx`
- Model bytes: `239233841` (228.1 MiB), below the 500 MB requirement.
- Model SHA256: `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`
- Tokens bytes: `315894`
- Tokens SHA256: `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc`
- Model directory total: `239554162` bytes.
- The GitHub release URL timed out from the Windows agent. The model was obtained from the Hugging Face mirror listed in `model-manifest.json`; the downloaded mirror ZIP SHA256 is `526304059c61db8c9eb677e40bb7105cf570a50bf419396d5d9d77f420e9a4bf`. The mirror README records the upstream release archive SHA256 as `7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e`; this upstream tarball hash was not independently verified against a separately downloaded tarball.
- The archive's `LICENSE` is an upstream pointer. Model-weight terms are separate from sherpa-onnx/runtime code; review the current model card and linked FunASR model license before external redistribution.

## Verification

### Build and unit tests

Command:

```powershell
.\gradlew.bat :app:testDebugUnitTest --tests io.github.bombless.htp_playpen.SenseVoicePcmResamplerTest :app:assembleDebug --no-daemon
```

- Result: `BUILD SUCCESSFUL`.
- `SenseVoicePcmResamplerTest`: 4 tests, 0 failures, 0 errors.
- Debug APK size at the final build: `628934891` bytes; SHA256 `773C0FD74C75B7521FDFC27FE9F5BC5928291DFFDBE2B9E0F1EB711E4A79E4C4`. This includes the project's other existing assets; it is not the SenseVoice model size.
- Release command: `gradlew.bat :app:assembleRelease --no-daemon` — `BUILD SUCCESSFUL` (including `lintVitalRelease`).
- Release artifact: `app/build/outputs/apk/release/app-release-unsigned.apk`, `618359762` bytes, SHA256 `2837FC0751F6EBC931EBD4E31618DB1E5C5B216ECFD6CBD0771DEBB44B86EF6C`. It is unsigned and must be signed with the project's release key before distribution.
- Verified the release APK contains the INT8 model, tokens, original license pointer, attribution, and manifest; debug-only WAV fixtures are absent.

### Real-device smoke test

- Device: Realme RMX3800, SoC `SM8650` (Snapdragon 8 Gen 3), Android 16 / API 36, arm64-v8a.
- Confirmed the packaged model initializes through sherpa-onnx 1.13.8 on CPU with `language=auto`.
- Debug-only smoke activity decoded the release's mono PCM16 16 kHz `zh.wav` and `ja.wav` fixtures. Both returned non-empty text.
- Two-thread warm decode: Mandarin `zh.wav` (5.592 s) in `182 ms` (RTF ≈ `0.033`); Japanese `ja.wav` (7.200 s) in `215 ms` (RTF ≈ `0.030`). Single smoke run per file; not a statistically controlled benchmark.
- Device log ended with `SENSEVOICE_SMOKE: SMOKE_PASS`.
- The app's SenseVoice provider selector and ready status were also verified in the main UI.

## Not yet verified / remaining work

- Live microphone capture was not exercised because microphone permission was not granted during device validation. Grant microphone permission to test recording end-to-end.
- No manually corrected reference transcripts were supplied, so CER/WER was not calculated. The smoke test checks model load and non-empty decode only.
- The Realme package installer blocked the separate instrumentation-test APK with `PC install attack detected` (`-99`). The same WAV inference was therefore run through the debug-only activity inside the main app APK instead.
- 1/4-thread comparison, 30-minute soak, peak RSS/PSS, battery/thermal measurements, and QNN execution-provider partition/fallback analysis remain outstanding. SenseVoice currently uses CPU by design.
- The existing Whisper QNN path reported missing external `encoder.bin` and `decoder.bin` on the test device. Its code was left intact; this is separate from SenseVoice.
- The original GitHub tarball was not downloaded directly due network timeout; see the mirror provenance/hash note above.

## Files added or updated for STT

- `app/src/main/java/io/github/bombless/htp_playpen/asr/SenseVoiceAsrEngine.kt`
- `app/src/main/java/io/github/bombless/htp_playpen/asr/SenseVoiceRecordingSession.kt`
- `app/src/main/java/io/github/bombless/htp_playpen/asr/AsrEngineFactory.kt`
- `app/src/main/java/io/github/bombless/htp_playpen/asr/AsrEngine.kt`
- `app/src/main/java/io/github/bombless/htp_playpen/ui/MainScreen.kt`
- `app/src/test/java/io/github/bombless/htp_playpen/SenseVoicePcmResamplerTest.kt`
- `app/src/androidTest/java/io/github/bombless/htp_playpen/asr/SenseVoiceAndroidSmokeTest.kt`
- `app/src/debug/java/io/github/bombless/htp_playpen/asr/SenseVoiceDebugSmokeActivity.kt`
- `lib/assets/models/asr/sense_voice_small_int8/`