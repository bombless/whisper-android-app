# Snapdragon Voice Lab

Experimental Kotlin + Jetpack Compose app for validating Qualcomm QNN/HTP inference on Snapdragon 8 Gen 3.

## Current status

- M0 PC prerequisites: Java 25, Python 3.11, Android SDK API 35, Gradle 9.1, and an ADB-connected device are present.
- `qai-hub-models` 0.62.2 is installed; its executable is available under the Python 3.11 Scripts directory.
- Current CLI inspection confirms Whisper-Tiny has a Snapdragon 8 Gen 3 `qnn_context_binary` float asset using QAIRT 2.45.0.260326154327.
- Current CLI inspection confirms MeloTTS-ZH has a Snapdragon 8 Gen 3 `voice_ai` mixed_with_float asset using QAIRT 2.45.0.260326154327; it is not exposed as a generic `qnn_context_binary` asset.
- QNN/QAIRT runtime location has not yet been verified.
- Android scaffold and explicit ASR/TTS engine boundaries are created.
- No Qualcomm runtime/model binaries are fabricated or bundled.

## Build

```powershell
.\gradlew.bat :app:assembleDebug
```

The first build may need network access to resolve Gradle/Maven dependencies.

## Next blocker

Install/verify `qai-hub-models`, then run model info for `Whisper-Tiny` and `MeloTTS-ZH`. Record the exact model/runtime/device/precision metadata before implementing QNN graph loading.

## 2026-09-15 progress

- QNN/HTP smoke test passes on RMX3800 / SM8650 with CPU fallback disabled.
- Verified Whisper-Tiny Snapdragon 8 Gen 3 QAIRT 2.45.0.260326154327 context assets are packaged: ncoder.bin 19,832,832 bytes and decoder.bin 97,452,032 bytes.
- Android APK rebuild/install and post-change HTP smoke test pass (median 1.843 ms, P95 7.081 ms in the latest run).
- QnnAsrEngine now validates the real Whisper asset contract; tokenizer boundary is present.
- Remaining M3 work: invoke the encoder/decoder QNN context binaries with the exact QAIRT Android API, then add Whisper log-mel preprocessing, token sampling, and timestamp/language handling before connecting AudioRecord.

## 2026-09-21 progress: Whisper-Large-V3-Turbo on QNN

- The app now runs **Whisper-Large-V3-Turbo** on QNN/HTP. Whisper-Tiny is retained and
  selectable through `WhisperVariant`; both share one runner because the graph topology is
  identical and only mel bins (80/128), KV heads (6/20) and vocab (51865/51866) differ.
- Turbo assets added: `encoder_ctx.onnx` + `decoder_ctx.onnx` EPContext wrappers generated
  from the asset's `metadata.json`, the Large-V3 tokenizer, and an incremental cache on the
  128-bin mel frontend for the live loop.
- On-device results: encoder 1390.9 ms for a 30 s window, ~20 ms per decoder step, ~1.6 s per
  1-second live update, `cpu_fallback: disabled`, `dsp_crash: false`.
- The same WAV through Tiny and Turbo yields an identical token sequence and text, which
  cross-validates both paths. Turbo decodes `你好呀 你会说话吗 你应该不会说话吧你说几句来听听呀`.
- Details, contract table, evidence and known limitations: `TURBO_QNN_RESULTS.md`.

