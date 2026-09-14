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
