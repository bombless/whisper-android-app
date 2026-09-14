# Snapdragon 8 Gen 3 本地语音 AI Android App 计划

## 1. 项目目标

构建一个 **Kotlin + Jetpack Compose** Android 应用，目标设备为 **Snapdragon 8 Gen 3 手机**，优先验证 Qualcomm AI Hub / `qai_hub_models_cli` 提供的模型能否通过 **QNN / Hexagon HTP（NPU）** 在设备端运行。

第一阶段不追求完整产品，而是做一个可测量、可替换模型的最小实验应用：

```text
麦克风
  ↓
AudioRecord / PCM 16 kHz
  ↓
ASR（第一目标：Whisper-Tiny）
  ↓
文本
  ↓
TTS（第二目标：MeloTTS-ZH）
  ↓
PCM / AudioTrack
  ↓
扬声器
```

核心原则：**先跑通 Qualcomm 已经准备好的模型资产，再逐步接入自定义模型；先证明 NPU/HTP 真正在工作，再做实时语音链路。**

---

## 2. 技术路线

### PC 端：模型准备

使用 Qualcomm AI Hub Models 的 CLI：

```powershell
python -m pip install -U qai-hub-models
qai-hub-models info Whisper-Tiny
qai-hub-models info MeloTTS-ZH
```

根据 CLI 当前版本支持的参数，查询/获取 Snapdragon 8 Gen 3 对应的 QNN/HTP 资产。不要硬编码一个未经验证的 CLI 参数；先执行 `--help` 和模型 README，确认当前版本的命令格式。

目标资产优先级：

1. `Whisper-Tiny` 的 QNN / HTP 可部署资产
2. `MeloTTS-ZH` 的 QNN / HTP 可部署资产
3. 如果某模型暂时无法直接生成 Android 可用资产，再退回 ONNX → QNN 的路径

建议把 **PC 端模型转换/下载** 与 **Android Runtime** 分离。Android APK 不负责运行 `qai_hub_models_cli`；CLI 是开发机上的模型准备工具。

---

## 3. Android 技术栈

- Kotlin
- Jetpack Compose
- Android Gradle Plugin / Gradle Wrapper
- Min SDK：根据 QNN Android runtime 支持情况确定，优先现代 Android 手机版本
- Target SDK：使用当前 Android Studio 稳定版对应 SDK
- `androidx.activity:activity-compose`
- Compose Material 3
- Kotlin Coroutines
- Android `AudioRecord`
- Android `AudioTrack`
- QNN Android runtime / Qualcomm AI Engine Direct
- HTP backend

不要第一版引入过多第三方框架。语音采集、推理调度、播放、benchmark 尽量自己控制，方便定位性能瓶颈。

---

## 4. 推荐项目结构

```text
whisper-android-app/
├── PLAN.md
├── README.md
├── settings.gradle.kts
├── build.gradle.kts
├── gradle.properties
├── gradlew
├── gradlew.bat
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/example/whisperapp/
│       │   ├── MainActivity.kt
│       │   ├── ui/
│       │   │   ├── App.kt
│       │   │   ├── MainScreen.kt
│       │   │   └── theme/
│       │   ├── audio/
│       │   │   ├── AudioRecorder.kt
│       │   │   └── AudioPlayer.kt
│       │   ├── asr/
│       │   │   ├── AsrEngine.kt
│       │   │   ├── WhisperTokenizer.kt
│       │   │   └── QnnAsrEngine.kt
│       │   ├── tts/
│       │   │   ├── TtsEngine.kt
│       │   │   └── QnnTtsEngine.kt
│       │   └── benchmark/
│       │       └── BenchmarkRunner.kt
│       └── assets/
│           ├── asr/
│           └── tts/
└── tools/
    ├── download_models.ps1
    ├── inspect_qnn_assets.ps1
    └── README.md
```

实际 package 名称可以在创建项目时统一修改。

---

## 5. 第一阶段：确认 PC 环境

在 Windows 开发机检查：

```powershell
java -version
adb version
python --version
python -m pip --version
qai-hub-models --help
```

还需要确认：

- Android Studio / SDK / build-tools
- JDK 版本与 Android Gradle Plugin 兼容
- 手机开启开发者选项和 USB debugging
- `adb devices` 能看到 Snapdragon 8 Gen 3 手机
- Qualcomm AI Engine Direct / QNN SDK 是否已经安装
- QNN SDK 的 `libQnnHtp.so`、backend、必要的 skel/相关 runtime 文件是否可获得

如果 QNN SDK 尚未安装，不要假设 AI Hub CLI 本身会把 Qualcomm Android runtime 自动放进 APK；需要明确区分 **模型资产** 和 **设备 runtime**。

---

## 6. 第二阶段：CLI 模型验证

### 6.1 Whisper-Tiny

先查看模型信息：

```powershell
qai-hub-models info Whisper-Tiny
```

确认：

- 输入采样率
- 输入 tensor shape
- tokenizer / vocabulary 要求
- 支持的 runtime
- 支持的 device
- 是否有 Snapdragon 8 Gen 3 / 对应 Qualcomm device target
- FP16 / INT8 等可用 precision

然后按照该版本 `qai-hub-models` README 执行模型导出、编译或 fetch。

### 6.2 MeloTTS-ZH

同样执行：

```powershell
qai-hub-models info MeloTTS-ZH
```

确认文本输入、speaker、音频输出以及 QNN/HTP 支持情况。

### 6.3 资产归档

不要把整个 Python/AI Hub 环境复制到 Android 项目。只把最终需要的部署资产整理到：

```text
models/
├── whisper-tiny/
└── melotts-zh/
```

并记录：

```text
model name
model version / commit
precision
input/output shapes
QNN SDK version
HTP backend version
target device
compile date
```

---

## 7. 第三阶段：先做“无 UI”的 Android NPU Smoke Test

这是整个项目最重要的里程碑。

先不要做实时录音和漂亮 UI，而是在 Android 中完成：

```text
加载 QNN runtime
    ↓
初始化 HTP backend
    ↓
加载编译好的模型 / context binary
    ↓
创建 QNN context / graph
    ↓
喂固定测试 tensor
    ↓
执行 inference
    ↓
读取 output tensor
```

用一份固定输入做 deterministic smoke test，并保存 expected output / checksum。

成功标准：

- APK 成功加载 Qualcomm runtime
- HTP backend 初始化成功
- graph/context binary 成功加载
- inference 成功
- output 数值合理
- log 中明确显示使用 HTP，而不是静默 fallback 到 CPU

**只有这一阶段成功，才进入 ASR。**

---

## 8. 第四阶段：Whisper-Tiny ASR

### 音频链路

使用 `AudioRecord`：

- mono
- PCM 16-bit
- 16 kHz
- buffer 按模型要求设置

第一版不要直接做无限流式 Whisper。先实现：

```text
点击录音
  ↓
录 3~10 秒
  ↓
停止
  ↓
feature extraction
  ↓
Whisper encoder
  ↓
decoder
  ↓
Tokenizer decode
  ↓
Compose 显示文字
```

这样容易验证模型输入是否正确。

### ASR 模块接口

建议抽象成：

```kotlin
interface AsrEngine {
    suspend fun transcribe(pcm16: ShortArray, sampleRate: Int): AsrResult
    fun close()
}
```

以后可以替换：

```text
QnnWhisperAsrEngine
CpuWhisperAsrEngine
MockAsrEngine
```

这样可以直接比较 NPU / CPU。

---

## 9. 第五阶段：MeloTTS-ZH

先做离线文本输入：

```text
Compose TextField
       ↓
QnnTtsEngine
       ↓
PCM
       ↓
AudioTrack
```

确认 TTS 的 QNN graph / context binary 能在 HTP 上运行以后，再和 ASR 串起来：

```text
麦克风 → Whisper → 文本 → MeloTTS → 扬声器
```

TTS 同样抽象：

```kotlin
interface TtsEngine {
    suspend fun synthesize(text: String): ShortArray
    fun close()
}
```

---

## 10. Compose UI

第一版页面保持简单：

```text
┌─────────────────────────────┐
│ Snapdragon Voice Lab        │
│                             │
│ Backend: HTP / QNN          │
│ ASR: Whisper-Tiny           │
│ TTS: MeloTTS-ZH             │
│                             │
│ [ 🎙 Start Recording ]      │
│                             │
│ Recognized Text              │
│ ┌─────────────────────────┐ │
│ │                         │ │
│ └─────────────────────────┘ │
│                             │
│ [ 🔊 Speak ]                │
│                             │
│ Inference: 842 ms           │
│ Audio: 5.0 s                │
│ RTF: 0.168                  │
│ Backend: HTP                │
└─────────────────────────────┘
```

另外提供一个 Diagnostics 区域显示：

- SoC / device
- Android version
- QNN version
- backend
- model
- precision
- input shape
- inference latency
- CPU time
- total latency
- peak memory（能可靠取得时）

---

## 11. Benchmark 设计

至少测以下指标：

### ASR

```text
audio duration = 1 / 3 / 5 / 10 / 30 sec
inference latency
real-time factor = inference_time / audio_duration
CPU utilization
NPU/HTP utilization（如果 profiling 工具可获得）
RAM
```

分别跑：

```text
CPU baseline
HTP FP16
HTP INT8（如果模型支持）
```

### TTS

测试：

```text
20 字
50 字
100 字
200 字
```

记录：

```text
first audio latency
full synthesis latency
audio duration
RTF
```

每个测试至少重复多次，丢弃第一次 warm-up，然后报告 median / p90，而不是只看一次结果。

---

## 12. QNN / HTP 集成注意事项

### 不要默认 ONNX Runtime 就等于 NPU

可以把 ONNX Runtime QNN EP 作为一种路线，但最终目标是确认 Qualcomm HTP backend 实际执行。

优先顺序：

```text
AI Hub 已编译 QNN asset
        ↓
Android QNN / QAIRT runtime
        ↓
HTP backend
```

必要时再考虑：

```text
ONNX
 ↓
ONNX Runtime QNN EP
 ↓
QNN HTP
```

### Runtime 与 model asset 必须版本匹配

特别记录：

- QNN / QAIRT SDK version
- HTP backend version
- model compilation target
- Android ABI（Snapdragon 8 Gen 3 通常是 arm64-v8a）

避免出现“模型能编译，但手机加载失败”的版本/ABI问题。

---

## 13. APK 中 native library 的规划

预计最终需要类似：

```text
app/src/main/jniLibs/arm64-v8a/
├── libQnnSystem.so
├── libQnnHtp.so
├── libQnnHtpVxxSkel.so   # 实际文件名以 SDK 为准
└── 其他 Qualcomm runtime libraries
```

**这里禁止凭记忆硬编码库文件名。** 实现阶段应该根据实际 QNN SDK 目录确认文件，并记录来源和版本。

模型 context binary / compiled asset 放：

```text
app/src/main/assets/models/
```

或者根据 runtime API 的加载方式使用 app-private files 目录。

---

## 14. Manifest / 权限

至少需要：

```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
```

运行时动态申请麦克风权限。

如果模型全部打包进 APK，则第一版不需要网络权限。目标是 **完全离线**。

---

## 15. 开发里程碑

### M0 — 环境

- [ ] Android Studio / Gradle 可用
- [ ] JDK 可用
- [ ] adb 连接手机
- [ ] Python 可用
- [ ] `qai-hub-models` 可执行
- [ ] QNN/QAIRT SDK 可定位

### M1 — AI Hub CLI

- [ ] `Whisper-Tiny` info 成功
- [ ] 找到 Snapdragon 8 Gen 3 对应 target
- [ ] 获得 QNN/HTP asset
- [ ] `MeloTTS-ZH` info 成功
- [ ] 获得 TTS asset

### M2 — Android HTP Smoke Test

- [ ] Android 工程编译
- [ ] QNN runtime 打包
- [ ] HTP backend 初始化
- [ ] 固定 tensor inference 成功
- [ ] 验证没有 CPU fallback

### M3 — Whisper

- [ ] PCM 录音
- [ ] feature extraction
- [ ] Whisper inference
- [ ] tokenizer decode
- [ ] 中文文本显示

### M4 — TTS

- [ ] 文本输入
- [ ] MeloTTS inference
- [ ] PCM output
- [ ] AudioTrack 播放

### M5 — 完整语音对话链路

- [ ] 一键录音
- [ ] ASR
- [ ] 显示文本
- [ ] TTS
- [ ] 播放
- [ ] 全程离线

### M6 — Benchmark

- [ ] CPU baseline
- [ ] HTP FP16
- [ ] HTP INT8（如可用）
- [ ] latency
- [ ] RTF
- [ ] memory
- [ ] profiler 数据

---

## 16. 第一版明确不做的事情

为了避免项目一开始变复杂，暂时不做：

- 实时 streaming Whisper
- LLM 对话
- 云端 API
- 用户账号
- 网络模型下载
- 多语言 UI
- 复杂音频降噪
- VAD 的高级算法
- 后台持续监听
- TTS 声音克隆

这些全部放到 M5 以后。

---

## 17. 推荐的实际执行顺序

不要直接开始写完整 App。按下面顺序推进：

```text
① 检查 Windows / Android / Python / ADB
        ↓
② 安装并验证 qai-hub-models
        ↓
③ 查看 Whisper-Tiny 当前 README 和支持的 Qualcomm device
        ↓
④ 获取/编译 Snapdragon 8 Gen 3 可用 QNN asset
        ↓
⑤ 确认 QNN/QAIRT SDK 和 HTP runtime
        ↓
⑥ 创建最小 Kotlin Android 工程
        ↓
⑦ 做 QNN HTP smoke test
        ↓
⑧ 接 Whisper-Tiny
        ↓
⑨ 接 AudioRecord
        ↓
⑩ 接 MeloTTS-ZH
        ↓
⑪ Compose UI
        ↓
⑫ Benchmark
```

**每一步都先验证，再进入下一步。** 如果某个模型/SDK API 在当前版本发生变化，以实际 CLI help、AI Hub 模型 README 和本机 QNN SDK 为准，不根据旧教程猜 API。

---

## 18. 第一轮交付标准

第一轮不要求“语音助手”，只要求做到：

> 在 Snapdragon 8 Gen 3 真机上，点击按钮录一段中文 → Whisper-Tiny 通过 Qualcomm QNN/HTP 完成推理 → Compose 显示中文识别结果，并显示实际 inference latency。

然后：

> 输入一段中文 → MeloTTS-ZH 通过 QNN/HTP 生成 PCM → 手机扬声器播放。

最终页面能明确显示：

```text
ASR backend: HTP
TTS backend: HTP
Offline: YES
Device: Snapdragon 8 Gen 3
```

这两个闭环跑通以后，再优化模型、量化和实时性。
