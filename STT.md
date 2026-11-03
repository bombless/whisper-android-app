# STT 接入计划：SenseVoiceSmall INT8

> 本文供后续开发上下文执行。本次只新增计划文档，不改业务代码、不下载模型、不提交代码。

## 1. 当前项目情况

- Android：Kotlin + Jetpack Compose，minSdk 29，仅 arm64-v8a。
- ASR 接口：app/src/main/java/io/github/bombless/htp_playpen/asr/AsrEngine.kt，提供 transcribe(pcm16, sampleRate) 和 close()。
- 现有主识别流程在 ui/MainScreen.kt，当前 sttVariant = WhisperVariant.LARGE_V3_TURBO，直接启动 QnnWhisperRealAudioRunner。
- Whisper 实时流程在 audio/LiveTranscriber.kt，依赖 Whisper 专用 Mel 缓存、QNN encoder/decoder 和 tokenizer。SenseVoice 不可复用 Whisper Mel/tokenizer/自回归解码逻辑。
- 运行时包含 lib/runtime/sherpa-onnx-1.13.8-no-ort.aar，Gradle 另有 ONNX Runtime Android 1.27.0 和 Qualcomm QNN 依赖。
- 项目已有 AsrEngine 抽象，但 MainScreen 还没有统一的 ASR engine factory/切换层。

## 2. 目标模型

首选官方 sherpa-onnx INT8 导出：

- 模型目录：sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
- 权重：model.int8.onnx，官方文档列出约 228 MB（小于 500 MB）。
- tokens.txt 约 308 KB。
- 支持普通话、粤语、英语、日语、韩语。
- 从 FunAudioLLM SenseVoiceSmall 导出为 sherpa-onnx 格式。
- 下载包：https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
- 官方说明：https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html

模型文档给出的权重大小约 228 MB；下载后必须重新用字节数确认，并记录模型、tokens、license、测试 wav 的 SHA256/来源。不要用 FP32 约 894 MB 的文件替代 INT8。

## 3. 建议资产布局

~~~
lib/assets/models/asr/sense_voice_small_int8/
  model.int8.onnx
  tokens.txt
  LICENSE
  README.md
  test_wavs/                 # 可选；开发测试，发布 APK 可排除
~~~

- 不把 tar.bz2 压缩包放入 APK。
- 保留官方文件名，便于升级和核对。
- 生产 APK 默认只打包模型、tokens 和必要许可证/manifest；测试音频放在 QA 目录，避免增加 APK 体积。
- 增加 model-manifest.json，记录来源 URL、release/tag、文件长度、SHA256、量化类型、语言和许可证。

## 4. 实施步骤

### P0：依赖与模型验证

- [ ] 下载官方 INT8 tarball，核对模型文件确实小于 500 MB，记录实际字节数和 SHA256。
- [ ] 检查许可证及模型分发要求。
- [ ] 检查现有 sherpa-onnx 1.13.8 AAR 是否包含 SenseVoice JNI/native 实现，以及 API 是否暴露 OfflineSenseVoiceModelConfig、OfflineRecognizer 等所需类。
- [ ] 不要只依据最新版文档假设 1.13.8 API 一致；必要时整体升级 AAR 与 JNI so 并跑回归。
- [ ] 用 sherpa-onnx CLI 或官方 Android 示例对中文、日文测试音频做基线识别，保存参考输出。
- [ ] 先用 CPU provider 跑通，再评估 NNAPI/QNN；初始验收不依赖 NPU。

### P1：实现 AsrEngine

- [ ] 新增 SenseVoiceAsrEngine.kt，实现 AsrEngine。
- [ ] 使用 sherpa-onnx offline SenseVoice 模型配置（model、tokens、language、use_itn、numThreads、provider）和 OfflineRecognizer。
- [ ] 默认 language = auto，允许调用端显式指定 zh 或 ja；对照当前 AAR API 确认语言参数格式。
- [ ] 输入统一转为 16 kHz mono PCM。若调用方传入其他采样率，必须真正重采样，不能只修改 sampleRate 参数。
- [ ] transcribe() 将 ShortArray 转成 API 所需 FloatArray（归一化到 [-1, 1]），创建/复用 stream，acceptWaveform 后 decode，读取文本。
- [ ] use_itn = true 作为默认候选以输出标点/规范化文本；中日语料实测后确定默认值。
- [ ] 封装为现有 AsrResult(text, latencyMs)。如需语言、情绪等元数据，可新增扩展数据类型，但避免破坏现有调用点。
- [ ] close() 释放 recognizer、stream 和 native 资源；并发调用需串行化或每请求独立 stream。
- [ ] 惰性加载模型并报告失败，避免主线程初始化。

### P2：接入主录音/转写流程

- [ ] 抽出统一 ASR engine factory/selection，避免 MainScreen 直接耦合 QnnWhisperRealAudioRunner。
- [ ] 增加 provider 选择：SenseVoice（CPU）/ Whisper QNN。
- [ ] 将录音 PCM 分发给所选引擎。SenseVoice 先走句段/按需离线路径；不要复用 Whisper 的 Mel cache、token step 或 QNN decoder。
- [ ] 如需边说边出字，使用 VAD/固定窗口模拟流式：按静音分句，或参考 sherpa-onnx SenseVoice microphone 示例。OfflineRecognizer 不应被包装成真正 token-level streaming。
- [ ] 录音线程与推理线程解耦：采集不可等待推理；队列需有背压/丢帧统计。
- [ ] 识别结果继续走现有 UI 状态流；切换引擎时停止识别、关闭旧实例，再初始化新引擎。
- [ ] Whisper QNN 保持原样可选，不删除现有 Mel/decoder/JNI 文件。

### P3：性能和设备加速

- [ ] Snapdragon 8 Gen 3 先测 CPU baseline（1/2/4 threads）。
- [ ] 检查当前 ONNX Runtime/QNN EP 对 SenseVoice 算子的覆盖率，记录 provider assignment、fallback 节点和初始化日志。
- [ ] 只有图分区正确且速度/耗电更优时才启用 QNN；否则保持 CPU。
- [ ] 对比 INT8 与 FP32（正式使用 INT8），记录中文/日文 CER、RTF、首句延迟、峰值内存和温升。
- [ ] 长音频按 VAD 分段，设定单段最大时长；合并时保留标点和语言切换处文本。

## 5. 测试与验收

### 语言准确率

- [ ] 普通话：安静/噪声环境、数字日期专有名词、较快语速。
- [ ] 日语：日常会话、敬语、汉字/假名混合、片假名外来词、数字和日期。
- [ ] 中日混说：同一句或相邻句切换语言，确认 auto 识别和文本拼接。
- [ ] 可选测试粤语、英语、韩语。
- [ ] 使用人工校对 reference transcript，报告中文 CER、日文 CER（或明确日文分词规则后的 WER），不能只靠主观判断。

### 稳定性/性能

- [ ] 冷启动模型加载成功，连续识别至少 30 分钟无 crash/明显内存增长。
- [ ] 录音采集不因推理阻塞；统计 dropped chunks、队列峰值和端到端延迟。
- [ ] 同一批音频对比 SenseVoice INT8 与当前 Whisper QNN 的准确率、RTF、峰值 RSS/PSS、耗电和温升。
- [ ] 设备：Snapdragon 8 Gen 3、arm64-v8a；记录 Android/系统版本。
- [ ] 模型文件小于 500 MB；APK 增量和完整模型目录大小分别记录。

### 交付

- [ ] 新增 SenseVoiceAsrEngine.kt、engine factory/selection 和必要 UI 设置。
- [ ] 单元测试覆盖 PCM 归一化、重采样、语言参数、错误处理、资源释放。
- [ ] instrumentation/smoke test 从 assets 加载真实模型并识别中日测试音频。
- [ ] 保留可复现测试音频、reference transcript、模型 SHA256 和性能报告。
- [ ] 新引擎真机验收通过后，才考虑把默认 STT 从 Whisper QNN 切换到 SenseVoice。

## 6. 关键注意事项

1. SenseVoiceSmall INT8 官方 sherpa-onnx 模型是离线模型。可做 VAD 分段/模拟流式，不应声称原生 token streaming。
2. ASR 输入目标为 16 kHz mono；确认录音采样率，必要时真正重采样。
3. 官方模型支持普通话和日语；language=auto 及显式 zh/ja 的参数行为需对照当前 AAR API 验证。
4. SenseVoice 输出可能包含 lang/emotion/event/timestamps；当前 AsrResult 只有 text/latency，先解析并保留文本，避免标签泄漏到 UI。
5. 不复用 Whisper 的 log-Mel、tokenizer、自回归 decode 和 QNN context 文件。
6. 228 MB 是权重文件约值，不代表完整模型包或 APK 增量只有 228 MB。

## 7. 官方资料

- sherpa-onnx SenseVoice 预训练模型及文件大小：https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html
- SenseVoiceSmall 原始模型：https://www.modelscope.cn/models/iic/SenseVoiceSmall
- sherpa-onnx Android 示例：https://k2-fsa.github.io/sherpa/onnx/android/index.html
- sherpa-onnx 项目：https://github.com/k2-fsa/sherpa-onnx

## 8. 本次执行记录（2026-10-30）

- 已完成 SenseVoice INT8 引擎、ASR factory、STT 引擎选择 UI、CPU 离线识别及 VAD 分句录音队列；Whisper QNN 保留为可选项，默认值未切换。
- 已在 RMX3800 / Snapdragon 8 Gen 3 / Android 16 上验证模型加载，并用 debug-only smoke activity 对 zh/ja WAV 完成实际推理。
- JVM PCM 重采样测试 4/4 通过，Debug APK 构建成功。
- 麦克风采集、CER/WER、30 分钟稳定性、1/2/4 线程对比和 QNN EP 评估仍待完成；详见 `SENSEVOICE_IMPLEMENTATION_REPORT.md`。
- 模型来源、SHA256、文件长度和许可证指针见 `lib/assets/models/asr/sense_voice_small_int8/model-manifest.json` 与 `MODEL_ATTRIBUTION.md`。
