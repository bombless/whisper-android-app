# TTS 接入计划：Kokoro ONNX

> 本文供后续开发上下文执行。本计划已根据用户授权执行完整接入；实施与验收进度见第 8 节。

## 1. 当前项目情况

- Android：Kotlin + Jetpack Compose，minSdk 29，仅 arm64-v8a。
- TTS 接口：app/src/main/java/io/github/bombless/htp_playpen/tts/TtsEngine.kt，提供 prepare()、synthesize(text): TtsResult、close()。
- 现有实现：SherpaOnnxTtsEngine.kt，通过 sherpa-onnx OfflineTts 生成 PCM16。
- 当前模型：Piper Xiao-Ya INT8，ONNX 权重约 18.65 MB；另有 VITS 中文模型资源。
- 依赖：lib/runtime/sherpa-onnx-1.13.8-no-ort.aar、ONNX Runtime Android 1.27.0。
- UI 在 ui/MainScreen.kt 创建 SherpaOnnxTtsEngine 并通过 TtsQueue 播放。保持 TtsEngine/TtsResult/TtsQueue 合约。

## 2. 目标模型与语言支持门槛

优先评估 Kokoro 82M ONNX 的 sherpa-onnx 多语言 INT8 模型包：

- 官方模型说明：https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
- 官方 TTS API：https://k2-fsa.github.io/sherpa/onnx/c-api/html/tts.html
- 官方候选：kokoro-int8-multi-lang-v1_1.tar.bz2。
- 对照包：kokoro-multi-lang-v1_0.tar.bz2。官方文档列出的 v1.0 model.onnx 约 310 MB，voices.bin 约 26 MB，另有 tokens、词典、espeak-ng 数据和中文 FST。

**重要：不要误判日语支持。** sherpa-onnx 官方 Kokoro 文档明确说明 v1.0/v1.1 当前集成支持中文 + 英文。speaker 列表出现 jf_* 日语声线名称，并不代表该 sherpa 文本前端已支持日语。不能直接把官方 sherpa 配置宣称为中日双语。

如果项目仍要求中文 + 日文，执行时必须先验证以下候选之一：

1. 找到并验证一个 Kokoro ONNX 导出包，其模型和文本前端确实支持 zh + ja。可评估 https://huggingface.co/soniqo/Kokoro-82M-ONNX 及 Android 实现 https://github.com/soniqo/speech-android。核对 ONNX 主权重、外置 .onnx.data、voices、tokenizer/phonemizer、语言资源总大小、revision、许可证和 SHA256。该导出不一定兼容 sherpa-onnx Kokoro 配置；若不兼容，使用独立 ONNX Runtime + 对应 phonemizer，不要硬套 sherpa OfflineTts。
2. 若只采用 sherpa 官方 Kokoro 包，则如实定位为中文/英文 TTS，日语继续由 Piper 日文 voice 或其他经验证模型负责；UI 不得宣称 Kokoro 已支持日语。

在代码实现前，先做一条中文和一条日文的离线合成 smoke test。语言前端、音素映射或 voice 任一环节不能正确处理日语，就判定该候选不满足中日双语验收。

## 3. 建议模型资产布局

~~~
lib/assets/models/tts/kokoro/
  model.onnx
  model.onnx.data             # 若模型使用 external data
  voices.bin                  # 若模型格式需要
  tokens.txt
  espeak-ng-data/             # 若对应前端需要
  lexicon-zh.txt
  lexicon-ja.txt              # 仅当所选导出明确要求
  date-zh.fst
  number-zh.fst
  README.md                   # 来源、revision、许可证、SHA256、语言支持
~~~

不要把压缩包直接放入 APK。不要遗漏 ONNX external data。确认 assets 路径与模型配置完全一致。权重目标小于 500 MB；同时记录模型目录总大小，phonemizer/词典资源也会影响 APK 体积。

## 4. 实施步骤

### P0：模型与运行时可行性

- [ ] 固定模型来源、revision/tag、量化类型及许可证。
- [ ] 核对主 ONNX、external data、voices、tokens、词典、语言前端资源实际字节数。
- [ ] 用 ONNX Runtime 检查输入/输出签名，确认 Android arm64 runtime 兼容。
- [ ] 检查 sherpa-onnx 1.13.8 Java API 是否包含 Kokoro 配置类及 native 实现；不能假设当前 AAR 与最新版文档一致。
- [ ] 桌面/Android 分别跑中文、日文 smoke test，保存输入、WAV、采样率、耗时和日志。
- [ ] 检查许可证是否允许本项目分发。

### P1：引擎封装

- [ ] 新增 KokoroOnnxTtsEngine.kt，实现现有 TtsEngine。
- [ ] 若模型兼容 sherpa Kokoro API，优先复用现有 AAR；否则使用 ONNX Runtime Android，并隔离模型推理和 phonemizer。
- [ ] prepare() 惰性初始化 session/引擎，重复调用幂等。
- [ ] synthesize() 做文本规范化、语言/voice 选择、合成及 float-to-PCM16 转换，返回真实采样率和耗时。
- [ ] close() 释放 session、native handle、缓存，避免 Activity 重建后重复加载或泄漏。
- [ ] 多语言模型按句子或显式语言参数选择语言与 speaker；不要只用字符比例猜测混合中日文本。必要时先分句，再逐句合成并拼接 PCM，确保采样率一致并处理句间静音。
- [ ] 保持 TtsResult(pcm16, sampleRate, latencyMs) 不变，尽量不改 TtsQueue 和 AudioTrack 播放路径。

### P2：UI 和配置

- [ ] 在 TTS variant/engine 选择中增加 Kokoro，保留 Piper Xiao-Ya 和 VITS。
- [ ] 暴露语言（自动/中文/日文）和 voice 选择；默认值明确且可持久化。
- [ ] UI 显示模型未安装/加载失败状态，不在主线程下载或初始化。
- [ ] 日语能力未通过 P0 前，UI 不得显示“日语已支持”。

### P3：性能优化

- [ ] Snapdragon 8 Gen 3 真机先测 CPU，再评估可用 ONNX EP；不能假设 Kokoro 全图可由 QNN 加速。
- [ ] 测量冷启动加载时间、首句延迟、RTF、峰值 RSS/PSS、生成音频时长和耗电。
- [ ] 对比 INT8 与 FP16/FP32 音质；优先采用可复现的官方量化包。
- [ ] 长文本分句生成，限制单次文本长度，避免内存峰值过高。

## 5. 测试与验收

### 功能测试

- [ ] 中文：短句、长句、数字/日期/标点、中英混读。
- [ ] 日文：平假名、片假名、常见汉字、长音/促音/拗音、数字日期、日英混读。
- [ ] 中日混合文本：语言切换点、专有名词、连续标点。
- [ ] 连续生成至少 20 段，无 native crash、资源泄漏或播放中断。
- [ ] 引擎关闭后重新创建可正常初始化。

### 性能/体积

- [ ] 每个权重文件 < 500 MB；同时记录全部模型资源总大小。
- [ ] 在目标 arm64 真机记录加载时间、RTF、峰值内存。
- [ ] 与 Piper Xiao-Ya 用同一批中文文本比较音质和延迟。
- [ ] 若日语由另一个模型承担，明确记录路由规则及每种语言所需权重。

### 交付

- [ ] 新增引擎及必要模型 manifest/校验工具。
- [ ] 单元测试覆盖配置、语言路由、PCM 转换和资源释放。
- [ ] 真机报告包含设备、Android 版本、模型 revision、文件 SHA256。
- [ ] 不删除旧 Piper/VITS；新引擎验收通过后再考虑调整默认项。

## 6. 风险

1. 官方 sherpa Kokoro 当前公开支持列表是中文 + 英文，不是中文 + 日文；必须验证日文前端/导出。
2. 不同 Kokoro ONNX 导出格式不一定兼容 sherpa-onnx OfflineTts。
3. ONNX external data 可能是必需文件，不能只检查 .onnx 大小。
4. 当前依赖为 sherpa-onnx 1.13.8 no-ort AAR 和 ONNX Runtime Android 1.27.0，需检查 native ABI/API 兼容，避免符号冲突。
5. 先保证离线正确性，再做 NPU 加速。

## 7. 资料

- sherpa-onnx Kokoro：https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
- sherpa-onnx TTS API：https://k2-fsa.github.io/sherpa/onnx/c-api/html/tts.html
- Kokoro-82M ONNX 候选：https://huggingface.co/soniqo/Kokoro-82M-ONNX
- Android 候选实现：https://github.com/soniqo/speech-android

## 8. 2026-10-30 实施记录

### 已实现

- 采用独立的 `audio.soniqo:speech:0.0.20` Kokoro Android JNI/ONNX 后端，不把不支持日语的 sherpa Kokoro 配置误当成中日双语。
- Kotlin/Compose 插件升级到 2.2.21；保留项目现有 Sherpa 1.13.8 与 ONNX Runtime Android 1.27.0。
- 新增 `KokoroOnnxTtsEngine.kt`，实现现有 `TtsEngine` 合约；`TtsResult`、`TtsQueue` 和 AudioTrack 播放接口未改变。
- 新增 `KokoroModelRepository.kt`：固定模型 revision，支持断点续传、文件大小及 SHA-256 校验、原子落盘和已验证标记。
- 新增 `KokoroTextNormalizer.kt`：中文使用 Android ICU Han-Latin 转数字声调拼音；日文使用 Any-Hiragana 转换为假名，避免把汉字直接交给仅处理假名的日文 G2P。
- UI 增加 Kokoro 引擎、自动/中文/日文语言选择，以及自动、`jf_alpha`、`zf_xiaobei` 声线选择。
- 保留 Piper/VITS 和原有默认引擎；Kokoro 权重首次使用时下载到 app 私有目录。

### 模型资产核验

- 来源：`soniqo/Kokoro-82M-ONNX`。
- 固定 revision：`b561586381f1b212a8d6bd2a46483e04f54f4ebf`。
- 仓库声明许可证：Apache-2.0。
- manifest 共 19 个文件，总计 174,878,328 bytes（约 166.7 MiB）。
- 已将每个文件的 SHA-256 写入 `lib/assets/models/tts/kokoro/manifest.json`，并与独立下载的同 revision 文件逐项比对一致。
- APK 只包含 manifest/README，不包含 ONNX 权重、external data、词典或 voice embedding。
- Gradle native merge 已确认最终保留应用的 ONNX Runtime 1.27.0（用于现有 QNN/Java ORT），并打包 `libspeech_android.so` 和 `libc++_shared.so`。

### 已验证

- `:app:assembleDebug`：通过。
- `:app:assembleRelease`：通过，生成 unsigned release APK。
- `:app:assembleDebugAndroidTest`：通过，包含中日文及 20 段 native smoke test。
- 中文 Han-to-Pinyin instrumentation test：通过（RMX3800 / Android 16）。
- JVM 单测：104 项中 103 项通过；唯一失败是原有 `TurboQnnContractTest.turboAssetDirectoryIsComplete` 找不到 `models/whisper_large_v3_turbo/encoder.bin`。单独排除该测试类后，其余 18 个测试类全部通过。
- APK 内容检查：Kokoro manifest 和 native library 存在，Kokoro 权重未被打包。
- manifest 的 19 个 SHA-256、文件大小及总字节数：与独立下载清单一致。

### 待完成验收

- 日文汉字转假名 instrumentation test 尚未完成。
- 中日文 native synthesis、连续 20 段生成、引擎关闭/重建测试已写入 `KokoroOnnxTtsSmokeTest.kt`，尚未在设备执行。
- RMX3800 主应用已获使用者确认并启动；ColorOS 对独立 instrumentation APK `io.github.bombless.htp_playpen.test`（约 1.2 MiB）再次弹出“继续安装”确认。Gradle 真机测试因 OEM 安装拦截（错误码 -99）未能启动；需要使用者确认测试包后重新运行。
- 真机加载耗时、RTF、峰值 RSS/PSS、耗电和音质对比尚未测量；CPU 路径已实现，未宣称 NPU/QNN 加速。
