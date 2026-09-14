# ONNX Runtime + QNN EP + HTP 极小 CNN Smoke Test

## 1. 目标

在现有 Android 项目中增加一个**最小、可验证、可重复**的 ONNX Runtime + Qualcomm QNN Execution Provider smoke test。

目标不是测试模型精度，而是确认：

```text
Android App
    ↓
ONNX Runtime Android
    ↓
QNN Execution Provider
    ↓
Qualcomm HTP / NPU
    ↓
ONNX inference
    ↓
正确 output
```

最终需要得到三个结论：

1. ONNX Runtime Android 可以正常加载。
2. QNN Execution Provider 可以正常初始化。
3. inference 确实使用 Qualcomm HTP/NPU，而不是偷偷 fallback 到 CPU。

---

# 2. 当前设备与项目约束

## 真机

目标设备已经确认：

```text
SoC:        SM8650
Platform:   pineapple
Product:    RMX3800
Chipset:    Snapdragon 8 Gen 3
Android:    API 36 / Android 16
ABI:        arm64-v8a
```

必须使用这台真机验证。

不要用 emulator 代替。

## 当前项目

现有项目：

```text
D:\mcp-agent-workspace\whisper-android-app
```

当前项目已经可以：

```text
gradle installDebug
```

并且：

```text
com.example.whisperapp
```

可以正常安装、启动。

原则：

> 不重建 Android 项目，不重写现有 UI，不破坏现有 Audio/ASR/TTS 结构。

---

# 3. 为什么第一阶段不用翻译模型

翻译模型虽然小，但第一阶段变量太多：

```text
tokenizer
↓
embedding
↓
Transformer
↓
attention
↓
dynamic sequence
↓
decoder
↓
autoregressive loop
```

如果失败，很难判断到底是：

* ORT
* QNN EP
* HTP
* unsupported operator
* dynamic shape
* tokenizer
* decoder

哪个出了问题。

所以第一阶段使用极小 CNN。

CNN smoke test 成功以后，再进入 Transformer/翻译模型。

---

# 4. 模型选择原则

选择一个：

* 标准 ONNX
* 极小
* CPU/ORT 能正常运行
* 输入 shape 固定
* 输出 shape 固定
* 不需要 tokenizer
* 不需要动态 sequence
* 不需要额外 preprocessing
* 最好只包含 Conv / Relu / Pool / Gemm 等基础算子

优先考虑：

```text
SqueezeNet
```

或者更小的：

```text
MobileNetV3-small
```

如果能找到更简单的 tiny CNN，则优先 tiny CNN。

## 不允许

第一阶段不要使用：

```text
Whisper
MeloTTS
BERT
T5
MarianMT
LLM
```

也不要使用一个需要几十个输入 tensor 的模型。

---

# 5. 先验证 ONNX Runtime Android 的 QNN 包

重点调查 Android Maven dependency。

优先验证当前官方 QNN Android package：

```text
com.qualcomm.qti:onnxruntime-android-qnn
```

同时确认与它匹配的：

```text
ONNX Runtime Android
QNN Runtime
```

版本。

不要凭记忆硬编码版本。

在真正修改 Gradle 前：

1. 查当前官方文档/仓库。
2. 确认 package 当前可用。
3. 确认 Android arm64-v8a 支持。
4. 确认 QNN EP 是否包含/依赖必要 runtime。
5. 确认当前 Android API 36 可以使用。

如果 Maven package 当前不可用，再考虑其他方案。

---

# 6. 不要依赖本机 Qualcomm SDK

当前开发机：

```text
QNN_SDK_ROOT=
QAIRT_SDK_ROOT=
```

而且没有发现完整 Qualcomm SDK。

本阶段的目标就是验证：

> 是否可以通过官方预编译 ONNX Runtime/QNN Android package，在不安装完整 Qualcomm SDK 的情况下完成设备端 inference。

如果官方 Android package 明确要求本地 SDK 才能构建，则不要伪造 SDK。

记录：

```text
SDK required for build: yes/no
SDK required at runtime: yes/no
```

---

# 7. Gradle 最小改动

只增加 QNN/ONNX Runtime 所需 dependency。

不要大规模升级：

```text
AGP
Gradle
Kotlin
Compose
Android SDK
```

除非 dependency resolution 明确要求。

修改前记录：

```text
gradle dependencies
```

至少确认：

```text
onnxruntime
onnxruntime-qnn
qnn runtime
```

没有产生明显版本冲突。

---

# 8. 添加 Smoke Test 类

建议：

```text
app/src/main/java/com/example/whisperapp/qnn/
```

增加：

```text
QnnSmokeTest.kt
```

职责只有一个：

```text
初始化 ORT
↓
初始化 QNN EP
↓
指定 HTP
↓
加载 tiny CNN ONNX
↓
创建 session
↓
执行一次 warmup
↓
执行 N 次 benchmark
↓
检查 output
↓
打印 execution/provider 信息
```

不要让这个类依赖：

```text
MainActivity
AudioRecorder
QnnAsrEngine
QnnTtsEngine
```

保持完全独立。

---

# 9. QNN EP 配置

核心目标：

```text
QNN Execution Provider
        ↓
HTP backend
```

不能只写：

```text
ONNX Runtime
```

也不能使用：

```text
CPUExecutionProvider
```

作为最终测试。

需要显式配置 QNN EP，并明确指定：

```text
backend = HTP
```

如果当前 ORT/QNN Android API 使用：

```text
backend path
```

或者：

```text
HTP
```

相关配置，则根据当前官方 API 实际写法实现。

不要自行猜 API。

---

# 10. 严格禁止 CPU fallback 当作成功

这是本计划最重要的验收规则。

以下情况：

```text
QNN EP 初始化失败
↓
ORT fallback CPU
↓
模型成功输出
```

必须判定：

```text
FAIL
```

而不是：

```text
PASS
```

因为我们的测试目标是：

```text
HTP/NPU
```

不是：

```text
ONNX Runtime 能运行
```

---

# 11. 模型输入

第一版使用固定 input。

例如：

```text
float32
shape = [1, 3, 224, 224]
```

或者模型实际要求的固定 shape。

测试代码直接生成 deterministic input：

```text
seed = fixed
```

例如：

```text
0.0
0.01
0.02
...
```

不要第一版接 Camera。

不要第一版接 Bitmap。

不要第一版接 UI。

这样 CPU/QNN output 可以稳定比较。

---

# 12. 正确性验证

先在 PC 上使用普通 ONNX Runtime 得到 reference output：

```text
reference_output
```

然后 Android：

```text
QNN/HTP output
```

比较：

```text
max_abs_error
mean_abs_error
```

以及：

```text
top-1
```

如果模型输出适合分类。

验收建议：

```text
shape identical
finite values
no NaN
no Inf
top-1 identical
max error within reasonable tolerance
```

不要要求 QNN 与 CPU bit-exact。

FP32 / mixed precision / HTP 数值差异是允许的。

---

# 13. Benchmark

第一次运行：

```text
session creation
```

单独记录。

不要把它混入 inference latency。

执行：

```text
warmup = 5
iterations = 20
```

记录：

```text
min
median
mean
p95
max
```

至少记录：

```text
warmup latency
inference latency
```

最终输出类似：

```text
QNN Smoke Test
--------------
Model: tiny_cnn.onnx
Input: [1,3,224,224]

Execution Provider:
QNN

Backend:
HTP

Warmup:
OK

Iterations:
20

Latency:
median = xx ms
p95    = xx ms

Output:
shape = [...]
top1 = ...

CPU fallback:
NO

Result:
PASS
```

---

# 14. Android logcat

给 smoke test 使用明确 tag：

```text
QNN_SMOKE
```

所有关键步骤都打印：

```text
QNN_SMOKE: loading model
QNN_SMOKE: creating QNN EP
QNN_SMOKE: backend=HTP
QNN_SMOKE: creating session
QNN_SMOKE: warmup
QNN_SMOKE: inference
QNN_SMOKE: output validation
QNN_SMOKE: benchmark
QNN_SMOKE: PASS
```

同时保存：

```text
adb logcat
```

中的 QNN / ORT 相关日志。

如果 QNN EP 报：

```text
unsupported operator
backend unavailable
library load failure
HTP initialization failure
```

必须原样记录。

不要用 fallback 掩盖。

---

# 15. 最好先做一个 Debug-only 入口

第一版不需要复杂 UI。

可以增加一个非常简单的 debug action，例如：

```text
Run QNN Smoke Test
```

点击后执行：

```text
QnnSmokeTest.run()
```

然后在屏幕显示：

```text
PASS / FAIL
```

以及简短结果。

完整信息仍然写 logcat。

如果这样改 UI 影响现有结构，也可以暂时不改 UI，直接通过：

```text
instrumentation test
```

或 debug-only runner 执行。

优先保持现有 App 不受影响。

---

# 16. 推荐测试方式

优先顺序：

## A. Unit/host reference

PC：

```text
ONNX Runtime CPU
```

得到 reference。

↓

## B. Android instrumentation

真机：

```text
ORT + QNN EP + HTP
```

执行模型。

↓

## C. Compare

比较：

```text
shape
numerical output
top1
latency
```

↓

## D. Logcat

确认：

```text
QNN
HTP
```

没有 CPU fallback。

---

# 17. 模型文件管理

模型放到：

```text
app/src/main/assets/models/
```

例如：

```text
tiny_cnn.onnx
```

不要直接把模型 URL 写进 App。

不要运行时下载。

第一阶段完全 offline。

同时建立：

```text
models/SHA256SUMS.txt
```

记录：

```text
tiny_cnn.onnx <sha256>
```

这样以后不会因为模型文件变化导致测试结果不一致。

---

# 18. 如果模型过大

如果选出的 CNN：

```text
> 20~30 MB
```

不要纠结。

第一 smoke test 可以换更小的模型。

理想目标：

```text
< 10 MB
```

甚至：

```text
< 5 MB
```

因为我们真正想测试的是：

```text
QNN EP + HTP
```

而不是模型下载/安装能力。

---

# 19. 验收标准

## PASS

必须同时满足：

```text
[PASS] APK builds
[PASS] App installs
[PASS] ORT initializes
[PASS] QNN EP initializes
[PASS] HTP backend selected
[PASS] Session created
[PASS] Inference succeeds
[PASS] Output shape correct
[PASS] Output numerically reasonable
[PASS] No CPU fallback
[PASS] Multiple iterations succeed
```

尤其：

```text
HTP backend selected
```

和：

```text
No CPU fallback
```

是硬条件。

---

# 20. FAIL 分类

失败时按照下面分类，不要直接进入 Whisper。

### FAIL-A

```text
Gradle dependency resolution failed
```

说明 Maven/package/版本问题。

---

### FAIL-B

```text
ORT native library load failed
```

检查：

```text
arm64-v8a
Android packaging
JNI libraries
```

---

### FAIL-C

```text
QNN EP unavailable
```

说明 QNN runtime/package 没有正确进入 APK。

---

### FAIL-D

```text
HTP backend unavailable
```

说明 QNN runtime 与设备/backend 存在问题。

---

### FAIL-E

```text
Model unsupported by HTP
```

说明模型算子/shape 不适合第一 smoke test。

换更简单的 CNN。

---

### FAIL-F

```text
QNN initialization succeeds
but inference falls back to CPU
```

判定：

```text
FAIL
```

继续调查 execution provider 配置。

---

### FAIL-G

```text
QNN HTP inference works
but output differs
```

先调查：

```text
precision
quantization
layout
input preprocessing
```

不要马上认为 HTP 有问题。

---

# 21. 完成后的输出物

执行完成后，在项目中留下：

```text
QNN_SMOKE_TEST.md
```

内容包括：

```text
Device
-------
SM8650
Snapdragon 8 Gen 3
Android API 36

Software
--------
ONNX Runtime version
QNN package version
Gradle version
Kotlin version

Model
-----
name
size
sha256
input shape
output shape

Execution
---------
EP = QNN
Backend = HTP
CPU fallback = NO

Benchmark
---------
warmup
iterations
median
p95
max

Correctness
-----------
reference output
QNN output
max abs error
top1

Result
------
PASS / FAIL
```

同时保留：

```text
adb logcat
```

关键日志。

---

# 22. Smoke Test 成功之后

不要直接跳到 MeloTTS。

下一阶段：

```text
tiny CNN
   ↓
PASS
   ↓
tiny Transformer
   ↓
PASS
   ↓
small translation model
   ↓
PASS
   ↓
Whisper-Tiny
```

原因：

CNN 只能证明：

```text
QNN EP + HTP
```

能工作。

Transformer 才能提前验证 Whisper 所需要的大量：

```text
MatMul
Attention
LayerNorm
GEMM
Reshape
Transpose
```

---

# 23. 最终路线

整个项目后续按照：

```text
                 ┌── Tiny CNN ───────── PASS
                 │
Android ORT ─────┼── Tiny Transformer ─ PASS
                 │
                 ├── Translation ────── PASS
                 │
                 └── Whisper Tiny ───── TARGET
                                           │
                                           ↓
                                      AudioRecord
                                           │
                                           ↓
                                       16 kHz PCM
                                           │
                                           ↓
                                      Whisper ASR
                                           │
                                           ↓
                                          Text
                                           │
                                           ↓
                                      MeloTTS-ZH
                                           │
                                           ↓
                                       Speaker
```

当前只做第一格：

```text
Tiny CNN
   ↓
ONNX Runtime
   ↓
QNN EP
   ↓
HTP
```

不要提前把 Whisper、AudioRecord、MeloTTS 混进来。
