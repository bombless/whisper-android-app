# SDXL on ONNX Runtime + Qualcomm QNN + HTP (异构执行方案)

本目录提供了将 **Stable Diffusion XL (SDXL)** 跑在 **高通 Hexagon NPU (HTP)** 上的完整端到端工程实现。

---

## 1. 核心异构计算设计

| 组件 | 运行硬件 | Execution Provider | 模型文件 | 优势与原因 |
| :--- | :--- | :--- | :--- | :--- |
| **UNet** (2.6B) | **Qualcomm HTP (NPU)** | `QnnExecutionProvider` | `unet.bin` + `unet_ctx.onnx` | 承担 90% 以上计算量，W8A16 量化释放 HTP 最大算力与能效 |
| **VAE Decoder** | **CPU** | `CPUExecutionProvider` | `taesdxl_decoder.onnx` | 彻底规避 NPU 片上 SRAM (TCM) 溢出，FP32 运行保证出图画质不失真 |
| **Scheduler** | **CPU** | Python / 原生代码 | 4-step Euler 调度算法 | 配合 SDXL-Lightning / Turbo 仅需 4 步迭代，总耗时极短 |

---

## 2. 目录文件结构

* `build_sdxl_epcontext.py`：生成包装 `unet.bin` 的 ONNX `EPContext` 模型，遵循 Qualcomm QNN EP 规范。
* `taesdxl_decoder.onnx`：轻量级 TAESDXL VAE 模型（仅 4.9MB，替代 84MB 的原版庞大 VAE）。
* `unet_ctx.onnx`：已生成的 SDXL UNet EPContext 包装器。
* `sdxl_pipeline.py`：端到端推理运行脚本。

---

## 3. 运行方式

### ① 本机快速验证（CPU / Mock 测试 Pipeline）
验证整个调度循环、张量形状传递以及 CPU 上的 TAESDXL 图像解码：
```bash
python tools/sdxl_qnn/sdxl_pipeline.py --backend cpu --output tools/sdxl_qnn/output_sample.png
```

### ② 部署到 Snapdragon 8 Gen 3 真机 (Android) 或 Snapdragon X Elite (Windows ARM)
1. 从 Hugging Face 获取适配高通 HTP (v73/v75) 的 SDXL UNet context binary：
   * 推荐预编译权重：[`CelesteImperia/SDXL-QNN`](https://huggingface.co/CelesteImperia/SDXL-QNN) 中的 `unet.bin`（约 2.39GB，W8A16 量化）。
2. 将 `unet.bin` 放置在与 `unet_ctx.onnx` 相同目录下。
3. 执行 HTP 模式推理：
   ```bash
   python tools/sdxl_qnn/sdxl_pipeline.py --backend htp --output result.png
   ```
