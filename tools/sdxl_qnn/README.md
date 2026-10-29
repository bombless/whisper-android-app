# SDXL 文生图

应用的“文生图”标签在 Android 进程内运行 SDXL：两个 CLIP 文本编码器和 TAESDXL 解码器使用 ONNX Runtime CPU，UNet 使用 QNN HTP。当前模型是 **SDXL Base**，不是 Lightning/Turbo；默认使用 20 步 Euler 和 CFG 7。模型固定为 1024×1024，目标设备是 Snapdragon 8 Gen 3 / SM8650。当前不支持其他模型、其他分辨率或提示词权重语法。

在 RMX3800（SM8650）上，已验证从提示词生成图片以及保存到相册。一次端到端诊断用时约 72 秒，实际时间取决于设备内存和温度；不应采用旧计划中 4.5 秒的估计。

## 下载与部署

需要 Python 3、`onnx`、`curl`、`adb`、已连接的设备，以及已安装的本项目 **debug** APK。主机应预留约 9 GB，设备至少预留约 5 GB。模型文件不打包到 APK，也不提交到 Git。

```powershell
python -m pip install onnx
python tools/sdxl_qnn/prepare_sdxl.py --proxy http://127.0.0.1:7897
```

脚本优先从 `hf-mirror.com` 下载，仅当镜像下载失败且指定 `--proxy` 时才通过代理访问 Hugging Face。它校验 SHA-256、从模型包中提取固定文件、生成符合真实图接口的包装器，并将文件传到应用专属目录。部署时会先停止应用，避免覆盖正在使用的模型。`--download-only` 仅准备主机缓存，不部署设备。若下载中断，可重运行；如果校验失败且断点续传无法恢复，删除报错的缓存文件再试。

下载源固定到以下版本：

- UNet 和 tokenizer：`xororz/sdxl-qnn` 的 `sdxl_base_qnn2.28_8gen3.zip`，修订版 `ead90f4635e21e7412b8200a5efd220b0193beeb`。
- 两个 CLIP ONNX 编码器：`Cyronius/sdxl-lightning-4step-onnx-web-int8`，修订版 `16aee6ccb10aee640bb37c7072763ad4d090e4ca`；此处只使用其 SDXL 文本编码器，不使用 Lightning UNet。
- TAESDXL ONNX 解码器：本目录中的 `taesdxl_decoder.onnx`。

模型分发和使用需遵守相应模型的许可条款。UNet 仓库未声明明确的模型包许可，使用前还应核对其上游模型和仓库的权利与限制。

设备文件位于应用的 `getExternalFilesDir(null)/sdxl`，包括 `unet.bin`、`unet_ctx.onnx`、`tokenizer.json`、`vae.onnx`，以及 `text_encoder/` 和 `text_encoder_2/` 内的模型和外部权重。卸载应用通常会删除这些文件。部署脚本只支持本项目 debug 包，因为它使用 `run-as` 启动准备步骤。

## 图接口与旧原型

`unet-metadata.json` 来自 Qualcomm `qnn-context-binary-utility` 对上述 UNet 的实际检查。图名为 `model`，输入依次是 `sample`（FP32）、`encoder_hidden_states`（FP32）、`timestamp`（INT32）、`time_ids`（FP32）和 `text_embeds`（FP32），输出为 `output`（FP32）。`build_sdxl_epcontext.py` 读取该元数据生成包装器，不推测二进制的输入规格。

目录中旧的 `sdxl_pipeline.py` 和 `unet_ctx.onnx` 是早期实验品：前者使用随机文本特征和模拟去噪，后者也不匹配此模型的实际接口。它们不能证明真实生成，也不被 Android 应用使用。
