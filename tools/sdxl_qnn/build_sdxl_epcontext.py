"""Generate ONNX Runtime EPContext wrapper for SDXL UNet QNN Context Binary.

This allows ONNX Runtime with QnnExecutionProvider to directly load and execute
the Qualcomm HTP-compiled `unet.bin` context binary.
"""
from __future__ import annotations

import argparse
import os
import onnx
from onnx import TensorProto, helper

DTYPES = {
    "float16": TensorProto.FLOAT16,
    "float32": TensorProto.FLOAT,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
}

def make_sdxl_unet_epcontext(
    bin_name: str = "unet.bin",
    sdk_version: str = "2.28.0.241029",
    dtype: str = "float32",
    batch_size: int = 1,
) -> onnx.ModelProto:
    dt = DTYPES[dtype]
    
    # Standard SDXL UNet 1024x1024 contract (Latents: 128x128)
    inputs = [
        helper.make_tensor_value_info("sample", dt, [batch_size, 4, 128, 128]),
        helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("encoder_hidden_states", dt, [batch_size, 77, 2048]),
        helper.make_tensor_value_info("text_embeds", dt, [batch_size, 1280]),
        helper.make_tensor_value_info("time_ids", TensorProto.FLOAT, [batch_size, 6]),
    ]
    
    outputs = [
        helper.make_tensor_value_info("out_sample", dt, [batch_size, 4, 128, 128]),
    ]
    
    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs],
        outputs=[v.name for v in outputs],
        name="qnn_sdxl_unet",
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_name,
        ep_sdk_version=sdk_version,
        source="Qnn",
    )
    
    graph = helper.make_graph(
        [node],
        "qnn-sdxl-unet-model",
        inputs,
        outputs,
    )
    
    model = helper.make_model(
        graph,
        producer_name="Qualcomm QNN SDXL Wrapper",
        producer_version="1.0",
        opset_imports=[
            helper.make_opsetid("", 19),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-name", default="unet.bin", help="Context binary file name")
    parser.add_argument("--sdk-version", default="2.28.0", help="QNN SDK version")
    parser.add_argument("--output", default="tools/sdxl_qnn/unet_ctx.onnx", help="Output ONNX path")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = parser.parse_args()

    model = make_sdxl_unet_epcontext(
        bin_name=args.bin_name,
        sdk_version=args.sdk_version,
        dtype=args.dtype,
    )
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"Generated {args.output} successfully! (Targets {args.bin_name}, SDK {args.sdk_version})")

if __name__ == "__main__":
    main()
