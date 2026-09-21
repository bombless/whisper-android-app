"""Generate ONNX Runtime EPContext wrappers for the Whisper-Large-V3-Turbo QNN asset.

The Qualcomm AI Hub asset ships raw QAIRT context binaries (encoder.bin / decoder.bin)
plus a metadata.json that declares their I/O contract. ONNX Runtime's QNN execution
provider cannot consume a bare .bin; it loads it through an `EPContext` node whose
`ep_cache_context` attribute points at the binary.

This mirrors the already-validated Tiny wrappers: same opset, same attribute set, same
producer/graph naming, same input/output ordering as metadata.json. Only the tensor
shapes differ (128 mel bins, 20 attention heads, 51866-token vocabulary).
"""
from __future__ import annotations

import argparse
import json
import os

import onnx
from onnx import TensorProto, helper

DTYPES = {
    "float16": TensorProto.FLOAT16,
    "float32": TensorProto.FLOAT,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "uint16": TensorProto.UINT16,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
}

# Mirrors the validated Tiny wrappers exactly, so the ORT/QNN load path is identical.
# encoder_ctx.onnx in the Tiny asset uses producer_name "Whisper QNN Android" with an
# empty producer_version; the decoder uses "com.microsoft"/"qnn-onnx-model".
PRODUCERS = {
    "encoder.bin": ("Whisper QNN Android", ""),
    "decoder.bin": ("com.microsoft", "qnn-onnx-model"),
}
NODE_NAMES = {
    "encoder.bin": "hf_whisper_encoder",
    "decoder.bin": "hf_whisper_decoder",
}


def value_info(name: str, spec: dict):
    dtype = DTYPES[spec["dtype"]]
    shape = [int(d) for d in spec["shape"]]
    return helper.make_tensor_value_info(name, dtype, shape)


def build_wrapper(bin_name: str, spec: dict, sdk_version: str) -> onnx.ModelProto:
    producer_name, producer_version = PRODUCERS[bin_name]
    inputs = [value_info(n, s) for n, s in spec["inputs"].items()]
    outputs = [value_info(n, s) for n, s in spec["outputs"].items()]

    node = helper.make_node(
        "EPContext",
        inputs=[v.name for v in inputs],
        outputs=[v.name for v in outputs],
        name=NODE_NAMES[bin_name],
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_name,
        ep_sdk_version=sdk_version,
        source="Qnn",
    )

    graph = helper.make_graph(
        [node],
        "qnn-onnx-model",
        inputs,
        outputs,
    )
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version=producer_version,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 11
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app", "src", "main", "assets", "models", "whisper_large_v3_turbo",
        ),
    )
    args = parser.parse_args()

    metadata_path = os.path.join(args.asset_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    sdk_version = metadata["tool_versions"]["qairt"]
    outputs = []
    for bin_name in ("encoder.bin", "decoder.bin"):
        binary_path = os.path.join(args.asset_dir, bin_name)
        if not os.path.isfile(binary_path):
            raise SystemExit(f"missing context binary: {binary_path}")
        spec = metadata["model_files"][bin_name]
        model = build_wrapper(bin_name, spec, sdk_version)
        onnx.checker.check_model(model)
        out_path = os.path.join(args.asset_dir, bin_name.replace(".bin", "_ctx.onnx"))
        onnx.save(model, out_path)
        outputs.append(
            (bin_name, os.path.basename(out_path), os.path.getsize(out_path),
             len(spec["inputs"]), len(spec["outputs"]), os.path.getsize(binary_path))
        )

    for bin_name, name, size, n_in, n_out, bin_size in outputs:
        print(f"OK {name} bytes={size} from={bin_name}({bin_size}) inputs={n_in} outputs={n_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
