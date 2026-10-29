"""Build an ONNX EPContext wrapper from QNN context-binary metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper


DTYPES = {
    "QNN_DATATYPE_FLOAT_32": TensorProto.FLOAT,
    "QNN_DATATYPE_FLOAT_16": TensorProto.FLOAT16,
    "QNN_DATATYPE_INT_32": TensorProto.INT32,
    "QNN_DATATYPE_INT_64": TensorProto.INT64,
    "QNN_DATATYPE_UINT_8": TensorProto.UINT8,
    "QNN_DATATYPE_UINT_16": TensorProto.UINT16,
}


def build_wrapper(metadata: dict, bin_name: str) -> onnx.ModelProto:
    info = metadata["info"]
    graphs = info["graphs"]
    if len(graphs) != 1:
        raise ValueError(f"Expected one QNN graph, got {len(graphs)}")
    graph = graphs[0]["info"]

    def value_info(tensor: dict):
        spec = tensor["info"]
        dtype = DTYPES.get(spec["dataType"])
        if dtype is None:
            raise ValueError(f"Unsupported QNN tensor type: {spec['dataType']}")
        return helper.make_tensor_value_info(spec["name"], dtype, spec["dimensions"])

    inputs = [value_info(tensor) for tensor in graph["graphInputs"]]
    outputs = [value_info(tensor) for tensor in graph["graphOutputs"]]
    node = helper.make_node(
        "EPContext",
        inputs=[value.name for value in inputs],
        outputs=[value.name for value in outputs],
        name=graph["graphName"],
        domain="com.microsoft",
        embed_mode=0,
        ep_cache_context=bin_name,
        ep_sdk_version=info["buildId"].removeprefix("v").split("_")[0],
        source="Qnn",
    )
    model = helper.make_model(
        helper.make_graph([node], "qnn-onnx-model", inputs, outputs),
        producer_name="Whisper QNN Android",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--bin-name", default="unet.bin")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if Path(args.bin_name).name != args.bin_name:
        parser.error("--bin-name must be a file name in the wrapper directory")
    with args.metadata.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    onnx.save(build_wrapper(metadata, args.bin_name), args.output)
    print(f"Generated {args.output} for {args.bin_name}")


if __name__ == "__main__":
    main()
