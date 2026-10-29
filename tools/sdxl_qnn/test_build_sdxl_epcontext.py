import copy
import json
import unittest
from pathlib import Path

from onnx import TensorProto

from build_sdxl_epcontext import build_wrapper


class SdxlContextTest(unittest.TestCase):
    def setUp(self):
        self.metadata = json.loads((Path(__file__).parent / "unet-metadata.json").read_text(encoding="utf-8"))

    def test_uses_real_qnn_graph_contract(self):
        model = build_wrapper(self.metadata, "unet.bin")
        self.assertEqual(model.graph.node[0].name, "model")
        self.assertEqual([value.name for value in model.graph.input],
                         ["sample", "encoder_hidden_states", "timestamp", "time_ids", "text_embeds"])
        self.assertEqual(model.graph.input[2].type.tensor_type.elem_type, TensorProto.INT32)
        self.assertEqual([value.name for value in model.graph.output], ["output"])
        self.assertEqual([dim.dim_value for dim in model.graph.input[1].type.tensor_type.shape.dim], [1, 77, 2048])

    def test_rejects_unsupported_tensor_type(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["info"]["graphs"][0]["info"]["graphInputs"][0]["info"]["dataType"] = "unknown"
        with self.assertRaises(ValueError):
            build_wrapper(metadata, "unet.bin")

    def test_rejects_multiple_graphs(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["info"]["graphs"].append(metadata["info"]["graphs"][0])
        with self.assertRaises(ValueError):
            build_wrapper(metadata, "unet.bin")


if __name__ == "__main__":
    unittest.main()
