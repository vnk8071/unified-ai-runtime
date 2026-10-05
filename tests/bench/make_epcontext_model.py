# SPDX-License-Identifier: Apache-2.0
"""Wraps a QNN context binary in an "EPContext" ONNX model so ONNX Runtime's QNN execution
provider can run it. ONNX Runtime and UAIRT then execute the same compiled graph.

    python make_epcontext_model.py model.bin graph_name model_ctx.onnx \
        --input image:uint8:1x640x640x3 \
        --output class_idx:uint8:1x8400 --output boxes:uint8:1x8400x4 --output scores:uint8:1x8400

`graph_name` is the QNN graph inside the binary (for example the name printed by `strings`
or `qnn-context-binary-utility`). The binary is referenced by file name, so keep it next to
the ONNX file. Tensor names, types and shapes must match the graph's I/O (`run_model --info`).
Needs the `onnx` Python package.
"""
import argparse

import onnx
from onnx import TensorProto, helper

TYPES = {"uint8": TensorProto.UINT8, "int8": TensorProto.INT8, "uint16": TensorProto.UINT16,
         "int32": TensorProto.INT32, "float32": TensorProto.FLOAT, "float16": TensorProto.FLOAT16}


def value_info(spec):
    name, dtype, shape = spec.split(":")
    return helper.make_tensor_value_info(name, TYPES[dtype], [int(d) for d in shape.split("x")])


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("binary")
parser.add_argument("graph_name")
parser.add_argument("output")
parser.add_argument("--input", action="append", required=True)
parser.add_argument("--output", dest="outputs", action="append", required=True)
parser.add_argument("--source", default="Qnn")
args = parser.parse_args()

inputs = [value_info(s) for s in args.input]
outputs = [value_info(s) for s in args.outputs]
node = helper.make_node(
    "EPContext", [i.name for i in inputs], [o.name for o in outputs], name=args.graph_name,
    domain="com.microsoft", main_context=1, embed_mode=0, ep_cache_context=args.binary,
    source=args.source, onnx_model_filename=args.output, partition_name=args.graph_name, notes="")
model = helper.make_model(helper.make_graph([node], "epcontext", inputs, outputs),
                          opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10
onnx.save(model, args.output)
print("wrote", args.output)
