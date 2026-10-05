# SPDX-License-Identifier: Apache-2.0
"""Generates the test models and checks them against Python ONNX Runtime."""
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def save(name, nodes, inputs, outputs):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, f"{name}.onnx")


def tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


save(
    "add_mul",
    [helper.make_node("Add", ["x", "y"], ["sum"]),
     helper.make_node("Mul", ["x", "y"], ["product"])],
    [tensor("x", [1, 4]), tensor("y", [1, 4])],
    [tensor("sum", [1, 4]), tensor("product", [1, 4])],
)
save(
    "dynamic",
    [helper.make_node("Identity", ["x"], ["out"])],
    [tensor("x", ["N", 4])],
    [tensor("out", ["N", 4])],
)

x = np.array([[1, 2, 3, 4]], dtype=np.float32)
y = np.array([[10, 20, 30, 40]], dtype=np.float32)
got = ort.InferenceSession("add_mul.onnx").run(None, {"x": x, "y": y})
np.testing.assert_allclose(got[0], x + y)
np.testing.assert_allclose(got[1], x * y)
print("models written and verified against Python ONNX Runtime", ort.__version__)
