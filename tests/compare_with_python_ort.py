# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's ONNX Runtime backend with Python ONNX Runtime on MobileNetV2.

The model is exported from torchvision with random weights (no download), so this
checks numerical agreement, not accuracy. Needs torch, torchvision, onnxruntime, numpy.

    python3 tests/compare_with_python_ort.py build/run_model build/libuairt_backend_onnxruntime.so
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision


def main(run_model, plugin):
    torch.manual_seed(0)
    network = torchvision.models.mobilenet_v2(weights=None).eval()
    sample = torch.from_numpy(
        np.random.default_rng(0).standard_normal((1, 3, 224, 224), dtype=np.float32)
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "mobilenet_v2.onnx"
        torch.onnx.export(network, sample, str(model), input_names=["input"],
                          output_names=["logits"], dynamo=False)
        sample.numpy().tofile(tmp / "in0.bin")

        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        expected = session.run(None, {"input": sample.numpy()})[0]

        start = time.perf_counter()
        subprocess.run([run_model, plugin, "onnxruntime", str(model),
                        str(tmp / "out"), str(tmp / "in0.bin")], check=True)
        elapsed = time.perf_counter() - start
        actual = np.fromfile(tmp / "out0.bin", dtype=np.float32).reshape(expected.shape)

    diff = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print(f"MobileNetV2 outputs match Python ONNX Runtime {ort.__version__}: "
          f"max abs diff {diff:.3g}, argmax {int(actual.argmax())} == {int(expected.argmax())}, "
          f"run_model wall time {elapsed:.2f}s (includes process start and model load)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
