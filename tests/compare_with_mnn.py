# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's MNN backend with MNN's own Python API (PyMNN) on the same model.

The same random float32 inputs go through both on the CPU; the outputs must be identical. Each engine option the backend takes
is checked as well, alone and together, against PyMNN configured the same way. Needs the `MNN` Python package (the same version
as the pinned submodule, third_party/MNN) and numpy; make the model with `mnnconvert -f ONNX --modelFile m.onnx --MNNModel m.mnn`.

    python tests/compare_with_mnn.py build/run_model build/libuairt_backend_mnn.so model.mnn
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import MNN
import numpy as np


def parse_info(text):
    tensors = {"input": [], "output": []}
    for line in text.splitlines():
        match = re.match(r"(input|output) (\d+) name=(\S*) dtype=(\d+) .* dims=(\S*)", line)
        if match:
            kind, _, name, dtype, dims = match.groups()
            assert int(dtype) == 1, f"{kind} {name} is not float32; this script compares float32 models"
            tensors[kind].append((name, tuple(int(d) for d in dims.split("x"))))
    return tensors


def main(run_model, plugin, model_path):
    info = subprocess.run([run_model, "--info", plugin, "mnn", model_path], check=True, capture_output=True, text=True).stdout
    print("\n".join(line for line in info.splitlines() if line.startswith(("input", "output"))))
    tensors = parse_info(info)
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal(shape).astype(np.float32) for _, shape in tensors["input"]]

    interpreter = MNN.Interpreter(model_path)

    def mnn_outputs(config):
        session = interpreter.createSession(config)
        for (name, shape), array in zip(tensors["input"], arrays):
            tensor = interpreter.getSessionInput(session, name)
            tensor.copyFrom(MNN.Tensor(shape, MNN.Halide_Type_Float, array, tensor.getDimensionType()))
        interpreter.runSession(session)
        results = []
        for name, shape in tensors["output"]:
            tensor = interpreter.getSessionOutput(session, name)
            host = MNN.Tensor(shape, MNN.Halide_Type_Float, np.zeros(shape, np.float32), tensor.getDimensionType())
            tensor.copyToHostTensor(host)
            results.append(np.array(host.getData(), np.float32).reshape(shape))
        return results

    def uairt_outputs(options):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = []
            for i, array in enumerate(arrays):
                array.tofile(tmp / f"in{i}.bin")
                paths.append(str(tmp / f"in{i}.bin"))
            flags = [x for k, v in options.items() for x in ("--option", f"{k}={v}")]
            subprocess.run([run_model, *flags, plugin, "mnn", model_path, str(tmp / "out"), *paths], check=True,
                           capture_output=True)
            return [np.fromfile(tmp / f"out{i}.bin", dtype=np.float32).reshape(shape)
                    for i, (_, shape) in enumerate(tensors["output"])]

    cases = [({}, {"numThread": 4, "precision": "normal"}),
             ({"num_threads": "2"}, {"numThread": 2, "precision": "normal"}),
             ({"precision": "high"}, {"numThread": 4, "precision": "high"}),
             ({"precision": "low"}, {"numThread": 4, "precision": "low"}),
             ({"num_threads": "1", "precision": "high"}, {"numThread": 1, "precision": "high"})]
    informative = 0
    for options, config in cases:
        want, got = mnn_outputs(config), uairt_outputs(options)
        for i, (w, g) in enumerate(zip(want, got)):
            np.testing.assert_array_equal(g, w)
            informative += np.unique(w).size > 1
        print(f"options {options or '{}'}: identical to PyMNN with {config}")
    assert informative, "every output is constant; the comparison proves nothing"
    print("UAIRT MNN backend matches MNN")


if __name__ == "__main__":
    main(*sys.argv[1:4])
