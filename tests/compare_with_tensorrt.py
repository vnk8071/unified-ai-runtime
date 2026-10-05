# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's TensorRT backend with TensorRT's own Python API on the same engine.

Same engine and same random inputs go through both; the outputs must be identical. Needs the `tensorrt` and
`torch` (CUDA) Python packages, a GPU, and numpy.

    python tests/compare_with_tensorrt.py build/run_model build/libuairt_backend_tensorrt.so model.engine
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

NUMPY_TO_DTYPE = {"float32": 1, "float16": 2, "int8": 4, "uint8": 5, "int32": 7, "int64": 8, "bool": 9}


def reference(engine_path, inputs):
    logger = trt.Logger(trt.Logger.ERROR)
    data = Path(engine_path).read_bytes()
    length = int.from_bytes(data[:4], "little")
    if 0 < length < len(data) - 4 and data[4:5] == b"{" and data[3 + length:4 + length] == b"}":
        data = data[4 + length:]  # ultralytics metadata header
    engine = trt.Runtime(logger).deserialize_cuda_engine(data)
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    out_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    buffers = {}
    for name, array in zip(in_names, inputs):
        buffers[name] = torch.from_numpy(array).cuda()
        context.set_tensor_address(name, buffers[name].data_ptr())
    for name in out_names:
        shape = tuple(engine.get_tensor_shape(name))
        dtype = torch.from_numpy(np.zeros(0, trt.nptype(engine.get_tensor_dtype(name)))).dtype
        buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return in_names, [buffers[n].cpu().numpy() for n in out_names]


def main(run_model, plugin, engine_path):
    info = subprocess.run([run_model, "--info", plugin, "tensorrt", engine_path], check=True,
                          capture_output=True, text=True).stdout
    print(info.strip())
    shapes = []
    for line in info.splitlines():
        match = re.match(r"input (\d+) name=(\S*) dtype=(\d+) .* dims=(\S*)", line)
        if match:
            dtype = [k for k, v in NUMPY_TO_DTYPE.items() if v == int(match.group(3))][0]
            shapes.append((dtype, tuple(int(d) for d in match.group(4).split("x"))))
    rng = np.random.default_rng(0)
    inputs = [(rng.random(s) * (255 if d == "uint8" else 1)).astype(d) if d.startswith(("float", "uint"))
              else rng.integers(0, 4, s).astype(d) for d, s in shapes]

    _, expected = reference(engine_path, inputs)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, array in enumerate(inputs):
            array.tofile(tmp / f"in{i}.bin")
            paths.append(str(tmp / f"in{i}.bin"))
        subprocess.run([run_model, plugin, "tensorrt", engine_path, str(tmp / "out"), *paths], check=True)
        informative = 0
        for i, want in enumerate(expected):
            got = np.fromfile(tmp / f"out{i}.bin", dtype=want.dtype).reshape(want.shape)
            np.testing.assert_array_equal(got, want)
            informative += np.unique(want).size > 1
            print(f"output {i}: {want.shape} {want.dtype} identical to TensorRT ({np.unique(want).size} distinct values)")
    assert informative, "every output is constant; the comparison proves nothing"
    print("UAIRT TensorRT backend matches TensorRT")


if __name__ == "__main__":
    main(*sys.argv[1:4])
