# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's NCNN backend with NCNN's own Python API on the same model.

The same random input goes through both. The reference runs on the CPU; a Vulkan run uses fp32 here (`fp16=false`),
because NCNN's default fp16 on the GPU differs from the CPU by far more than rounding. Needs the `ncnn` Python package
and numpy.

    python tests/compare_with_ncnn.py build/run_model build/libuairt_backend_ncnn.so model_ncnn_model 3x640x640 \
        [cpu|vulkan] [rtol] [atol]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import ncnn
import numpy as np


def main(run_model, plugin, model_dir, shape, device="cpu", rtol=None, atol=None):
    rtol = rtol or ("1e-4" if device == "cpu" else "1e-3")
    atol = atol or ("1e-3" if device == "cpu" else "1e-2")
    dims = tuple(int(d) for d in shape.split("x"))
    param = next(Path(model_dir).glob("*.param"))
    weights = param.with_suffix(".bin")
    x = np.random.default_rng(0).random(dims, dtype=np.float32)

    with ncnn.Net() as net:
        net.opt.use_vulkan_compute = False
        net.load_param(str(param))
        net.load_model(str(weights))
        with net.create_extractor() as ex:
            ex.input(net.input_names()[0], ncnn.Mat(x).clone())
            expected = []
            for name in net.output_names():
                _, mat = ex.extract(name)
                expected.append(np.array(mat))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        x.tofile(tmp / "in0.bin")
        subprocess.run([run_model, "--option", f"input_shapes={shape}", "--option", f"device={device}",
                        "--option", "fp16=false", plugin, "ncnn", model_dir, str(tmp / "out"), str(tmp / "in0.bin")], check=True)
        informative = 0
        for i, want in enumerate(expected):
            got = np.fromfile(tmp / f"out{i}.bin", dtype=np.float32).reshape(want.shape)
            np.testing.assert_allclose(got, want, rtol=float(rtol), atol=float(atol))
            informative += np.unique(want).size > 1
            print(f"output {i}: {want.shape} matches NCNN on the CPU within rtol={rtol} atol={atol} "
                  f"(max abs diff {np.abs(got - want).max():.3g})")
    assert informative, "every output is constant; the comparison proves nothing"
    print(f"UAIRT NCNN backend ({device}) matches NCNN")


if __name__ == "__main__":
    main(*sys.argv[1:8])
