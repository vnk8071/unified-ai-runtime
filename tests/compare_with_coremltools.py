# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's CoreML backend with CoreML itself (through coremltools).

Builds three small models (Add/Mul, a conv net with random weights, and a conv net with an
image input), compiles them with coremlcompiler, and checks UAIRT's outputs against
MLModel.predict. Needs macOS, coremltools, numpy, pillow and Xcode.

    python tests/compare_with_coremltools.py build/run_model build/libuairt_backend_coreml.so
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

DTYPES = {1: np.float32, 2: np.float16, 5: np.uint8, 7: np.int32}


def build_models(directory):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4)), mb.TensorSpec(shape=(1, 4))])
    def add_mul(x, y):
        return mb.add(x=x, y=y, name="sum"), mb.mul(x=x, y=y, name="product")

    rng = np.random.default_rng(0)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 64, 64))])
    def convnet(image):
        w1 = rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.2
        y = mb.conv(x=image, weight=w1, strides=[2, 2], pad_type="same")
        y = mb.relu(x=y)
        y = mb.reduce_mean(x=y, axes=[2, 3], keep_dims=False)
        w2 = rng.standard_normal((10, 8)).astype(np.float32)
        b2 = rng.standard_normal(10).astype(np.float32)
        return mb.linear(x=y, weight=w2, bias=b2, name="logits")

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 64, 64))])
    def imagenet(image):
        w = rng.standard_normal((8, 3, 3, 3)).astype(np.float32) * 0.2
        y = mb.relu(x=mb.conv(x=image, weight=w, strides=[2, 2], pad_type="same"))
        return mb.reduce_mean(x=y, axes=[2, 3], keep_dims=False, name="features")

    image = ct.ImageType(name="image", shape=(1, 3, 64, 64), scale=1 / 255, color_layout=ct.colorlayout.BGR)
    models = {}
    for name, program, inputs in [("add_mul", add_mul, None), ("convnet", convnet, None),
                                  ("imagenet", imagenet, [image])]:
        package = directory / f"{name}.mlpackage"
        ct.convert(program, inputs=inputs, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                   minimum_deployment_target=ct.target.macOS13).save(str(package))
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(directory)],
                       check=True, capture_output=True)
        models[name] = (package, directory / f"{name}.mlmodelc")
    return models


def parse_info(text):
    tensors = {"input": [], "output": []}
    for line in text.splitlines():
        m = re.match(r"(input|output) (\d+) name=(\S*) dtype=(\d+) .* dims=(\S*)", line)
        if m:
            kind, _, name, dtype, dims = m.groups()
            tensors[kind].append((name, DTYPES[int(dtype)], tuple(int(d) for d in dims.split("x"))))
    return tensors


def compare(run_model, plugin, package, compiled, units, tmp):
    base = [run_model, "--option", f"compute_units={units}"]
    info = subprocess.run([*base, "--info", plugin, "coreml", str(compiled)], check=True,
                          capture_output=True, text=True).stdout
    tensors = parse_info(info)
    names_in = [name for name, _, _ in tensors["input"]]
    assert names_in == sorted(names_in), "inputs must be ordered by name"

    rng = np.random.default_rng(1)
    arrays, paths = {}, []
    raw = {}
    for i, (name, dtype, shape) in enumerate(tensors["input"]):
        if dtype == np.uint8:  # an image input: BGRA bytes for UAIRT, the same pixels as an image for CoreML
            from PIL import Image
            raw[name] = rng.integers(0, 256, shape, dtype=np.uint8)
            arrays[name] = Image.fromarray(raw[name][0][..., [2, 1, 0]], "RGB")
        else:
            raw[name] = arrays[name] = rng.standard_normal(shape).astype(dtype)
        path = tmp / f"in{i}.bin"
        raw[name].tofile(path)
        paths.append(str(path))
    prefix = tmp / f"out_{package.stem}_{units}_"
    subprocess.run([*base, plugin, "coreml", str(compiled), str(prefix), *paths], check=True)

    unit = {"cpu_only": ct.ComputeUnit.CPU_ONLY, "all": ct.ComputeUnit.ALL}[units]
    expected = ct.models.MLModel(str(package), compute_units=unit).predict(arrays)
    for i, (name, dtype, shape) in enumerate(tensors["output"]):
        actual = np.fromfile(f"{prefix}{i}.bin", dtype=dtype).reshape(shape)
        diff = float(np.max(np.abs(actual.astype(np.float64) - np.asarray(expected[name], np.float64))))
        np.testing.assert_allclose(actual, np.asarray(expected[name]), rtol=1e-4, atol=1e-5)
        assert np.unique(actual).size > 1, f"output {name} is constant"
        print(f"{package.stem} [{units}] output {i} {name} {shape}: max abs diff {diff:.3g}")


def main():
    run_model, plugin = sys.argv[1], sys.argv[2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, (package, compiled) in build_models(tmp).items():
            for units in ("cpu_only", "all"):
                compare(run_model, plugin, package, compiled, units, tmp)
    print("UAIRT CoreML backend matches CoreML")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main()
