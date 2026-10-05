# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's OpenVINO backend with OpenVINO's own Python API on the same model.

The same random inputs go through both on the CPU device; the outputs must be identical. Needs the `openvino` Python
package (the same version the plugin was built against) and numpy.

    python tests/compare_with_openvino.py build/run_model build/libuairt_backend_openvino.so model.xml
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov

DTYPES = {1: np.float32, 2: np.float16, 4: np.int8, 5: np.uint8, 6: np.int16, 7: np.int32, 8: np.int64,
          9: np.bool_, 10: np.uint16}


def main(run_model, plugin, model_path):
    info = subprocess.run([run_model, "--info", plugin, "openvino", model_path], check=True,
                          capture_output=True, text=True).stdout
    print(info.strip())
    inputs = []
    for line in info.splitlines():
        match = re.match(r"input (\d+) name=(\S*) dtype=(\d+) .* dims=(\S*)", line)
        if match:
            inputs.append((DTYPES[int(match.group(3))], tuple(int(d) for d in match.group(4).split("x"))))
    rng = np.random.default_rng(0)
    arrays = [rng.random(shape).astype(dtype) if np.issubdtype(dtype, np.floating)
              else rng.integers(0, 4, shape).astype(dtype) for dtype, shape in inputs]

    compiled = ov.Core().compile_model(model_path, "CPU")
    expected = [np.asarray(v) for v in compiled(arrays).values()]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, array in enumerate(arrays):
            array.tofile(tmp / f"in{i}.bin")
            paths.append(str(tmp / f"in{i}.bin"))
        subprocess.run([run_model, plugin, "openvino", model_path, str(tmp / "out"), *paths], check=True)
        informative = 0
        for i, want in enumerate(expected):
            got = np.fromfile(tmp / f"out{i}.bin", dtype=want.dtype).reshape(want.shape)
            np.testing.assert_array_equal(got, want)
            informative += np.unique(want).size > 1
            print(f"output {i}: {want.shape} {want.dtype} identical to OpenVINO ({np.unique(want).size} distinct values)")
    assert informative, "every output is constant; the comparison proves nothing"
    print("UAIRT OpenVINO backend matches OpenVINO")


if __name__ == "__main__":
    main(*sys.argv[1:4])
