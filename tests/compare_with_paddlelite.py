# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's Paddle Lite backend with Paddle Lite's own C++ API on the same .nb model.

The same random float32 inputs go through both on the CPU; the outputs must be identical. Each engine option the backend takes is
checked as well, alone and together, against the reference run with the same settings. The reference is tests/paddlelite_reference.cpp,
built by CMake with the plugin.

    python tests/compare_with_paddlelite.py build/run_model build/libuairt_backend_paddlelite.so build/paddlelite_reference model.nb 1x3x224x224
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main(run_model, plugin, reference, model, shapes_text):
    shapes = [tuple(int(d) for d in s.split("x")) for s in shapes_text.split(";")]
    rng = np.random.default_rng(0)
    arrays = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    cases = [({}, ("1", "normal")),
             ({"num_threads": "1"}, ("1", "normal")),
             ({"num_threads": "4"}, ("4", "normal")),
             ({"num_threads": "2", "power_mode": "high"}, ("2", "normal")),
             ({"precision": "low"}, ("1", "low"))]
    informative = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = []
        for i, array in enumerate(arrays):
            array.tofile(tmp / f"in{i}.bin")
            paths.append(str(tmp / f"in{i}.bin"))
        for options, (threads, precision) in cases:
            flags = [x for k, v in {"input_shapes": shapes_text, **options}.items() for x in ("--option", f"{k}={v}")]
            subprocess.run([run_model, *flags, plugin, "paddlelite", model, str(tmp / "ours"), *paths], check=True, capture_output=True)
            subprocess.run([reference, model, threads, precision, str(tmp / "ref"),
                            *[f"{shape_text}={path}" for shape_text, path in zip(shapes_text.split(";"), paths)]], check=True,
                           capture_output=True)
            i = 0
            while (tmp / f"ref{i}.bin").exists():
                want, got = (tmp / f"ref{i}.bin").read_bytes(), (tmp / f"ours{i}.bin").read_bytes()
                assert want == got, f"output {i} differs for options {options}"
                informative += np.unique(np.frombuffer(want, np.float32)).size > 1
                (tmp / f"ref{i}.bin").unlink()
                (tmp / f"ours{i}.bin").unlink()
                i += 1
            assert i, "the reference wrote no outputs"
            print(f"options {options or '{}'}: identical to Paddle Lite's own API with threads={threads} precision={precision}")
    assert informative, "every output is constant; the comparison proves nothing"
    print("UAIRT Paddle Lite backend matches Paddle Lite")


if __name__ == "__main__":
    main(*sys.argv[1:6])
