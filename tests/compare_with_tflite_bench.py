# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's TFLite backend with tflite_bench (TFLite's own C API) on one model.

The same random native-dtype input goes through both, with matching delegate options, and the
outputs must be identical. Run on the target device (see tests/bench/README.md).

    python3 tests/compare_with_tflite_bench.py build/run_model build/libuairt_backend_tflite.so \\
        tflite_bench model.tflite [--delegate qnn --delegate-library libQnnTFLiteDelegate.so \\
        --backend-lib libQnnHtp.so --skel-dir DIR --perf default]
"""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DTYPES = {1: np.float32, 2: np.float16, 4: np.int8, 5: np.uint8, 6: np.int16, 7: np.int32,
          8: np.int64, 9: np.bool_, 10: np.uint16}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_model")
    parser.add_argument("plugin")
    parser.add_argument("tflite_bench")
    parser.add_argument("model")
    parser.add_argument("--delegate", choices=["none", "qnn"], default="none")
    parser.add_argument("--delegate-library")
    parser.add_argument("--backend-lib")
    parser.add_argument("--skel-dir", default="")
    parser.add_argument("--perf", default="default")
    args = parser.parse_args()

    options = ["--option", f"delegate={args.delegate}"]
    bench = [args.tflite_bench]
    env = dict(os.environ)
    if args.delegate == "qnn":
        options += ["--option", f"delegate_library={args.delegate_library}",
                    "--option", f"backend_library={args.backend_lib}",
                    "--option", f"skel_dir={args.skel_dir}",
                    "--option", f"htp_performance_mode={args.perf}",
                    "--option", f"adsp_library_path={args.skel_dir};/usr/lib/rfsa/adsp"]
        bench += ["--delegate-backend", "htp", "--backend-lib", args.backend_lib,
                  "--skel-dir", args.skel_dir, "--perf", args.perf]
        env["ADSP_LIBRARY_PATH"] = f"{args.skel_dir};/usr/lib/rfsa/adsp"

    info = subprocess.run([args.run_model, *options, "--info", args.plugin, "tflite", args.model],
                          check=True, capture_output=True, text=True, env=env).stdout
    print(info.strip())
    tensors = {"input": [], "output": []}
    for line in info.splitlines():
        m = re.match(r"(input|output) \d+ name=(\S*) dtype=(\d+) .* dims=(\S*)", line)
        if m:
            kind, name, dtype, dims = m.groups()
            tensors[kind].append((name, DTYPES[int(dtype)], tuple(int(d) for d in dims.split("x"))))
    assert len(tensors["input"]) == 1, "this comparison takes models with one input"

    rng = np.random.default_rng(0)
    name, dtype, shape = tensors["input"][0]
    if np.issubdtype(dtype, np.floating):
        data = rng.random(shape, dtype=np.float32).astype(dtype)
    else:
        info_ = np.iinfo(dtype)
        data = rng.integers(max(info_.min, 0), min(info_.max, 255) + 1, shape).astype(dtype)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data.tofile(tmp / "in.bin")
        subprocess.run([*bench, args.model, str(tmp / "in.bin"), str(tmp / "ref_"), "1"],
                       check=True, capture_output=True, env=env)
        subprocess.run([args.run_model, *options, args.plugin, "tflite", args.model,
                        str(tmp / "uairt_"), str(tmp / "in.bin")], check=True, capture_output=True, env=env)
        non_constant = 0
        for i, (name, dtype, shape) in enumerate(tensors["output"]):
            expected = np.fromfile(tmp / f"ref_{i}.bin", dtype=dtype).reshape(shape)
            actual = np.fromfile(tmp / f"uairt_{i}.bin", dtype=dtype).reshape(shape)
            assert np.array_equal(actual, expected), f"output {i} ({name}) differs from tflite_bench"
            distinct = int(np.unique(actual).size)
            non_constant += distinct > 1
            print(f"output {i} {name}: {shape} {np.dtype(dtype).name} identical to tflite_bench "
                  f"({distinct} distinct values)")
    assert non_constant, "every output is constant; the comparison proves nothing"
    print(f"UAIRT TFLite backend matches tflite_bench ({non_constant}/{len(tensors['output'])} outputs are non-constant)")


if __name__ == "__main__":
    main()
