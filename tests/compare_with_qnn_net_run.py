# SPDX-License-Identifier: Apache-2.0
"""Compares UAIRT's QNN backend with the SDK's qnn-net-run.

Accepts a DLC (.dlc) or a context binary (.bin) and the CPU or HTP backend. The same
native-dtype inputs (random, or zeros with --zeros) go through both; outputs must
match exactly.

    python3 tests/compare_with_qnn_net_run.py <QNN_SDK_ROOT> <model.dlc|model.bin> \
        build/run_model build/libuairt_backend_qnn.so \
        [--target aarch64-ubuntu-gcc9.4] [--backend cpu|htp] [--dsp-arch v68] [--zeros] [--dmabuf]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DTYPES = {1: np.float32, 2: np.float16, 4: np.int8, 5: np.uint8, 6: np.int16,
          7: np.int32, 8: np.int64, 9: np.bool_, 10: np.uint16}


def parse_info(text):
    tensors = {"input": [], "output": []}
    for line in text.splitlines():
        match = re.match(r"(input|output) (\d+) name=(\S*) dtype=(\d+) .* dims=(\S*)", line)
        if match:
            kind, _, name, dtype, dims = match.groups()
            shape = tuple(int(d) for d in dims.split("x")) if dims else ()
            tensors[kind].append((name, DTYPES[int(dtype)], shape))
    return tensors


def random_array(dtype, shape, rng):
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape, dtype=np.float32).astype(dtype)
    info = np.iinfo(dtype) if dtype != np.bool_ else None
    if info is None:
        return rng.integers(0, 2, shape).astype(dtype)
    return rng.integers(max(info.min, 0), min(info.max, 255) + 1, shape).astype(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sdk_root")
    parser.add_argument("dlc")
    parser.add_argument("run_model")
    parser.add_argument("plugin")
    parser.add_argument("--target", default="aarch64-windows-msvc" if sys.platform == "win32"
                        else "aarch64-ubuntu-gcc9.4")
    parser.add_argument("--backend", choices=["cpu", "htp"], default="cpu")
    parser.add_argument("--dsp-arch", default="v68")
    parser.add_argument("--graph-name")
    parser.add_argument("--zeros", action="store_true", help="use all-zero inputs")
    parser.add_argument("--dmabuf", action="store_true", help="give UAIRT zero-copy DMABUF buffers")
    args = parser.parse_args()

    sdk = Path(args.sdk_root)
    lib_dir = sdk / "lib" / args.target
    windows = sys.platform == "win32"
    net_run = sdk / "bin" / args.target / ("qnn-net-run.exe" if windows else "qnn-net-run")
    prefix, suffix = ("", ".dll") if windows else ("lib", ".so")
    backend_lib = lib_dir / (prefix + ("QnnCpu" if args.backend == "cpu" else "QnnHtp") + suffix)
    options = ["--option", f"backend_library={backend_lib}",
               "--option", f"system_library={lib_dir / (prefix + 'QnnSystem' + suffix)}"]
    if args.graph_name:
        options += ["--option", f"graph_name={args.graph_name}"]
    uairt = [args.run_model, *options]
    env = {**os.environ}
    env["PATH" if windows else "LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + os.environ.get(
        "PATH" if windows else "LD_LIBRARY_PATH", "")
    if args.backend == "htp":
        skel = sdk / "lib" / ("hexagon-" + args.dsp_arch) / "unsigned"
        env["ADSP_LIBRARY_PATH"] = str(skel) if windows else f"{skel};/usr/lib/rfsa/adsp"
    is_binary = args.dlc.endswith(".bin")

    info = subprocess.run([*uairt, "--info", args.plugin, "qnn", args.dlc], check=True,
                          capture_output=True, text=True, env=env).stdout
    tensors = parse_info(info)
    print(info.strip())

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inputs, list_line = [], []
        for i, (name, dtype, shape) in enumerate(tensors["input"]):
            array = np.zeros(shape, dtype) if args.zeros else random_array(dtype, shape, rng)
            path = tmp / f"in{i}.bin"
            array.tofile(path)
            inputs.append(str(path))
            list_line.append(f"{name}:={path}")
        (tmp / "list.txt").write_text(" ".join(list_line) + "\n")

        model_flag = ["--retrieve_context" if is_binary else "--dlc_path", args.dlc]
        subprocess.run([str(net_run), "--backend", str(backend_lib), *model_flag,
                        "--input_list", str(tmp / "list.txt"),
                        "--output_dir", str(tmp / "ref"), "--use_native_input_files",
                        "--use_native_output_files", "--log_level", "error"],
                       check=True, env=env, cwd=tmp, capture_output=True)
        subprocess.run([*uairt, *(["--dmabuf"] if args.dmabuf else []), args.plugin, "qnn", args.dlc,
                        str(tmp / "uairt_out"), *inputs], check=True, env=env)

        reference_files = sorted((tmp / "ref" / "Result_0").glob("*.raw"))
        assert len(reference_files) == len(tensors["output"]), (
            f"qnn-net-run wrote {len(reference_files)} outputs, expected {len(tensors['output'])}")
        by_name = {re.sub(r"_native$", "", p.stem): p for p in reference_files}
        non_constant = 0
        for i, (name, dtype, shape) in enumerate(tensors["output"]):
            reference_path = by_name.get(name) or by_name.get(re.sub(r"\W", "_", name))
            assert reference_path is not None, (
                f"no qnn-net-run output for '{name}' among {sorted(by_name)}")
            expected = np.fromfile(reference_path, dtype=dtype).reshape(shape)
            actual = np.fromfile(tmp / f"uairt_out{i}.bin", dtype=dtype).reshape(shape)
            diff = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            assert diff == 0, f"output {i} ({name}) differs from qnn-net-run: max abs diff {diff}"
            distinct = int(np.unique(actual).size)
            non_constant += distinct > 1
            print(f"output {i} {name}: {shape} {np.dtype(dtype).name} identical to qnn-net-run "
                  f"({distinct} distinct values)")
    assert non_constant, "every output is constant; the comparison proves nothing"
    print(f"UAIRT QNN backend matches qnn-net-run ({non_constant}/{len(tensors['output'])} "
          "outputs are non-constant)")


if __name__ == "__main__":
    main()
