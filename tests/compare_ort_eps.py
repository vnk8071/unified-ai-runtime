# SPDX-License-Identifier: Apache-2.0
"""Compares ONNX Runtime's CPU execution provider with a plugin execution provider through UAIRT.

Exports a torchvision model with random weights (no download), runs it with the CPU EP and with the
plugin EP (ep_library / ep_name options), checks that the outputs agree and that the top-1 classes are
equal, and prints both latencies. Needs torch, torchvision and numpy.

    python tests/compare_ort_eps.py build/run_model build/libuairt_backend_onnxruntime.so \\
        /path/libonnxruntime_mlx_ep.dylib MLXExecutionProvider [--model resnet50] [--batch 8]
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision


def run(run_model, plugin, model, input_path, prefix, runs, options):
    command = [run_model, *options, "--repeat", str(runs), plugin, "onnxruntime", str(model), str(prefix), str(input_path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    latency = re.search(r"latency_ms .*median=([\d.]+)", result.stderr)
    init = re.search(r"init_ms=([\d.]+)", result.stderr)
    return float(latency.group(1)), float(init.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_model")
    parser.add_argument("plugin")
    parser.add_argument("ep_library")
    parser.add_argument("ep_name")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "mobilenet_v2"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    network = getattr(torchvision.models, args.model)(weights=None).eval()
    sample = torch.from_numpy(np.random.default_rng(1).standard_normal((args.batch, 3, 224, 224), dtype=np.float32))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.onnx"
        torch.onnx.export(network, sample, str(model), input_names=["input"], output_names=["logits"], dynamo=False)
        sample.numpy().tofile(tmp / "in.bin")

        cpu_ms, cpu_init = run(args.run_model, args.plugin, model, tmp / "in.bin", tmp / "cpu_", args.runs, [])
        ep_options = ["--option", f"ep_library={args.ep_library}", "--option", f"ep_name={args.ep_name}"]
        ep_ms, ep_init = run(args.run_model, args.plugin, model, tmp / "in.bin", tmp / "ep_", args.runs, ep_options)

        expected = np.fromfile(tmp / "cpu_0.bin", np.float32).reshape(args.batch, -1)
        actual = np.fromfile(tmp / "ep_0.bin", np.float32).reshape(args.batch, -1)
    relative = float(np.abs(expected - actual).max() / np.abs(expected).max())
    assert relative < 1e-4, f"outputs differ: relative max abs diff {relative:.3g}"
    assert (expected.argmax(1) == actual.argmax(1)).all(), "top-1 classes differ"
    print(f"{args.model} batch {args.batch}: outputs agree (relative max abs diff {relative:.2g}, top-1 equal)")
    print(f"CPU EP    median {cpu_ms:8.2f} ms   init {cpu_init:7.1f} ms")
    print(f"{args.ep_name:9s} median {ep_ms:8.2f} ms   init {ep_init:7.1f} ms   ({cpu_ms / ep_ms:.2f}x)")


if __name__ == "__main__":
    main()
