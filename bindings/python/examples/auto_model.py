# SPDX-License-Identifier: Apache-2.0
"""Loads any supported model with AutoModel, runs it once on random inputs and prints what was chosen.

    python auto_model.py <model> [--device cpu|gpu|npu|cuda|tensorrt|vulkan[:N]] [--backend NAME] [--runs N] [key=value ...]

The backend comes from the file (and device), and its plugin is found on UAIRT_PLUGIN_PATH, in the package, or next to
libuairt. Extra key=value arguments are backend options, for example input_shapes=3x640x640 for an NCNN model.
"""
import argparse
import statistics
import time

import numpy as np

import uairt


def random_input(info, rng):
    if np.issubdtype(info.dtype, np.floating):
        return rng.random(info.shape, dtype=np.float32).astype(info.dtype)
    return rng.integers(0, 4, info.shape).astype(info.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--device")
    parser.add_argument("--backend")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("options", nargs="*", help="backend options as key=value")
    args = parser.parse_intermixed_args()
    options = dict(item.split("=", 1) for item in args.options)

    print("backends:", {name: ("registered" if s["registered"] else str(s["plugin"] or "-"))
                        for name, s in uairt.available_backends().items()})
    started = time.perf_counter()
    try:
        model = uairt.AutoModel.from_file(args.model, device=args.device, backend=args.backend, options=options)
    except uairt.UairtError as error:
        raise SystemExit(f"{type(error).__name__}: {error}")
    with model:
        print(f"{model}  options={model.options}  load={(time.perf_counter() - started) * 1e3:.0f} ms")
        for i, info in enumerate(model.inputs):
            print(f"  input {i}: {info.name} {info.dtype} {info.shape}")
        inputs = [random_input(info, np.random.default_rng(0)) for info in model.inputs]
        times = []
        for _ in range(args.runs + 3):
            start = time.perf_counter()
            outputs = model.run(*inputs)
            times.append((time.perf_counter() - start) * 1e3)
        for i, output in enumerate(outputs):
            print(f"  output {i}: {output.dtype} {output.shape}, {np.unique(output).size} distinct values")
        times = times[3:]
        print(f"latency_ms runs={len(times)} min={min(times):.2f} median={statistics.median(times):.2f}")


if __name__ == "__main__":
    main()
