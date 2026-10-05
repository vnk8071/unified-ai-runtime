# SPDX-License-Identifier: Apache-2.0
"""Runs a model through the QNN backend plugin with random inputs and prints the latency.

    python run_qnn.py <plugin> <sdk-root> <model> [npu|cpu|gpu] [runs] [cache-dir]
"""
import os
import statistics
import sys
import time

import numpy as np
import uairt


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    plugin, sdk_root, model_path = sys.argv[1:4]
    device = sys.argv[4] if len(sys.argv) > 4 else "npu"
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    if device == "npu" and sys.platform == "win32" and not os.environ.get("ADSP_LIBRARY_PATH"):
        # Without it the NPU libraries fall back to a path that crashed the process (heap corruption).
        sys.exit(r"set ADSP_LIBRARY_PATH to <QAIRT root>\lib\hexagon-v73\unsigned before starting Python")
    options = {"device": device, "sdk_root": sdk_root}
    if len(sys.argv) > 6:
        options["cache_dir"] = sys.argv[6]

    uairt.load_backend_library(plugin)
    started = time.perf_counter()
    with uairt.Engine("qnn", options) as engine, engine.load_model(model_path) as model:
        print(f"init_ms={(time.perf_counter() - started) * 1e3:.1f}")
        rng = np.random.default_rng(0)
        inputs = []
        for info in model.inputs:
            print("input", info.name, info.dtype, info.shape)
            inputs.append(rng.integers(0, 256, info.shape).astype(info.dtype))
        latencies = []
        for _ in range(max(runs, 1)):
            start = time.perf_counter()
            outputs = model.run(*inputs)
            latencies.append((time.perf_counter() - start) * 1e3)
        print(f"latency_ms runs={len(latencies)} min={min(latencies):.2f} "
              f"median={statistics.median(latencies):.2f}")
        for i, output in enumerate(outputs):
            print(f"output {i}: {output.shape} {output.dtype} byte sum {int(output.astype(np.uint64).sum())}")


if __name__ == "__main__":
    main()
