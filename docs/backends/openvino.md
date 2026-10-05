<!-- SPDX-License-Identifier: Apache-2.0 -->
# OpenVINO backend (Intel CPUs, GPUs and NPUs)

Source: `src/backends/openvino.c`. Plugin: `libuairt_backend_openvino.so`.

## You install

OpenVINO (Apache-2.0) with its C API: either a release from https://github.com/openvinotoolkit/openvino/releases
(the `runtime` directory) or a pip install (`pip install openvino`; its `site-packages/openvino` directory has the headers
and libraries). Set `OPENVINO_ROOT` to that directory. For the `GPU` and `NPU` devices you also need Intel's drivers
for them; without them OpenVINO offers only `CPU`.

## Build

```bash
export OPENVINO_ROOT=/path/to/site-packages/openvino        # or a release's runtime directory
cmake -S . -B build -DUAIRT_BUILD_OPENVINO=ON && cmake --build build
```

## Run

The model is an OpenVINO IR (`.xml` with its `.bin`), an ONNX file, or a directory that holds one `.xml` (an ultralytics
`YOLO(...).export(format="openvino")` directory loads as is).

```bash
build/run_model --info build/libuairt_backend_openvino.so openvino yolov8n_openvino_model
build/run_model --repeat 50 --option device=GPU build/libuairt_backend_openvino.so openvino model.xml out in0.bin
```

Engine options: `device` (default `CPU`; also `GPU`, `NPU`, `GPU.1`, `AUTO`, `HETERO:...`), `performance_hint`
(`latency` or `throughput`), `num_threads` and `cache_dir` (keeps compiled models, which makes GPU startup much faster).
A device OpenVINO does not list on this machine fails at engine creation with `UAIRT_ERR_BACKEND_UNAVAILABLE` and the
list of devices that are available.

## Limits

- Models are loaded from a path, not from memory.
- Static shapes only (a model with a dynamic dimension is rejected at load).
- Host memory only. Outputs are written into the caller's buffers without a copy where the device allows it.

## Check

`ctest` runs `openvino_errors`. For the comparison with OpenVINO's Python API set `UAIRT_OPENVINO_TEST_MODEL=/path/model.xml`
and `UAIRT_OPENVINO_PYTHON=<python with openvino and numpy>` before running cmake; the outputs must be identical, so use
the same OpenVINO version for both.

## Result

Verified on the CPU device only (Intel Core i9-9900K, OpenVINO 2026.4.1, Ubuntu 24.04): a yolov8n IR model is
bit-identical to OpenVINO's Python API, takes about 79 ms per run, and finds the same detections as the other backends
on a test image. The GPU and NPU devices are not verified: the test machine has no Intel GPU or NPU.
