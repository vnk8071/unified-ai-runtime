<!-- SPDX-License-Identifier: Apache-2.0 -->
# TensorRT backend (NVIDIA GPUs)

Source: `src/backends/tensorrt.cpp`, a C++ plugin behind the C backend ABI. Plugin: `libuairt_backend_tensorrt.so`.
Verified on a GTX 1650 with TensorRT 10.9.0.34 and CUDA 12.8 (see the last section).

## You install

An NVIDIA GPU with its driver, the CUDA toolkit, and TensorRT 8.5 or newer with its headers
(`NvInfer.h`) and libraries (`libnvinfer`), under NVIDIA's licence. Set `TENSORRT_ROOT` to the directory that
contains `include/NvInfer.h` and `lib/libnvinfer.so`; a system install under `/usr/include/<arch>-linux-gnu` is also
found. The Python `tensorrt` wheel has the libraries but not the headers, so it is not enough to build.

## Build

```bash
export TENSORRT_ROOT=/path/to/TensorRT
cmake -S . -B build -DUAIRT_BUILD_TENSORRT=ON && cmake --build build
```

## Run

The model is a serialized TensorRT engine (`.engine` or `.plan`). An engine is built for one GPU model and one
TensorRT version, so build it on the machine that will run it (for example with `trtexec`, or ultralytics'
`YOLO(...).export(format="engine")`).

```bash
build/run_model --info build/libuairt_backend_tensorrt.so tensorrt model.engine
build/run_model --repeat 50 build/libuairt_backend_tensorrt.so tensorrt model.engine out in0.bin
```

Engine options: `device` (CUDA device index, default 0).

Engines exported by ultralytics start with a 4-byte length and a JSON metadata block before the plan; the backend skips that header, so `YOLO(...).export(format="engine")` output loads as is.

## Limits

- Static shapes only: an engine with a dynamic dimension is rejected at load.
- Host memory only. Inputs and outputs are copied through device buffers the model owns on every run.
- One execution context per model; do not run one model from several threads at once.
- Supported element types: float32, float16, bfloat16, int8, uint8, int32, int64 and bool.

## Check

`ctest` runs `tensorrt_errors` (error paths, no engine needed). To compare against TensorRT's own Python API set
`UAIRT_TENSORRT_TEST_ENGINE=/path/model.engine` and `UAIRT_TENSORRT_PYTHON=<python with tensorrt, torch and numpy>`
before running cmake; the outputs must be identical.

## Result

A yolov8n engine exported by ultralytics 8.4.173 (FP32, 640x640, GTX 1650, TensorRT 10.9.0.34): the outputs are bit-identical
to TensorRT's Python API. Median time per run over 200 runs: 5.14 ms for TensorRT with tensors already on the GPU, 6.63 ms for
ultralytics' engine call, and 7.93 ms through this backend, which copies the 4.9 MB input and 2.8 MB output through pageable host
memory on every run. Pinned or device-resident zero-copy buffers would close most of that gap.
