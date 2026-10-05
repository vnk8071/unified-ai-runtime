<!-- SPDX-License-Identifier: Apache-2.0 -->
# ONNX Runtime backend

Source: `src/backends/onnxruntime.c`. Plugin: `libuairt_backend_onnxruntime.so`.

## You install

An ONNX Runtime release (MIT licensed) from https://github.com/microsoft/onnxruntime/releases.
It must contain `include/onnxruntime_c_api.h` and `lib/libonnxruntime`. The pip wheel is not
enough: it has the library but no headers.

For the GPU providers use a GPU release (for example `onnxruntime-linux-x64-gpu_cuda12-<version>.tgz`) whose CUDA
major version matches the CUDA toolkit you have installed, plus that toolkit's runtime libraries and cuDNN 9. The
`tensorrt` provider also needs TensorRT's libraries and its ONNX parser (`libnvonnxparser`). The default `onnxruntime-gpu`
wheel on PyPI can target a different CUDA major version than your toolkit, so check before using it as a reference.

## Build

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>
cmake -S . -B build -DUAIRT_BUILD_ONNXRUNTIME=ON && cmake --build build
```

The plugin is built against the header version and fails to load with a clear error if the
installed library is older.

## Engine options

`intra_op_threads`, `execution_provider` (`cpu`, `cuda` or `tensorrt`), `log_level` (`verbose`, `info`, `warning`,
`error`, `fatal`), and `ep_option.<key>`, which is passed to the provider (CUDA, TensorRT or a plugin). For a plugin
execution provider also `ep_library` and `ep_name`.

## GPU providers (CUDA and TensorRT)

`execution_provider=cuda` or `tensorrt` appends that provider with the `ep_option.<key>` settings; nodes it cannot take
run on the CPU provider, which ONNX Runtime adds last. A CPU-only build, or missing CUDA, cuDNN or TensorRT libraries,
fail at engine creation with `UAIRT_ERR_BACKEND_UNAVAILABLE` and the loader's message.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$ONNXRUNTIME_ROOT/lib
build/run_model --repeat 100 --option execution_provider=cuda \
  build/libuairt_backend_onnxruntime.so onnxruntime model.onnx out in0.bin
# TensorRT provider: the first run compiles an engine, so cache it
build/run_model --option execution_provider=tensorrt --option ep_option.trt_engine_cache_enable=1 \
  --option ep_option.trt_engine_cache_path=/tmp/trt_cache \
  build/libuairt_backend_onnxruntime.so onnxruntime model.onnx out in0.bin
```

Check that the GPU really ran: `--option log_level=verbose` prints the node placement, for example
`All nodes placed on [CUDAExecutionProvider]. Number of nodes: 175`. Set `UAIRT_TEST_ORT_PROVIDER=cuda` (or `tensorrt`)
before `ctest` to run the backend test on that provider.

Verified on a GTX 1650 (CUDA 12.8, cuDNN 9) with ONNX Runtime 1.30.0: a yolov8n model (`.onnx`, 640x640, FP32) takes
10.0 ms on `cuda` against 59.9 ms on 8 CPU threads, all 175 nodes on the GPU, and the detections on a test image
are identical to the CPU provider's (outputs differ by at most 7e-4 on values up to 637).

The `tensorrt` provider on the same model: the first run compiles an engine (116 s here), a load from the engine cache
takes 3.5 s, and a run takes 8.1 ms with identical detections (outputs differ from the CPU's by at most 1.4e-3). Both
providers copy through ordinary host memory; the standalone `tensorrt` backend with pinned buffers is faster still
(6.4 ms for the same network, see `docs/backends/tensorrt.md`).

## Plugin execution providers (for example MLX)

A plugin EP needs an ONNX Runtime core of the version it targets (`onnxruntime-ep-mlx` 0.29.x needs
ONNX Runtime 1.29.x). Build UAIRT against that release's headers and keep the provider library
(the `onnxruntime-ep-mlx` wheel bundles `libonnxruntime_mlx_ep.dylib` and the MLX runtime) together.
To run the plugin test under `ctest`, set `UAIRT_TEST_ORT_EP_LIBRARY` and `UAIRT_TEST_ORT_EP_NAME`.

## Check

```bash
ctest --test-dir build --output-on-failure
build/run_model --info build/libuairt_backend_onnxruntime.so onnxruntime model.onnx
```

`tests/data/make_models.py` regenerates the test models and checks them against Python ONNX
Runtime (needs `onnx`, `onnxruntime`, `numpy`). `tests/compare_with_python_ort.py` compares
outputs with Python ONNX Runtime.
