<!-- SPDX-License-Identifier: Apache-2.0 -->
# ONNX Runtime backend

Source: `src/backends/onnxruntime.c`. Plugin: `libuairt_backend_onnxruntime.so`.

## You install

An ONNX Runtime release (MIT licensed) from https://github.com/microsoft/onnxruntime/releases.
It must contain `include/onnxruntime_c_api.h` and `lib/libonnxruntime`. The pip wheel is not
enough: it has the library but no headers.

## Build

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>
cmake -S . -B build -DUAIRT_BUILD_ONNXRUNTIME=ON && cmake --build build
```

The plugin is built against the header version and fails to load with a clear error if the
installed library is older.

## Engine options

`intra_op_threads`, `execution_provider` (`cpu`), and for a plugin execution provider
`ep_library`, `ep_name` and `ep_option.<key>`.

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
