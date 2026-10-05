<!-- SPDX-License-Identifier: Apache-2.0 -->
# Unified AI Runtime (UAIRT)

One C API to run compiled AI models on QNN (Qualcomm NPUs), ONNX Runtime, TFLite, CoreML, TensorRT (NVIDIA GPUs), OpenVINO (Intel) and NCNN (CPU and Vulkan GPUs).
Write the application code once; choose the backend by name when you create an engine.

```c
#include <uairt/uairt.h>

uairt_load_backend_library("libuairt_backend_onnxruntime.so");   /* a plugin per backend */

uairt_engine* engine;
uairt_engine_create("onnxruntime", NULL, 0, &engine);

uairt_model_source source = {.struct_size = sizeof(source), .path = "model.onnx"};
uairt_model* model;
uairt_model_load(engine, &source, &model);

uairt_tensor in, out;
uairt_model_input_info(model, 0, &in);      /* dtype, shape, quantization, name */
uairt_model_output_info(model, 0, &out);
in.data = input_buffer;   in.nbytes = input_size;
out.data = output_buffer; out.nbytes = output_size;
uairt_model_run(model, &in, 1, &out, 1);
```

A complete runnable program is in [examples/quickstart.c](examples/quickstart.c).

## What it is, and is not

UAIRT loads artifacts that each vendor's toolchain already produced (an ONNX file, a QNN
DLC or context binary, a CoreML model) and runs them through that vendor's runtime. It
does not convert, quantize or optimize models, and it does not hide that artifacts are
tied to a device or SDK version: a mismatch returns an error instead of falling back
quietly. It is not an agent runtime. If your models are all ONNX, ONNX Runtime and its
execution providers may already be enough; UAIRT is for mixing artifact formats behind one
API, with a plain C ABI that other languages can bind to.

## Status

Version 0.1.0, pre-release: the API and ABI may change. See [CHANGELOG.md](CHANGELOG.md).

| Backend | Models | Verified on |
|---|---|---|
| ONNX Runtime | `.onnx`, static shapes; CPU, and the CUDA and TensorRT providers on NVIDIA GPUs | macOS arm64 (CPU) and a GTX 1650 (CUDA provider); matches Python ONNX Runtime. Compiles on Windows ARM64, not run there |
| QNN | `.dlc`, context binaries; CPU and HTP; DMABUF zero-copy; compiled-context cache | Qualcomm QCS6490 and QCS8550, and Windows 11 ARM64 on a Snapdragon X Elite NPU; byte-identical to `qnn-net-run` |
| TFLite | `.tflite`; CPU kernels, optional QNN delegate | Qualcomm QCS6490 (CPU path only); identical to `tflite_bench` |
| CoreML | `.mlmodelc`, `.mlpackage`, `.mlmodel`; multi-array I/O, image inputs | Apple M5; identical to CoreML itself |
| OpenVINO | IR (`.xml` + `.bin`) and `.onnx`, static shapes; any OpenVINO device | an Intel Core CPU, and an x64 build under emulation on a Snapdragon X Elite laptop (there is no Windows ARM64 OpenVINO); bit-identical to OpenVINO's Python API. GPU and NPU devices untested |
| NCNN | `.param` + `.bin`, float32, static shapes; CPU and Vulkan | i9-9900K CPU and a GTX 1650 through Vulkan (matches NCNN's Python API); a Snapdragon X Elite laptop, CPU and Adreno GPU through Vulkan, same detections |
| TensorRT | serialized `.engine` / `.plan` (ultralytics exports load as is), static shapes; host and pinned buffers | NVIDIA GTX 1650, TensorRT 10.9; bit-identical to TensorRT's Python API |

"Verified" in this table and in the docs means a real model ran through that backend on the listed hardware and matched the
vendor's own tool. A passing `ctest` with only the reference backend proves the core, loader and bindings, not a vendor runtime;
the two are reported separately (see [docs/agents/run-model.md](docs/agents/run-model.md)).

Known issues and the exact configurations tested are in [docs/api.md](docs/api.md) and
[docs/design.md](docs/design.md). Read them before relying on a backend.

## Install (Python)

```bash
pip install "uairt[onnxruntime]"
```

[`uairt` on PyPI](https://pypi.org/project/uairt/) has wheels for Linux (x86_64, aarch64), macOS (arm64) and Windows (x64) for
Python 3.9 and later. Each carries the native library and the ONNX Runtime plugin; drop the `[onnxruntime]` extra if you bring
your own `onnxruntime`. The other backends (QNN, TensorRT, CoreML, OpenVINO, NCNN, TFLite) are not in the wheel: build
their plugins from this repository (below) and point `UAIRT_PLUGIN_PATH` at them. See
[bindings/python/README.md](bindings/python/README.md).

## Build

Needs CMake 3.16+ and a C11 compiler.

```bash
scripts/doctor.sh                      # reports what is available on this machine
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
build/quickstart
```

On Windows use the presets (Visual Studio 2022 with the C++ tools, and Git Bash for one test). Check the machine with
`scripts/doctor.ps1` in PowerShell, or `scripts/doctor.sh --target windows` in Git Bash; WSL is Linux and is rejected.

```powershell
cmake --preset windows-arm64-debug      # or windows-x64-debug; add -DUAIRT_BUILD_QNN=ON and so on
cmake --build --preset windows-arm64-debug
ctest --preset windows-arm64-debug
```

### ONNX Runtime

Download a release from https://github.com/microsoft/onnxruntime/releases and unpack it.

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>
cmake -S . -B build -DUAIRT_BUILD_ONNXRUNTIME=ON && cmake --build build
```

Engine options: `intra_op_threads`, `execution_provider` (`cpu`, `cuda` or `tensorrt` with a GPU release), `log_level`, and for a plugin execution provider
`ep_library`, `ep_name` and `ep_option.<key>` (for example Apple's MLX with ONNX Runtime 1.29;
see [docs/design.md](docs/design.md)).

### TFLite

Needs TFLite's C API headers and a `libtensorflowlite_c` built for your target. The QNN delegate is
optional and needs `QnnTFLiteDelegate.h` from your QAIRT SDK at build time:

```bash
export TFLITE_INCLUDE_DIR=/path/containing/tensorflow/lite/c/c_api.h
export TFLITE_LIBRARY=/path/to/libtensorflowlite_c.so
export QNN_SDK_ROOT=/path/to/qairt/<version>        # optional, enables delegate=qnn
cmake -S . -B build -DUAIRT_BUILD_TFLITE=ON && cmake --build build
build/run_model --option num_threads=4 --info build/libuairt_backend_tflite.so tflite model.tflite
```

With `--option delegate=qnn` plus `delegate_library`, `backend_library`, `skel_dir` and
`htp_performance_mode`. Both paths are verified on a QCS8550 (Hexagon v73); the delegate crashes
on Hexagon v68 (see [docs/design.md](docs/design.md)). `UAIRT_TFLITE_TEST_MODEL` and `UAIRT_TFLITE_BENCH` (a built
`tests/bench/tflite_bench`) enable the comparison test under `ctest`.

### CoreML (macOS)

```bash
cmake -S . -B build -DUAIRT_BUILD_COREML=ON && cmake --build build
build/run_model --option compute_units=all --info build/libuairt_backend_coreml.so coreml model.mlmodelc
```

Needs Xcode. To check against CoreML itself, point `UAIRT_COREML_PYTHON` at a Python with
`coremltools` and `numpy` before running cmake.

### OpenVINO (Intel CPUs, GPUs, NPUs)

`pip install openvino` (or a release) provides the C API; point `OPENVINO_ROOT` at `site-packages/openvino`. Ultralytics
exports (`format="openvino"`) load as a directory.

```bash
export OPENVINO_ROOT=/path/to/site-packages/openvino
cmake -S . -B build -DUAIRT_BUILD_OPENVINO=ON && cmake --build build
build/run_model --repeat 50 --option device=CPU build/libuairt_backend_openvino.so openvino yolov8n_openvino_model out in0.bin
```

`device=GPU` or `NPU` runs on Intel hardware when OpenVINO lists it; see [docs/backends/openvino.md](docs/backends/openvino.md).

### NCNN (CPU and Vulkan GPUs)

Build NCNN with `-DNCNN_VULKAN=ON` (same C++ toolchain as the plugin) and set `NCNN_ROOT` to its install prefix. NCNN's
`.param` records no input shapes, so pass them:

```bash
export NCNN_ROOT=$HOME/ncnn-install
cmake -S . -B build -DUAIRT_BUILD_NCNN=ON && cmake --build build
build/run_model --repeat 100 --option input_shapes=3x640x640 --option device=vulkan --option fp16=false \
  build/libuairt_backend_ncnn.so ncnn yolov8n_ncnn_model out in0.bin
```

On a GTX 1650 a yolov8n model runs in 21.9 ms through Vulkan against 79 ms on the CPU; `device=vulkan` fails instead of
quietly using the CPU when no GPU is usable. See [docs/backends/ncnn.md](docs/backends/ncnn.md).

### TensorRT (NVIDIA GPUs)

Install an NVIDIA driver, the CUDA toolkit and TensorRT 8.5+ with its headers yourself (the apt packages
`libnvinfer-dev` or NVIDIA's tarball). An engine is built for one GPU and one TensorRT version, so build it on the GPU
that runs it, with `trtexec` or ultralytics (`YOLO("yolov8n.pt").export(format="engine")`).

```bash
export TENSORRT_ROOT=/path/to/TensorRT            # not needed for a system install
cmake -S . -B build -DUAIRT_BUILD_TENSORRT=ON && cmake --build build
build/run_model --info build/libuairt_backend_tensorrt.so tensorrt yolov8n.engine
build/run_model --repeat 100 --pinned build/libuairt_backend_tensorrt.so tensorrt yolov8n.engine out in0.bin
```

`--pinned` allocates the inputs and outputs as page-locked buffers (`UAIRT_MEM_PINNED`), which made a yolov8n run
8.1 ms to 6.4 ms on a GTX 1650. A complete detection program, with the Python binding and pinned buffers, is
[bindings/python/examples/detect_tensorrt.py](bindings/python/examples/detect_tensorrt.py); see
[docs/backends/tensorrt.md](docs/backends/tensorrt.md).

### QNN (Qualcomm AI Runtime)

Install QAIRT from Qualcomm yourself. The SDK has no macOS libraries, so build and run on
the Linux or Windows machine or device that has `lib/<target>/libQnnCpu.so`, `libQnnHtp.so`
and `libQnnSystem.so`.

```bash
export QNN_SDK_ROOT=/path/to/qairt/<version>
cmake -S . -B build -DUAIRT_BUILD_QNN=ON && cmake --build build
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/<target>
build/run_model \
  --option backend_library=$QNN_SDK_ROOT/lib/<target>/libQnnHtp.so \
  --option system_library=$QNN_SDK_ROOT/lib/<target>/libQnnSystem.so \
  --option adsp_library_path="$QNN_SDK_ROOT/lib/hexagon-v68/unsigned;/usr/lib/rfsa/adsp" \
  --info build/libuairt_backend_qnn.so qnn model.serialized.bin
```

On Windows ARM64 or any machine where the SDK layout is standard, `--option device=npu --option sdk_root=$QNN_SDK_ROOT`
replaces the three library options (`cpu` and `gpu` work too), and `--option cache_dir=<dir>` keeps the compiled
context of a `.dlc` so later loads skip the compile (3.1 s to 0.4 s on a Snapdragon X Elite).

Add `--option htp_performance_mode=burst` for short graphs (about 4x faster at 10 ms),
`--option graph_name=<name>` when a model holds several graphs, `--repeat N` to time
runs, and `--dmabuf` for zero-copy buffers on HTP. `UAIRT_QNN_TEST_DLC`,
`UAIRT_QNN_TEST_BACKEND` and `UAIRT_QNN_TARGET` enable the comparison with `qnn-net-run`
under `ctest` (see [docs/agents/setup.md](docs/agents/setup.md)).

Per-backend setup pages are in [docs/backends/](docs/backends/README.md).

## Language bindings

Python (`bindings/python`), Rust (`bindings/rust`) and a header-only C++17 wrapper
(`include/uairt/uairt.hpp`) are done, and run on Windows ARM64 too. Each has a `run_qnn` example, and Python also has
`detect_tensorrt.py`. The Python binding has `AutoModel`, which picks the backend from the model file and device and finds
backend plugins on its own (`uairt.AutoModel.from_file("model.dlc", device="npu")`). A wheel with the native library and the
ONNX Runtime plugin is on PyPI (`pip install "uairt[onnxruntime]"`; see [docs/releasing.md](docs/releasing.md)). See
[bindings/README.md](bindings/README.md) and [bindings/python/README.md](bindings/python/README.md).

## Documentation

- [docs/api.md](docs/api.md): every function, status codes, options, tested configurations
- [docs/writing-a-backend.md](docs/writing-a-backend.md): the plugin ABI and how to add a backend
- [docs/design.md](docs/design.md): design decisions, per-backend notes, measurements
- [docs/LICENSING.md](docs/LICENSING.md): how vendor licenses shape this repository
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md)
- [AGENTS.md](AGENTS.md) and [docs/agents/](docs/agents/run-model.md): instructions for coding agents, including a
  playbook for building UAIRT and running a model on a target device (`scripts/doctor.sh` reports the device)

## Vendor SDKs

QAIRT (Qualcomm), Xcode and CoreML (Apple) and TensorRT (NVIDIA) are proprietary and are
never part of this repository. You install them yourself under their own licenses and
point the build at them. See [docs/LICENSING.md](docs/LICENSING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
