<!-- SPDX-License-Identifier: Apache-2.0 -->
# Unified AI Runtime (UAIRT)

One C API to run compiled AI models on QNN, ONNX Runtime and CoreML (TensorRT planned).
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
| ONNX Runtime | `.onnx`, static shapes, CPU | macOS arm64; matches Python ONNX Runtime |
| QNN | `.dlc`, context binaries; CPU and HTP; DMABUF zero-copy | Qualcomm QCS6490 and QCS8550; byte-identical to `qnn-net-run` |
| TFLite | `.tflite`; CPU kernels, optional QNN delegate | Qualcomm QCS6490 (CPU path only); identical to `tflite_bench` |
| CoreML | `.mlmodelc`, `.mlpackage`, `.mlmodel`; multi-array I/O | Apple M5; identical to CoreML itself |
| TensorRT | not implemented | needs an NVIDIA device |

Known issues and the exact configurations tested are in [docs/api.md](docs/api.md) and
[docs/design.md](docs/design.md). Read them before relying on a backend.

## Build

Needs CMake 3.16+ and a C11 compiler.

```bash
scripts/doctor.sh                      # reports what is available on this machine
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
build/quickstart
```

### ONNX Runtime

Download a release from https://github.com/microsoft/onnxruntime/releases and unpack it.

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>
cmake -S . -B build -DUAIRT_BUILD_ONNXRUNTIME=ON && cmake --build build
```

Engine options: `intra_op_threads`, `execution_provider` (`cpu`), and for a plugin execution provider
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

Add `--option htp_performance_mode=burst` for short graphs (about 4x faster at 10 ms),
`--option graph_name=<name>` when a model holds several graphs, `--repeat N` to time
runs, and `--dmabuf` for zero-copy buffers on HTP. `UAIRT_QNN_TEST_DLC`,
`UAIRT_QNN_TEST_BACKEND` and `UAIRT_QNN_TARGET` enable the comparison with `qnn-net-run`
under `ctest` (see [docs/agents/setup.md](docs/agents/setup.md)).

Per-backend setup pages are in [docs/backends/](docs/backends/README.md).

## Language bindings

Python (`bindings/python`), Rust (`bindings/rust`) and a header-only C++17 wrapper
(`include/uairt/uairt.hpp`) are done. See [bindings/README.md](bindings/README.md).

## Documentation

- [docs/api.md](docs/api.md): every function, status codes, options, tested configurations
- [docs/writing-a-backend.md](docs/writing-a-backend.md): the plugin ABI and how to add a backend
- [docs/design.md](docs/design.md): design decisions, per-backend notes, measurements
- [docs/LICENSING.md](docs/LICENSING.md): how vendor licenses shape this repository
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md)
- [AGENTS.md](AGENTS.md): instructions for coding agents working in this repository

## Vendor SDKs

QAIRT (Qualcomm), Xcode and CoreML (Apple) and TensorRT (NVIDIA) are proprietary and are
never part of this repository. You install them yourself under their own licenses and
point the build at them. See [docs/LICENSING.md](docs/LICENSING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

## For agents

`AGENTS.md` and `docs/agents/` direct a coding agent to build UAIRT and run a model on a target device: `run-model.md` is the playbook, `setup.md` covers each backend's SDK, `troubleshooting.md` maps symptoms to fixes, and `scripts/doctor.sh` reports the device and what is ready. A matching Claude Code skill is in `.claude/skills/run-model-on-device/`.
