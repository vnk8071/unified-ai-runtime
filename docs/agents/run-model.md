# Build and run a model on the target device

Goal: the user names a model and a device ("run this on the NPU"). You build UAIRT there, run the model on that
device, check the result, and report what actually ran. Backend installation details live in `setup.md`;
symptoms and fixes live in `troubleshooting.md`. The rules in `AGENTS.md` apply throughout.

## 1. Find the target

Run `scripts/doctor.sh` and read every line. The `device` section gives the OS, chip, whether an NPU is present and
a Hexagon architecture hint (it is a hint: check it against the chip before relying on it).

- The target is the machine you are on unless the user says otherwise. For another board or phone, ask how to reach
  it. Build and run there; never assume access.
- If `doctor.sh` says `missing` for a core tool, tell the user what to install. Do not use `sudo`.

## 2. Pick the backend from the model

| Model file | Backend | Devices | The user provides |
|---|---|---|---|
| `.dlc`, `.bin` (QNN context binary) | `qnn` | NPU (HTP), CPU, GPU | QAIRT SDK, new enough for `.dlc` |
| `.onnx` | `onnxruntime` | CPU; NVIDIA GPU with the `cuda` or `tensorrt` provider; NPU through a plugin execution provider | an ONNX Runtime release matching the headers (1.22+), a GPU release for the GPU providers |
| `.tflite` | `tflite` | CPU; NPU through the QNN delegate (`libQnnTFLiteDelegate.so`, so Linux or Android) | a TFLite C library built for the target |
| OpenVINO IR (`.xml` + `.bin`), also `.onnx` | `openvino` | Intel CPU, GPU, NPU | OpenVINO (pip or release), Intel GPU/NPU drivers for those devices |
| NCNN `.param` + `.bin` | `ncnn` | CPU, any Vulkan GPU | NCNN built with Vulkan, a Vulkan driver; `input_shapes` option |
| `.engine`, `.plan` (TensorRT) | `tensorrt` | NVIDIA GPU (CUDA) | TensorRT 8.5+ with headers, CUDA toolkit; build the engine on the GPU that runs it |
| `.mlmodel`, `.mlpackage` | `coreml` | Apple CPU, GPU, Neural Engine | Xcode (macOS) |
| PyTorch (`.pt`, `.pth`) | none | | export to ONNX first, then use `onnxruntime` |

- A PyTorch model has no backend of its own: `torch.onnx.export` it, then follow the `.onnx` row. `torch` has no
  Windows ARM64 wheel, so on that platform export from an x64 Python.
- A `.bin` context binary is tied to the chip and the QAIRT version it was built for. One built for another chip or
  an older SDK will not load. A `.dlc` is portable but is compiled on the device at load (see the cache below).
- Producing a `.dlc` needs `qairt-converter`, which runs on x86_64 Linux only. If the user has only an ONNX or
  PyTorch model and the target is the NPU, say what that conversion needs and who does it.

## 3. Get the SDK in place

If `doctor.sh` reports the backend as `skip` or `missing`, follow `setup.md`: give the user the vendor download
page and the environment variable to set, wait for them to install and set it, then rerun `doctor.sh`. Never
download an SDK or accept its licence for them, and never put SDK files in this repository.

## 4. Build and test

Linux and macOS:

```bash
cmake -S . -B build -DUAIRT_BUILD_QNN=ON      # add the backend options you need
cmake --build build && ctest --test-dir build --output-on-failure
```

Windows: use the presets, which carry the Visual Studio generator and the platform. Run `scripts/doctor.ps1` (PowerShell) or
`scripts/doctor.sh --target windows` (Git Bash) first; WSL is Linux and does not count. Both reject the wrong shell.

```bash
cmake --preset windows-arm64-debug -DUAIRT_BUILD_QNN=ON      # or windows-x64-debug
cmake --build --preset windows-arm64-debug
ctest --preset windows-arm64-debug
```

Set the backend's environment variables (`QNN_SDK_ROOT`, `ONNXRUNTIME_ROOT`, and the `UAIRT_*_TEST_*` variables from
`setup.md`) before the `cmake` configure step, or the tests that need them are not registered. Check that `ctest`
lists every test you expect.

## 5. Run the model

`build/run_model` takes raw input files, one per model input, and writes raw outputs. Look at the model first:

```bash
build/run_model [--option key=value]... --info <plugin> <backend> <model>
```

The plugin is `build/libuairt_backend_<name>.so` (`.dll` on Windows, under `build/Debug`). `--info` prints each
input's name, dtype, quantization and dims. Make inputs of exactly that dtype and size, for example with numpy
(`rng.integers(0, 256, shape).astype(np.uint8).tofile("in0.bin")`), then run:

```bash
build/run_model --repeat 50 [--option ...] <plugin> <backend> <model> <out-prefix> in0.bin
```

It prints `init_ms` and `latency_ms` (min, median, mean). Outputs go to `<out-prefix><index>.bin`.

QNN on the NPU, the shortest form: `--option device=npu --option sdk_root=<QAIRT root>` (optionally
`hexagon_arch=v73`, `htp_performance_mode=burst`, `cache_dir=<dir>`). `device` finds the backend and system
libraries and the Hexagon skel directory for you. `cache_dir` keeps the compiled context of a `.dlc` so later loads
skip the compile; writes are off by default on Windows (see `docs/backends/qnn.md`).

From a binding: `bindings/cpp/run_qnn.cpp`, `bindings/python/examples/run_qnn.py` and
`bindings/rust/uairt/examples/run_qnn.rs` do the same through the C++, Python and Rust APIs. Python and Rust on
Windows need `ADSP_LIBRARY_PATH` set before the process starts for the NPU (see `troubleshooting.md`).

## 6. Check the result

- Correctness is agreement with the vendor tool, not agreement between devices. For QNN, run the comparison against
  `qnn-net-run` through `ctest` (`UAIRT_QNN_TEST_DLC`, `UAIRT_QNN_TEST_BACKEND=htp`, `UAIRT_QNN_DSP_ARCH`); it must be
  bit-identical. Quantized models differ slightly between CPU and NPU, so do not use that as a pass or fail test.
- Confirm the device that really ran. For the NPU the backend library must be `QnnHtp`, and latency should be far
  below the CPU run (a small vision model went from about 790 ms on the CPU to under 1 ms on the NPU). A run on
  `QnnCpu` is not an NPU run.
- Run with a second, different input and check the outputs are not constant.

## Two levels of "verified"

Say which one you mean, for each backend, in every report.

- **Core verified (level 1):** the build and `ctest` pass with the reference backend and the tests that need no vendor SDK.
  The library, plugin loader, bindings and error paths work on this OS and architecture. It says nothing about a vendor runtime.
- **Backend verified (level 2):** a real model ran through the real backend on the real device, with the vendor runtime
  installed, the output was compared with the vendor's own tool (or an identical-input reference), and the device that
  executed it was confirmed (NPU: the `QnnHtp` library; GPU: the provider and a placement log; Vulkan: the device name).

A passing `ctest` on a laptop is level 1. "Runs on the NPU" needs level 2. List what is not verified, for example a device a
backend supports but the machine does not have (OpenVINO's GPU and NPU), or a backend that only compiled.

## 7. Report

Give the level reached (see above) and the actual command and its output: `init_ms`, the median latency, the device, the backend library used, and
the `ctest` result. Say plainly what you did not verify (for example a backend that only compiled, or a model you
did not compare against the vendor tool). The work is done only when `ctest` passes, `doctor.sh` reports the
backend as `ok`, and a run on the requested device is shown.

## Stop and ask when

- an SDK, model conversion or licence acceptance is needed;
- a step needs `sudo` or a system setting changed;
- the target is another machine and you do not have access;
- the model's format needs a backend that is not built here and cannot be built without the user's SDK.
