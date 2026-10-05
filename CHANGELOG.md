<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

## 0.1.0 (unreleased)

First public version. Pre-1.0: the API and ABI may change.

### Added
- C11 API (`uairt/uairt.h`): engines, models, tensors, zero-copy buffers, status codes.
- Backend plugin ABI (`uairt/uairt_backend.h`): backends are shared libraries that fill in
  a function table and do not link the core.
- Backends: `reference` (tests), ONNX Runtime (CPU), QNN (CPU and HTP; DLC and context
  binaries; DMABUF zero-copy), CoreML (`.mlmodelc`, `.mlpackage`, `.mlmodel`).
- `run_model` command-line tool, `quickstart` example, `scripts/doctor.sh`.
- Tests that compare each backend with the vendor's own tool: Python ONNX Runtime,
  `qnn-net-run`, CoreML through `coremltools`.

### Added (Python wheels)
- A root `pyproject.toml` (scikit-build-core) builds the `uairt` wheel: the Python package, `libuairt` and the ONNX Runtime plugin,
  tagged `py3-none-<platform>` so one wheel per platform serves every Python 3.x. `.github/workflows/wheels.yml` builds and tests
  the wheels with cibuildwheel (Linux x86_64 and aarch64, macOS arm64, Windows x64, Windows ARM64 when a runner is enabled) and
  publishes to TestPyPI and then PyPI with trusted publishing, behind an approval. See `docs/releasing.md`.
- The ONNX Runtime plugin loads ONNX Runtime at run time instead of linking it, so it works with any release from 1.22 on,
  including a pip-installed `onnxruntime`, which the Python package locates itself. `UAIRT_ONNXRUNTIME_LIBRARY` names a library
  explicitly. Verified on Windows ARM64 with `onnxruntime` 1.24 and 1.30 (the backend had not run on Windows before).
- `scripts/ci/check_release_version.py` fails a release when the tag, `pyproject.toml` and the C header disagree.

### Added (Python AutoModel)
- `uairt.AutoModel.from_file(path, device=...)` (and `uairt.load`) chooses the backend from the model file and the device,
  maps the device to that backend's options, and reports its choice in `model.backend`, `model.options` and `model.plan`.
  It is strict: no fallback to another runtime or the CPU, and clear errors saying why candidates were passed over.
- Plugin discovery: `Engine("onnxruntime")` loads `libuairt_backend_onnxruntime` from `UAIRT_PLUGIN_PATH`, the package's
  `plugins/` directory or the directory of libuairt, so plugin paths are no longer needed. `uairt.available_backends()`,
  `discover_plugins()` and `ensure_backend()` expose it.
- Verified end to end on a Snapdragon X Elite laptop: a `.dlc` on the NPU and an NCNN model on the Adreno GPU, each from one
  `AutoModel.from_file` call with no plugin paths in the code.

### Added (Windows as a build target)
- `CMakePresets.json` with `windows-arm64-debug` and `windows-x64-debug` (configure, build and test presets), so the Visual
  Studio generator and platform flags need not be remembered.
- `scripts/doctor.ps1`, a PowerShell version of the device and toolchain check, and `--target windows` for `doctor.sh`, which
  reports WSL and fails when the shell is not a Windows shell.
- CI jobs for Windows x64 (hosted runner) and Windows ARM64 (enabled with the `ENABLE_WINDOWS_ARM64_CI` repository variable,
  since it needs an ARM64 runner); both build with warnings as errors and run `ctest` and the vendor-file check.
- `.gitattributes` keeps text files, and shell scripts in particular, LF in every checkout.
- The docs and `AGENTS.md` separate "core verified" from "backend verified".

### Added (OpenVINO and NCNN)
- OpenVINO backend (`openvino`, C): IR, ONNX or a directory holding one `.xml`; any OpenVINO device; `device`,
  `performance_hint`, `num_threads`, `cache_dir`. Built with `-DUAIRT_BUILD_OPENVINO=ON`. Verified on an Intel CPU
  (bit-identical to OpenVINO's Python API); the GPU and NPU devices are untested.
- NCNN backend (`ncnn`, a C++ plugin): `.param` + `.bin`, float32, static shapes, `input_shapes` required; CPU and Vulkan
  (`device=vulkan` fails instead of falling back to the CPU). Built with `-DUAIRT_BUILD_NCNN=ON`. Verified on a CPU and a
  GTX 1650: a yolov8n model runs in 21.9 ms on the GPU against 79 ms on the CPU.
- `doctor.sh` reports OpenVINO, NCNN and the Vulkan device.
- OpenVINO runs on a Windows ARM64 laptop as an x64 build under emulation (CPU device): bit-identical to OpenVINO's Python
  API. `run_model` now flushes stdout after printing tensor info, because the OpenVINO runtime ends the process without
  flushing C stdio on Windows.
- NCNN builds and runs natively on Windows ARM64 (Release, Visual Studio): a yolov8n model takes 28 ms on the Snapdragon X
  Elite CPU and 22 to 26 ms on its Adreno GPU through Vulkan, with the same detections.

### Added (ONNX Runtime GPU)
- `execution_provider=cuda` and `tensorrt` for the ONNX Runtime backend, `ep_option.<key>` for them, and a `log_level`
  option. A provider that cannot load reports `UAIRT_ERR_BACKEND_UNAVAILABLE`. Verified with the CUDA provider on a
  GTX 1650: a yolov8n model runs in 10.0 ms against 59.9 ms on the CPU, with the same detections. The `tensorrt`
  provider runs it in 8.1 ms (engine compile 116 s on the first run, cached afterwards).

### Added (TensorRT)
- TensorRT backend (`tensorrt`, a C++ plugin): serialized engines (ultralytics exports load as is), static shapes, on
  an NVIDIA GPU. Built with `-DUAIRT_BUILD_TENSORRT=ON`. Verified on a GTX 1650 with TensorRT 10.9.0.34: outputs
  bit-identical to TensorRT's Python API.
- `UAIRT_MEM_PINNED` memory domain (appended, existing values unchanged): page-locked host buffers from
  `uairt_buffer_alloc`. On the TensorRT backend they cut a yolov8n run from 8.1 ms to 6.4 ms. `run_model --pinned`,
  and `Domain::Pinned` / `"pinned"` in the C++, Rust and Python bindings.
- Example: `bindings/python/examples/detect_tensorrt.py` (YOLOv8 detection through the Python binding).

### Added (TFLite)
- TFLite backend (`tflite`): `.tflite` from a path or from memory; optional QNN delegate on HTP.
  Built with `-DUAIRT_BUILD_TFLITE=ON`. The CPU path is verified on a QCS6490 and the delegate path on a QCS8550
  (Hexagon v73); outputs are identical to TFLite's C API used directly.

### Added (bindings)
- Python binding (`bindings/python`, `ctypes` and NumPy) and a header-only C++17 wrapper
  (`include/uairt/uairt.hpp`, RAII and exceptions) and a Rust binding
  (`bindings/rust`, `uairt-sys` and `uairt`).

### Added (ONNX Runtime plugin execution providers)
- The ONNX Runtime backend accepts `ep_library`, `ep_name` and `ep_option.<key>` to load a plugin execution
  provider. Verified with the MLX provider on an Apple M5: ResNet50 at batch 8 ran 3.1 to 3.2 times faster than
  the CPU EP with the same outputs; MobileNetV2 at batch 1 showed no gain.

### Added (llama.cpp and sessions)
- `llamacpp` backend: GGUF language models through llama.cpp, a pinned submodule (`third_party/llama.cpp`, MIT), built with
  `-DUAIRT_BUILD_LLAMACPP=ON`; Metal on Apple.
- Public session API: `uairt_model_vocab_size/tokenize/detokenize` and `uairt_session_*`, backed by an optional `session`
  field appended to `uairt_backend_api`. The host now validates plugins against the v1 struct size, so existing plugins
  load unchanged.
- Python: `Model.tokenize/detokenize`, `Session`, `generate()` (sampling in NumPy), `.gguf` in `AutoModel`.
- `examples/llm_generate.c` and a test that compares greedy output with llama.cpp's own.

### Changed
- QNN: `htp_performance_mode` option (`default`, `burst`, `high_performance`, `balanced`,
  `power_saver`). Without it HTP runs a small graph about 3.8 times slower than with
  `burst`.

### Verified
- ONNX Runtime: Add/Mul and MobileNetV2 (random weights) match Python ONNX Runtime
  1.24.4 exactly.
- QNN on a QCS6490 (Hexagon v68): a DLC on the CPU backend and a 680 MB Qwen2-0.5B
  prefill context binary on the HTP backend give byte-identical outputs to
  `qnn-net-run`. DMABUF zero-copy reduced the run time by about 5% on that model.
- QNN on a QCS8550 (Hexagon v73, Qualcomm Linux): a DLC and a context binary, on the CPU
  and HTP backends, host and DMABUF buffers, byte-identical to `qnn-net-run`. A context
  binary loaded about 3.3 times faster than the DLC.
- QNN versus TFLite with the QNN delegate on the QCS8550, same quantized network: the same
  speed within a few percent in every HTP performance mode (see `docs/design.md`).
- QNN versus ONNX Runtime with its QNN EP on the QCS8550, same compiled graph: byte-identical
  outputs and the same speed within a few percent where the performance mode names mean the
  same thing (see `docs/design.md`).
- TFLite backend on a QCS6490 with TFLite's CPU kernels: outputs identical to `tflite_bench`, same
  run time, sanitizers clean.
- CoreML on an Apple M5: three models, identical to `MLModel.predict` on CPU-only and
  all compute units.
- llamacpp on an Apple M5 (Metal and CPU): Llama 3.2 1B Q4, greedy output identical to llama.cpp's own
  `llama-completion`. Not run on Linux, Windows, CUDA or Vulkan.

### Known issues
- A DLC on the QNN HTP backend crashes in Qualcomm's `libQnnHtpPrepare.so` on the
  QCS6490 (Hexagon v68) with QAIRT 2.46.0.260424, including with the SDK's own
  `qnn-net-run`. The same DLC works on the QCS8550 (Hexagon v73). Context binaries work on
  both.
- `QnnSystemDlc_composeGraphs` allocates about 1.3 KB per DLC load that the QNN headers
  do not say how to release; LeakSanitizer reports it.
- Static shapes only, one graph per QNN model, no asynchronous execution, no dynamic
  shapes, no Windows plugin loading.
- TensorRT is not implemented.
- The TFLite backend's QNN delegate crashes on Hexagon v68 like the DLC path (also with the
  reference `tflite_bench`); it works on Hexagon v73.
- Plugin execution providers depend on ONNX Runtime's device discovery: the MLX provider works on macOS,
  but the `onnxruntime-qnn` plugin cannot be used on Linux (ONNX Runtime 1.24.4 does not discover an
  NPU); the comparison used a source build with the QNN EP built in (details in `docs/design.md`).
- Agreement with TFLite's outputs was measured only on random-noise input; there is no
  detection-accuracy comparison.
- Tested only on the configurations listed in `docs/api.md`.
