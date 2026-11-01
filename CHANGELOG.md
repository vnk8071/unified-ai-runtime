<!-- SPDX-License-Identifier: Apache-2.0 -->
# Changelog

## Unreleased

On `main` after the `v0.2.0` tag; not in the 0.2.0 wheels.

### Fixed
- `whispercpp`: `language=auto` returned an empty transcript, because the backend set whisper.cpp's `detect_language`, which detects the language and stops without transcribing. Found through a comment on a Reddit post; `tests/compare_with_whisper_cli.py` now compares `auto` as well as `en` with `whisper-cli`, and fails on the old code. Checked only on the English-only `tiny.en`.
- A shared build (`-DBUILD_SHARED_LIBS=ON`) on Linux failed to link `test_llm` (`undefined reference to symbol 'logf' ... DSO missing from command line`): the test now links `m`. Seen on Colab and reproduced on an Ubuntu 20.04 aarch64 board; the default static build and macOS were not affected.

### Added (generic ONNX Runtime provider)
- `onnxruntime` backend: `execution_provider=<name>` passes any provider in the loaded ONNX Runtime build (`CoreML`, `QNN`, `OpenVINO`,
  `XNNPACK`, ...) to ONNX Runtime by name, with `ep_option.<key>` as its options. A provider the build lacks, or an unknown name, is
  now `UAIRT_ERR_BACKEND_UNAVAILABLE` (it was `UAIRT_ERR_UNSUPPORTED`). Only CoreML was run (Apple M5). `cpu_fallback=0` (ONNX Runtime's `session.disable_cpu_ep_fallback`) makes a model that would run on ONNX Runtime's CPU provider fail to load.

### Added (Transformers.js backend)
- `transformersjs` backend (POSIX): runs a Transformers.js model directory in a Node worker (`-DUAIRT_BUILD_TRANSFORMERSJS=ON`; you install Node and `@huggingface/transformers`). Options `package_dir`, `node`, `worker`, `device` (`cpu`, `webgpu`), `dtype`, `input_shapes`. `device=webgpu` is refused unless ONNX Runtime logged nodes on `WebGpuExecutionProvider`. Verified on an Apple M5 with all-MiniLM-L6-v2: identical to Transformers.js called from Node on `cpu` and `webgpu`. Not run on Linux or other models; see `docs/backends/transformersjs.md`.

### Added (WebGPU through ONNX Runtime)
- `onnxruntime` backend: `execution_provider=webgpu` is an alias for ONNX Runtime's `WebGPU` provider. An ONNX Runtime build without it fails at engine creation with `UAIRT_ERR_BACKEND_UNAVAILABLE`. Run on an Apple M5 with the ONNX Runtime 1.24.4 macOS release (has WebGPU): all nodes on `WebGpuExecutionProvider`, outputs equal the CPU provider's (add_mul exactly, a small conv net within 1.1e-8), and slower than the CPU on that tiny net. A fixed-shape `all-MiniLM-L6-v2` (Transformers.js's ONNX export, 1x16 tokens) also ran: 328 of 331 nodes on WebGPU, output within 2.3e-6 of ONNX Runtime's CPU run, slower than the CPU. The pip ONNX Runtime 1.31.0 has no WebGPU provider (the error path). WebCL has no runtime anywhere (Khronos dropped it); use `device=opencl` in `mnn` or `device=GPUOpenCL` in `llamacpp`, neither run on OpenCL here.

### Added (benchmark tool)
- `scripts/bench_model.py`: exports a PyTorch module or ONNX file to every target of `export_model.py`, runs each backend and option variant in its own process through UAIRT,
  checks the outputs against the original and ranks latency (the core's own timing), with a verdict that keeps exact and reduced-precision results apart; `--artifact` adds files
  you already have (a QNN `.dlc`, a TensorRT engine). `docs/agents/benchmark-model.md` is the playbook. `export_model.py` gained `align_inputs` and `onnx_input_info` for it.

### Added (Paddle Lite backend)
- `paddlelite` backend (`-DUAIRT_BUILD_PADDLELITE=ON`, `PADDLE_LITE_ROOT`): `.nb` models, float32, static shapes (`input_shapes` required), CPU; options `num_threads`, `power_mode`,
  `precision`. Byte-identical to Paddle Lite's own C++ API in `tests/compare_with_paddlelite.py`; `ctest -R paddlelite` runs `tests/test_paddlelite.c` on `tests/data/add_mul.nb`.
  Paddle Lite is built by the user (the recipe and its workarounds are in `docs/backends/paddlelite.md`); not run on Android or Linux. `uairt.AutoModel` maps `.nb` to it.

### Added (MNN backend)
- `mnn` backend (`-DUAIRT_BUILD_MNN=ON`, MNN 3.6.1 as the pinned submodule `third_party/MNN`, or `MNN_ROOT`): `.mnn` models, CPU and Metal, options `device`, `num_threads`,
  `precision` and `input_shapes`. A GPU device that MNN did not end up on is an error. Identical to MNN's Python API in `tests/compare_with_mnn.py`;
  `ctest -R mnn` runs `tests/test_mnn.c` on `tests/data/add_mul.mnn`. The Android build compiled; it was not run on a device.
- `scripts/export_model.py` can write `.mnn` (from ONNX); `uairt.AutoModel` maps `.mnn` to the backend; `doctor.sh` knows MNN.

### Added (macOS notebook)
- `examples/notebooks/uairt_macos.ipynb`: builds every Apple-relevant backend, runs `ctest`, runs ResNet-18 through all variants and proves the Metal runs; `scripts/bench_model.py` gained three variants for ONNX Runtime's CoreML provider (`cpu_fallback=0`).

### Added (agent skills and the front page)
- `.claude/skills/`: `export-model`, `benchmark-model`, `write-uairt-app`, `use-onnxruntime-providers`, `add-backend`, `test-on-remote-device`, `test-on-colab-gpu` and `release-uairt`, next to `run-model-on-device` and `check-references`.
- The README became an open-source front page: badges, an architecture diagram, a testing and CI section, and contributing, citation and licence pointers. Added `CODE_OF_CONDUCT.md`, `CITATION.cff`, a pull request template and issue templates.
- A README hero image generated from measured numbers (`docs/images/hero-data.json`, `scripts/make_hero.py`; light and dark SVGs, and a 1280x640 social preview card) and the release notes for 0.2.0 (`docs/releases/v0.2.0.md`).
- `examples/notebooks/uairt_tensorrt_colab.ipynb` gained sections 9 and 10: the `onnxruntime` backend with its CUDA and TensorRT providers (the CUDA provider ran on a Colab T4; the TensorRT provider rows are not yet verified).

## 0.2.0 (2026-10-19)

Pre-1.0: the API and ABI may change. 0.1.0 was published to PyPI on 2026-10-05; this section grew while it was still
labelled unreleased, so entries from before that date were already in 0.1.0 and are not marked separately. Since 0.1.0: the
llama.cpp, whisper.cpp, ExecuTorch and llmserver backends, the C++17 core behind the unchanged C API, the Windows
ARM64 wheel, vendor version checks in `doctor`, and the export, application and pre/post-processing guides for agents and people.

### Added
- C11 API (`uairt/uairt.h`): engines, models, tensors, zero-copy buffers, status codes.
- Backend plugin ABI (`uairt/uairt_backend.h`): backends are shared libraries that fill in
  a function table and do not link the core.
- Backends: `reference` (tests), ONNX Runtime (CPU), QNN (CPU and HTP; DLC and context
  binaries; DMABUF zero-copy), CoreML (`.mlmodelc`, `.mlpackage`, `.mlmodel`).
- `run_model` command-line tool, `quickstart` example, `scripts/doctor.sh`.
- Tests that compare each backend with the vendor's own tool: Python ONNX Runtime,
  `qnn-net-run`, CoreML through `coremltools`.

### Added (vendor versions and agent guides)
- `scripts/vendors.tsv` and `docs/vendors.md`: tested and minimum vendor versions and download pages. `doctor.sh` and
  `doctor.ps1` detect the installed version, flag one older than the minimum, point at a newer installed QAIRT, and print
  the download page for a missing SDK (they never download anything). Dependabot proposes llama.cpp submodule bumps.
- `docs/agents/export-model.md` (choose a format and export a model for a device) and `docs/agents/write-app.md` (write
  application code against UAIRT).
- The Windows ARM64 CI job runs without a repository variable, on Visual Studio 2026 (the runner image has no 2022).
- `doctor` no longer reports a device whose name merely contains "npu" (such as "Hyper-V Input") as an NPU; Visual Studio 2026 is accepted.
- The llama.cpp backend builds with MSVC; `llm_generate` flushes stdout (OpenCL on Windows ends the process without flushing).

### Added (Python wheels)
- A root `pyproject.toml` (scikit-build-core) builds the `uairt` wheel: the Python package, `libuairt` and the ONNX Runtime plugin,
  tagged `py3-none-<platform>` so one wheel per platform serves every Python 3.x. `.github/workflows/wheels.yml` builds and tests
  the wheels with cibuildwheel (Linux x86_64 and aarch64, macOS arm64, Windows x64, Windows ARM64) and
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

### Added (agent guide for pre- and post-processing)

- `docs/agents/pre-post-processing.md`: how an agent goes from a model's printed inputs and outputs to a verified result: images (resize, crop, letterbox, colour
  order, range, normalization, quantized inputs), audio, box decoding with NMS, classification, embeddings, CTC, tokenizers (GGUF and Hugging Face `tokenizers`), chat
  templates read from the GGUF, stop tokens, sampling, and how to verify against the original pipeline. Linked from `AGENTS.md`, `write-app.md`, `run-model.md`, the
  `run-model-on-device` skill and the README.
- `examples/python/inspect_model.py` prints a model's inputs and outputs and a heuristic guess of what it is; `prepost.py` holds the tested NumPy helpers.
  `tests/test_prepost.py` and `tests/test_inspect_model.py` need no native library. Checked on an Apple M5: NMS equals `torchvision.ops.nms` (150 random cases), the ImageNet
  recipe equals torchvision's (7e-7, 8 image shapes), a ResNet-18 through UAIRT equals PyTorch (2e-6), UAIRT's GGUF tokenizer ids equal Hugging Face's, and the hand-built chat
  prompt equals the GGUF's own template. Not run: log-mel features, CTC with a real model, segmentation, quantized models, a trained detector.

### Changed (core language)

- The core (`src/uairt.cpp`, `src/loader.cpp`) is now C++17 instead of C11; the public headers, the plugin ABI and every exported symbol are unchanged (38 before and
  after, same names, C linkage). It is built without exceptions or RTTI and uses no STL or `new`, so `libuairt` still depends only on the system C library and a
  plain C program links the static archive with the C driver. The builtin reference backend and all plugins are unchanged. Verified on macOS arm64 (shared and static
  builds, the C, C++ and Python binding tests, ASan and UBSan); not yet built with GCC, MSVC or on Windows.

### Added (notebooks)

- `examples/notebooks/uairt_backends_tour.ipynb`: builds UAIRT and runs a tiny model through each backend it can; a section whose runtime is missing
  skips itself with the reason. Executed end to end on an Apple M5: `reference`, `onnxruntime`, `ncnn`, `whispercpp`, `llamacpp`, `coreml` and `openvino`
  ran and matched a NumPy, PyTorch, Whisper or llama reference (`executorch` too, with an ExecuTorch install); `tflite`, `qnn` and `tensorrt` skipped (no runtime here).
  Both notebooks are committed with their saved outputs.
- `examples/notebooks/uairt_tensorrt_colab.ipynb`: TensorRT 10.9 on a Colab GPU (install, build the plugin, ResNet-18 engine, comparison with PyTorch,
  timing). Ran to completion on a Colab Tesla T4 (Python 3.13, CUDA 13.0 toolkit, TensorRT 10.9.0.34): the FP32 engine matched PyTorch to 1.5e-6, the FP16
  engine to 2.6e-3, same top-1 class.

### Added (submodules)

- NCNN and whisper.cpp are pinned git submodules like llama.cpp (`third_party/ncnn`, `third_party/whisper.cpp`), built statically into
  their plugins with only `uairt_backend_get_api` visible. `NCNN_ROOT` and `WHISPER_ROOT` still select your own install instead.
  `docs/LICENSING.md` and `THIRD_PARTY_NOTICES.md` list them.

### Added (whisper.cpp)

- whisper.cpp backend (`whispercpp`, a C plugin): speech to text with ggml Whisper models. Float32 16 kHz audio in, UTF-8 text and
  its length out; `n_samples` and `max_text_bytes` set the static shapes. Built with `-DUAIRT_BUILD_WHISPERCPP=ON` against a
  shared whisper.cpp install. Verified on an Apple M5 (CPU and Metal, `tiny.en`) against `whisper-cli`; `doctor.sh` reports it.

### Added (ExecuTorch)

- ExecuTorch `num_threads` option. On a tiny model the default thread pool costs about 45 µs per run against 9 µs with one thread.

- Plugins may export `uairt_backend_global_symbols` to be opened `RTLD_GLOBAL` instead of `RTLD_LOCAL` (POSIX only, opt-in per
  plugin; the default is unchanged). The ExecuTorch Metal build uses it.

- ExecuTorch backend (`executorch`, a C++ plugin): `.pte`, tensor I/O, static shapes, `method` option. Built with
  `-DUAIRT_BUILD_EXECUTORCH=ON` against an install you build. Verified on an Apple M5 CPU only (portable kernels and the
  XNNPACK delegate, a small synthetic net); `doctor.sh` reports it.

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
- Example: `examples/python/detect_tensorrt.py` (YOLOv8 detection through the Python binding).

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

### Added (exporting models)
- `scripts/export_model.py`: exports a PyTorch module or an ONNX file to ONNX Runtime, CoreML, ExecuTorch, OpenVINO or NCNN and checks each result through UAIRT against the
  original (one process per target). `docs/agents/export-model.md` is rewritten around it: a source-by-target table, what was run here, and the traps found (half-precision defaults in
  OpenVINO and `pnnx`, `qnn-net-run` needing native-file flags for quantized models).
- OpenVINO backend: option `inference_precision` (`f32`, `f16`, `bf16`).

### Fixed (OpenVINO)
- Engine options of the OpenVINO backend (`performance_hint`, `num_threads`, `cache_dir`): any single option made the model fail to load ("invalid c input parameters"), and with several only
  the first took effect, because the property count passed to OpenVINO's C API is an argument count (two per property). `tests/compare_with_openvino.py` now checks each option.

### Added (iOS and Android examples)
- `examples/ios`: an app that links the core and the CoreML backend statically and runs bundled models on an iPhone, with a live camera detector (button, compute-unit switch); checked on an iPhone 12 against a Mac. `examples/android`: `run_on_device.sh`, and an app (built into an APK without Gradle, `app/build_apk.sh`) that detects objects in a picture through the TFLite backend; checked on a Qualcomm board.

### Added (Android)
- QNN and TFLite backends cross-compile for Android; the QNN plugin keeps `libQnnHtp.so` mapped at exit there (closing it crashed). Checked on an SM8850 board against `qnn-net-run`.
- NDK build (`cmake --preset android-arm64`) for `arm64-v8a`; checked on a Qualcomm device, including llama.cpp against `llama-completion`.
- The plugin loader reloads an opt-in (`uairt_backend_global_symbols`) plugin with `RTLD_GLOBAL` on Android, whose linker ignores `RTLD_GLOBAL` when
  reopening a loaded library.

### Fixed
- `prepost.nms` returns no boxes for no boxes (it crashed when classes were passed).

### Added (generation)
- `libuairt_llm` (`uairt/uairt_llm.h`): sampling (temperature, top-k, top-p, seed) and a streaming generation loop over the session
  API, for any backend that has sessions, stopping where the model ends its text. `examples/llm_stream.c`, and C++, Python and Rust
  examples under `examples/`.
- `uairt_model_is_end_of_generation` (core) and an optional `is_end_of_generation` entry at the end of the session table; `llamacpp` implements it.

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
  shapes.
- The TFLite backend's QNN delegate crashes on Hexagon v68 like the DLC path (also with the
  reference `tflite_bench`); it works on Hexagon v73.
- Plugin execution providers depend on ONNX Runtime's device discovery: the MLX provider works on macOS,
  but the `onnxruntime-qnn` plugin cannot be used on Linux (ONNX Runtime 1.24.4 does not discover an
  NPU); the comparison used a source build with the QNN EP built in (details in `docs/design.md`).
- Agreement with TFLite's outputs was measured only on random-noise input; there is no
  detection-accuracy comparison.
- Tested only on the configurations listed in `docs/api.md`.
