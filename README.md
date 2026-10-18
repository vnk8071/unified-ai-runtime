<!-- SPDX-License-Identifier: Apache-2.0 -->
# Unified AI Runtime (UAIRT)

One C API to run compiled AI models on QNN (Qualcomm NPUs), ONNX Runtime, TFLite, CoreML, llama.cpp (GGUF language models), TensorRT (NVIDIA GPUs), OpenVINO (Intel), NCNN (CPU and Vulkan GPUs), ExecuTorch (`.pte`), whisper.cpp (speech to text), and language model servers (Ollama, LM Studio, SGLang).
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

A complete runnable program is in [examples/quickstart.c](examples/quickstart.c); there are also C++, Python and Rust examples in [examples/](examples/README.md).

## Use it with a coding agent

This repository is written to be driven by a coding agent such as Claude Code, Codex or OpenCode: start the agent in the repository root (`claude`, `codex` or `opencode`) and say what you want, for example

> Build UAIRT and run `model.onnx` on this machine, on the fastest device it has, and tell me what actually ran.
> Run `face_det.dlc` on this board's NPU and compare the result with `qnn-net-run`.
> Export my PyTorch model to a format UAIRT loads, then run it and show me the output.
> Write a small program that uses UAIRT to transcribe `meeting.wav`.

The agent follows the playbook in [docs/agents/run-model.md](docs/agents/run-model.md):

1. **Find the device.** `scripts/doctor.sh` reports the operating system, CPU, GPU, NPU, toolchain, which vendor SDKs are installed and at what version, and which backends are ready.
2. **Pick the backend** from the model file and the device you asked for, and say what it needs.
3. **Ask you to install what it may not install.** Vendor SDKs (QAIRT, TensorRT, Xcode and so on) are yours to download and accept the licence for: the agent points you at the vendor page, waits, and
   checks with `doctor.sh`. It never downloads an SDK, accepts a licence for you, uses `sudo` or changes system settings without asking.
4. **Build and test** with CMake and `ctest`, then **run** the model on the device you asked for. There is no silent fallback to the CPU.
5. **Check the answer** against the original framework or the vendor's own tool on the same input.
6. **Report honestly.** "Core verified" (builds, `ctest` with the reference backend) is reported separately from "backend verified" (a real model on the real device, compared with the vendor tool), and the
   report says what was not verified.

Where the instructions live, so any agent finds them:

| File | For |
|---|---|
| [`AGENTS.md`](AGENTS.md) | the job and the rules (ABI, no vendor files, no SDK downloads, no `sudo`); read by Codex, OpenCode and other agents that follow the AGENTS.md convention |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code's entry point; it imports `AGENTS.md`. (OpenCode reads `AGENTS.md` first and uses `CLAUDE.md` only when there is none, so it is not read twice.) |
| [`.claude/skills/run-model-on-device`](.claude/skills/run-model-on-device/SKILL.md) | a skill that starts this workflow when you ask to run a model on a device; Claude Code and OpenCode both load skills from `.claude/skills/` |
| [`docs/agents/`](docs/agents/run-model.md) | the playbooks: run a model, set up a backend, export a model, write an application, troubleshoot, and [pre- and post-processing, tokenizers and decoding](docs/agents/pre-post-processing.md) driven by the model's printed inputs and outputs |
| [`docs/vendors.md`](docs/vendors.md) and `scripts/doctor.sh` | which vendor versions were tested and the minimum each backend needs |

These files were written and used with Claude Code. For OpenCode, `opencode debug skill` (version 1.3.17) lists this repository's two skills, and its documentation says it
reads `AGENTS.md`; a full OpenCode session has not been run. Codex reads the same `AGENTS.md` by convention, but a Codex run has not been tried here.

### Example: a Hugging Face model on a Mac

The prompt, typed into the agent in the repository root:

> Run the Hugging Face model `bartowski/SmolLM2-135M-Instruct-GGUF` (the `Q8_0` file) on this Mac with UAIRT and show me what it says to "The capital of France is".
> Check it against llama.cpp's own tool.

What it does, and what that looked like on an Apple M5 (macOS, Claude Code):

1. **Find the device.** `scripts/doctor.sh` printed `os: Darwin arm64`, `chip: Apple M5`, `cmake` and `cc` ok, and `llamacpp: ok (pinned submodule)`.
2. **Pick the backend.** A `.gguf` file means the `llamacpp` backend (the table in [docs/agents/run-model.md](docs/agents/run-model.md)). llama.cpp is a submodule, so there is no SDK for you to install, and
   the agent says so instead of stopping to ask.
3. **Get the file.** It tells you the download is 145 MB, fetches `SmolLM2-135M-Instruct-Q8_0.gguf` from the Hugging Face repository you named, and compares its SHA-256 with the one
   Hugging Face publishes for it (`5a1395716f79...`). (In the run shown here the file was already on disk from an earlier download, so only the hash check ran.)
4. **Build.** `UAIRT_LLAMACPP_TEST_MODEL=<file> cmake -S . -B build -DUAIRT_BUILD_LLAMACPP=ON && cmake --build build` (about 25 seconds here; setting the model variable also builds llama.cpp's own
   `llama-completion` for the comparison).
5. **Run it.** `build/llm_generate build/libuairt_backend_llamacpp.so SmolLM2-135M-Instruct-Q8_0.gguf "The capital of France is" 24`. On the Metal GPU (the default) it printed:

   ```text
   The capital of France is Paris. Paris is the largest city in France and the capital of the French department of department FRANCE. It is
   ```

   and on the CPU (`n_gpu_layers` set to `0`):

   ```text
   The capital of France is Paris. Paris is a city known for its historical landmarks, art, fashion, and culture. It is the political and
   ```

   The two differ after about ten tokens. That is floating-point behaviour on different devices with a model this small, not a bug, and it is why the check below compares like with like.
6. **Check it.** `ctest` ran `llamacpp_vs_llama_completion`: for 30 greedy tokens UAIRT's text is identical to `llama-completion`'s, once on the CPU and once on Metal. `llamacpp_errors`, `llamacpp_model` and
   `llamacpp_symbols` passed too.
7. **Report.** "Backend verified: this one model on this Mac, on the CPU and Metal, matching llama.cpp's own tool. Not verified: other models, CUDA or Vulkan, a Hugging Face repository that has no GGUF."

If a repository has only PyTorch weights, the agent follows [docs/agents/export-model.md](docs/agents/export-model.md) (convert to GGUF with llama.cpp's `convert_hf_to_gguf.py`, or export to ONNX).
That conversion path has not been run in this repository yet, so expect the agent to tell you what it could not check.

## Getting started

1. **Bring a model in the format its runtime reads.** UAIRT does not convert or quantize: use an `.onnx`, `.dlc`, `.tflite`,
   `.mlpackage`, `.gguf` and so on. [docs/agents/export-model.md](docs/agents/export-model.md) shows how to produce one.
2. **Install.** For Python, `pip install "uairt[onnxruntime]"` (ONNX Runtime only; see [Install (Python)](#install-python)). For
   any other backend, build from source with that backend's option and its SDK, after checking what the machine supports:

   ```bash
   scripts/doctor.sh
   cmake -S . -B build -DUAIRT_BUILD_<BACKEND>=ON && cmake --build build     # llama.cpp, whisper.cpp and NCNN are submodules: git submodule update --init --depth 1 third_party/<name> first
   ```

   Each backend's page in [docs/backends/](docs/backends/README.md) lists what to install and the build flag.
3. **Look at the model.** `build/run_model --info <plugin> <backend> <model>` prints each input's name, dtype and shape.
4. **Run it.** The backend comes from the model file and the device in Python, or from a name in C:

   ```python
   import uairt
   with uairt.AutoModel.from_file("model.onnx", device="cpu") as model:          # or gpu, npu where the backend has one
       (y,) = model.run(preprocess(image).astype(model.inputs[0].dtype))
   ```

   Language models (`.gguf`) use a session: `for piece in model.generate("Hello", max_tokens=40): print(piece, end="")`.
   In C, copy the call order from [examples/quickstart.c](examples/quickstart.c) or [examples/llm_generate.c](examples/llm_generate.c).
5. **Check it.** Compare the output with the original framework or the vendor's own tool on the same input.

To see it run before writing anything, open [examples/notebooks/uairt_backends_tour.ipynb](examples/notebooks/uairt_backends_tour.ipynb): it builds UAIRT and runs a tiny model through
each backend it can (a section whose runtime is missing skips itself and says why). A [Google Colab notebook](examples/notebooks/uairt_tensorrt_colab.ipynb) does the same for TensorRT on a GPU.
Both are saved with the outputs of a real run; see [examples/notebooks/README.md](examples/notebooks/README.md).

What you still write: pre- and post-processing (resize, layout, normalization, box decoding, chat templates; [docs/agents/pre-post-processing.md](docs/agents/pre-post-processing.md) walks through it, starting from `examples/python/inspect_model.py`). Inputs must match the
model's dtype and shape exactly, because UAIRT never converts silently. If a backend cannot serve the device you ask for, the call fails
instead of falling back to the CPU. [docs/agents/write-app.md](docs/agents/write-app.md) has the details and the pitfalls.

## The problem it solves: porting a large application to another accelerator

Say you built your application on TensorRT because the first target was an NVIDIA GPU. Months later the project is large, and
the product has to ship on a Qualcomm NPU. With the vendor APIs that means starting over on QNN: a new engine and context
lifecycle, new tensor and memory handling, new error handling, and a port of every place the application runs a model.

With UAIRT the inference code stays. The application already talks to `uairt_engine`, `uairt_model` and `uairt_tensor`, so
moving to QNN is a change of inputs, not of code:

| Changes | Stays |
|---|---|
| The backend name: `"tensorrt"` to `"qnn"` (and its plugin library) | Engine, model and run calls, error handling, threading |
| The model artifact: a TensorRT `.engine` to a QNN `.dlc` or context binary, made with Qualcomm's tools | Application logic, buffers and tensor-shape handling |
| Backend options, such as the HTP device | The rest of the project |

This is a port of the model, not of the program. Be aware of what UAIRT does not do for you:

- **Converting the model is still your job** (see [docs/agents/export-model.md](docs/agents/export-model.md)), and a quantized QNN model
  can need different input dtypes and scales than the FP32 TensorRT one. UAIRT never converts silently, so a mismatch fails loudly;
  read the dtype, shape and quantization from `uairt_model_input_info` rather than hard-coding them.
- **Pre- and post-processing is yours**, and it may change with the model format
  ([docs/agents/pre-post-processing.md](docs/agents/pre-post-processing.md)).
- **Verify the new backend** against the old one on the same inputs and compare the raw outputs, not just the top result.

## What it is, and is not

UAIRT loads artifacts that each vendor's toolchain already produced (an ONNX file, a QNN
DLC or context binary, a CoreML model) and runs them through that vendor's runtime. It
does not convert, quantize or optimize models, and it does not hide that artifacts are
tied to a device or SDK version: a mismatch returns an error instead of falling back
quietly. It is not an agent runtime. If your models are all ONNX, ONNX Runtime and its
execution providers may already be enough; UAIRT is for mixing artifact formats behind one
API, with a plain C ABI that other languages can bind to. (Behind that ABI the core is C++17 with no C++ runtime dependency; plugins stay C or C++ as they like.)

## Status

Version 0.1.0, pre-release: the API and ABI may change. See [CHANGELOG.md](CHANGELOG.md).

| Backend | Models | Verified on |
|---|---|---|
| ONNX Runtime | `.onnx`, static shapes; CPU, and the CUDA and TensorRT providers on NVIDIA GPUs | macOS arm64 (CPU) and a GTX 1650 (CUDA provider); matches Python ONNX Runtime. Compiles on Windows ARM64, not run there |
| QNN | `.dlc`, context binaries; CPU and HTP; DMABUF zero-copy; compiled-context cache | Qualcomm QCS6490 and QCS8550, and Windows 11 ARM64 on a Snapdragon X Elite NPU; byte-identical to `qnn-net-run` |
| TFLite | `.tflite`; CPU kernels, optional QNN delegate | Qualcomm QCS6490 (CPU path only); identical to `tflite_bench` |
| CoreML | `.mlmodelc`, `.mlpackage`, `.mlmodel`; multi-array I/O, image inputs | Apple M5; identical to CoreML itself |
| llama.cpp | `.gguf`; sessions (tokenize, append, logits); text completion (`uairt_model_complete`); `device` picks a llama.cpp device (`CUDA0`, `Vulkan0`, `GPUOpenCL`, `HTP0`) | Apple M5 (Metal); Ubuntu 24.04 x86_64 RTX 3060 (CUDA and Vulkan); Windows 11 ARM64 Snapdragon X Elite (CPU, OpenCL on the Adreno GPU, Hexagon NPU as `HTP0`). `ctest` compares with llama.cpp's own greedy output; some GPU and NPU outputs differ from the CPU's after a few tokens (see [docs/backends/llamacpp.md](docs/backends/llamacpp.md)) |
| LLM servers | Ollama, LM Studio, SGLang through one `llmserver` backend (attach to a running server over HTTP; text in, streamed text out; no tensors or sessions) | macOS arm64 with Ollama only: output identical to a direct request to the server. LM Studio, SGLang and Modular MAX (`max serve`, via `base_url`) untested (see [docs/backends/llmserver.md](docs/backends/llmserver.md)) |
| OpenVINO | IR (`.xml` + `.bin`) and `.onnx`, static shapes; any OpenVINO device | an Intel Core CPU, and an x64 build under emulation on a Snapdragon X Elite laptop (there is no Windows ARM64 OpenVINO); bit-identical to OpenVINO's Python API. GPU and NPU devices untested |
| NCNN | `.param` + `.bin`, float32, static shapes; CPU and Vulkan | i9-9900K CPU and a GTX 1650 through Vulkan (matches NCNN's Python API); a Snapdragon X Elite laptop, CPU and Adreno GPU through Vulkan, same detections |
| whisper.cpp | ggml `.bin` Whisper models; audio in (float32 PCM, 16 kHz), UTF-8 text out; CPU and Metal | Apple M5, `tiny.en`: transcript equal to `whisper-cli` on both CPU and Metal. Other models, CUDA and Vulkan untested |
| ExecuTorch | `.pte`, tensor I/O, static shapes; CPU (XNNPACK) or whatever delegates your ExecuTorch build has | Apple M5 CPU (XNNPACK): equal to ExecuTorch's Python runtime on a small synthetic net. The experimental Metal delegate ran the same net, within 1.2e-8 of the CPU result. No NPU delegate tested |
| TensorRT | serialized `.engine` / `.plan` (ultralytics exports load as is), static shapes; host and pinned buffers | NVIDIA GTX 1650, TensorRT 10.9; bit-identical to TensorRT's Python API. Also a Colab Tesla T4 (TensorRT 10.9.0.34 from pip): a ResNet-18 engine matched PyTorch to 1.5e-6 (FP32) and 2.6e-3 (FP16) |

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
your own `onnxruntime`. The other backends (QNN, TensorRT, CoreML, OpenVINO, NCNN, TFLite, ExecuTorch, whisper.cpp, llama.cpp and the LLM-server
backend) are not in the wheel: build their plugins from this repository (below) and point `UAIRT_PLUGIN_PATH` at them. See
[bindings/python/README.md](bindings/python/README.md).

## Build

Needs CMake 3.16+ and a C11 and C++17 compiler (the core is C++17 behind the C API; nothing links a C++ runtime).

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

### Android (NDK)

The core, the plugin loader and the C tests cross-compile with the Android NDK (r30 tested) for `arm64-v8a`; plugins are ordinary shared
libraries, so `dlopen` works as on Linux. With `ANDROID_STL=none` (the preset) `libuairt.so` depends only on `libdl`, `libm` and `libc`.

```bash
export ANDROID_NDK_HOME=/opt/homebrew/share/android-ndk       # wherever the NDK is
cmake --preset android-arm64 && cmake --build --preset android-arm64
adb push build-android-arm64/{libuairt.so,libuairt_llm.so,test_core,uairt_reference_plugin.so} /data/local/tmp/uairt/
adb shell 'cd /data/local/tmp/uairt && LD_LIBRARY_PATH=. ./test_core uairt_reference_plugin.so'
```

The C++ binding and examples need the C++ library: build with `-DANDROID_STL=c++_static` and push the NDK's `libc++_shared.so` next to them (the core then
depends on it too). For llama.cpp add `-DUAIRT_BUILD_LLAMACPP=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm`
(drop the `+i8mm` part on a CPU without it; check `Features` in `/proc/cpuinfo`). Checked on a Qualcomm "canoe" board (Android 16): the core and C/C++ tests pass,
and llama.cpp (CPU) generates text identical to llama.cpp's own `llama-completion` on the same model. Not done: Android CI, GPU (Vulkan/OpenCL) and Hexagon NPU
runs, the QNN backend on Android, and an app packaged as an APK.

### ONNX Runtime

Download a release from https://github.com/microsoft/onnxruntime/releases and unpack it.

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-<version>
cmake -S . -B build -DUAIRT_BUILD_ONNXRUNTIME=ON && cmake --build build
```

The plugin needs only the ONNX Runtime header to build and loads the library at run time, so the lighter route is `pip install onnxruntime` plus
`scripts/ci/fetch_onnxruntime_headers.sh` (headers into the git-ignored `.ort/`, used when `ONNXRUNTIME_ROOT` is unset).

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

NCNN is a pinned submodule: `git submodule update --init --depth 1 third_party/ncnn`, then build it into the plugin with
`-DUAIRT_BUILD_NCNN=ON` (CPU; add `-DUAIRT_NCNN_VULKAN=ON` and `--init --recursive` for Vulkan). Or build NCNN yourself and set
`NCNN_ROOT` to its install prefix. NCNN's `.param` records no input shapes, so pass them:

```bash
git submodule update --init --depth 1 third_party/ncnn
cmake -S . -B build -DUAIRT_BUILD_NCNN=ON -DUAIRT_NCNN_VULKAN=ON && cmake --build build
build/run_model --repeat 100 --option input_shapes=3x640x640 --option device=vulkan --option fp16=false \
  build/libuairt_backend_ncnn.so ncnn yolov8n_ncnn_model out in0.bin
```

On a GTX 1650 a yolov8n model runs in 21.9 ms through Vulkan against 79 ms on the CPU; `device=vulkan` fails instead of
quietly using the CPU when no GPU is usable. See [docs/backends/ncnn.md](docs/backends/ncnn.md).

### llama.cpp (GGUF language models)

llama.cpp is a pinned submodule (MIT), so there is nothing to install; Metal is built in on a Mac. CUDA, Vulkan, OpenCL (Adreno) and the Hexagon NPU
(`HTP0`) take their own flags and are described on the backend page.

```bash
git submodule update --init --depth 1 third_party/llama.cpp
cmake -S . -B build -DUAIRT_BUILD_LLAMACPP=ON && cmake --build build
build/llm_generate build/libuairt_backend_llamacpp.so model.gguf "The capital of France is" 40
```

Options: `n_gpu_layers` (default 99; `0` stays on the CPU) and `device` (one llama.cpp device by name). See [docs/backends/llamacpp.md](docs/backends/llamacpp.md).

`build/llm_stream` is the same with sampling and streaming text (`libuairt_llm`, [docs/api.md](docs/api.md#generation-uairtuairt_llmh-libuairt_llm)):
`build/llm_stream build/libuairt_backend_llamacpp.so model.gguf "The capital of France is" 40 0.8 40 0.95 7` (temperature, top-k, top-p, seed).

### LLM servers (Ollama, LM Studio, SGLang)

Attaches to a server you have already started and gives you the same text API (`uairt_model_complete`) as llama.cpp. It needs libcurl and downloads no model.

```bash
cmake -S . -B build -DUAIRT_BUILD_LLMSERVER=ON && cmake --build build
build/llm_complete build/libuairt_backend_llmserver.so ollama llama3.2:1b "Name three colors." max_tokens=64
```

See [docs/backends/llmserver.md](docs/backends/llmserver.md).

### whisper.cpp (speech to text)

whisper.cpp is a pinned submodule (MIT). You need a ggml Whisper model (for example `ggml-tiny.en.bin`, 78 MB, from the whisper.cpp model list). The tensor API has no
strings, so the model takes `audio` (float32 `[n_samples]`, mono 16 kHz) and returns `text` (UTF-8 bytes) and `text_length`:

```bash
git submodule update --init --depth 1 third_party/whisper.cpp
cmake -S . -B build -DUAIRT_BUILD_WHISPERCPP=ON && cmake --build build
build/run_model --option n_samples=176000 build/libuairt_backend_whispercpp.so whispercpp ggml-tiny.en.bin out audio.f32
```

Options include `language`, `translate`, `num_threads`, `use_gpu` and `flash_attn` (on by default, as in whisper.cpp). See [docs/backends/whispercpp.md](docs/backends/whispercpp.md).

### ExecuTorch (`.pte`)

Build and install ExecuTorch yourself with its Module and Tensor extensions (the exact flags are on the backend page), then:

```bash
export EXECUTORCH_ROOT=$HOME/executorch-install
cmake -S . -B build -DUAIRT_BUILD_EXECUTORCH=ON && cmake --build build
build/run_model --option num_threads=1 build/libuairt_backend_executorch.so executorch model.pte out in0.bin
```

Which hardware runs the model is decided when you export the `.pte` (XNNPACK for the CPU; the experimental Metal delegate is described on the page). `num_threads` matters for tiny
models: a microsecond-scale net took 45.7 µs per run with ExecuTorch's default pool and 8.9 µs with one thread. See [docs/backends/executorch.md](docs/backends/executorch.md).

### TensorRT (NVIDIA GPUs)

Install an NVIDIA driver, the CUDA toolkit and TensorRT 8.5+ with its headers yourself (the apt packages
`libnvinfer-dev` or NVIDIA's tarball). An engine is built for one GPU and one TensorRT version, so build it on the GPU
that runs it, with `trtexec` or ultralytics (`YOLO("yolov8n.pt").export(format="engine")`).

**Without apt or a tarball** (for example on Google Colab): `pip install tensorrt-cu12==10.9.0.34` supplies the libraries, and the headers come from
NVIDIA's public repository (the `include/` directory of its `v10.9.0` tag). Tell CMake where both are:

```bash
cmake -S . -B build -DUAIRT_BUILD_TENSORRT=ON \
  -DUAIRT_TRT_INCLUDE=/path/to/TensorRT/include \
  -DUAIRT_TRT_LIBRARY=$(python -c "import tensorrt_libs, pathlib; print(next(pathlib.Path(tensorrt_libs.__file__).parent.glob('libnvinfer.so.10')))") \
  -DCMAKE_MODULE_LINKER_FLAGS=-Wl,-rpath,$(python -c "import tensorrt_libs, pathlib; print(pathlib.Path(tensorrt_libs.__file__).parent)")
```

[examples/notebooks/uairt_tensorrt_colab.ipynb](examples/notebooks/uairt_tensorrt_colab.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vnk8071/unified-ai-runtime/blob/main/examples/notebooks/uairt_tensorrt_colab.ipynb)
does all of it on a free GPU runtime: install, build, export ResNet-18, build FP32 and FP16 engines, run them through the Python binding and compare with
PyTorch. Its saved run, on a Colab Tesla T4 (driver 580.82.07, CUDA 13.0 toolkit, Python 3.13), gave outputs within 1.5e-6 (FP32) and 2.6e-3 (FP16) of PyTorch,
the same top-1 class, and per-run times of 1.79 ms (1.54 ms with pinned buffers) for FP32 and 0.76 ms (0.60 ms) for FP16, against 3.32 ms for eager PyTorch.
That is one GPU and a random-weight model, and PyTorch was not given TensorRT: it shows the engine is faster than eager PyTorch, not UAIRT's overhead over
TensorRT used directly (the yolov8n measurement below does compare with it).

```bash
export TENSORRT_ROOT=/path/to/TensorRT            # not needed for a system install
cmake -S . -B build -DUAIRT_BUILD_TENSORRT=ON && cmake --build build
build/run_model --info build/libuairt_backend_tensorrt.so tensorrt yolov8n.engine
build/run_model --repeat 100 --pinned build/libuairt_backend_tensorrt.so tensorrt yolov8n.engine out in0.bin
```

`--pinned` allocates the inputs and outputs as page-locked buffers (`UAIRT_MEM_PINNED`), which made a yolov8n run
8.1 ms to 6.4 ms on a GTX 1650. A complete detection program, with the Python binding and pinned buffers, is
[examples/python/detect_tensorrt.py](examples/python/detect_tensorrt.py); see
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
- [examples/notebooks/](examples/notebooks/README.md): a notebook with one runnable example per backend, and a Google Colab notebook that runs TensorRT on a GPU
- [docs/writing-a-backend.md](docs/writing-a-backend.md): the plugin ABI and how to add a backend
- [docs/design.md](docs/design.md): design decisions, per-backend notes, measurements
- [docs/LICENSING.md](docs/LICENSING.md): how vendor licenses shape this repository
- [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md)
- [AGENTS.md](AGENTS.md) and [docs/agents/](docs/agents/run-model.md): instructions for coding agents, including a
  playbook for building UAIRT and running a model on a target device (`scripts/doctor.sh` reports the device), choosing
  and exporting a model format ([docs/agents/export-model.md](docs/agents/export-model.md)) and writing application code
  ([docs/agents/write-app.md](docs/agents/write-app.md)). People can follow the same pages
- [docs/vendors.md](docs/vendors.md): which vendor SDK versions were tested, the minimum versions, and where to get them

## Vendor SDKs

QAIRT (Qualcomm), Xcode and CoreML (Apple) and TensorRT (NVIDIA) are proprietary and are
never part of this repository. You install them yourself under their own licenses and
point the build at them. The open-source runtimes (llama.cpp and whisper.cpp under MIT, NCNN under BSD-3-Clause) are different: they are pinned git submodules, fetched only
when you ask, and their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See [docs/LICENSING.md](docs/LICENSING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
