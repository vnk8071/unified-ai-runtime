<!-- SPDX-License-Identifier: Apache-2.0 -->
# Notebooks

| Notebook | What it does | Where it runs |
|---|---|---|
| [`uairt_backends_tour.ipynb`](uairt_backends_tour.ipynb) | Builds UAIRT and runs a tiny model through each backend it can: `reference`, `onnxruntime`, `ncnn`, `executorch`, `whispercpp`, `llamacpp`, `coreml`, `openvino`, `tflite`, `qnn`, `tensorrt`. A section whose runtime is missing skips itself with the reason, and a summary table closes the notebook. | Your machine (Jupyter, VS Code) or Colab |
| [`uairt_tensorrt_colab.ipynb`](uairt_tensorrt_colab.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vnk8071/unified-ai-runtime/blob/main/examples/notebooks/uairt_tensorrt_colab.ipynb) | On an NVIDIA GPU: installs TensorRT 10.9, builds UAIRT with its TensorRT plugin, exports ResNet-18, builds FP32 and FP16 engines, runs them through UAIRT, compares with PyTorch and times them. Sections 9 and 10 add the `onnxruntime` backend with its CUDA and TensorRT execution providers on the same model (the CUDA provider ran on a Colab T4; the TensorRT provider is not yet verified there). | Google Colab with a GPU runtime |
| [`uairt_macos.ipynb`](uairt_macos.ipynb) | The macOS counterpart: builds UAIRT once with every backend that uses a Mac's CPU, GPU (Metal) or Neural Engine (`coreml`, `onnxruntime` with its CoreML provider, `mnn` with Metal, `ncnn`, `openvino`, llama.cpp and whisper.cpp on Metal), runs the repo's `ctest` including the comparisons with the vendor tools, runs ResNet-18 through every backend with `scripts/bench_model.py`, proves the Metal runs offloaded to the GPU, and ends with a pass/fail/skip table. | A Mac (Apple silicon tested) |

## Running the tour on your machine

```bash
pip install jupyter                                # the notebook installs the rest from pip itself (INSTALL_FROM_PIP = False to opt out)
# what you install yourself is found through these variables (see the table in the notebook); all optional:
export EXECUTORCH_ROOT=... UAIRT_WHISPER_MODEL=.../ggml-tiny.en.bin
jupyter lab examples/notebooks/uairt_backends_tour.ipynb
```

`ncnn`, `whispercpp` and `llamacpp` need nothing installed (they are pinned submodules the notebook fetches), but the last two need a model file. Put
a ggml Whisper model (`ggml-*.bin`) and a small GGUF (`*.gguf`) in a `models/` folder **next to the notebook**: the setup cell finds them by a path relative to
the notebook (`WHISPER_MODEL`, `LLAMACPP_MODEL`; an absolute path or the `UAIRT_WHISPER_MODEL` / `UAIRT_LLAMACPP_MODEL` variables work too). Git ignores
`*.bin` and `*.gguf`, so a model there is never committed. Instead of finding the files yourself, set `DOWNLOAD_WHISPER_MODEL = True` and/or `DOWNLOAD_LLAMACPP_MODEL = True` in the setup cell: it fetches the
78 MB Whisper `tiny.en` file and the 145 MB SmolLM2-135M-Instruct Q8 GGUF into `models/` when none is there, and checks each against its published hash (both off by default).
A cell near the top installs the backends that have a wheel: `numpy`, `onnxruntime`, `openvino`, and `coremltools` on macOS. It uses `pip`, or `uv pip`
when the environment has no pip, and installs only what is missing. The notebook fetches ONNX Runtime's header itself. The first build compiles NCNN and
whisper.cpp and takes several minutes. Run from outside a checkout, the notebook clones the repository itself.
Vendor SDKs (QAIRT, TensorRT, Xcode) are never fetched for you except in the Colab notebook, which says so and pins the version.

## Status

- The tour was executed end to end on an Apple M5 (macOS), including from outside a checkout, and in a fresh Python 3.12.13 venv with no pip and no packages (it installed numpy,
  onnxruntime, openvino and coremltools itself, then `reference`, `onnxruntime`, `ncnn`, `whispercpp`, `llamacpp`, `coreml` and `openvino` ran, the two
  model sections from files in a `models/` folder next to the notebook, which the notebook itself downloaded and hash-checked with both download flags on). It has not been run on Linux or on Colab. Sections for hardware that machine does not have
  (`qnn`, `tflite`, `tensorrt`, `llamacpp` without a model) are written from the documented procedures and have **not** been run.
- Sections 1 to 8 of the Colab notebook ran to completion on a Colab Tesla T4 (Python 3.13, CUDA 13.0 driver, TensorRT 10.9.0.34, UAIRT 0.2.0): the plugin built, and for ResNet-18 the FP32
  engine matched PyTorch to 1.5e-6 and the FP16 engine to 2.7e-3, with the same top-1 class. Median time per run: UAIRT 1.61 ms (1.43 ms with pinned buffers) for FP32,
  0.74 ms (0.59 ms) for FP16, PyTorch eager 3.11 ms. One GPU, one run, a random-weight model: a sanity check, not a benchmark.
- Sections 9 and 10 of the Colab notebook ran on a Colab Tesla T4 (ONNX Runtime 1.31.0 from `onnxruntime-gpu`): the plugin built, and the CUDA provider matched PyTorch to 1.7e-6
  with every node on the GPU in 3.2 ms (the CPU provider took 34 ms). The TensorRT provider rows failed in that run because the cell did not create the engine cache
  directory; the cell now does, and its output is cleared until it is run again, so the TensorRT provider on Colab is **not yet verified**.
- `uairt_macos.ipynb` ran end to end on an Apple M5 (macOS 26.6, Python 3.12.15, Xcode 27, CMake 4.4): 47 checks passed, none failed, 2 skipped (ExecuTorch, which needs your own install, and the `references_scripts` test, which trips on untracked local files under `docs/`); 23 of its 24 `ctest` tests passed (the 24th is the skipped `references_scripts`) and the ONNX Runtime test passed again through the CoreML provider, and ResNet-18 ran through 15 backend variants (CoreML, ONNX Runtime with its CoreML provider, MNN with Metal, NCNN, OpenVINO). Its outputs are saved, with machine paths replaced. It has not been run on an Intel Mac.
- The Colab TensorRT notebook and the tour are committed **with** their saved outputs, so you can read the results without running anything. The tour's outputs come from one run on an
  Apple M5 (macOS, Python 3.12.13, the `sam3` virtual environment), with the Whisper and llama models taken from files outside the repository, and they contain
  that machine's local paths. The Colab notebook's outputs come from the Colab T4 run above. Rerunning a notebook replaces them; clear the
  outputs before committing if you do not want your own paths in the repository.
