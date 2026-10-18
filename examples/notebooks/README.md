<!-- SPDX-License-Identifier: Apache-2.0 -->
# Notebooks

| Notebook | What it does | Where it runs |
|---|---|---|
| [`uairt_backends_tour.ipynb`](uairt_backends_tour.ipynb) | Builds UAIRT and runs a tiny model through each backend it can: `reference`, `onnxruntime`, `ncnn`, `executorch`, `whispercpp`, `llamacpp`, `coreml`, `openvino`, `tflite`, `qnn`, `tensorrt`. A section whose runtime is missing skips itself with the reason, and a summary table closes the notebook. | Your machine (Jupyter, VS Code) or Colab |
| [`uairt_tensorrt_colab.ipynb`](uairt_tensorrt_colab.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vnk8071/unified-ai-runtime/blob/main/examples/notebooks/uairt_tensorrt_colab.ipynb) | On an NVIDIA GPU: installs TensorRT 10.9, builds UAIRT with its TensorRT plugin, exports ResNet-18, builds FP32 and FP16 engines, runs them through UAIRT, compares with PyTorch and times them. | Google Colab with a GPU runtime |

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
`*.bin` and `*.gguf`, so a model there is never committed. `DOWNLOAD_WHISPER_MODEL = True` fetches the 78 MB Whisper `tiny.en` file into `models/`.
A cell near the top installs the backends that have a wheel: `numpy`, `onnxruntime`, `openvino`, and `coremltools` on macOS. It uses `pip`, or `uv pip`
when the environment has no pip, and installs only what is missing. The notebook fetches ONNX Runtime's header itself. The first build compiles NCNN and
whisper.cpp and takes several minutes. Run from outside a checkout, the notebook clones the repository itself.
Vendor SDKs (QAIRT, TensorRT, Xcode) are never fetched for you except in the Colab notebook, which says so and pins the version.

## Status

- The tour was executed end to end on an Apple M5 (macOS), including from outside a checkout, and in a fresh Python 3.12.13 venv with no pip and no packages (it installed numpy,
  onnxruntime, openvino and coremltools itself, then `reference`, `onnxruntime`, `ncnn`, `whispercpp`, `llamacpp`, `coreml` and `openvino` ran, the two
  model sections from files in a `models/` folder next to the notebook). It has not been run on Linux or on Colab. Sections for hardware that machine does not have
  (`qnn`, `tflite`, `tensorrt`, `llamacpp` without a model) are written from the documented procedures and have **not** been run.
- The Colab notebook was run on a Colab T4 (Python 3.13, CUDA 13.0 toolkit) up to the model export: TensorRT 10.9.0.34 installed, its headers were
  fetched and **the UAIRT TensorRT plugin built**. The export then failed because Colab has no `onnx` package; that is fixed, and the engine build,
  the UAIRT run and the comparison with PyTorch have **not run yet**. Please report what the next run shows.
- The notebooks are committed without outputs.
