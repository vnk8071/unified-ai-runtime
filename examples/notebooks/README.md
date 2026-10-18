<!-- SPDX-License-Identifier: Apache-2.0 -->
# Notebooks

| Notebook | What it does | Where it runs |
|---|---|---|
| [`uairt_backends_tour.ipynb`](uairt_backends_tour.ipynb) | Builds UAIRT and runs a tiny model through each backend it can: `reference`, `onnxruntime`, `ncnn`, `executorch`, `whispercpp`, `llamacpp`, `coreml`, `openvino`, `tflite`, `qnn`, `tensorrt`. A section whose runtime is missing skips itself with the reason, and a summary table closes the notebook. | Your machine (Jupyter, VS Code) or Colab |
| [`uairt_tensorrt_colab.ipynb`](uairt_tensorrt_colab.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vnk8071/unified-ai-runtime/blob/main/examples/notebooks/uairt_tensorrt_colab.ipynb) | On an NVIDIA GPU: installs TensorRT 10.9, builds UAIRT with its TensorRT plugin, exports ResNet-18, builds FP32 and FP16 engines, runs them through UAIRT, compares with PyTorch and times them. | Google Colab with a GPU runtime |

## Running the tour on your machine

```bash
pip install jupyter numpy onnxruntime openvino     # the last two are optional; each one adds a backend
# what you install yourself is found through these variables (see the table in the notebook); all optional:
export EXECUTORCH_ROOT=... UAIRT_WHISPER_MODEL=.../ggml-tiny.en.bin
jupyter lab examples/notebooks/uairt_backends_tour.ipynb
```

`ncnn` and `whispercpp` need nothing installed (they are pinned submodules the notebook fetches); `whispercpp` also needs a model file.
`onnxruntime` and `openvino` are picked up when they are `pip install`ed (the notebook fetches ONNX Runtime's header itself). On Colab run
`!pip install onnxruntime openvino` in a cell above the setup cell; the first build compiles NCNN and whisper.cpp and takes several minutes.
Run from outside a checkout, the notebook clones the repository itself.
Vendor SDKs (QAIRT, TensorRT, Xcode) are never fetched for you except in the Colab notebook, which says so and pins the version.

## Status

- The tour was executed end to end on an Apple M5 (macOS), including from outside a checkout with only `pip install onnxruntime openvino` done;
  see `CHANGELOG.md` for which sections ran there. It has not been run on Linux or on Colab. Sections for hardware that machine does not have
  (`qnn`, `tflite`, `tensorrt`, `llamacpp` without a model) are written from the documented procedures and have **not** been run.
- The Colab notebook was run on a Colab T4 (Python 3.13, CUDA 13.0 toolkit) up to the model export: TensorRT 10.9.0.34 installed, its headers were
  fetched and **the UAIRT TensorRT plugin built**. The export then failed because Colab has no `onnx` package; that is fixed, and the engine build,
  the UAIRT run and the comparison with PyTorch have **not run yet**. Please report what the next run shows.
- The notebooks are committed without outputs.
