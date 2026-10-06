# Choose and export a model for a device

Goal: the user has a model (a PyTorch checkpoint, a Hugging Face model, an ONNX file) and a device, and the model is
not yet in a format a UAIRT backend loads. You pick the format, say who runs which conversion tool, and check the
result before it reaches UAIRT. The rules in `AGENTS.md` apply. Running the model is [run-model.md](run-model.md).

The commands below come from the tools' own documentation. Unless a line says "run here", they have not been run in this
repository: read the tool's `--help` for the installed version first, and report what you actually ran.

## 1. Ask first

- What is the model, where does it come from, and under what licence? A gated model (for example Llama) needs the user to
  accept its licence on the publisher's page. Never accept it for them and never use their token without being asked.
- Which device: CPU, GPU (which vendor), or NPU (which chip)? `scripts/doctor.sh` tells you what this machine has.
- Is there a ready-made file in the target format from the model's publisher (a `.gguf`, a `.dlc`, an OpenVINO IR)? Use it
  before converting; say where it came from.
- What accuracy or speed does the user need? A quantized model is faster and smaller and differs from the original.

## 2. Pick the format from the target

| Target | Format | Backend | Where the conversion runs |
|---|---|---|---|
| Any CPU, portable | `.onnx` | `onnxruntime` | any machine with Python |
| Language model, CPU or GPU | `.gguf` | `llamacpp` | any machine with Python and the llama.cpp source |
| Snapdragon NPU (HTP) | `.dlc` or a context `.bin` | `qnn` | `qairt-converter` runs on x86_64 Linux only; a `.bin` is tied to the chip and QAIRT version |
| Intel CPU, GPU, NPU | OpenVINO IR (`.xml` + `.bin`) | `openvino` | any machine with the `openvino` Python package |
| NVIDIA GPU | `.engine` or `.plan` | `tensorrt` | the GPU that will run it (an engine is not portable) |
| Vulkan GPU, small CNNs | NCNN `.param` + `.bin` | `ncnn` | any machine with `pnnx` |
| Apple CPU, GPU, Neural Engine | `.mlpackage` | `coreml` | macOS with `coremltools` |
| Mobile CPU, QNN delegate | `.tflite` | `tflite` | a machine with the TFLite or `ai-edge-torch` converter |

Use a throwaway virtual environment for conversion packages; do not install into the user's own environments without
asking. On Windows ARM64 there is no `torch` wheel, so export from an x64 Python (run-model.md).

## 3. Recipes

### PyTorch to ONNX (the common first step)

```python
import torch
model = Model().eval()
example = torch.zeros(1, 3, 640, 640)                      # the shape and dtype you will really feed
torch.onnx.export(model, example, "model.onnx", opset_version=17,
                  input_names=["images"], output_names=["output"])
```

- Fix the input shape unless the user needs a variable one; add `dynamic_axes` only for axes that really change. NPU and
  GPU compilers are slower and less reliable with dynamic shapes.
- Check the file: `onnx.checker.check_model(onnx.load("model.onnx"))`, then run the original and the ONNX model on the same
  input with Python ONNX Runtime and compare (see section 4). `tests/data/make_models.py` does this for the test models.
- Pre- and post-processing (resize, normalize, NMS) usually stay outside the model, in the application: see
  [write-app.md](write-app.md).

### Hugging Face language model to GGUF

1. Prefer a GGUF the publisher or a trusted quantizer already made. Pick the quantization for the device: `Q8_0` and
   `Q4_0` are the safe choices for the Hexagon NPU path (GenieX's notes say `Q4_K_M` leaves some tensors on the CPU);
   `Q8_0` has been run on this repository's CPU and GPU path (Qwen3-0.6B).
2. To convert yourself, use the converter in the pinned source, `third_party/llama.cpp/convert_hf_to_gguf.py`, in a
   virtual environment with that tree's `requirements`, then `llama-quantize` from the same tree
   (`llama-quantize model-f16.gguf model-q8_0.gguf Q8_0`). Building `llama-quantize` needs llama.cpp's tools: configure
   with `UAIRT_LLAMACPP_TEST_MODEL` set, as `docs/backends/llamacpp.md` describes. Not yet run in this repository.
3. Check it loads and generates sensibly through `examples/llm_generate.c` and compares with llama.cpp's own tool.

### Snapdragon NPU (QAIRT)

- The user (or a Linux x86_64 machine) runs `qairt-converter` on the ONNX file to make a `.dlc`; for the NPU the model usually
  needs quantizing with `qairt-quantizer` and calibration inputs that look like real data. The option names change
  between QAIRT versions: use the SDK's documentation and `--help`.
- A `.dlc` is portable and is compiled on the device at load (use `cache_dir`, see `docs/backends/qnn.md`). A context
  `.bin` is built for one chip and one QAIRT version.
- If the only machine is Windows ARM64, say this conversion needs x86_64 Linux and ask who does it. Do not try
  emulation tricks.
- Cloud compilation (Qualcomm AI Hub) needs the user's own account and token: ask, do not assume.

### Other targets

- **OpenVINO:** `ovc model.onnx` writes the IR. Check with `openvino` Python on the same input.
- **TensorRT:** `trtexec --onnx=model.onnx --saveEngine=model.engine` on the GPU that runs it, with the TensorRT version
  UAIRT is built against (`scripts/doctor.sh`).
- **NCNN:** `pnnx model.pt inputshape=[1,3,640,640]` writes `.param` and `.bin`; UAIRT needs the `input_shapes` option.
- **CoreML:** `coremltools.convert(...)` then save as `.mlpackage`; compute units are an engine option.

## 4. Check the export before UAIRT

Run the original and the converted model on the same input, outside UAIRT, and compare:

- float models: maximum absolute difference around `1e-4` to `1e-3`;
- quantized models: compare the task result (class, boxes, tokens), not raw values. If the answer changed, say so.

A conversion that fails or changes the answer is the conversion's problem, not a reason to edit UAIRT. Then run it through
UAIRT and compare with the vendor's tool as in run-model.md section 6.

## Stop and ask when

- a licence or token is needed, or a large download (state the size);
- the conversion needs a machine this one is not (QAIRT converter, a TensorRT GPU, macOS);
- the user's accuracy target is not met after quantizing;
- the model has operators the target compiler rejects: report which, do not rewrite the model silently.
