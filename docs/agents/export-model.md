# Choose and export a model for a device

Goal: the user has a model (a PyTorch module or checkpoint, an ONNX file, a Hugging Face model) and a device, and the model is not
yet in a format a UAIRT backend loads. You pick the route, say who runs which conversion tool, convert, and **check the result
through UAIRT against the original** before declaring it done. The rules in `AGENTS.md` apply. Running the model afterwards is
[run-model.md](run-model.md).

Each route below is marked. **Run here** means it was run in this repository on the machine named, with the outputs compared with the
original; **not run here** means the command comes from the tool's documentation, so read its `--help` for the installed
version first and report what you actually ran.

## 1. Ask first

- What is the model, where does it come from, and under what licence? A gated model (for example Llama) needs the user to accept
  its licence on the publisher's page. Never accept it for them and never use their token without being asked.
- Which device: CPU, GPU (which vendor), or NPU (which chip)? `scripts/doctor.sh` tells you what this machine has.
- Is there a ready-made file in the target format from the model's publisher (a `.gguf`, a `.dlc`, an OpenVINO IR, a `.mlpackage`)?
  Use it before converting; say where it came from.
- What accuracy or speed does the user need? A quantized or half-precision model is faster and smaller and differs from the original.
- What is the input shape you will really feed? Export with that fixed shape (section 4 explains why).

## 2. Pick the route

Find the target backend in the first column, then the row for the source you have.

| Target backend | From a PyTorch module | From an ONNX file | State |
|---|---|---|---|
| `onnxruntime` | `torch.onnx.export` (section 3) | already ONNX | **Run here** (macOS arm64) |
| `coreml` | `coremltools.convert` of a `torch.jit.trace` | none: coremltools does not read ONNX, so go back to the PyTorch model | **Run here** |
| `executorch` | `torch.export` then `to_edge_transform_and_lower` (XNNPACK) | none | **Run here** |
| `openvino` | export ONNX first | `openvino.convert_model` (or `ovc`) | **Run here**, from both |
| `ncnn` | `pnnx` on the module | `pnnx model.onnx inputshape=[1,3,224,224] fp16=0` | **Run here**, from both |
| `paddlelite` | save a Paddle model (`paddle.jit.save`, legacy format), then `opt` | ONNX to Paddle with `x2paddle`, then `opt` | `opt` from a Paddle model **run here** (PaddlePaddle 3.3.1, Paddle Lite v2.14-rc); `x2paddle` not run |
| `mnn` | export ONNX first | `mnnconvert -f ONNX --modelFile m.onnx --MNNModel m.mnn` | **Run here**, from both |
| `tensorrt` | export ONNX first | `trtexec --onnx` on the GPU that runs it | not run here (needs an NVIDIA GPU) |
| `qnn` (Snapdragon NPU) | export ONNX first | `qairt-converter` then `qairt-quantizer`, on x86_64 Linux | conversion not run here; loading a `.dlc` and a context `.bin` is, see below |
| `tflite` | `ai-edge-torch`, or ONNX first | `onnx2tf` | not run here |
| `llamacpp` (language models) | Hugging Face checkpoint: `convert_hf_to_gguf.py` | none | not run here (loading a GGUF is) |
| `whispercpp` | `models/convert-pt-to-ggml.py` of whisper.cpp, or its download script | none | not run here (loading a model is) |

An ONNX Runtime `.ort` file (the mobile format) should load through the same backend because ONNX Runtime reads it; that has not been tried.

Conversion packages go in a **throw-away virtual environment**; do not install into the user's own environments without asking.
On Windows ARM64 there is no `torch` wheel, so export from an x64 Python (run-model.md).

## 3. The tool: `scripts/export_model.py`

It exports a PyTorch module or an ONNX file to the targets marked **Run here** and checks each result through UAIRT:

```bash
# a torchvision model with random weights (add --state-dict weights.pt for trained ones), five targets, checked one by one
scripts/export_model.py torchvision:resnet18 --shape 1,3,224,224 --to onnx coreml executorch openvino ncnn --out exported

# your own module: a function that returns an nn.Module
scripts/export_model.py my_model.py:build --shape 1,3,640,640 --to onnx --state-dict weights.pt

# an ONNX file (the shape is read from the file, which must have fixed input shapes)
scripts/export_model.py model.onnx --to openvino ncnn mnn
```

What it does for each target: writes `<out>/<name>.<ext>`, runs the reference (PyTorch, or ONNX Runtime for an ONNX source) and the
converted file **through UAIRT's Python binding** on the same random input, and compares the **raw outputs**. A top-1 match is a weak
check. It fails the target on a difference above `--atol` plus 0.2% of the output's magnitude, on NaN, and on a constant output.
The exit status is 0 only if every target passed.

What it needs, none of which it installs: the converter packages it names when one is missing (`torch`, `onnx`, `coremltools`,
`executorch`, `openvino`, `pnnx`, `MNN`, `onnxruntime`), and UAIRT built shared with the backend plugins (`UAIRT_LIBRARY`,
`UAIRT_PLUGIN_PATH`, see `bindings/python/README.md`). Each target runs in its **own process**: ONNX Runtime, OpenVINO and PyTorch
together deadlocked once (OpenVINO blocked in an abseil mutex after an ONNX Runtime run), and a crashing converter then costs only
its own target. Pass `--option key=value` for a backend option in the check and `--no-verify` to export only.

Run here, on macOS arm64 (torch 2.14, coremltools 9.0, executorch 1.5.1, openvino 2026.4, pnnx 20260526; ONNX Runtime 1.24 backend):
ResNet-18 with random weights, input 1x3x224x224, largest difference from PyTorch on the 1000 logits (magnitude 5.9):

| Target | Max difference | Note |
|---|---|---|
| onnx (onnxruntime backend) | 5e-6 | |
| coreml | 2e-6 | compute precision float32 |
| executorch (XNNPACK) | 5e-6 | |
| openvino | 4e-6 | needs `inference_precision=f32`: its default on an Arm CPU is half precision and gave 0.048 |
| ncnn | 3e-6 | needs `fp16=False` in `pnnx` (weights) and `fp16=false` in the backend: with the defaults 0.02 |
| mnn | 1e-4 | `mnnconvert` fuses and optimises layers; the check runs MNN at `precision=high` |
| an ONNX file to openvino, and to ncnn | 4e-6 and 5e-6 | ONNX Runtime as the reference |

The script already sets those two precision choices for its check. When you export for speed, choose half precision on purpose and
widen `--atol` knowingly; say in the report that you did.

## 4. Per-target notes (learned the hard way)

**ONNX.** Use `torch.onnx.export(..., dynamo=False)` (the TorchScript exporter: no extra package, simple graphs), opset 17, and
`onnx.checker.check_model`. Fix the input shape unless a variable one is needed; add `dynamic_axes` only for axes that really change.
NPU and GPU compilers are slower and less reliable with dynamic shapes, and several backends here refuse them. Pre- and
post-processing (resize, normalize, NMS) usually stays outside the model: [write-app.md](write-app.md),
[pre-post-processing.md](pre-post-processing.md).

**CoreML.** `coremltools` warns when torch is newer than the version it was tested with (9.0 with torch 2.7) and still worked with 2.14
here; if it fails, try the tested torch. Outputs come back sorted by name, so match by shape. A detector exported by Ultralytics with
`nms=True` is a pipeline with scalar inputs and variable-length outputs that UAIRT refuses: extract the network stage and do the NMS
yourself ([backends/coreml.md](../backends/coreml.md#models-with-a-built-in-nms-stage)). Image inputs reach UAIRT as `uint8` BGRA bytes.

**ExecuTorch.** `.pte` files are tied to the delegates named at export; the delegate must be built into the `executorch` plugin's
runtime (XNNPACK is). `--delegate none` exports for the portable CPU kernels.

**OpenVINO.** `ovc model.onnx` writes the IR (`openvino.convert_model` then `save_model(..., compress_to_fp16=False)` in Python).
Without the `inference_precision` option OpenVINO runs the model in the device's default precision, half on an Arm CPU.

**NCNN.** `pnnx` stores weights as float16 unless told otherwise (the `.bin` is half the size and the logits move by 0.02). NCNN's
`input_shapes` option has no batch dimension (`3x224x224`), and its outputs have none either. The `pnnx` Python wrapper ran a
self-check on a generated file that failed to parse after the NCNN files were written; the script accepts that only when the files
exist, and the check through UAIRT is what counts.

**MNN.** `mnnconvert` ships in the `MNN` pip package and replaced packages (`protobuf`) the first time it ran, so use the throw-away environment. MNN's inputs and outputs are numbered by name, not by the model's order: match by name, or let the script match by shape. A converted ResNet-18 differs from PyTorch by 1e-4 on logits of size 6, five times the others' spread but harmless.

**TensorRT.** `trtexec --onnx=model.onnx --saveEngine=model.engine` on the GPU that will run it, with the TensorRT version UAIRT is built
against (`scripts/doctor.sh`). An engine is not portable across GPUs or TensorRT versions.

**Snapdragon NPU (QAIRT).**
- The user (or a Linux x86_64 machine) runs `qairt-converter` on the ONNX file to make a `.dlc`; for the NPU the model usually needs
  quantizing with `qairt-quantizer` and calibration inputs that look like real data. Option names change between QAIRT versions: use the
  SDK's documentation and `--help`.
- A `.dlc` is portable and is compiled on the device at load (use `cache_dir`, see [backends/qnn.md](../backends/qnn.md)). A context `.bin`
  is built for one chip and one QAIRT version, yet a float context binary built for a QCS8550 loaded and ran on an SM8850 board here, so
  try before ruling it out. A `.dlc` exported with a newer QAIRT than the installed SDK also loaded.
- **Checking a quantized model against `qnn-net-run`:** give it `--use_native_input_files` and `--use_native_output_files`. Without them it
  treats a raw `uint8` file as float32, quantizes it again, and its outputs differ from UAIRT's for that reason alone.
- If the only machine is Windows ARM64, say this conversion needs x86_64 Linux and ask who does it. Do not try emulation tricks.
- Cloud compilation (Qualcomm AI Hub) needs the user's own account and token: ask, do not assume.

**Hugging Face language model to GGUF.** Prefer a GGUF the publisher or a trusted quantizer already made. Pick the quantization for the
device: `Q8_0` and `Q4_0` are the safe choices for the Hexagon NPU path (GenieX's notes say `Q4_K_M` leaves some tensors on the CPU);
`Q8_0` has been run on this repository's CPU and GPU path. To convert yourself, use `third_party/llama.cpp/convert_hf_to_gguf.py` in a
virtual environment with that tree's requirements, then `llama-quantize model-f16.gguf model-q8_0.gguf Q8_0` from the same tree (building
it needs llama.cpp's tools: configure with `UAIRT_LLAMACPP_TEST_MODEL` set, as [backends/llamacpp.md](../backends/llamacpp.md) describes).
Then check that it generates sensibly and compare with llama.cpp's own tool (`ctest -R llamacpp_vs_llama_completion`).

## 5. Check the export before UAIRT

If you used the script, the first check is done. Otherwise run the original and the converted model on the same input, outside UAIRT, and
compare:

- float models: maximum absolute difference around `1e-4` to `1e-3` of the output's size; much larger means a conversion problem or a
  half-precision default (see the table above);
- quantized models: compare the task result (class, boxes, tokens), not raw values. If the answer changed, say so.

A conversion that fails or changes the answer is the conversion's problem, not a reason to edit UAIRT. Then run it through UAIRT on the real
device and compare with the vendor's tool as in run-model.md section 6. If a backend option does not seem to do anything, test it: the
OpenVINO backend once passed its engine options in a way that made any single one fail and ignored all but the first of several, and
nothing had noticed.

## Stop and ask when

- a licence or token is needed, or a large download (state the size);
- the conversion needs a machine this one is not (QAIRT converter, a TensorRT GPU, macOS);
- the user's accuracy target is not met after quantizing;
- the model has operators the target compiler rejects: report which, do not rewrite the model silently;
- the check fails and you cannot say whether the converter, the precision, or UAIRT is responsible: show the numbers.
