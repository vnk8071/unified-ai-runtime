# Write application code that uses UAIRT

Goal: the user wants a program (or a snippet) that runs their model through UAIRT: detect objects, classify an image,
chat with a language model. You write it, build it, run it on the real device and show the output. The rules in
`AGENTS.md` apply. Getting the model into the right format is [export-model.md](export-model.md); running and checking
it on a device is [run-model.md](run-model.md).

## 1. Choose the language

| Language | Use when | Start from |
|---|---|---|
| Python | scripts, demos, pre/post-processing with NumPy | `bindings/python/examples/auto_model.py` (any model), `llm_generate.py` (GGUF) |
| C | a library or tool with no dependencies | `examples/quickstart.c`, `examples/run_model.c`, `examples/llm_generate.c` |
| C++ | RAII and exceptions on the caller's side | `bindings/cpp/run_qnn.cpp` |
| Rust | safe wrapper | `bindings/rust/uairt/examples/run_qnn.rs` |

Do not invent API names. Read `docs/api.md` and the header `include/uairt/uairt.h`, and copy the call order from the
example nearest to the task. Never put C++ types, exceptions or STL across the C ABI.

## 2. Find out what the model expects

Ask the model, do not guess: `build/run_model --info <plugin> <backend> <model>` prints each input's name, dtype,
quantization and dims; in Python, `model.inputs` and `model.outputs`. Inputs must match dtype and shape exactly; UAIRT
never converts silently. A quantized model takes integers with a scale and zero point: apply them in the app.

What the model does not do is yours to write, and it comes from the model's own documentation, not from UAIRT:

- **Vision:** decode the image, resize or letterbox to the input size, convert the layout (HWC to CHW), scale or
  normalize, and for detectors decode and filter boxes (NMS). Check the original pipeline's mean, std and channel order.
- **Language models:** the chat template. UAIRT tokenizes text but does not apply a template: build the prompt the way the
  model's card says, and stop on its end-of-turn token (see below).

## 3. The pattern

Python, any non-language model:

```python
import numpy as np, uairt

with uairt.AutoModel.from_file("model.onnx", device="cpu") as model:   # device: cpu, gpu, npu, cuda, ...
    x = preprocess(image).astype(model.inputs[0].dtype)                # shape and dtype from model.inputs
    (y,) = model.run(x)
    print(postprocess(y))
```

C: load the plugin if the backend is not built in, create an engine, load the model, fill `uairt_tensor` inputs from
`uairt_model_input_info`, run, destroy in the order engine-last. Check every `uairt_status` and print
`uairt_status_string` with `uairt_last_error` as `examples/quickstart.c` does.

Language models are session-based (`docs/backends/llamacpp.md`): tokenize, `session_append`, read `session_logits`, pick a
token, append it, repeat. `AutoModel.generate` does this in Python. Things that bite:

- stop on the model's end-of-turn token: tokenize its text (`model.tokenize("<|eot_id|>", add_special=False)` for Llama 3
  style models) and pass it as `stop_tokens`, or limit `max_tokens`;
- the context is limited (`n_ctx`, default 2048, capped by the model); a generation that reaches it fails mid-stream;
- `n_gpu_layers=0` runs on the CPU; the default `99` offloads to the GPU if the plugin was built with one.

## 4. Platform notes that cost time

- **Plugins:** found through `UAIRT_PLUGIN_PATH`, next to `libuairt`, or loaded by path. The file is `libuairt_backend_<name>`
  with `.so`, `.dylib` or `.dll`; on Windows that is under `build/Debug` or `Release`. A plugin still needs the vendor runtime
  installed; loading reports what is missing.
- **Windows C programs:** some runtimes (OpenVINO, OpenCL) end the process without flushing C stdio, so redirected output is
  lost. Call `fflush(stdout)` before returning, as `examples/run_model.c` and `examples/llm_generate.c` do.
- **Windows ARM64 Python:** use an ARM64 Python for the ARM64 `uairt.dll`; QNN on the NPU needs `ADSP_LIBRARY_PATH` set before
  Python starts (`docs/agents/troubleshooting.md`).
- **Threads:** one model, one thread at a time. Use one model per thread.
- **Licences:** do not copy vendor code or headers into the app or this repository.

## 5. Test what you wrote

1. Build and run it on the requested device with a real input, and show the real output (not a description of it).
2. Run it a second time with a different input and check the output changes in the expected way.
3. Compare with a reference: the original framework on the same input, or the vendor tool (run-model.md section 6).
4. Say what you did not verify: a device the machine does not have, a model you only ran once, a quantized model's accuracy.

If the code is meant to be kept in this repository, put examples under `examples/` or `bindings/<lang>/examples/`, match
the surrounding style, and add a test only if it can run without a vendor SDK.
