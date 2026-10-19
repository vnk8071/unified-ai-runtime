---
name: run-model-on-device
description: >-
  Build UAIRT and run an AI model on a target device (CPU, GPU or NPU), then verify and report the result.
  Use when the user asks to run, benchmark or test a model (.dlc, .bin, .onnx, .tflite, .mlmodel, .mlpackage, .gguf,
  .pte, .engine, a Whisper ggml model, or a PyTorch model) on this machine or a board, to set up a backend (QNN/QAIRT,
  ONNX Runtime, TFLite, CoreML, OpenVINO, NCNN, MNN, Paddle Lite, TensorRT, llama.cpp, whisper.cpp, ExecuTorch, or a language model
  server), to use the NPU or GPU, or when a build, load or NPU run fails.
---

# Run a model on the target device

1. Read `AGENTS.md` (the rules: ABI, no vendor files, never download SDKs or accept licences, no `sudo`).
2. Run `scripts/doctor.sh` and read every line; it reports the device, the NPU, the toolchain and which backends
   are ready.
3. Follow `docs/agents/run-model.md`: pick the backend from the model format, get the SDK in place with
   `docs/agents/setup.md`, build, run with `run_model`, verify against the vendor tool, report.
4. On any error, look it up in `docs/agents/troubleshooting.md` before trying anything else.
5. If the model is not in a loadable format, follow `docs/agents/export-model.md` (`scripts/export_model.py` exports a PyTorch
   module or an ONNX file to ONNX Runtime, CoreML, ExecuTorch, OpenVINO or NCNN and checks it through UAIRT); if the user wants a program, follow
   `docs/agents/write-app.md`. To prepare inputs or decode outputs (images, audio, tokenizers, chat templates, boxes), start from
   `python examples/python/inspect_model.py <model>` and follow `docs/agents/pre-post-processing.md`. Installed versus tested vendor
   versions: `docs/vendors.md`.

Done means `ctest` passes, `doctor.sh` reports the backend as `ok`, and a run on the requested device is shown with
its real output. Say what you did not verify. Stop and ask when an SDK, a model conversion, a licence, `sudo` or
access to another machine is needed.
