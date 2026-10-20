---
name: use-onnxruntime-providers
description: >-
  Run a model on a GPU, NPU or other accelerator through ONNX Runtime's execution providers (CoreML, CUDA, TensorRT, QNN, OpenVINO,
  XNNPACK, DirectML, plugin providers) using the UAIRT onnxruntime backend. Use when the user has an .onnx file and wants a specific
  provider, asks why a provider is not used, or reports an ONNX Runtime run that is not faster on the accelerator.
---

# ONNX Runtime execution providers through UAIRT

Read `docs/backends/onnxruntime.md`. Options of the `onnxruntime` backend:

- `execution_provider=cpu|cuda|tensorrt|<ONNX Runtime provider name>`: a name goes to ONNX Runtime by name (case-sensitive: `CoreML`, `QNN`,
  `OpenVINO`, `XNNPACK`, `WebGPU`, `DML`) with `ep_option.<key>` as its options. The provider must be in the ONNX Runtime build that is
  loaded (`UAIRT_ONNXRUNTIME_LIBRARY`). A missing one is `BACKEND_UNAVAILABLE`.
- `ep_library` + `ep_name`: a **plugin** provider library (QNN 2.x, MLX); needs an ONNX Runtime core of a matching version and a hardware
  device ONNX Runtime can see.
- `cpu_fallback=0`: fail the load if any node would run on ONNX Runtime's CPU provider.
- There is no Metal provider: on Apple hardware use `CoreML` (`ep_option.MLComputeUnits=CPUAndGPU|CPUAndNeuralEngine|ALL`).

**Never trust "it ran".** A provider that is present but cannot start still lets ONNX Runtime run everything on its CPU provider (a QNN
provider without its FastRPC library did exactly that, at CPU speed). Run once with `cpu_fallback=0`, or with `log_level=verbose` and look
for `All nodes placed on [<Provider>]`, and compare latency with `execution_provider=cpu`.

GPU and Neural Engine results are usually half precision: compare with the CPU output and report the difference as reduced precision, not exact.
For vendor NPUs prefer the native backend (`qnn`, `openvino`, `tensorrt`, `coreml`) unless the user already ships ONNX Runtime; say which
you used and why. `docs/backends/onnxruntime.md` lists what was run (CoreML, CUDA) and what was not (QNN through ONNX Runtime did not work on the QCS6490 board that was tried); say which applies.
