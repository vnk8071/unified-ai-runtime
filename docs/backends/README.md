<!-- SPDX-License-Identifier: Apache-2.0 -->
# Backend setup

One page per backend. Each lists what you install yourself, the build flags, the engine options
and how to check the result. Vendor SDKs are never part of this repository
(see [../LICENSING.md](../LICENSING.md)).

| Backend | Page | Platforms |
|---|---|---|
| ONNX Runtime | [onnxruntime.md](onnxruntime.md) | Linux, macOS, Windows |
| TFLite | [tflite.md](tflite.md) | Linux (verified on Qualcomm boards) |
| CoreML | [coreml.md](coreml.md) | macOS, Apple platforms |
| QNN | [qnn.md](qnn.md) | Linux, Windows (no macOS libraries) |
| OpenVINO | [openvino.md](openvino.md) | Linux, Windows (verified on a Linux CPU) |
| NCNN | [ncnn.md](ncnn.md) | Linux (CPU and Vulkan GPU, verified on a GTX 1650) |
| TensorRT | [tensorrt.md](tensorrt.md) | Linux with an NVIDIA GPU (verified on a GTX 1650) |
| llama.cpp | [llamacpp.md](llamacpp.md) | GGUF language models on the CPU, Metal, CUDA, Vulkan, OpenCL (Adreno) and the Hexagon NPU (`HTP0`). Verified on macOS (Metal, CPU), Linux x86_64 with an RTX 3060 (CUDA, Vulkan) and Windows 11 ARM64 on a Snapdragon X Elite (CPU, OpenCL, `HTP0`) |
| reference | [reference.md](reference.md) | any, test only |

Run `scripts/doctor.sh` first: it reports which backends can be built on the current machine.
