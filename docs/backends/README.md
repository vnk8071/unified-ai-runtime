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
| TensorRT | [tensorrt.md](tensorrt.md) | Linux with an NVIDIA GPU (verified on a GTX 1650) |
| reference | [reference.md](reference.md) | any, test only |

Run `scripts/doctor.sh` first: it reports which backends can be built on the current machine.
