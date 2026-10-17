# Licensing policy

UAIRT is Apache-2.0. It wraps vendor runtimes that are not open source.

| Backend | Header | Library | In this repo? |
|---|---|---|---|
| ONNX Runtime | MIT | MIT | Not vendored; user supplies it |
| QNN (QAIRT) | Qualcomm proprietary | Qualcomm proprietary | Never |
| CoreML | Apple proprietary | Apple system framework | Never |
| TensorRT | Apache-2.0 (NVIDIA/TensorRT repo) | NVIDIA proprietary | Never |
| llama.cpp | MIT | MIT | As a pinned git submodule at `third_party/llama.cpp` (only the pin is in the repo) |
| whisper.cpp | MIT | MIT | As a pinned git submodule at `third_party/whisper.cpp` |
| NCNN | BSD-3-Clause, plus bundled zlib, BSD-2 and BSD-3 components | same | As a pinned git submodule at `third_party/ncnn` |

## Rules

1. The repository contains only our own code. Backends are built against an SDK
   directory the user provides (`QNN_SDK_ROOT`, `ONNXRUNTIME_ROOT`,
   `TENSORRT_ROOT`, Xcode) and load vendor libraries at runtime.
2. Do not copy vendor headers or struct definitions, even partially.
3. Do not publish vendor SDK files in releases, Docker images or CI artifacts.
4. Do not reverse engineer vendor binaries or artifact formats.
5. Do not use TensorRT in the same binary as copyleft-licensed code.
6. Document in each backend that the vendor SDK and its license are required.
7. Exception to rule 1: llama.cpp, whisper.cpp (both MIT) and NCNN (BSD-3-Clause) are open source, not vendor SDKs, and are
   pinned submodules under `third_party/`, built only when `UAIRT_BUILD_LLAMACPP`, `UAIRT_BUILD_WHISPERCPP` or `UAIRT_BUILD_NCNN`
   is `ON`. The repository holds the submodule pins, not their files, and releases that include a plugin must ship
   `THIRD_PARTY_NOTICES.md`. A user-provided install (`NCNN_ROOT`, `WHISPER_ROOT`) overrides the submodule. Nothing else
   becomes a submodule without a licence check and an entry here.

## Notes from the license texts reviewed (not legal advice)

- QAIRT SDK 2.46 (AI Stack License): redistribution only in object code, only as
  incorporated in your own application, never standalone; no reverse engineering;
  no patent license; revocable and terminable by Qualcomm without cause; broad
  indemnification; prohibited and high-risk use cases listed in section 2.
- Xcode and Apple SDKs Agreement: no redistribution of Apple SDKs or headers;
  iOS distribution to third parties and iOS libraries fall under the Apple
  Developer Program License Agreement, which was not reviewed. Ship source, not
  prebuilt iOS binaries, until it has been.
- NVIDIA TensorRT: only files NVIDIA marks distributable (runtime libraries) may be
  redistributed, and only inside an application with material added functionality;
  no use that subjects the SDK to an open source license; no safety-critical use.
  The EULA was read only through a summary.

`scripts/check_no_vendor_files.sh` runs under ctest to catch accidental commits.
