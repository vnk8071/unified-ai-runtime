# Licensing policy

UAIRT is Apache-2.0. It wraps vendor runtimes that are not open source.

| Backend | Header | Library | In this repo? |
|---|---|---|---|
| ONNX Runtime | MIT | MIT | Not vendored; user supplies it |
| QNN (QAIRT) | Qualcomm proprietary | Qualcomm proprietary | Never |
| CoreML | Apple proprietary | Apple system framework | Never |
| TensorRT | Apache-2.0 (NVIDIA/TensorRT repo) | NVIDIA proprietary | Never |

## Rules

1. The repository contains only our own code. Backends are built against an SDK
   directory the user provides (`QNN_SDK_ROOT`, `ONNXRUNTIME_ROOT`,
   `TENSORRT_ROOT`, Xcode) and load vendor libraries at runtime.
2. Do not copy vendor headers or struct definitions, even partially.
3. Do not publish vendor SDK files in releases, Docker images or CI artifacts.
4. Do not reverse engineer vendor binaries or artifact formats.
5. Do not use TensorRT in the same binary as copyleft-licensed code.
6. Document in each backend that the vendor SDK and its license are required.

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
