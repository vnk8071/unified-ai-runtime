<!-- SPDX-License-Identifier: Apache-2.0 -->
# C API reference

Header: `#include <uairt/uairt.h>`. Plain C11, usable from C++ (`extern "C"`).

## Lifecycle

```
uairt_load_backend_library(plugin)      once per backend plugin (not needed for built-ins)
  uairt_engine_create(name, options)    one engine per backend configuration
    uairt_model_load(source)            one model per artifact
      uairt_buffer_alloc(...)           optional zero-copy buffers
      uairt_model_run(...)              as often as needed
    uairt_model_destroy
    uairt_buffer_free
  uairt_engine_destroy
```

Destroy models before freeing the buffers they ran with, and free buffers before the
engine that allocated them. Destroy every model before its engine.

## Status codes

Every function that can fail returns `uairt_status`. `uairt_status_string()` gives a short
name; `uairt_last_error()` gives a detailed message.

| Code | Meaning |
|---|---|
| `UAIRT_OK` | success |
| `UAIRT_ERR_INVALID_ARGUMENT` | bad argument, shape or buffer size mismatch, unknown option |
| `UAIRT_ERR_NOT_FOUND` | no backend with that name |
| `UAIRT_ERR_UNSUPPORTED` | the backend does not support this model or feature |
| `UAIRT_ERR_BACKEND_UNAVAILABLE` | the vendor runtime could not be loaded |
| `UAIRT_ERR_INCOMPATIBLE_MODEL` | the artifact is invalid or does not match this device or SDK |
| `UAIRT_ERR_OUT_OF_MEMORY` | allocation failed |
| `UAIRT_ERR_RUNTIME` | the backend failed while running |
| `UAIRT_ERR_VERSION_MISMATCH` | ABI or vendor API version mismatch |
| `UAIRT_ERR_IO` | a file could not be opened or read |

`uairt_last_error()` is thread-local and valid until the next UAIRT call on the same
thread. Do not free the returned pointer.

## Types

### `uairt_tensor`

Describes a model input or output (`data == NULL`) and binds a buffer to it.

| Field | Meaning |
|---|---|
| `struct_size` | set by `uairt_tensor_init`; keep it when copying |
| `dtype` | `UAIRT_DTYPE_FLOAT32`, `FLOAT16`, `BFLOAT16`, `INT8`, `UINT8`, `INT16`, `UINT16`, `INT32`, `INT64`, `BOOL` |
| `rank`, `dims[UAIRT_MAX_RANK]` | shape; static shapes only in this version |
| `quant_scale`, `quant_zero_point` | per-tensor quantization of the stored values; `0` when not quantized. Real value = `(q - zero_point) * scale` |
| `domain` | `UAIRT_MEM_HOST`, `UAIRT_MEM_DMABUF` or `UAIRT_MEM_PINNED` |
| `data`, `dmabuf_fd`, `nbytes` | the buffer; `nbytes` must cover the whole tensor |
| `name` | set by `uairt_model_input_info` / `uairt_model_output_info`; owned by the model |

### `uairt_model_source`

`{ struct_size, path, data, size, format }`. Set exactly one of `path` or (`data`,
`size`). `format` is an optional hint; backends also infer it from the file extension.

### `uairt_option`

`{ key, value }`, both strings. Options are validated by the backend; unknown options
return `UAIRT_ERR_INVALID_ARGUMENT`.

## Functions

| Function | Notes |
|---|---|
| `uairt_version_string()` | same as `UAIRT_VERSION_STRING` in the header |
| `uairt_status_string(status)`, `uairt_last_error()` | see above |
| `uairt_dtype_size(dtype)` | bytes per element, `0` for unknown |
| `uairt_tensor_init(t)` | zero a tensor, set `struct_size`, host domain, `dmabuf_fd = -1` |
| `uairt_tensor_num_elements(t)` | `-1` if the shape is invalid |
| `uairt_backend_count()`, `uairt_backend_name(i)` | registered backends |
| `uairt_load_backend_library(path)` | `dlopen` a plugin and register it; plugins stay loaded for the process lifetime. On Windows it uses `LoadLibrary`. |
| `uairt_engine_create(name, options, n, &engine)` | |
| `uairt_model_load(engine, &source, &model)` | |
| `uairt_model_num_inputs/outputs(model)` | |
| `uairt_model_input_info/output_info(model, i, &tensor)` | fills dtype, shape, quantization and name |
| `uairt_buffer_alloc(engine, nbytes, domain, &buffer)` | `DMABUF` and `PINNED` need backend support (`PINNED`: TensorRT), otherwise `UAIRT_ERR_UNSUPPORTED` |
| `uairt_buffer_data/fd/size(buffer)`, `uairt_tensor_use_buffer(&t, buffer)` | bind a buffer to a tensor |
| `uairt_model_run(model, inputs, n_in, outputs, n_out)` | blocking. Counts, dtypes, shapes, buffer sizes and memory domains are checked first |
| `*_destroy`, `uairt_buffer_free` | accept `NULL` |

Outputs are allocated by the caller, sized from `uairt_model_output_info`.

## Threading

The backend registry is protected by a lock, but multi-threaded use has not been
stress-tested. A model must not be run concurrently from several threads; use one model
per thread. Engines, models and buffers are not otherwise synchronised. This is a
conservative contract: individual backends may allow more, but the API does not promise it.

## Ordering of inputs and outputs

Backends report inputs and outputs in the order the model defines them. CoreML has no
such order, so its inputs and outputs are sorted by name.

## Backends and options

| Backend | Name | Models | Options | Memory domains |
|---|---|---|---|---|
| Reference (tests only) | `reference` | none (fixed float32 [1,4] copy) | none | host |
| ONNX Runtime | `onnxruntime` | `.onnx` (path or memory), static shapes | `intra_op_threads`, `execution_provider` (`cpu`, `cuda`, `tensorrt`; the GPU ones need a GPU build), `log_level`, `ep_library` + `ep_name` (a plugin execution provider; CPU stays the fallback), `ep_option.<key>` (passed to the CUDA, TensorRT or plugin provider) | host |
| QNN | `qnn` | `.dlc`, context binary `.bin` (one graph; others via `graph_name`) | `backend_library`*, `system_library`* (or `device` instead: `cpu`, `gpu`, `npu`, with `sdk_root` or `QNN_SDK_ROOT`, `target`, `hexagon_arch`, default `v73`), `graph_name`, `adsp_library_path`, `htp_performance_mode` (`default`, `burst`, `high_performance`, `balanced`, `power_saver`; HTP only), `cache_dir` and `cache_write` (compiled-context cache for `.dlc` on HTP; writes default to on, but off on Windows) | host, DMABUF (HTP) |
| TFLite | `tflite` | `.tflite` (path or memory), static shapes; optional QNN delegate (HTP) | `num_threads`, `delegate` (`none`, `qnn`), and with `qnn`: `delegate_library`*, `backend_library`*, `skel_dir`, `htp_performance_mode` (`default`, `burst`, `high_performance`, `sustained`, `balanced`, `power_saver`), `htp_precision` (`quantized`, `fp16`), `adsp_library_path` | host |
| TensorRT | `tensorrt` | serialized engine `.engine` / `.plan` (ultralytics exports load as is), static shapes | `device` (CUDA device index) | host, pinned |
| CoreML | `coreml` | `.mlmodelc`, `.mlpackage`, `.mlmodel` (path only), multi-array I/O, static shapes | `compute_units`: `all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_ne` | host |

\* required. Backends other than `reference` are separate plugins: load
`libuairt_backend_<name>` with `uairt_load_backend_library` first.

## Tested configurations

These are what the author ran, not a support matrix.

| Backend | Where | SDK / runtime |
|---|---|---|
| reference, core | macOS arm64, Linux aarch64 (Ubuntu 20.04) | Apple clang 21, GCC 9.4 (Ubuntu 20.04) |
| ONNX Runtime | macOS arm64 | ONNX Runtime 1.24.4 |
| ONNX Runtime + MLX execution provider | macOS 26.6 on an Apple M5 | ONNX Runtime 1.29.1, onnxruntime-ep-mlx 0.29.6 |
| QNN CPU, HTP | Qualcomm QCS6490 (Hexagon v68), Ubuntu 20.04 aarch64 | QAIRT 2.46.0.260424 |
| QNN CPU, HTP | Qualcomm QCS8550 (kalama, soc_id 603, Hexagon v73), Qualcomm Linux LE.PRODUCT.2.1, glibc 2.35 | QAIRT 2.46.0.260424 |
| TFLite, CPU kernels | Qualcomm QCS6490, Ubuntu 20.04 aarch64 | TFLite 2.16.1 built on the device |
| CoreML | macOS 26.6 on an Apple M5 | Xcode with the macOS 27.0 SDK, coremltools 9.0 |
| core, reference, C++, Python and Rust bindings | Windows 11 ARM64 (Snapdragon X Elite) | Visual Studio 2022, ARM64 Python 3.11, Rust 1.96 |
| QNN CPU, HTP (NPU) | Windows 11 ARM64, Snapdragon X Elite (Hexagon v73) | QAIRT 2.45.0.260326; outputs bit-identical to `qnn-net-run` |
| ONNX Runtime CUDA provider | Ubuntu 24.04 x86_64, NVIDIA GTX 1650 (compute 7.5), driver 580.178 | ONNX Runtime 1.30.0 GPU (CUDA 12), CUDA 12.8, cuDNN 9; a yolov8n model, 10.0 ms against 59.9 ms on the CPU, same detections. The `tensorrt` provider also ran it (8.1 ms, TensorRT 10.9.0.34 with its ONNX parser) |
| TensorRT | Ubuntu 24.04 x86_64, NVIDIA GTX 1650 (compute 7.5), driver 580.178; built and error-path tested also on an RTX 2060 | TensorRT 10.9.0.34, CUDA 12.8; a yolov8n engine, outputs bit-identical to TensorRT's Python API |

Not tested: Android, iOS, other Snapdragon parts, the TFLite backend's QNN delegate path (see
`docs/design.md`), the ONNX Runtime backend on Windows (it compiles; no matching runtime was available), the TFLite
backend on Windows (ported, not compiled), and TensorRT with dynamic shapes or on GPUs other than the two above.

## ABI stability

Version 0.x makes no ABI promise. From 1.0, public structs only grow at the end and
carry `struct_size`, functions are not removed or changed, and the backend plugin ABI
(`UAIRT_BACKEND_ABI_VERSION`) changes only with a major version.
