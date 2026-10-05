# Design

## Scope

Run already-compiled models through one C API and choose the backend by name.
Out of scope: model conversion, quantization, graph building, custom kernels,
agent orchestration.

## Layers

```
application --> uairt.h (public C API, validation) --> uairt_backend_api (vtable)
                                                          |
                                  built-in or dlopen'd plugin: onnxruntime, qnn, ...
```

The core validates argument counts, dtypes, shapes, buffer sizes and memory
domains before a backend runs, so backends do not repeat it.

## ABI rules

- Plain C11. No exceptions, no C++ types, no ownership transfer across the boundary.
- `struct_size` first in every caller-filled struct; backends expose `abi_version`
  and `struct_size`. Add fields at the end only.
- Status codes are `int32_t` constants; details come from `uairt_last_error()`
  (thread-local) and backends report them through `uairt_host_api.set_error`.
- Plugins export one symbol, `uairt_backend_get_api`, and do not link the core
  library. The host passes a function table in.
- Loaded plugin libraries are never unloaded.
- Destroy models before the engine that created them.

## Memory domains

`UAIRT_MEM_HOST`, `UAIRT_MEM_DMABUF` and `UAIRT_MEM_PINNED` are bit flags. A backend advertises the set it
accepts in `supported_domains`; the core rejects other bindings with
`UAIRT_ERR_UNSUPPORTED`. Device pointers (CUDA) and IOSurface (CoreML) are added later
as new bits. `PINNED` is page-locked host memory: the pointer is usable like host memory, and a
device can DMA from and to it without an extra copy.

Zero-copy buffers come from the engine: `uairt_buffer_alloc(engine, n, UAIRT_MEM_DMABUF)`
asks the backend (optional `alloc_buffer`/`free_buffer` in `uairt_backend_api`) and
`uairt_tensor_use_buffer` binds the result to a tensor. Destroy models before freeing
the buffers they ran with, and free buffers before the engine. Which backends accept
DMABUF at run time is decided by the backend: QNN HTP does, QNN CPU reports
`UAIRT_ERR_UNSUPPORTED`. Pinned buffers (`UAIRT_MEM_PINNED`) come the same way and only the TensorRT backend
provides them; on a GTX 1650 they cut a yolov8n run from 8.1 ms to 6.4 ms. Staging the copy through a pinned
buffer inside the backend did not help (8.9 ms): the single-threaded host copy cost more than the faster DMA.

## QNN notes

- The SDK ships no macOS libraries. The CPU backend runs on Linux x86_64, Linux or
  Android aarch64 and Windows, so QNN is built and tested on the target device.
- Two model formats: DLC (`.dlc`, public `QnnSystemDlc_*` API) and context binaries
  (`.bin`, `QnnContext_createFromBinary`, memory-mapped from a path). One graph is
  bound per model; if a model has several, select it with the `graph_name` option.
- HTP needs `libQnnHtp.so` plus its stub/prepare libraries on `LD_LIBRARY_PATH` and the
  Hexagon skel directory in `ADSP_LIBRARY_PATH` (the `adsp_library_path` option sets it).
- A DLC on HTP triggers on-device graph preparation, and whether that works depends on
  the device. On a QCS6490 (Hexagon v68, Ubuntu 20.04) with QAIRT 2.46.0.260424 it
  crashes inside `libQnnHtpPrepare.so` (`GraphPrepare::sequencing_stage`); the SDK's own
  `qnn-net-run` crashes the same way on two different DLCs, with or without an explicit
  v68 device config. On a QCS8550 (Hexagon v73, Qualcomm Linux) the same SDK prepares and
  runs the same DLC, through both `qnn-net-run` and UAIRT. Context binaries avoid
  on-device preparation and load faster: generate them with
  `qnn-context-binary-generator`.
- A context binary built by QNN 2.29 loaded and ran on the 2.46 HTP backend here.
- Zero-copy: DMABUF buffers are FastRPC `rpcmem` allocations (`libcdsprpc.so`, loaded at
  runtime). Each model input/output registers its buffer once with `QnnMem_register`
  (type ION, by fd) and reuses the handle while the same buffer is bound; handles are
  released when the model is destroyed. On the QCS6490 with the Qwen2-0.5B prefill
  graph, the median run time fell from 1056 ms to 1005 ms (about 5%), in two rounds of
  10 runs; outputs stayed byte-identical to `qnn-net-run`.
- Quantized tensors keep their native dtype; `quant_scale` and `quant_zero_point`
  are reported. UAIRT's zero point is `-offset` of QNN's `Qnn_ScaleOffset_t`.
- Known limitation: `QnnSystemDlc_composeGraphs` allocates graph info (about 1.3 KB
  for the tested model) that the headers do not say how to release, and
  `QnnSystemDlc_free` does not release it. LeakSanitizer reports it per model load.
  It is left alone instead of guessing a deep free; confirm with Qualcomm.

## CoreML notes

- The backend is Objective-C (`coreml.m`, ARC) behind the C boundary; it links
  `CoreML.framework` and builds only on Apple platforms.
- Models load from a path: `.mlmodelc`, or `.mlpackage`/`.mlmodel` which are compiled
  with `MLModel compileModelAtURL` on load (the compiled copy is deleted with the
  model). Loading from memory is not supported because CoreML needs a file URL.
- Only multi-array inputs and outputs (float32, float16, int32) with static shapes.
  Image and other feature types, and flexible shapes, return `UAIRT_ERR_UNSUPPORTED`.
  CoreML reports a fixed shape as one enumerated shape, which counts as static.
- CoreML has no ordering of inputs and outputs, so UAIRT orders both by name.
- Inputs wrap the caller's memory without copying. Outputs are bound with
  `outputBackings`; if CoreML returns its own (possibly strided) array, the backend
  copies it into the caller's buffer, honouring strides.
- `compute_units` engine option: `all` (default), `cpu_only`, `cpu_and_gpu`,
  `cpu_and_ne`.
- Measured on an Apple M5 with a small fp16 convolutional net (224x224 input): median
  0.29 ms on CPU, 0.31 ms CPU+GPU, 0.16 ms with the Neural Engine; `coremltools` gave
  0.292, 0.345 and 0.165 ms. UAIRT's overhead is within noise, but the model is tiny and
  these are millisecond-fraction timings, not a rigorous benchmark.
- Outputs are identical to `MLModel.predict` for the same compute units. fp16
  execution on GPU/Neural Engine differs from CPU by about 0.06 on logits spanning
  -45 to 22, which is fp16 rounding.

## QNN measurements on a QCS8550

Qualcomm Linux (LE.PRODUCT.2.1, glibc 2.35), Hexagon v73, QAIRT 2.46.0.260424, a 10 MB
uint8 detection model (1x640x640x3 input). The binaries were cross-built on the Ubuntu
20.04 QCS6490 and copied over; nothing was compiled on the device.

| | Median |
|---|---|
| HTP run, host buffers (DLC or context binary) | 11.2 to 11.4 ms |
| HTP run, DMABUF zero-copy | 10.6 to 10.7 ms (about 5% faster) |
| CPU backend run | 92 ms |
| Process time to load the model and print its I/O: DLC | 1063 ms |
| Same, context binary | 321 ms |

Outputs were byte-identical to `qnn-net-run` for the CPU backend and for HTP with host
and DMABUF buffers, from a DLC and from a context binary. Fifty runs per timing; the
load times are five whole-process runs, so they include process start.

## TFLite notes

- The backend uses TFLite's C API (`libtensorflowlite_c`, which you build for your target) and
  builds as a plugin. Models load from a path or from memory; I/O is static-shape and host memory,
  copied in and out with the C API, in the order the model defines. Quantization (scale, zero point)
  is reported as TFLite stores it.
- `delegate=qnn` creates the QNN TFLite delegate (HTP) for each model. The delegate library is loaded
  at run time from `delegate_library`; the plugin must have been built with `QnnTFLiteDelegate.h`
  from your QAIRT SDK (`QNN_SDK_ROOT`), otherwise `delegate=qnn` returns `UAIRT_ERR_UNSUPPORTED`.
  `htp_performance_mode` maps to the delegate's own modes (it has no CPU backend).
- Verified: on the QCS6490 with TFLite 2.16.1 and `model.tflite`, outputs are identical to
  `tflite_bench` (TFLite's C API directly), and the run time matches it (302.7 ms with TFLite's CPU
  kernels). Loading from memory gives identical outputs. The error-path tests and AddressSanitizer,
  UBSan and LeakSanitizer runs are clean.
- `delegate=qnn` on the QCS8550 (Hexagon v73): outputs are identical to `tflite_bench` with the
  same delegate (3 of 3 outputs non-constant), as are the CPU-kernel outputs. On the QCS6490
  (Hexagon v68) it segfaults in the QNN library, in both `tflite_bench` and the UAIRT backend, the
  same crash as a DLC there (see above). Not run under the sanitizers (no compiler on the device).
- TFLite prints its own message to stderr when it is given a file that is not a model.

## QNN HTP performance modes and the TFLite delegate

HTP clocks and sleep behaviour depend on votes made through the HTP performance
infrastructure. With no vote, a small graph runs several times slower than it can. The
`htp_performance_mode` engine option makes the votes (DCVS v3, plus RPC polling time and
control latency for `burst` and `high_performance`); they last as long as the engine and
are released when it is destroyed. The mapping from mode name to votes is UAIRT's own, so
the names are not guaranteed to mean what Qualcomm's identically named modes mean.

Measured on the QCS8550 (Hexagon v73), the 10 MB detection model, 100 runs, median ms, in
one session. The TFLite numbers come from `tests/bench/tflite_bench.c` with TFLite 2.16.1
and the QNN TFLite delegate from the same QAIRT 2.46.0.260424, using `model.tflite` (the
same quantized network as `model.dlc`: same tensor names, shapes and quantization):

| HTP mode | UAIRT (`.dlc`) | TFLite + QNN delegate (`.tflite`) |
|---|---|---|
| default (no vote) | 10.98, 10.97 | 10.83, 10.73 |
| burst | 2.91 | 2.85 |
| high_performance | 3.07 | 3.00 |
| balanced | 5.00 | 4.56 |
| power_saver | 6.59 | 6.71 |

A later session on a QCS8550 compared `tflite_bench` with the TFLite backend of UAIRT, both with
the delegate, plus UAIRT's `tflite` and `qnn` backends side by side (median ms, 100 runs, two
runs each):

| HTP mode | `tflite_bench` + delegate | UAIRT `tflite` + delegate | UAIRT `qnn` (`.dlc`) |
|---|---|---|---|
| default (no vote) | 10.92, 10.83 | 11.62, 11.67 | 11.01, 10.83 |
| burst | 2.85 | 2.90 | 2.92 |
| high_performance | 3.00 | 3.05 | 3.07 |
| balanced | 4.49 | 4.95 | 4.95 |
| power_saver | 6.71 | 6.91 | 6.68 |

The UAIRT `tflite` backend is about 0.05 ms slower in the voted modes and about 0.7 ms slower
in the default mode; part of that is the per-run copy of inputs and outputs that the C API
backend does and `tflite_bench` (which reuses its buffers) does not. This was not investigated
further.

Other points from the first session: TFLite's own CPU kernels took 141 ms (UAIRT's QNN CPU
backend 92 ms); the delegate's setup took about 0.9 s, similar to loading the DLC.

- Within a few percent the two stacks are the same speed in every mode. The large
  difference between modes matters more than the difference between stacks.
- UAIRT's `burst` and `default` outputs are byte-identical; the mode does not change results.
- On the Qwen2-0.5B prefill graph (about 1 s of compute on a QCS6490, Hexagon v68) the
  mode made no measurable difference (about 1.06 s in every mode). The effect is
  large for short graphs and absent for long compute-bound ones.
- Output agreement between runtimes was checked on random-noise input, which is a poor
  input for a detector: tiny quantization differences become different boxes. UAIRT HTP
  versus TFLite HTP agreed on 64.6% of box values, 99.9% of scores and 92.4% of class
  indices; UAIRT CPU versus TFLite CPU agreed about as well (64.8%, 99.9%, 92.8%), and
  TFLite's own HTP versus its own CPU agreed less (54.4%, 99.9%, 89.8%). So UAIRT differs
  from TFLite about as much as the two compute units of TFLite differ from each other.
  This says nothing about detection accuracy; a test with real images and box overlap is
  still to do.

## ONNX Runtime with the QNN execution provider

Same device (QCS8550, Hexagon v73), same QAIRT 2.46.0.260424, and the same compiled graph: the
context binary made from `model.dlc` was wrapped in an EPContext ONNX model
(`tests/bench/make_epcontext_model.py`) for ONNX Runtime, and loaded directly by UAIRT. 100 runs
per cell, median ms, interleaved in one session. `init` is engine plus model creation in
UAIRT and environment plus session creation in ONNX Runtime, which are not the same scope.

| HTP mode | UAIRT | ONNX Runtime QNN EP |
|---|---|---|
| default | 10.95, 10.86 | 11.26, 11.44 |
| burst | 2.92 | 2.93 |
| high_performance | 3.07 | 4.27 |
| balanced | 4.96 | 4.76 |
| power_saver | 6.63 | 6.78 |
| init (range over the six runs) | 223 to 258 ms | 238 to 282 ms |

- The outputs were byte-identical (all three tensors, default mode).
- Within a few percent in `default`, `burst`, `balanced` and `power_saver`. The
  `high_performance` rows differ because each stack maps that name to its own votes; UAIRT's
  mapping is documented above, ONNX Runtime's is its own.
- ONNX Runtime was built from source (v1.24.4, Release, QNN EP as a shared provider library,
  against the QAIRT 2.46 headers) in an arm64 Ubuntu 22.04 container
  (`tests/bench/ort-build/`) and run with that image's libstdc++, because the board's lacks
  `GLIBCXX_3.4.30`.

The `onnxruntime-qnn` plugin from PyPI could not be used: on Linux, ONNX Runtime 1.24.4
discovers only CPU (`/proc/cpuinfo`) and GPU (`/sys/class/drm`) hardware, never an NPU, so the
plugin's QNN factory offers no device (only `CPUExecutionProvider` is listed), and the by-name
call is disabled in that build. Every Linux aarch64 `onnxruntime-qnn` wheel (2.1.0 to 2.6.0) is
such a plugin. Only the built-in EP, which does not depend on device discovery, works.

## ONNX Runtime plugin execution providers: MLX

The ONNX Runtime backend can load a plugin execution provider with `ep_library` (path to the
library), `ep_name` (the name to register it under, normally the provider's own, for example
`MLXExecutionProvider`) and `ep_option.<key>=<value>` options. It registers the library, picks the
device whose provider name matches, and appends it; ONNX Runtime keeps the CPU EP as the fallback.
The plugin must match the ONNX Runtime core: `onnxruntime-ep-mlx` 0.29.x needs ONNX Runtime 1.29.x,
and the UAIRT plugin must be built against headers no newer than that core (it asks for the header's
API level, so a build against 1.29.1 headers works with 1.29.1).

Whether a plugin gets a device depends on what ONNX Runtime discovers. On Linux it never finds an
NPU, so the QNN plugin has nothing to attach to (see above). On macOS ONNX Runtime 1.29.1 does offer
the Apple GPU, and the MLX provider reports "bound to GPU device".

Measured on an Apple M5 with ONNX Runtime 1.29.1 and `onnxruntime-ep-mlx` 0.29.6 (float32, random
weights, `tests/compare_ort_eps.py`, median of 10 runs, MLX claimed every node, 100% in both):

| Model | CPU EP | MLX EP | |
|---|---|---|---|
| MobileNetV2, batch 1 | 2.32 ms | 2.38 ms | no gain |
| ResNet50, batch 8 | 147 to 149 ms | 47 ms | 3.1 to 3.2x faster |

Outputs agreed with the CPU EP to a relative max abs difference of 8e-7 (MobileNetV2) and 1.4e-6
(ResNet50), and the top-1 classes were equal. The first MLX run in a fresh process took 1.6 s to
initialize for MobileNetV2 and about 68 ms for the later runs of the other model; I did not find out
why, so treat initialization time as unexplained.

Not tested: LLMs or other transformers (the provider's own benchmarks target those, and its README
reports larger gains there), quantized models, models with dynamic shapes, and a comparison with the
CoreML backend on the same model (there is no ONNX to CoreML conversion path here). The README
figures for the provider are the authors' own and were not reproduced.

## Not in v1

Asynchronous execution, dynamic shapes, per-channel quantization, profiling hooks,
Windows plugin loading. Each is added as an extension without breaking v1.

## Backend artifacts are not portable

A QNN context binary is tied to a SoC and HTP architecture, a TensorRT plan to a
GPU and TensorRT version. Backends must return `UAIRT_ERR_INCOMPATIBLE_MODEL` with a
clear message instead of falling back silently.

## Milestones

| | Content | Evidence |
|---|---|---|
| M0 | C ABI, plugin interface, reference backend, tests | `ctest` passes (done) |
| M1 | ONNX Runtime backend (CPU EP) | Add/Mul model and MobileNetV2 (random weights) match Python ORT 1.24.4, max abs diff 0 (done) |
| M2 | QNN backend, CPU backend, DLC models | A 9.9 MB uint8 detection DLC on a QCS6490 gives byte-identical outputs to `qnn-net-run` (done). Context binaries come with M3 |
| M3 | QNN HTP on a Snapdragon device, ORT QNN EP | HTP context binary on a QCS6490 NPU: 49/49 outputs byte-identical to `qnn-net-run`, latency within noise of it, and ~5% faster with DMABUF zero-copy (done for QNN). ORT QNN EP not done |
| M4 | CoreML, TensorRT, docs, public release | CoreML: outputs identical to `MLModel.predict` on three models, ANE ~1.8x faster than CPU on a small net (done). TensorRT needs an NVIDIA device (todo). Docs and release (todo) |

## Prior art to read

ONNX Runtime plugin EP ABI (`onnxruntime_ep_c_api.h`), Intel oneAPI Unified Runtime
(C API, loader, adapters), `etri/unified-npu-sdk`.
