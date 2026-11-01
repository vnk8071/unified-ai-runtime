<!-- SPDX-License-Identifier: Apache-2.0 -->
# Transformers.js backend (a Node worker; CPU, and WebGPU where onnxruntime-node has it)

Source: `src/backends/transformersjs.c` (the plugin) and `src/backends/transformersjs_worker.mjs` (the Node side). Plugin:
`libuairt_backend_transformersjs.so`. POSIX only (macOS, Linux); `-DUAIRT_BUILD_TRANSFORMERSJS=ON` fails on Windows.

Transformers.js runs ONNX models on ONNX Runtime, which UAIRT's `onnxruntime` backend already wraps. Use that backend for speed and for
`.onnx` files. This one exists to run a **Transformers.js model directory** the way Transformers.js itself loads it (its `AutoModel`,
its dtype variants, its WebGPU device), from the C API, and to check UAIRT against Transformers.js.

## You install

- Node.js (26.10 was used) on `PATH`, or pass the `node` option.
- `@huggingface/transformers` (Apache-2.0) in a directory of your choice, for example `npm install --prefix ~/uairt-transformersjs @huggingface/transformers`.
  Point the backend at it with the `package_dir` option or `UAIRT_TRANSFORMERSJS_DIR`. Version 4.3.2 was used; it brings `onnxruntime-node` 1.30.0.
- A model directory: `config.json`, `onnx/model.onnx` (or the dtype variant you pass as `dtype`) and, if you use a tokenizer, `tokenizer.json`.
  Transformers.js hosts many on Hugging Face (check each model's licence); the backend never downloads: remote models are switched off.

## Build and run

```bash
cmake -S . -B build -DUAIRT_BUILD_TRANSFORMERSJS=ON && cmake --build build
export UAIRT_TRANSFORMERSJS_DIR=$HOME/uairt-transformersjs
build/run_model --option device=webgpu --option input_shapes='input_ids:1x16;attention_mask:1x16;token_type_ids:1x16' \
  build/libuairt_backend_transformersjs.so transformersjs path/to/model-dir out input_ids.bin attention_mask.bin token_type_ids.bin
```

## Options

`package_dir` (default `UAIRT_TRANSFORMERSJS_DIR`, else the current directory), `node` (default `node`), `worker` (the script; default the one in the
source tree the plugin was built from, or `UAIRT_TRANSFORMERSJS_WORKER`), `device` (`cpu` default, or `webgpu`), `dtype` (Transformers.js dtype,
default `fp32`), `input_shapes` (`name:1x16;name:1x16`, needed for any input with a dynamic dimension, which is most Transformers.js models).

## Behaviour

- One Node process per loaded model, started at `uairt_model_load` and stopped at `uairt_model_destroy`. Tensors cross a socket pair as raw bytes, so
  every run copies the inputs and outputs twice; expect more overhead than the `onnxruntime` backend.
- Shapes are static. The inputs are what `input_shapes` (or the model) says. The output shapes come from one run on zeros at load, so a model that fails
  on all-zero inputs fails to load.
- Types: float32, int64, int32, int8, uint8, bool. Anything else (float16 outputs, strings) is `UAIRT_ERR_UNSUPPORTED` naming the tensor. One ONNX session per
  model: encoder-decoder models (several sessions) are `UAIRT_ERR_UNSUPPORTED`. There is no text or session API; this is a tensor backend.
- `device=webgpu` is checked, not trusted: the worker asks ONNX Runtime for verbose logs and the plugin refuses the load with
  `UAIRT_ERR_BACKEND_UNAVAILABLE` unless the log says nodes were placed on `WebGpuExecutionProvider`. Some nodes still run on the CPU provider (7 of 192 for
  all-MiniLM-L6-v2 here); this is ONNX Runtime's partitioning, not a failure.
- Errors: a missing model directory is `UAIRT_ERR_NOT_FOUND`; no Node, no worker script or no installed package is `UAIRT_ERR_BACKEND_UNAVAILABLE`; a dynamic
  dimension without `input_shapes` is `UAIRT_ERR_INVALID_ARGUMENT`; a worker that dies is `UAIRT_ERR_RUNTIME` with the end of its stderr.

## Verified

Apple M5, macOS, Node 26.10.0, `@huggingface/transformers` 4.3.2 (`onnxruntime-node` 1.30.0), `Xenova/all-MiniLM-L6-v2` fp32 at 1x16 tokens:

| Device | Placement | `tests/compare_with_transformersjs.py` against Transformers.js called from Node |
|---|---|---|
| `cpu` | all 183 nodes on the CPU provider | identical (difference 0), output std 0.39 |
| `webgpu` | 185 nodes on `WebGpuExecutionProvider`, 7 on the CPU provider | identical (difference 0; both sides run the same kernels), 3.8 ms per run through UAIRT |

WebGPU output differs from the CPU's by about 3e-6. `ctest` runs `tests/test_transformersjs.c`: the error paths, a protocol check against a stand-in worker
(`tests/data/fake_transformersjs_worker.mjs`, needs only Node) including the refusal of `webgpu` when no WebGPU placement is logged, and, when
`UAIRT_TRANSFORMERSJS_DIR` is set, a run of a two-output model directory on `cpu` and `webgpu`. Run under AddressSanitizer and UBSan: passed (leak detection is not available on macOS arm64).

**Not verified:** Linux, any GPU but the M5's, other models or tasks, dtype variants (fp16, q4, q4f16), a language model with a key-value cache,
dynamic shapes at run time, concurrent runs on one model (not supported; runs are serialized by the caller), Node versions other than 26.10.
