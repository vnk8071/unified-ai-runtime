<!-- SPDX-License-Identifier: Apache-2.0 -->
# llama.cpp backend

Source: `src/backends/llamacpp.c`. Plugin: `libuairt_backend_llamacpp.so` (a dylib on macOS, still named `.so`). Runs GGUF language models
through llama.cpp, which is a pinned git submodule at `third_party/llama.cpp` (MIT; see
[../LICENSING.md](../LICENSING.md) and `THIRD_PARTY_NOTICES.md`). The backend is off by default.

## You install

Nothing beyond a C and C++ toolchain. Fetch the pinned llama.cpp once: `git submodule update --init third_party/llama.cpp`
(or point `UAIRT_LLAMACPP_ROOT` at another checkout, at your own risk: `llama.h` changes often).

## Build

```bash
cmake -S . -B build -DUAIRT_BUILD_LLAMACPP=ON && cmake --build build
```

llama.cpp is built static inside the plugin, with Metal (shader library embedded) on Apple. The plugin exports only
`uairt_backend_get_api`. CUDA and Vulkan are left to you (`-DGGML_CUDA=ON`) and have not been verified.

### Adreno GPU on Windows ARM64 (OpenCL)

Snapdragon X's GPU is reached through llama.cpp's OpenCL backend, not Vulkan. Install the Adreno OpenCL SDK yourself
(`OPENCL_SDK_ROOT`, for example `C:\Qualcomm\OpenCL_SDK.3.2.3.2`), then:

```powershell
cmake -S . -B build-opencl -G "Visual Studio 17 2022" -A ARM64 -DUAIRT_BUILD_LLAMACPP=ON -DGGML_OPENCL=ON "-DCMAKE_PREFIX_PATH=$env:OPENCL_SDK_ROOT"
cmake --build build-opencl --config Release
```

Run with the default `n_gpu_layers` (99); `0` stays on the CPU. The HTP (Hexagon NPU) path of llama.cpp (`ggml-hexagon`) is not
wired into this plugin: it needs the Hexagon SDK, a test-signing certificate and a way to select the device.

## Use

A `.gguf` model has no tensors. Load it as usual, then use a session:

```c
uairt_load_backend_library("libuairt_backend_llamacpp.so");
uairt_option layers = {"n_gpu_layers", "99"};            /* 0 = CPU only; the default is 99 */
uairt_engine_create("llamacpp", &layers, 1, &engine);
uairt_model_load(engine, &(uairt_model_source){.struct_size = sizeof(uairt_model_source), .path = "model.gguf"}, &model);
uairt_model_tokenize(model, text, len, 1, tokens, capacity, &count);
uairt_session_create(model, NULL, 0, &session);          /* option n_ctx, default 2048, capped by the model */
uairt_session_append(session, tokens, count);
uairt_session_logits(session, logits, vocab, &n);        /* pick a token, append it, repeat */
```

`examples/llm_generate.c` is a complete greedy generator. From Python: `uairt.AutoModel.from_file("model.gguf",
device="gpu")`, then `model.generate(prompt, max_tokens=..., temperature=..., top_k=..., top_p=..., seed=...)`, or
`model.tokenize`, `model.session()` and `session.logits()` for your own loop (`bindings/python/examples/llm_generate.py`).

- `uairt_model_run` returns `UAIRT_ERR_UNSUPPORTED`: the backend is session-only.
- `generate()` and `examples/llm_generate.c` do not stop at an end-of-generation token by themselves (the C API exposes no special-token ids); pass `stop_tokens` in Python, or limit the token count.
- `uairt_model_tokenize` parses special tokens in the text (`<|eot_id|>` becomes a control token), as llama.cpp's own
  CLI does. Do not tokenize untrusted text if that matters to you.
- To stop at the end-of-generation token, tokenize its text: `stop = model.tokenize("<|eot_id|>", add_special=False)`
  returns one id to pass in `stop_tokens` (Llama 3 style models; other models use their own end token string).
- `n_ctx` is capped by the model's training context and cannot be read back. A generation that reaches the context
  size fails mid-stream with "context full" (`InvalidArgument` in Python).
- Different sessions of one model may be used from different threads; one session must not be.
- A token id outside the vocabulary returns `UAIRT_ERR_INVALID_ARGUMENT`. Appending past `n_ctx` fails with "context
  full" and changes nothing. If llama.cpp itself fails while decoding, the session is reset and `UAIRT_ERR_RUNTIME` is
  returned.
- llama.cpp's logging is silenced except errors.
- Sessions are independent: each has its own KV cache and context memory (about `n_ctx` times the model's per-token cache).

## Check

`ctest --test-dir build --output-on-failure` runs the error-path tests and the symbol check. Set
`UAIRT_LLAMACPP_TEST_MODEL` to a GGUF path before `cmake` to also run the model tests and
`tests/compare_with_llamacpp.py`, which checks that greedy output equals llama.cpp's own `llama-completion` for the same
model (it needs `UAIRT_BUILD_EXAMPLES=ON`, the default), on the CPU and with all layers offloaded (Metal on a Mac). `UAIRT_LLAMACPP_TEST_MODEL` is read when `cmake` configures, so set it before configuring (and reconfigure after changing it).
Verified on macOS with an Apple M5 (Metal and the CPU path) with Llama 3.2 1B Q4.

Verified on Windows 11 ARM64 (Snapdragon X Elite) with Qwen3-0.6B Q8_0: the CPU build passes `ctest` including the comparison with
`llama-completion`. The OpenCL build ran the model on the Adreno GPU (the process showed GPU engine use, none with `n_gpu_layers=0`)
and generated text, but its greedy output differs from the CPU's after the first few tokens and has not been compared with
llama.cpp's own tool on the GPU, and `ctest` has not been run for it. Linux, CUDA, Vulkan and the Hexagon NPU have not been run.
