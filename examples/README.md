<!-- SPDX-License-Identifier: Apache-2.0 -->
# Examples

Each language has the same two programs: a **quickstart** that runs the built-in `reference` backend (no SDK, runs anywhere) and an
**llm_stream** that streams text from a GGUF model through the `llamacpp` backend with sampling. Build a shared library set first;
the paths below use `build-shared`.

```bash
git submodule update --init --depth 1 third_party/llama.cpp          # only for the llm_stream programs
cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON -DUAIRT_BUILD_LLAMACPP=ON && cmake --build build-shared
```

| Folder | What is there | Run |
|---|---|---|
| `c` (this folder's `*.c`) | `quickstart.c`, `run_model.c` (any model), `llm_generate.c` (your own greedy loop), `llm_stream.c` (`uairt_llm`), `llm_complete.c` (text servers) | `build-shared/quickstart` |
| [`cpp/`](cpp) | `uairt.hpp`, a header-only RAII wrapper with exceptions, and `quickstart.cpp`, `llm_stream.cpp` | `build-shared/examples/cpp/quickstart_cpp` |
| [`python/`](python) | `quickstart.py`, `llm_stream.py`, `session_loop.py` (your own sampling loop over the logits, with NumPy); `auto_model.py` (any model), `llm_generate.py`, `run_qnn.py`, `detect_tensorrt.py`; `inspect_model.py` and `prepost.py`, the helpers behind [docs/agents/pre-post-processing.md](../docs/agents/pre-post-processing.md) | `PYTHONPATH=bindings/python UAIRT_LIBRARY=build-shared/libuairt.dylib python examples/python/quickstart.py` |
| [`rust/`](rust) | A crate with hand-written FFI declarations, a safe wrapper (`Engine`, `Model` that borrows its engine) and the two programs | `UAIRT_LIB_DIR=$PWD/build-shared cargo run --release --manifest-path examples/rust/Cargo.toml --bin quickstart` |

The llm_stream programs take `<plugin> <model.gguf> "<prompt>" <max_tokens> [temperature [top_k [top_p [seed]]]]`, for example
`build-shared/examples/cpp/llm_stream_cpp build-shared/libuairt_backend_llamacpp.so model.gguf "The capital of France is" 40 0.8 40 0.95 7`.
The C, C++ and Rust programs call the same `uairt_llm_generate`, so the same seed gives the same text. The Python one samples in
Python (NumPy's random numbers), so its sampled text differs; greedy text (temperature 0) is the same everywhere.

These programs are checked on macOS arm64 with the SmolLM2-135M GGUF model. On Linux use `.so` instead of `.dylib`.
The C++ wrapper and the Rust crate are examples of how an application can wrap the C API, not part of the supported ABI.
