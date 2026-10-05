<!-- SPDX-License-Identifier: Apache-2.0 -->
# Language bindings

Thin wrappers over the C API in `include/uairt/uairt.h`. Each wrapper needs the shared library:

```bash
cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON && cmake --build build-shared
```

Backends are plugins loaded at run time (`uairt_load_backend_library`); the built-in `reference`
backend works everywhere and is what the binding tests use unless a plugin is given.

| Language | Directory | Mechanism | Status |
|---|---|---|---|
| Python | `python/` | `ctypes` + NumPy | done; 9 tests pass on macOS arm64 |
| C++ | `../include/uairt/uairt.hpp` | header-only C++17 over the C API (RAII, exceptions) | done; tested on macOS arm64 with ONNX Runtime, clean under ASan and UBSan |
| Rust | `rust/` | `uairt-sys` (FFI) + `uairt` (safe API) | done; 7 tests and a doc-test pass on macOS arm64 with ONNX Runtime, clippy and rustfmt clean |

The bindings share the same shape: `Engine` (a backend plus options) creates `Model`s; a model
reports its inputs and outputs (name, dtype, shape, quantization) and runs on host arrays or on
zero-copy buffers. They map status codes to each language's errors, keep engines alive for as long
as their models, and never convert dtypes silently.

`./bindings/test_all.sh` builds the shared library and runs the tests of every binding whose toolchain is
installed (set `ONNXRUNTIME_ROOT` to include the ONNX Runtime tests).

The bindings are not published to PyPI or crates.io.

Windows ARM64 (Snapdragon X Elite): the Python (ARM64 interpreter), C++ and Rust bindings pass their tests with the
reference backend, and each has a `run_qnn` example that runs a DLC on the NPU (C++ and `run_model` need no setup; Python
and Rust need `ADSP_LIBRARY_PATH` set before the process starts). The C++ example builds as `run_qnn_cpp` with `-DUAIRT_BUILD_QNN=ON`.
