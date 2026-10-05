<!-- SPDX-License-Identifier: Apache-2.0 -->
# uairt (Rust)

Two crates, no external dependencies (so it builds offline):

- `uairt-sys`: raw `extern "C"` declarations for `include/uairt/uairt.h`, written by hand (no bindgen or libclang).
- `uairt`: the safe API. `Model<'e>` and `Buffer<'e>` borrow their `Engine`, so the compiler enforces that the
  engine outlives them. Failures are `uairt::Error` values with a `Status` and a message.

```bash
cmake -S ../.. -B ../../build-shared -DBUILD_SHARED_LIBS=ON && cmake --build ../../build-shared
cargo test                       # looks for libuairt in ../../build-shared; override with UAIRT_LIB_DIR
UAIRT_TEST_ORT_PLUGIN=../../build-shared/libuairt_backend_onnxruntime.so cargo test   # adds the ONNX Runtime test
```

```rust
uairt::load_backend_library("libuairt_backend_onnxruntime.so")?;
let engine = uairt::Engine::new("onnxruntime", &[("intra_op_threads", "1")])?;
let mut model = engine.load_model("model.onnx")?;
let input = vec![0f32; model.inputs()[0].num_elements()];
let mut output = vec![0f32; model.outputs()[0].num_elements()];
model.run(&[uairt::as_bytes(&input)], &mut [uairt::as_bytes_mut(&mut output)])?;
```

- `run` takes one byte slice per input and output, sized as `TensorInfo::nbytes()` reports; `as_bytes` views typed
  slices as bytes without copying.
- The raw handles make `Engine`, `Model` and `Buffer` `!Send` and `!Sync`, matching the C API's rule that one
  model must not run from several threads at once. Free a buffer only after the models that used it.
- Windows: build libuairt as a DLL (`cmake -S ../.. -B ../../build-shared -G "Visual Studio 17 2022" -DBUILD_SHARED_LIBS=ON`,
  then `--build ... --config Debug`), set `UAIRT_LIB_DIR` to `build-shared\Debug` and put that directory on `PATH` before
  `cargo test`; there is no rpath.
- `cargo run --example run_qnn --release -- <libuairt_backend_qnn.dll> <QAIRT root> <model.dlc> [npu|cpu|gpu] [runs] [cache-dir]`
  runs a model through the QNN plugin. On Windows ARM64, set `ADSP_LIBRARY_PATH` to
  `<QAIRT root>\lib\hexagon-v73\unsigned` before starting it for `npu`. The plugin sets it too, and that works from the
  C tool, but from a Rust process the NPU session failed to open unless the variable was already in the environment at
  startup (cause not found).
- A program that links `uairt` needs `libuairt` on its runtime library path (an rpath, `DYLD_LIBRARY_PATH` or
  `LD_LIBRARY_PATH`); this workspace's own tests get an rpath from the build scripts.
