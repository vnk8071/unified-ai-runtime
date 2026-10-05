<!-- SPDX-License-Identifier: Apache-2.0 -->
# Framework benchmarks

Programs that time other frameworks on the same model and device as UAIRT, so the numbers in
`docs/design.md` can be reproduced. They are not built by CMake because they need headers and
libraries that are not part of this repository.

## `tflite_bench.c`

TFLite, with or without the QNN TFLite delegate.

Needs TFLite's C API headers and `libtensorflowlite_c.so` (built for your target), and
`QnnTFLiteDelegate.h` plus `libQnnTFLiteDelegate.so` from your QAIRT SDK.

```bash
gcc -std=gnu11 -O2 -o tflite_bench tests/bench/tflite_bench.c \
  -I<tensorflow-source-root> -I$QNN_SDK_ROOT/include/QNN/TFLiteDelegate \
  -L<dir-with-libtensorflowlite_c> -ltensorflowlite_c \
  -L$QNN_SDK_ROOT/lib/<target> -lQnnTFLiteDelegate -ldl -lpthread

./tflite_bench --delegate-backend htp --backend-lib $QNN_SDK_ROOT/lib/<target>/libQnnHtp.so \
  --skel-dir $QNN_SDK_ROOT/lib/hexagon-v73/unsigned --perf default \
  model.tflite input.bin out_ 100
```

The delegate has no CPU backend, so a delegated run is HTP (or GPU). Without
`--delegate-backend` TFLite uses its own CPU kernels. Set `LD_LIBRARY_PATH` to the TFLite and
delegate libraries and `ADSP_LIBRARY_PATH` to the Hexagon skel directory when running.
For a fair comparison with UAIRT use `--perf default` against UAIRT's default, or match the
modes (see `htp_performance_mode` in `docs/api.md`).

## `ort_bench.c`

ONNX Runtime with the QNN plugin execution provider (`onnxruntime-qnn` 2.x). The core library is
loaded with `dlopen`, so it builds against the ONNX Runtime 1.24 C header alone:

```bash
gcc -std=gnu11 -O2 -o ort_bench tests/bench/ort_bench.c -I<onnxruntime>/include -ldl
./ort_bench --core libonnxruntime.so --plugin libonnxruntime_providers_qnn.so \
  --opt backend_path=$QNN_SDK_ROOT/lib/<target>/libQnnHtp.so model.onnx input.bin out_ 100
```

Two ways to get the QNN EP:

- **Built-in** (works on Linux): build ONNX Runtime from source with `--use_qnn`
  (`ort-build/`, run in an arm64 Ubuntu 22.04 container). Run `ort_bench` without `--plugin`.
  Copy `libonnxruntime.so*` and the `libonnxruntime_providers_*.so` files together. The image's
  libstdc++ may be newer than the board's; ship it next to the library if the board lacks
  `GLIBCXX_3.4.30`.
- **Plugin** (`onnxruntime-qnn` 2.x from PyPI): does not work on Linux. ONNX Runtime 1.24.4
  discovers only CPU and GPU hardware there, never an NPU, so the plugin offers no device and
  the benchmark stops with "no QNN execution provider device found". See `docs/design.md`.

`make_epcontext_model.py` wraps a QNN context binary in an EPContext ONNX model, so ONNX
Runtime and UAIRT run the same compiled graph:

```bash
./ort_bench --core libonnxruntime.so --opt backend_path=$QNN_SDK_ROOT/lib/<target>/libQnnHtp.so \
  --opt htp_performance_mode=burst model_ctx.onnx input.bin out_ 100
```
