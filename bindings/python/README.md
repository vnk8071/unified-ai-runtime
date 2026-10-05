<!-- SPDX-License-Identifier: Apache-2.0 -->
# uairt (Python)

A `ctypes` binding for the Unified AI Runtime C API. Pure Python plus NumPy; no compiler is needed
to install it, but it needs the shared library `libuairt`.

```bash
cmake -S ../.. -B ../../build-shared -DBUILD_SHARED_LIBS=ON && cmake --build ../../build-shared
export UAIRT_LIBRARY=$PWD/../../build-shared/libuairt.dylib      # .so on Linux
pip install -e .
```

```python
import numpy as np
import uairt

uairt.load_backend_library("libuairt_backend_onnxruntime.so")   # a plugin per backend
with uairt.Engine("onnxruntime", intra_op_threads=1) as engine:
    with engine.load_model("model.onnx") as model:
        print(model.inputs, model.outputs)          # TensorInfo(name, dtype, shape, quant_scale, ...)
        x = np.zeros(model.inputs[0].shape, model.inputs[0].dtype)
        (y,) = model.run(x)
```

- Inputs must match the model's dtype and shape exactly; the binding never converts silently.
- Errors are `uairt.UairtError` subclasses (`NotFound`, `Unsupported`, `IncompatibleModel`, `IoFailure`, ...)
  carrying `.status` and `.message`.
- `Engine.alloc_buffer(n, "host" | "dmabuf")` and `Model.run_buffers` give zero-copy buffers;
  `Buffer.array(dtype, shape)` is a NumPy view.
- Models and buffers keep their engine alive. Close models before buffers they used.
- Threading follows the C API: do not run one model from several threads at once.

Tests: `UAIRT_LIBRARY=... pytest bindings/python/tests`. Set `UAIRT_TEST_ORT_PLUGIN` to the ONNX Runtime
plugin to also run the two-input/two-output test.

Windows ARM64: use an ARM64 Python (an x64 Python cannot load the ARM64 `uairt.dll`), build libuairt as a DLL and set
`UAIRT_LIBRARY` to `build-shared\Debug\uairt.dll`. `examples/run_qnn.py <plugin> <QAIRT root> <model.dlc> [npu|cpu|gpu]`
runs a model through the QNN plugin (0.8 ms on the Snapdragon X Elite NPU for `face_det_lite`). Set `ADSP_LIBRARY_PATH`
to `<QAIRT root>\lib\hexagon-v73\unsigned` before starting Python for `npu`; see the Rust README for the caveat.
