<!-- SPDX-License-Identifier: Apache-2.0 -->
# uairt (Python)

A `ctypes` binding for the Unified AI Runtime C API. Pure Python plus NumPy; no compiler is needed
to install it, but it needs the shared library `libuairt`.

From a wheel on [PyPI](https://pypi.org/project/uairt/) (see [docs/releasing.md](../../docs/releasing.md)), which carries the native library and the ONNX
Runtime plugin, so nothing else is built:

```bash
pip install "uairt[onnxruntime]"
```

From source, for development:

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

## AutoModel and plugin discovery

`AutoModel` picks the backend from the model file and the device, so you do not need to know which plugin or which options
a runtime takes. It is a thin layer over `Engine` and `Model`, which stay available.

```python
import uairt

with uairt.AutoModel.from_file("yolov8n.onnx", device="gpu") as model:
    print(model.backend, model.options)        # onnxruntime {'execution_provider': 'cuda'}
    (y,) = model.run(x)

uairt.AutoModel.from_file("face_det.dlc", device="npu")                  # qnn
uairt.AutoModel.from_file("yolov8n_ncnn_model", device="gpu", input_shapes="3x640x640")   # ncnn on Vulkan
uairt.AutoModel.resolve("m.onnx", device="gpu")                           # the plan only, nothing loaded
```

- **Backend by file:** `.dlc` and `.bin` go to `qnn`, `.onnx` to `onnxruntime` (then `openvino`), `.xml` to `openvino`, `.param` to
  `ncnn`, `.engine` and `.plan` to `tensorrt`, `.tflite` to `tflite`, `.mlmodelc`, `.mlpackage` and `.mlmodel` to `coreml`. A
  directory is judged by what it holds (an ultralytics `_openvino_model` or `_ncnn_model` export). Pass `backend=` to choose.
- **Device:** `cpu`, `gpu`, `npu`, `cuda`, `tensorrt` or `vulkan`, with an optional index (`gpu:1`), or `None` for the backend's
  default. It is mapped to that backend's own options (`device=npu` for QNN, `execution_provider=cuda` for ONNX Runtime,
  `device=GPU` for OpenVINO, `device=vulkan` for NCNN, and so on). QNN's default is `npu`.
- **Strict:** the first candidate that is available (registered, or with a plugin that can be found) and supports the device
  is used. If it fails to load, that error is raised: there is no fallback to another runtime or to the CPU. A device a
  backend cannot serve raises `Unsupported`; no usable plugin raises `NotFound`. Both say why each candidate was passed over.
  `model.plan.skipped` lists the candidates passed over.
- **Options:** keyword arguments and `options={...}` go to the backend and override the device mapping.

Backend plugins are found, and loaded when an engine asks for them, in this order: `UAIRT_PLUGIN_PATH` (directories separated
like `PATH`), the package's own `plugins/` directory, the directory of the loaded `libuairt`, and its `uairt/` subdirectory.
A plugin is a file named `libuairt_backend_<name>.so`, `.dylib` or `.dll`. `uairt.load_backend_library(path)` still loads one
by path, and `uairt.available_backends()` and `uairt.discover_plugins()` show what is registered and what was found. A plugin
still needs its vendor runtime installed; loading it reports that.

`examples/auto_model.py` loads any model this way and prints what was chosen.

- Inputs must match the model's dtype and shape exactly; the binding never converts silently.
- Errors are `uairt.UairtError` subclasses (`NotFound`, `Unsupported`, `IncompatibleModel`, `IoFailure`, ...)
  carrying `.status` and `.message`.
- `Engine.alloc_buffer(n, "host" | "dmabuf" | "pinned")` and `Model.run_buffers` give zero-copy buffers;
  `Buffer.array(dtype, shape)` is a NumPy view.
- Models and buffers keep their engine alive. Close models before buffers they used.
- Threading follows the C API: do not run one model from several threads at once.

Tests: `UAIRT_LIBRARY=... pytest bindings/python/tests`. Set `UAIRT_TEST_ORT_PLUGIN` to the ONNX Runtime
plugin to also run the two-input/two-output test.

Windows ARM64: use an ARM64 Python (an x64 Python cannot load the ARM64 `uairt.dll`), build libuairt as a DLL and set
`UAIRT_LIBRARY` to `build-shared\Debug\uairt.dll`. `examples/run_qnn.py <plugin> <QAIRT root> <model.dlc> [npu|cpu|gpu]`
runs a model through the QNN plugin (0.8 ms on the Snapdragon X Elite NPU for `face_det_lite`). Set `ADSP_LIBRARY_PATH`
to `<QAIRT root>\lib\hexagon-v73\unsigned` before starting Python for `npu`; see the Rust README for the caveat.
