<!-- SPDX-License-Identifier: Apache-2.0 -->
# MNN backend (CPU, and the GPU backends of an MNN built with them)

Source: `src/backends/mnn.cpp`, a C++ plugin behind the C backend ABI. Plugin: `libuairt_backend_mnn.so`.

## You install

Nothing for the CPU: MNN (Alibaba, Apache-2.0) is a pinned git submodule, built into the plugin and linked statically with its
symbols hidden, so it cannot clash with another MNN in your application.

```bash
git submodule update --init --depth 1 third_party/MNN        # about 260 MB on disk
```

To use your own MNN build instead (another version, or one you share with other code), install it and set `MNN_ROOT`; that takes
priority over the submodule. Build it with the **same C++ toolchain** as this plugin: the plugin uses MNN's C++ API.

A model is a `.mnn` file. MNN's converter makes it from ONNX, TFLite, TorchScript and Caffe models; `pip install MNN` installs it as
`mnnconvert` (it replaced packages such as `protobuf` on its first run, so use a throw-away virtual environment):

```bash
mnnconvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode uairt
```

## Build

```bash
cmake -S . -B build -DUAIRT_BUILD_MNN=ON && cmake --build build                          # the pinned submodule, CPU
cmake -S . -B build -DUAIRT_BUILD_MNN=ON -DUAIRT_MNN_METAL=ON && cmake --build build     # plus Metal (Apple platforms)
export MNN_ROOT=$HOME/mnn-install     # or your own install: include/MNN/Interpreter.hpp and lib/libMNN
```

`UAIRT_MNN_OPENCL` and `UAIRT_MNN_VULKAN` build the pinned MNN with those backends; neither has been run here. MNN has many more options
(CUDA, NNAPI, QNN, CoreML); this plugin names only the ones above.

## Run

```bash
build/run_model --info build/libuairt_backend_mnn.so mnn model.mnn
build/run_model --repeat 100 --option num_threads=4 --option precision=high build/libuairt_backend_mnn.so mnn model.mnn out in0.bin
```

Engine options:

- `device`: `cpu` (default), `metal`, `opencl`, `vulkan` or `cuda`. A GPU device fails with `UAIRT_ERR_BACKEND_UNAVAILABLE` when the
  session did not end up on it (this MNN was built without it, or it is not usable here), where MNN alone would quietly run on the CPU.
- `num_threads`: CPU threads.
- `precision`: `normal` (MNN's default), `high` (float32 everywhere), `low` (float16 kernels) or `bf16`. On the CPU `normal` is
  float32; on the GPU it is float16 (see Check).
- `input_shapes`: only for a model whose input shapes are not fixed, or to resize it: one shape per input in input order, such as
  `1x3x224x224`, separated by `;`.

## Behaviour to know

- **Order of inputs and outputs.** MNN's session API returns tensors in a map ordered by name, not in the order the model declared, so
  UAIRT numbers inputs and outputs by name (`x` before `y`, `product` before `sum`), as the CoreML backend does. Read the names with
  `--info`.
- **Layout.** Tensors are exchanged in the layout MNN reports for them, usually NCHW for a model converted from ONNX; MNN converts to and
  from its internal layout itself.
- **Types.** float32, float16, bfloat16 and the integer types of 8, 16, 32 and 64 bits (unsigned for 8 and 16). Static shapes only.
- **Log lines.** MNN prints to standard output with `printf`: one line about the CPU's instruction sets each time a session starts, and
  messages from a GPU backend when a kernel does not compile (the Metal backend printed two on macOS 27 before falling back). Do not parse
  the standard output of a program that uses this backend; on Android the lines go to logcat under `MNNJNI`.
- **Not a model.** A file that is not a `.mnn` model is `UAIRT_ERR_INCOMPATIBLE_MODEL`, a missing file `UAIRT_ERR_IO`.

## Limits

Host memory only; static shapes; no zero-copy buffers. `.mnn` models with dynamic input shapes load only with `input_shapes`.

## Check

`ctest` runs `mnn`, which loads `tests/data/add_mul.mnn` (made from `tests/data/add_mul.onnx` with `mnnconvert`) and checks the error paths,
the names and order of the tensors, the results of a run, `input_shapes` (resizing both inputs), and that `device=cuda` is refused.
For the comparison with MNN's own Python API set `UAIRT_MNN_TEST_MODEL` (a `.mnn` file) and `UAIRT_MNN_PYTHON` (a Python with `MNN` and numpy,
the version of the submodule) before cmake: `tests/compare_with_mnn.py` runs each engine option, alone and together, through both and wants
identical float32 outputs.

Run here, on an Apple M5 (macOS arm64), PyMNN 3.6.1 and the pinned MNN 3.6.1:

- CPU: identical to PyMNN on `add_mul`, a small convolutional net and ResNet-18, for five option combinations each (default, two threads,
  `precision` `high`, `low`, and one thread with `high`).
- Metal (`-DUAIRT_MNN_METAL=ON`): ResNet-18 identical to PyMNN's Metal result, and 4e-3 away from the CPU float32 output on logits of size 6,
  because the GPU computes in float16 by default.
- ResNet-18 converted from ONNX with `mnnconvert` agrees with PyTorch to 1e-4 on the logits (the other backends here agree to 5e-6; MNN
  fuses and optimises layers when it converts), through `scripts/export_model.py`.
- Cross-compiled with the Android NDK (r30, `arm64-v8a`, `c++_static`): the plugin and `run_model` built. It was **not run on a device**:
  the Qualcomm board's session expired before the run. OpenCL, Vulkan and CUDA were not run anywhere.
