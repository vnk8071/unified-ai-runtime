<!-- SPDX-License-Identifier: Apache-2.0 -->
# NCNN backend (CPU, and any Vulkan GPU)

Source: `src/backends/ncnn.cpp`, a C++ plugin behind the C backend ABI. Plugin: `libuairt_backend_ncnn.so`.

## You install

NCNN (BSD-3-Clause), built from https://github.com/Tencent/ncnn with `-DNCNN_VULKAN=ON` for the GPU path, and installed to a
prefix. Build it with the **same C++ toolchain** as this plugin: the plugin uses NCNN's C++ API, because NCNN's C API cannot
tell whether a GPU is usable. The Vulkan path also needs a Vulkan driver (on NVIDIA, the graphics libraries of the driver
package, for example `libnvidia-gl-<version>`; many containers lack them).

```bash
git clone --depth 1 --recurse-submodules https://github.com/Tencent/ncnn.git && cd ncnn
cmake -S . -B build -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/ncnn-install
cmake --build build -j && cmake --install build
```

## Build

```bash
export NCNN_ROOT=$HOME/ncnn-install        # contains include/ncnn/net.h and lib/libncnn
cmake -S . -B build -DUAIRT_BUILD_NCNN=ON && cmake --build build
```

## Run

The model is a `.param` file (with the `.bin` of the same stem) or a directory that holds one; an ultralytics
`YOLO(...).export(format="ncnn")` directory loads as is. NCNN's `.param` records no input shapes, so give them:

```bash
build/run_model --info --option input_shapes=3x640x640 build/libuairt_backend_ncnn.so ncnn yolov8n_ncnn_model
build/run_model --repeat 100 --option input_shapes=3x640x640 --option device=vulkan --option fp16=false \
  build/libuairt_backend_ncnn.so ncnn yolov8n_ncnn_model out in0.bin
```

Engine options:

- `input_shapes` (required): one shape per model input, in input order, separated by `;`. Shapes follow NCNN's layout with no
  batch dimension: `w`, `h x w`, `c x h x w` or `c x d x h x w`. Outputs are found by running the model once on zeros at load.
- `device`: `cpu` (default) or `vulkan`. `vulkan` fails with `UAIRT_ERR_BACKEND_UNAVAILABLE` when NCNN finds no usable
  Vulkan device, where NCNN alone would quietly run on the CPU. `vulkan_device` picks the GPU (default 0).
- `fp16`: `true` or `false`. NCNN's default uses fp16 on the GPU, which changes results noticeably (see below).
- `num_threads`.

## Limits

float32 tensors only; static shapes; host memory; models from a path, not from memory.

## Check

`ctest` runs `ncnn_errors`. For the comparison with NCNN's Python API (CPU reference) set `UAIRT_NCNN_TEST_MODEL`,
`UAIRT_NCNN_TEST_SHAPES` (for example `3x640x640`) and `UAIRT_NCNN_PYTHON` (a Python with `ncnn` and numpy) before cmake.

## Result

yolov8n exported by ultralytics 8.4.173, input 3x640x640, on a GTX 1650 (Vulkan 1.4, driver 580.178) and an Intel Core
i9-9900K, NCNN built from source (master):

| Device | Median per run | Difference from the CPU output |
|---|---|---|
| CPU | 79 ms | matches NCNN's Python API within 1.3e-3 |
| Vulkan, `fp16=false` | 21.9 ms | at most 1.9e-3 (mean 2.5e-6) |
| Vulkan, default fp16 | 21.6 ms | at most 7.6 on values up to about 600 (mean 0.008) |

The first Vulkan load takes about 12 s while NCNN builds its pipelines. All three find the same detections on a test image
(the fp16 run scores the bus 0.83 instead of 0.84), and fp16 was no faster than fp32 on this GPU, so the tests use
`fp16=false`.
