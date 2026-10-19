<!-- SPDX-License-Identifier: Apache-2.0 -->
# Paddle Lite backend (Baidu's mobile and edge engine)

Source: `src/backends/paddlelite.cpp`, a C++ plugin behind the C backend ABI. Plugin: `libuairt_backend_paddlelite.so`.

This wraps **Paddle Lite**, the engine for phones and boards (Arm CPUs, and GPUs and NPUs through its own backends). It does not wrap
**Paddle Inference**, the server engine inside the `paddlepaddle` package: that package ships no inference API headers and keeps the engine
inside its Python extension, so there is nothing to link against.

## You install

Paddle Lite (Apache-2.0), built by you; the repository does not carry it, because building it needs steps and a patch described below. The plugin
needs its published `cxx` directory: `include/paddle_api.h` and `lib/libpaddle_api_light_bundled.a`. Build it **with exceptions on**
(`LITE_WITH_EXCEPTION=ON`, which makes a model error a C++ exception instead of aborting the process) and with the **same C++ toolchain** as the plugin.

The recipe below built Paddle Lite `v2.14-rc` for macOS arm64 on an Apple M5 with Xcode 27 and CMake 4. Its own `lite/tools/build_macos.sh`
does nothing useful in that tag (with arguments it parses them and exits; without, it calls a function that does not exist), so these are the CMake
commands from inside the script:

```bash
git clone --depth 1 --branch v2.14-rc https://github.com/PaddlePaddle/Paddle-Lite && cd Paddle-Lite
git submodule update --init --recursive --depth 1
mkdir -p build/lite/gen_code && touch build/lite/gen_code/__generated_code__.cc && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DWITH_TESTING=OFF -DLITE_WITH_ARM=ON -DLITE_ON_TINY_PUBLISH=ON -DLITE_WITH_OPENMP=OFF \
  -DWITH_ARM_DOTPROD=ON -DLITE_WITH_X86=OFF -DLITE_WITH_M1=ON -DLITE_WITH_PYTHON=OFF -DLITE_WITH_LOG=ON -DLITE_WITH_EXCEPTION=ON \
  -DLITE_BUILD_EXTRA=OFF -DLITE_WITH_CV=OFF -DARM_TARGET_ARCH_ABI=armv8 -DARM_TARGET_OS=armmacos -DDEPLOYMENT_TARGET=11.0
make publish_inference -j8            # about a minute; the result is build/inference_lite_lib.armmacos.armv8/cxx
```

A model is a `.nb` file, made from a Paddle model by Paddle Lite's `opt` tool. On Linux x86_64 and a few other platforms `pip install paddlelite`
gives `paddle_lite_opt`; there is no wheel for macOS arm64, so build `opt` from the same checkout:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5       # reaches the old CMake projects of its dependencies
# Xcode's libc++ rejects one non-const comparator in the old protobuf it builds for the host (it is in a Java generator, unused):
sed -i '' 's/bool operator ()(const FieldDescriptor\* f1, const FieldDescriptor\* f2) {/bool operator ()(const FieldDescriptor* f1, const FieldDescriptor* f2) const {/' \
  third-party/protobuf-host/src/google/protobuf/compiler/java/java_file.cc
mkdir -p build.opt/lite/gen_code && touch build.opt/lite/gen_code/__generated_code__.cc && cd build.opt
cmake .. -DLITE_ON_MODEL_OPTIMIZE_TOOL=ON -DWITH_TESTING=OFF -DLITE_BUILD_EXTRA=ON -DLITE_WITH_X86=OFF -DWITH_MKL=OFF
make opt -j8
lite/api/opt --model_file=m.pdmodel --param_file=m.pdiparams --optimize_out_type=naive_buffer --optimize_out=m --valid_targets=arm   # writes m.nb
```

Paddle 3.x saves models in a new format by default; Paddle Lite reads the older `.pdmodel`. Make one with `FLAGS_enable_pir_api=0` set before
`import paddle`, then `paddle.jit.save(net, "m", input_spec=[paddle.static.InputSpec([1, 3, 224, 224], "float32", "x")])`. A model without any
parameter has no `.pdiparams` file, which `opt` requires.

## Build

```bash
export PADDLE_LITE_ROOT=/path/to/Paddle-Lite/build/inference_lite_lib.armmacos.armv8/cxx
cmake -S . -B build -DUAIRT_BUILD_PADDLELITE=ON && cmake --build build
```

Paddle Lite registers its operators and kernels from static initializers, which a normal link drops; the build therefore links the whole archive
(`-force_load` on Apple, `--whole-archive` elsewhere) and hides its symbols so they cannot clash with the application's own copy.

## Run

```bash
build/run_model --option input_shapes=1x3x224x224 --info build/libuairt_backend_paddlelite.so paddlelite m.nb
build/run_model --repeat 100 --option input_shapes=1x3x224x224 --option num_threads=4 build/libuairt_backend_paddlelite.so paddlelite m.nb out in0.bin
```

Engine options:

- `input_shapes` (required): a `.nb` file records no input shapes. One shape per input in the model's input order, separated by `;`, such as
  `1x3x224x224` or `1x4;1x4`. Output shapes are found by running the model once on zeros at load.
- `num_threads`: CPU threads.
- `power_mode`: `high`, `low`, `full`, `no_bind` (default), `rand_high` or `rand_low`: which cores Paddle Lite binds its threads to.
- `precision`: `normal` or `low` (float16 kernels, only in a Paddle Lite built with `LITE_WITH_ARM82_FP16`).

## Behaviour to know

- Inputs and outputs keep the model's order. Output names are what Paddle called them (`tmp_0`, `linear_0.tmp_1`), so read them with `--info`.
- float32 tensors only in this version; an output of another type (for example an argmax in int64) is refused at load.
- A file that is not a `.nb` model is `UAIRT_ERR_INCOMPATIBLE_MODEL`, a missing file `UAIRT_ERR_IO`.
- A Paddle Lite built without `LITE_WITH_EXCEPTION=ON` aborts the process on a model error instead of returning one.
- If a model fails with "can not create operator 'feed'" the kernels were dropped by the linker (see Build), or the library was built
  `tiny_publish` without the operators the model needs (`--with_extra`).
- **Apple platforms:** the linker warns that Paddle Lite's assembly kernels clobber register `x18`, which Apple reserves for the platform. Every
  model run here was correct, but Apple does not promise that; use Android or Linux on Arm for anything that matters.

## Limits

Host memory only; static shapes; float32; no zero-copy buffers; CPU (Paddle Lite's OpenCL, Metal, NPU and Huawei, MediaTek and Rockchip delegates are
not exposed, and none was run).

## Check

`ctest` runs `paddlelite`, which loads `tests/data/add_mul.nb` (a Paddle layer returning `x + y` and `x * y`, converted with `opt`) and checks the
error paths, the order and names of the tensors, a run with two threads from a path and from memory, and the shape-count check.
For the comparison with Paddle Lite's own API set `UAIRT_PADDLELITE_TEST_MODEL` (a `.nb` file) and `UAIRT_PADDLELITE_TEST_SHAPES` before cmake:
it builds `tests/paddlelite_reference.cpp`, which runs the model through Paddle Lite's C++ API, and `tests/compare_with_paddlelite.py` wants
byte-identical outputs for each engine option, alone and together.

Run here, on an Apple M5 (macOS arm64), Paddle Lite `v2.14-rc`, PaddlePaddle 3.3.1:

- CPU: byte-identical to Paddle Lite's own C++ API on a small convolutional net and on ResNet-18, for five option combinations each.
- End to end, from the Paddle model through `opt` and UAIRT, against PaddlePaddle's own output on the same input: 5e-7 on the small net and 1e-4 on
  ResNet-18 (random weights; logits of size 160, so 6e-7 relative). ResNet-18 ran in 23 ms.
- **Not run:** Android and Linux (not built), the GPU and NPU delegates, models with non-float32 tensors, and anything with Paddle Inference.
