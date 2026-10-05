# Setup workflow for agents

Goal: get a requested backend building and passing tests. Logic lives in
`scripts/`; this file only sequences it.

1. Run `scripts/doctor.sh` and read every line.
2. If `core` items are missing, tell the user what to install. Do not use `sudo`.
3. For the requested backend, if doctor reports `skip` or `missing`:
   - give the user the vendor download page and the environment variable to set
     (`QNN_SDK_ROOT`, `ONNXRUNTIME_ROOT`, `TENSORRT_ROOT`, or Xcode);
   - the user downloads the SDK and accepts its license themselves; never do it
     for them and never place SDK files inside this repository;
   - rerun `scripts/doctor.sh` until the backend is `ok`.
4. Build and test: `cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure`.
5. Report the result with the actual command output. If something failed, say so.

## ONNX Runtime

- Needs a release directory with `include/onnxruntime_c_api.h` and `lib/libonnxruntime`.
  The user downloads and unpacks it (MIT licensed); export it as `ONNXRUNTIME_ROOT`.
- The pip wheel is not enough: it has the library but no headers.
- Configure with `-DUAIRT_BUILD_ONNXRUNTIME=ON`. The plugin is built against the
  header version; it fails to load with a clear error if the installed library is older.
- `tests/data/make_models.py` regenerates the test models and checks them against
  Python ONNX Runtime (needs `onnx`, `onnxruntime`, `numpy`).
- GPU: use a GPU release whose CUDA major version matches the installed toolkit (for CUDA 12, the `gpu_cuda12` tarball)
  and cuDNN 9. Run with `execution_provider=cuda` or `tensorrt` (the latter also needs TensorRT and `libnvonnxparser`),
  and set `UAIRT_TEST_ORT_PROVIDER` for `ctest`. Confirm placement with `log_level=verbose`. The user installs these;
  see `docs/backends/onnxruntime.md`.

- Windows: configure with `-G "Visual Studio 17 2022"` and run `ctest -C Debug`. The unpacked release needs
  `lib/onnxruntime.lib` (import library) next to `onnxruntime.dll`. Model and plugin paths are converted to UTF-16.

## ONNX Runtime plugin execution providers

- A plugin execution provider needs an ONNX Runtime core of the version it targets (for MLX,
  `onnxruntime-ep-mlx` 0.29.x needs ONNX Runtime 1.29.x). Download that ONNX Runtime release yourself,
  build UAIRT against its headers (`ONNXRUNTIME_ROOT`), and get the provider library (for MLX, the
  `onnxruntime-ep-mlx` wheel on PyPI bundles `libonnxruntime_mlx_ep.dylib` and the MLX runtime; keep the files
  together). Neither is part of this repository.
- To run the plugin test under `ctest`, set `UAIRT_TEST_ORT_EP_LIBRARY` and `UAIRT_TEST_ORT_EP_NAME`.
  `tests/compare_ort_eps.py` compares the CPU EP with the plugin (needs `torch` and `torchvision`).

## CoreML

- Apple platforms only; needs Xcode (`CoreML.framework`, and `xcrun coremlcompiler` for
  the comparison test). The user installs Xcode and accepts its license.
- Configure with `-DUAIRT_BUILD_COREML=ON`. Engines take the optional `compute_units`
  (`all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_ne`).
- The comparison test needs `coremltools` and `numpy`. Do not install them into the
  user's own environments without asking; a throwaway virtualenv is fine. Set
  `UAIRT_COREML_PYTHON` to that interpreter before running cmake.

## TFLite

- You build TFLite's C library for the target (the repository does not). Set `TFLITE_INCLUDE_DIR`
  (contains `tensorflow/lite/c/c_api.h`) and `TFLITE_LIBRARY`, and configure with
  `-DUAIRT_BUILD_TFLITE=ON`. Set `QNN_SDK_ROOT` too if you want `delegate=qnn`.
- A library built on a newer system may need a newer glibc or libstdc++ than the board has; build it
  on the oldest system you target.
- To compare with TFLite itself, build `tests/bench/tflite_bench.c` (see `tests/bench/README.md`)
  and set `UAIRT_TFLITE_BENCH` and `UAIRT_TFLITE_TEST_MODEL` before running cmake, then `ctest`.

## TensorRT

- NVIDIA GPU, driver and CUDA toolkit, plus TensorRT 8.5+ with headers. The user installs TensorRT under NVIDIA's licence;
  never run `pip install tensorrt` or an installer for them. Set `TENSORRT_ROOT` (include/NvInfer.h, lib/libnvinfer).
- Configure with `-DUAIRT_BUILD_TENSORRT=ON`. Engines must be built on the GPU that runs them; see `docs/backends/tensorrt.md`.

## QNN (Qualcomm AI Runtime)

- The user installs QAIRT from Qualcomm and accepts its license; set `QNN_SDK_ROOT`
  to the SDK directory (it must contain `include/QNN/QnnInterface.h`).
- The SDK has no macOS libraries. Build and run on a Linux or Windows host or
  device that has `lib/<target>/libQnnCpu.so` and `libQnnSystem.so`.
- Configure with `-DUAIRT_BUILD_QNN=ON`. Engines need the options `backend_library`
  and `system_library` (absolute paths to `libQnnCpu.so` and `libQnnSystem.so`).
- To compare against the SDK's `qnn-net-run`, set `UAIRT_QNN_TEST_DLC=/path/model.dlc` (or a
  `.bin` context binary), `UAIRT_QNN_TEST_BACKEND=cpu|htp` (default `cpu`) and optionally
  `UAIRT_QNN_TARGET` (default `aarch64-ubuntu-gcc9.4`) before running cmake, then `ctest`.
  The test needs Python 3 with `numpy`. For HTP, `LD_LIBRARY_PATH` and the Hexagon skel
  directory must be reachable (the test script sets both).
- Windows ARM64 (Snapdragon X): use QAIRT 2.45 or newer (`QnnSystemDlc.h` is missing from 2.31) and the default
  target `aarch64-windows-msvc`. For the NPU set `UAIRT_QNN_TEST_BACKEND=htp` and `UAIRT_QNN_DSP_ARCH=v73`
  (X Elite), and run `ctest -C Debug`. Pass Windows-style absolute paths to the script when running it by hand.
- `device=npu|cpu|gpu` replaces `backend_library`, `system_library` and `adsp_library_path`: it takes them from
  `sdk_root` (default `QNN_SDK_ROOT`) and `hexagon_arch`. `cache_dir` stores the compiled context of a `.dlc`
  on HTP so later loads skip composing it (3.1 s to 0.4 s for `fastsam_s` on X Elite). The key covers the
  model path, size, mtime and backend library. Cache writes are off on Windows by default because another
  runtime reported heap corruption in `QnnContext_getBinary` on QAIRT 2.45 arm64x; enable with
  `cache_write=true` after checking. It was stable over repeated runs here.
- Zero-copy (`--dmabuf`, `uairt_buffer_alloc`) needs `libcdsprpc.so` (FastRPC) on the device and
  the HTP backend; pass `--dmabuf` to `compare_with_qnn_net_run.py` to test it.
- Producing a DLC needs `qairt-converter`, which runs on x86_64 Linux only.

Backend-specific steps for the other backends are added here as they land.
