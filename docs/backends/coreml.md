<!-- SPDX-License-Identifier: Apache-2.0 -->
# CoreML backend

Source: `src/backends/coreml.m`. Plugin: `libuairt_backend_coreml.so` (a dylib on macOS).

## You install

Xcode (provides `CoreML.framework` and `xcrun coremlcompiler`) and accept its license.
Apple platforms only.

## Build

```bash
cmake -S . -B build -DUAIRT_BUILD_COREML=ON && cmake --build build
```

## Run

```bash
build/run_model --option compute_units=all --info build/libuairt_backend_coreml.so coreml model.mlmodelc
```

Accepts `.mlmodelc`, `.mlpackage` and `.mlmodel`. The engine option `compute_units` is
`all`, `cpu_only`, `cpu_and_gpu` or `cpu_and_ne`.

## Check

`ctest --test-dir build --output-on-failure`. The comparison with CoreML itself needs Python with
`coremltools`, `numpy` and `pillow`; use a throwaway virtualenv and set `UAIRT_COREML_PYTHON` to its
interpreter before running cmake.
