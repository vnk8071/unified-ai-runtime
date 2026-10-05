#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds ONNX Runtime v1.24.4 with the QNN execution provider (shared provider library) inside the
# tool image. Mount your QAIRT SDK read-only at /qnn and a volume at /work:
#   docker build -t ort-builder tests/bench/ort-build
#   docker run --rm -v ortbuild:/work -v $PWD/tests/bench/ort-build/build_ort.sh:/build_ort.sh:ro \
#     -v $QNN_SDK_ROOT:/qnn:ro ort-builder bash /build_ort.sh
# The result is /work/out/libonnxruntime.so* and /work/build/Release/libonnxruntime_providers_*.so.
set -euo pipefail
cd /work
if [ ! -d onnxruntime ]; then
  git clone --depth 1 --branch v1.24.4 https://github.com/microsoft/onnxruntime onnxruntime
fi
cd onnxruntime
./build.sh --allow_running_as_root --config Release --build_shared_lib --parallel 5 --skip_tests \
  --use_qnn --qnn_home /qnn --cmake_generator Ninja --compile_no_warning_as_error \
  --build_dir /work/build --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5
mkdir -p /work/out
cp -a /work/build/Release/libonnxruntime.so* /work/out/
cp -a include/onnxruntime/core/session/onnxruntime_c_api.h /work/out/ 2>/dev/null || true
ls -la /work/out
echo BUILD_DONE
