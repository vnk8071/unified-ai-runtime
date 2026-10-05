#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds the shared library and runs the tests of every binding whose toolchain is installed.
# Set ONNXRUNTIME_ROOT to also build the ONNX Runtime plugin and run the two-input/two-output tests.
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

args=(-DBUILD_SHARED_LIBS=ON)
[[ -n "${ONNXRUNTIME_ROOT:-}" ]] && args+=(-DUAIRT_BUILD_ONNXRUNTIME=ON)
cmake -S . -B build-shared "${args[@]}" >/dev/null && cmake --build build-shared >/dev/null || { echo "build failed"; exit 1; }

ext=so; [[ "$(uname -s)" == "Darwin" ]] && ext=dylib
export UAIRT_LIBRARY="$root/build-shared/libuairt.$ext"
[[ -f "build-shared/libuairt_backend_onnxruntime.so" ]] && export UAIRT_TEST_ORT_PLUGIN="$root/build-shared/libuairt_backend_onnxruntime.so"

failed=0
run() {
  local name="$1"; shift
  if "$@" >/tmp/uairt_binding_$name.log 2>&1; then echo "ok       $name"; else echo "FAILED   $name (see /tmp/uairt_binding_$name.log)"; failed=1; fi
}
skip() { echo "skipped  $1 ($2 not found)"; }

command -v python3 >/dev/null && python3 -c "import numpy, pytest" 2>/dev/null \
  && run python bash -c "cd bindings/python && python3 -m pytest -q tests" || skip python "python3 with numpy and pytest"
command -v cargo >/dev/null && run rust bash -c "cd bindings/rust && cargo test --offline" || skip rust cargo
if command -v c++ >/dev/null; then
  run cpp bash -c "cmake -S . -B build-shared >/dev/null && cmake --build build-shared --target test_cpp >/dev/null && ./build-shared/test_cpp ${UAIRT_TEST_ORT_PLUGIN:-} ${UAIRT_TEST_ORT_PLUGIN:+tests/data/add_mul.onnx}"
else
  skip cpp "a C++ compiler"
fi
exit $failed
