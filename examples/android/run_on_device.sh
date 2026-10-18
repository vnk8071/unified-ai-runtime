#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Cross-compiles UAIRT with the Android NDK, pushes it to a device over adb and runs the checks and examples there.
#
#   ANDROID_NDK_HOME=/path/to/ndk examples/android/run_on_device.sh [--gguf model.gguf] [--serial SERIAL]
#
# Always runs: quickstart, the core tests, the generation tests. With --gguf (a small llama.cpp model) it also builds the
# llamacpp backend with dotprod+i8mm and generates text on the device. Needs adb with one device attached (a Qualcomm Device
# Cloud tunnel works: ssh -L 5037:<device>:5037 ..., then adb devices).
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to the Android NDK directory}"
gguf=""
serial=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gguf) gguf="$2"; shift 2 ;;
    --serial) serial=(-s "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
adb_() { adb ${serial[@]+"${serial[@]}"} "$@"; }
adb_ get-state >/dev/null || { echo "no adb device" >&2; exit 1; }

abi="$(adb_ shell getprop ro.product.cpu.abi | tr -d '\r')"
[ "$abi" = "arm64-v8a" ] || { echo "this script builds arm64-v8a; the device is $abi" >&2; exit 1; }
echo "device: $(adb_ shell getprop ro.product.model | tr -d '\r'), Android $(adb_ shell getprop ro.build.version.release | tr -d '\r')"

build="$root/build-android-example"
extra=()
[ -n "$gguf" ] && extra=(-DUAIRT_BUILD_LLAMACPP=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm
                         -DANDROID_STL=c++_static)
cmake -S "$root" -B "$build" -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_STL=none ${extra[@]+"${extra[@]}"} >/dev/null
cmake --build "$build" -j8 >/dev/null

dir=/data/local/tmp/uairt-example
adb_ shell "rm -rf $dir && mkdir -p $dir"
files=(libuairt.so libuairt_llm.so quickstart test_core test_session test_llm test_llm_utf8 test_global_symbols
       uairt_reference_plugin.so uairt_session_test_1.so uairt_session_test_2.so uairt_session_test_3.so uairt_session_test_4.so
       uairt_global_symbols_test_0.so uairt_global_symbols_test_1.so)
(cd "$build" && adb_ push -q "${files[@]}" "$dir/")
if [ -n "$gguf" ]; then  # llama.cpp needs the C++ library, which then also links into libuairt.so
  libcxx="$(find "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" -name libc++_shared.so -path '*aarch64-linux-android*' | head -1)"
  adb_ push -q "$libcxx" "$dir/"
fi

run() {  # name command...: runs on the device, prints PASS/FAIL and the last line
  local name="$1"; shift
  local out
  out="$(adb_ shell "cd $dir && LD_LIBRARY_PATH=$dir $* 2>&1; echo __exit=\$?" | tr -d '\r')"
  if [ "$(printf '%s' "$out" | tail -1)" = "__exit=0" ]; then
    printf 'PASS  %-24s %s\n' "$name" "$(printf '%s' "$out" | tail -2 | head -1)"
  else
    printf 'FAIL  %-24s\n%s\n' "$name" "$out"; failed=1
  fi
}
failed=0
run quickstart ./quickstart
run core ./test_core uairt_reference_plugin.so
run global_symbols_opt_out ./test_global_symbols uairt_global_symbols_test_0.so 0
run global_symbols_opt_in ./test_global_symbols uairt_global_symbols_test_1.so 1
run session ./test_session uairt_session_test_1.so uairt_session_test_2.so uairt_session_test_3.so uairt_session_test_4.so
run generation ./test_llm uairt_session_test_1.so uairt_session_test_4.so
run generation_utf8 ./test_llm_utf8

if [ -n "$gguf" ]; then
  (cd "$build" && adb_ push -q libuairt_backend_llamacpp.so llm_stream "$dir/")
  adb_ push -q "$gguf" "$dir/model.gguf"
  echo "--- llm_stream on the device (greedy, 30 tokens):"
  adb_ shell "cd $dir && LD_LIBRARY_PATH=$dir ./llm_stream ./libuairt_backend_llamacpp.so model.gguf 'The capital of France is' 30" | tr -d '\r'
fi
adb_ shell "rm -rf $dir"
exit $failed
