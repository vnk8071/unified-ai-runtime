#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reports the target device, build prerequisites and which backends can be built here.
# Output lines are "<status> <item>: <detail>" with status ok | missing | skip | info.
# Exits non-zero only when core prerequisites are missing.
#
#   scripts/doctor.sh [--target native|windows]
#
# --target windows says the build is for Windows: the check fails when this shell is not a Windows shell (WSL, Linux
# and macOS all report their own platform, which would mislead setup decisions). Use Git Bash, or scripts/doctor.ps1
# from PowerShell. WSL is always reported, because it is Linux even though it runs on a Windows machine.
set -uo pipefail

core_missing=0
target=native
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) target="${2:-}"; shift 2 ;;
    --target=*) target="${1#*=}"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [[ "$target" != native && "$target" != windows ]]; then
  echo "--target must be native or windows" >&2
  exit 2
fi

report() { printf '%-8s %s\n' "$1" "$2"; }

have() { command -v "$1" >/dev/null 2>&1; }

windows=0
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) windows=1 ;; esac

wsl=0
if [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then wsl=1; fi

ps() { powershell.exe -NoProfile -Command "$1" 2>/dev/null | tr -d '\r'; }

# Vendor versions: scripts/vendors.tsv holds the tested and minimum versions and the download pages (docs/vendors.md).
vendors_file="$(dirname "${BASH_SOURCE[0]}")/vendors.tsv"
vendor_field() { awk -F'\t' -v b="$1" -v c="$2" '$1 == b && $c != "-" { print $c }' "$vendors_file" 2>/dev/null; }
version_lt() { [[ "$1" != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$1" ]]; }
# Where to get a backend's SDK; printed when it is missing, so the user (never this script) downloads it.
vendor_hint() {
  local url min
  url="$(vendor_field "$1" 7)"; min="$(vendor_field "$1" 5)"
  [[ -n "$url" ]] && report info "$1: get it from $url${min:+ (minimum $min)}; you install it and accept its licence"
}
# A version taken from an install directory's name, such as .../QAIRT/2.45.0.260326 (empty when it has none).
dir_version() {
  local d="${1//\\//}"
  d="$(basename "${d%/}")"
  [[ "$d" =~ ^[0-9]+(\.[0-9]+)+ ]] && echo "${BASH_REMATCH[0]}"
  return 0
}
# A pip-installed package, found from its metadata without importing it (importing openvino starts a telemetry subprocess):
# prints "<version> <package directory>", or returns 1. Uses the first python3 or python on PATH, so an active virtualenv counts.
pip_package() {
  local py
  for py in python3 python; do have "$py" && break; done
  have "$py" || return 1
  "$py" - "$1" <<'PY' 2>/dev/null
import importlib.metadata as metadata
import importlib.util as util
import sys

name = sys.argv[1]
try:
    version = metadata.version(name)
except metadata.PackageNotFoundError:
    sys.exit(1)
spec = util.find_spec(name)
directory = spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else ""
print(version, directory)
PY
}

# Compares a detected version with the manifest. Older than the minimum is reported as missing; a version other than
# the tested one is information, not an error, and means "not verified here".
vendor_check() {
  local have="$2" tested min
  tested="$(vendor_field "$1" 4)"; min="$(vendor_field "$1" 5)"
  if [[ -z "$have" ]]; then
    report info "$1: version not detected${tested:+ (tested with $tested)}"
  elif [[ -n "$min" ]] && version_lt "$have" "$min"; then
    report missing "$1: version $have is older than the minimum $min"
  elif [[ -n "$tested" && "$tested" != submodule && "$have" != "$tested"* ]]; then
    report info "$1: version $have; tested with $tested, so other versions are unverified"
  else
    report info "$1: version $have"
  fi
}

# Maps a chip name to the Hexagon architecture the QNN skel directory is named after. A hint, not a fact.
hexagon_hint() {
  case "$1" in
    *X1E*|*X1P*|*"X Elite"*|*"X Plus"*|*QCS8550*|*"8 Gen 2"*|*SM8550*) echo v73 ;;
    *"8 Gen 3"*|*SM8650*) echo v75 ;;
    *"8 Elite"*|*SM8750*|*X2E*) echo v79 ;;
    *QCS6490*|*QCS5430*|*SM7325*) echo v68 ;;
    *) echo "" ;;
  esac
}

echo "== device"
report info "os: $(uname -s) $(uname -m)"
if [[ $wsl == 1 ]]; then
  report info "wsl: this shell is WSL (${WSL_DISTRO_NAME:-unknown distro}); it is Linux, not Windows, and cannot use the Windows NPU, GPU or Visual Studio"
fi
if [[ "$target" == windows && $windows == 0 ]]; then
  report missing "target: Windows requested but this shell is $(uname -s)$([[ $wsl == 1 ]] && echo " (WSL)"); run from Git Bash or use scripts/doctor.ps1 in PowerShell"
  core_missing=1
fi
if [[ $windows == 1 || "$target" == windows ]]; then
  report info "presets: cmake --preset windows-arm64-debug | windows-x64-debug (then cmake --build --preset ... and ctest --preset ...)"
fi
chip=""
npu=""
if [[ $windows == 1 ]]; then
  chip="$(ps '(Get-CimInstance Win32_Processor | Select-Object -First 1).Name')"
  npu="$(ps "Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { \$_.FriendlyName -match '\\bNPU\\b' -and \$_.Status -eq 'OK' } | Select-Object -First 1 -ExpandProperty FriendlyName")"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
else
  [[ -r /sys/firmware/devicetree/base/model ]] && chip="$(tr -d '\0' < /sys/firmware/devicetree/base/model)"
  [[ -z "$chip" ]] && chip="$(grep -m1 -i 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
  for node in /dev/fastrpc-* /dev/adsprpc-smd* /dev/accel/accel*; do
    [[ -e "$node" ]] && npu="$node" && break
  done
fi
[[ -n "$chip" ]] && report info "chip: $chip"
if [[ -n "$npu" ]]; then report ok "npu: $npu"; else report skip "npu: none detected"; fi
if have nvidia-smi; then
  gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
  [[ -n "$gpu" ]] && report ok "gpu: $gpu (name, driver, memory, compute capability)"
  have nvcc && report info "cuda toolkit: $(nvcc --version | tail -1)"
fi
if have vulkaninfo; then
  vulkan="$(vulkaninfo --summary 2>/dev/null | grep -m1 "deviceName" | sed 's/.*= *//')"
  if [[ -n "$vulkan" ]]; then report ok "vulkan: $vulkan"; else report skip "vulkan: vulkaninfo finds no device (the Vulkan driver may be missing)"; fi
fi
arch="$(hexagon_hint "$chip")"
[[ -n "$arch" ]] && report info "hexagon arch hint: $arch (QNN hexagon_arch option; verify against your chip)"

echo "== core"
for tool in cmake git; do
  if have "$tool"; then report ok "$tool: $("$tool" --version | head -1)"
  else report missing "$tool: not found"; core_missing=1; fi
done
vswhere="${PROGRAMFILES_X86:-/c/Program Files (x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
if have cc; then report ok "cc: $(cc --version | head -1)"
elif [[ -x "$vswhere" ]] \
     && vs="$("$vswhere" -latest -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName 2>/dev/null)" && [[ -n "$vs" ]]; then
  report ok "cc: $vs (use: cmake -G \"Visual Studio 17 2022\", or a preset)"
elif [[ -x "$vswhere" ]] \
     && vs="$("$vswhere" -latest -version '[18.0,19.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName 2>/dev/null)" && [[ -n "$vs" ]]; then
  report ok "cc: $vs (the presets name Visual Studio 2022; add -G \"Visual Studio 18 2026\" to the preset configure)"
elif [[ -x "$vswhere" ]] && other="$("$vswhere" -latest -products '*' -property displayName 2>/dev/null)" && [[ -n "$other" ]]; then
  report missing "cc: found $other, but the presets and docs use Visual Studio 2022 (generator 'Visual Studio 17 2022') with the C++ tools; install it or pass your own generator"; core_missing=1
else report missing "cc: no C compiler"; core_missing=1; fi

echo "== backends (SDKs are installed by you under their own licenses)"

qnn_root="${QNN_SDK_ROOT:-}"
found_hint=""
if [[ -z "$qnn_root" ]]; then
  for pattern in /c/Qualcomm/AIStack/QAIRT/* /opt/qcom/aistack/qairt/* "$HOME"/qairt/* "$HOME"/Qualcomm/AIStack/QAIRT/*; do
    [[ -f "$pattern/include/QNN/QnnInterface.h" ]] && found_hint="$pattern"
  done
fi
if [[ -n "$qnn_root" ]]; then
  if [[ ! -f "$qnn_root/include/QNN/QnnInterface.h" ]]; then
    report missing "qnn: QNN_SDK_ROOT set but include/QNN/QnnInterface.h not found"
  elif [[ ! -f "$qnn_root/include/QNN/System/QnnSystemDlc.h" ]]; then
    report missing "qnn: $qnn_root is too old: it lacks System/QnnSystemDlc.h (the DLC API); use a newer QAIRT"
    vendor_check qnn "$(dir_version "$qnn_root")"
  else
    report ok "qnn: $qnn_root"
    vendor_check qnn "$(dir_version "$qnn_root")"
  fi
  # Another QAIRT next to the selected one that is newer: the usual cause of "too old" when a newer one is installed.
  qnn_dir="${qnn_root//\\//}"
  qnn_have="$(dir_version "$qnn_dir")"
  qnn_best="" qnn_best_dir=""
  for candidate in "$(dirname "${qnn_dir%/}")"/*/; do
    candidate_version="$(dir_version "$candidate")"
    [[ -n "$candidate_version" && -f "$candidate/include/QNN/QnnInterface.h" ]] || continue
    if [[ -z "$qnn_best" ]] || version_lt "$qnn_best" "$candidate_version"; then qnn_best="$candidate_version"; qnn_best_dir="${candidate%/}"; fi
  done
  if [[ -n "$qnn_have" && -n "$qnn_best" ]] && version_lt "$qnn_have" "$qnn_best"; then
    report info "qnn: a newer QAIRT $qnn_best is installed at $qnn_best_dir; set QNN_SDK_ROOT to it to use it"
  fi
elif [[ -n "$found_hint" ]]; then
  report skip "qnn: found $found_hint; set QNN_SDK_ROOT to it (newer QAIRT releases are needed for .dlc models)"
else
  report skip "qnn: set QNN_SDK_ROOT to your QAIRT SDK directory"
  vendor_hint qnn
fi

if [[ -n "${ONNXRUNTIME_ROOT:-}" ]]; then
  if [[ -f "$ONNXRUNTIME_ROOT/include/onnxruntime_c_api.h" ]] ||
     [[ -f "$ONNXRUNTIME_ROOT/include/onnxruntime/onnxruntime_c_api.h" ]]; then
    report ok "onnxruntime: $ONNXRUNTIME_ROOT"
    vendor_check onnxruntime "$(tr -d '[:space:]' < "$ONNXRUNTIME_ROOT/VERSION_NUMBER" 2>/dev/null)"
  else
    report missing "onnxruntime: header not found under ONNXRUNTIME_ROOT"
    vendor_hint onnxruntime
  fi
else
  # No release directory: the plugin needs only ONNX Runtime's header to build (scripts/ci/fetch_onnxruntime_headers.sh puts it in
  # .ort/) and loads the library at run time, which a pip install provides.
  ort_header="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.ort/include/onnxruntime_c_api.h"
  ort_pip="$(pip_package onnxruntime || true)"
  ort_version="${ort_pip%% *}"
  ort_dir="${ort_pip#* }"
  if [[ -f "$ort_header" && -n "$ort_pip" ]]; then
    report ok "onnxruntime: header in .ort/ and pip onnxruntime $ort_version; build with -DUAIRT_BUILD_ONNXRUNTIME=ON"
    ort_library="$(ls "$ort_dir"/capi/libonnxruntime.*dylib "$ort_dir"/capi/libonnxruntime.so* "$ort_dir"/capi/onnxruntime.dll 2>/dev/null | head -n1)"
    [[ -n "$ort_library" ]] && report info "onnxruntime: the Python binding finds that library by itself; for run_model or C programs, export UAIRT_ONNXRUNTIME_LIBRARY=$ort_library"
    vendor_check onnxruntime "$ort_version"
  elif [[ -n "$ort_pip" ]]; then
    report skip "onnxruntime: pip onnxruntime $ort_version is installed; run scripts/ci/fetch_onnxruntime_headers.sh for the header (or set ONNXRUNTIME_ROOT to a release)"
  elif [[ -f "$ort_header" ]]; then
    report skip "onnxruntime: the header is in .ort/ but no pip onnxruntime is installed; pip install onnxruntime (or set ONNXRUNTIME_ROOT to a release)"
  else
    report skip "onnxruntime: pip install onnxruntime and run scripts/ci/fetch_onnxruntime_headers.sh (or set ONNXRUNTIME_ROOT to a release)"
    vendor_hint onnxruntime
  fi
fi

if [[ -n "${OPENVINO_ROOT:-}" ]]; then
  if [[ -f "$OPENVINO_ROOT/include/openvino/c/openvino.h" ]] ||
     [[ -f "$OPENVINO_ROOT/runtime/include/openvino/c/openvino.h" ]]; then
    report ok "openvino: $OPENVINO_ROOT"
    vendor_check openvino "$(dir_version "$OPENVINO_ROOT")"
  else
    report missing "openvino: openvino/c/openvino.h not found under OPENVINO_ROOT"
    vendor_hint openvino
  fi
else
  ov_pip="$(pip_package openvino || true)"
  ov_version="${ov_pip%% *}"
  ov_dir="${ov_pip#* }"
  if [[ -n "$ov_pip" && -f "$ov_dir/include/openvino/c/openvino.h" ]]; then
    report ok "openvino: pip openvino $ov_version at $ov_dir; build with OPENVINO_ROOT=$ov_dir"
    vendor_check openvino "$ov_version"
  elif [[ -n "$ov_pip" ]]; then
    report skip "openvino: pip openvino $ov_version is installed but $ov_dir has no C headers; use a release and set OPENVINO_ROOT"
    vendor_hint openvino
  else
    report skip "openvino: pip install openvino (it ships the C headers), then run this script again: it prints the OPENVINO_ROOT to use"
    vendor_hint openvino
  fi
fi

if [[ -n "${NCNN_ROOT:-}" ]]; then
  if [[ -f "$NCNN_ROOT/include/ncnn/net.h" ]]; then
    report ok "ncnn: $NCNN_ROOT"
    vendor_check ncnn "$(sed -n 's/^#define NCNN_VERSION_STRING "\(.*\)"/\1/p' "$NCNN_ROOT/include/ncnn/platform.h" 2>/dev/null | head -n1)"
  else
    report missing "ncnn: include/ncnn/net.h not found under NCNN_ROOT"
    vendor_hint ncnn
  fi
else
  ncnn_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/ncnn"
  if [[ -f "$ncnn_src/CMakeLists.txt" ]]; then
    report ok "ncnn: $ncnn_src (pinned submodule, built with -DUAIRT_BUILD_NCNN=ON; set NCNN_ROOT to use an install instead)"
    ncnn_commit="$(git -C "$ncnn_src" describe --tags --always 2>/dev/null || true)"
    [[ -n "$ncnn_commit" ]] && report info "ncnn: version $ncnn_commit (a pinned submodule; bumping it needs ctest and a model run, see docs/vendors.md)"
  else
    report skip "ncnn: run git submodule update --init --depth 1 third_party/ncnn (or set NCNN_ROOT to an install)"
  fi
fi
if [[ -n "${MNN_ROOT:-}" ]]; then
  if [[ -f "$MNN_ROOT/include/MNN/Interpreter.hpp" ]]; then
    report ok "mnn: $MNN_ROOT"
    vendor_check mnn "$(awk '/#define MNN_VERSION_(MAJOR|MINOR|PATCH) /{v=v (v?".":"") $3} END{print v}' "$MNN_ROOT/include/MNN/MNNDefine.h" 2>/dev/null)"
  else
    report missing "mnn: include/MNN/Interpreter.hpp not found under MNN_ROOT"
    vendor_hint mnn
  fi
else
  mnn_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/MNN"
  if [[ -f "$mnn_src/CMakeLists.txt" ]]; then
    report ok "mnn: $mnn_src (pinned submodule, built with -DUAIRT_BUILD_MNN=ON; set MNN_ROOT to use an install instead)"
    mnn_commit="$(git -C "$mnn_src" describe --tags --always 2>/dev/null || true)"
    [[ -n "$mnn_commit" ]] && report info "mnn: version $mnn_commit (a pinned submodule; bumping it needs ctest and a model run, see docs/vendors.md)"
  else
    report skip "mnn: run git submodule update --init --depth 1 third_party/MNN (or set MNN_ROOT to an install)"
  fi
fi
if [[ -n "${PADDLE_LITE_ROOT:-}" ]]; then
  if [[ -f "$PADDLE_LITE_ROOT/include/paddle_api.h" ]]; then
    report ok "paddlelite: $PADDLE_LITE_ROOT"
    vendor_check paddlelite ""
    [[ -f "$PADDLE_LITE_ROOT/lib/libpaddle_api_light_bundled.a" ]] || report missing "paddlelite: lib/libpaddle_api_light_bundled.a not found under PADDLE_LITE_ROOT"
  else
    report missing "paddlelite: include/paddle_api.h not found under PADDLE_LITE_ROOT (it is the cxx directory of a Paddle Lite build)"
    vendor_hint paddlelite
  fi
else
  report skip "paddlelite: build Paddle Lite (docs/backends/paddlelite.md) and set PADDLE_LITE_ROOT to its cxx directory"
fi

if [[ -n "${WHISPER_ROOT:-}" ]]; then
  if [[ -f "$WHISPER_ROOT/include/whisper.h" ]]; then
    report ok "whispercpp: $WHISPER_ROOT"
    vendor_check whispercpp "$(sed -n 's/^set(WHISPER_VERSION[[:space:]]*\(.*\))/\1/p' "$WHISPER_ROOT/lib/cmake/whisper/whisper-config.cmake" 2>/dev/null | head -n1)"
  else
    report missing "whispercpp: include/whisper.h not found under WHISPER_ROOT"
    vendor_hint whispercpp
  fi
else
  whisper_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/whisper.cpp"
  if [[ -f "$whisper_src/CMakeLists.txt" ]]; then
    report ok "whispercpp: $whisper_src (pinned submodule, built with -DUAIRT_BUILD_WHISPERCPP=ON; set WHISPER_ROOT to use an install instead)"
    whisper_commit="$(git -C "$whisper_src" describe --tags --always 2>/dev/null || true)"
    [[ -n "$whisper_commit" ]] && report info "whispercpp: version $whisper_commit (a pinned submodule; bumping it needs ctest and a model run, see docs/vendors.md)"
  else
    report skip "whispercpp: run git submodule update --init --depth 1 third_party/whisper.cpp (or set WHISPER_ROOT to a shared install)"
  fi
fi

if [[ -n "${EXECUTORCH_ROOT:-}" ]]; then
  if [[ -f "$EXECUTORCH_ROOT/lib/cmake/ExecuTorch/executorch-config.cmake" && -f "$EXECUTORCH_ROOT/include/executorch/extension/module/module.h" ]]; then
    report ok "executorch: $EXECUTORCH_ROOT"
    vendor_check executorch ""
  else
    report missing "executorch: lib/cmake/ExecuTorch and the Module extension headers not found under EXECUTORCH_ROOT"
    vendor_hint executorch
  fi
else
  report skip "executorch: set EXECUTORCH_ROOT (an install with -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON and _TENSOR=ON)"
  vendor_hint executorch
fi

system_trt=""
for header in /usr/include/x86_64-linux-gnu/NvInfer.h /usr/include/aarch64-linux-gnu/NvInfer.h /usr/include/NvInfer.h; do
  [[ -f "$header" ]] && system_trt="$header" && break
done
# NvInferVersion.h defines NV_TENSORRT_MAJOR, _MINOR and _PATCH (and _BUILD).
trt_version() {
  [[ -f "$1" ]] || return 0
  local f="$1" v
  v="$(for part in MAJOR MINOR PATCH BUILD; do sed -n "s/^#define NV_TENSORRT_$part \([0-9]*\).*/\1/p" "$f" | head -n1; done | paste -sd. -)"
  [[ "$v" =~ ^[0-9]+\.[0-9]+ ]] && echo "$v"
  return 0
}
if [[ -n "${TENSORRT_ROOT:-}" ]]; then
  if [[ -f "$TENSORRT_ROOT/include/NvInfer.h" ]]; then
    report ok "tensorrt: $TENSORRT_ROOT"
    vendor_check tensorrt "$(trt_version "$TENSORRT_ROOT/include/NvInferVersion.h")"
  else
    report missing "tensorrt: NvInfer.h not found under TENSORRT_ROOT"
    vendor_hint tensorrt
  fi
elif [[ -n "$system_trt" ]]; then
  report ok "tensorrt: system install ($system_trt)"
  vendor_check tensorrt "$(trt_version "$(dirname "$system_trt")/NvInferVersion.h")"
else
  report skip "tensorrt: set TENSORRT_ROOT (TensorRT with headers, installed by you)"
  vendor_hint tensorrt
fi

llama_root="${UAIRT_LLAMACPP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/llama.cpp}"
if [[ -f "$llama_root/include/llama.h" ]]; then
  report ok "llamacpp: $llama_root"
  llama_commit="$(git -C "$llama_root" describe --tags --always 2>/dev/null || true)"
  [[ -n "$llama_commit" ]] && report info "llamacpp: version $llama_commit (a pinned submodule; bumping it needs ctest and a model run, see docs/vendors.md)"
else
  report skip "llamacpp: run git submodule update --init --depth 1 third_party/llama.cpp (or set UAIRT_LLAMACPP_ROOT)"
fi

if have pkg-config && pkg-config --exists libcurl 2>/dev/null; then
  report ok "llmserver: libcurl $(pkg-config --modversion libcurl); the servers (Ollama, LM Studio, SGLang) are yours to start"
elif have curl-config; then
  report ok "llmserver: libcurl $(curl-config --version | awk '{print $2}'); the servers (Ollama, LM Studio, SGLang) are yours to start"
else
  report skip "llmserver: libcurl development files not found (libcurl4-openssl-dev on Debian and Ubuntu)"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  if have xcrun; then
    sdk="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)"
    if [[ -n "$sdk" && -d "$sdk/System/Library/Frameworks/CoreML.framework" ]]; then
      report ok "coreml: $sdk"
    else
      report missing "coreml: CoreML.framework not found in the macOS SDK"
    fi
  else
    report missing "coreml: xcrun not found (install Xcode)"
  fi
else
  report skip "coreml: requires macOS"
fi

exit "$core_missing"
