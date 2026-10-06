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
  else
    report ok "qnn: $qnn_root"
  fi
elif [[ -n "$found_hint" ]]; then
  report skip "qnn: found $found_hint; set QNN_SDK_ROOT to it (newer QAIRT releases are needed for .dlc models)"
else
  report skip "qnn: set QNN_SDK_ROOT to your QAIRT SDK directory"
fi

if [[ -n "${ONNXRUNTIME_ROOT:-}" ]]; then
  if [[ -f "$ONNXRUNTIME_ROOT/include/onnxruntime_c_api.h" ]] ||
     [[ -f "$ONNXRUNTIME_ROOT/include/onnxruntime/onnxruntime_c_api.h" ]]; then
    report ok "onnxruntime: $ONNXRUNTIME_ROOT"
  else
    report missing "onnxruntime: header not found under ONNXRUNTIME_ROOT"
  fi
else
  report skip "onnxruntime: set ONNXRUNTIME_ROOT"
fi

if [[ -n "${OPENVINO_ROOT:-}" ]]; then
  if [[ -f "$OPENVINO_ROOT/include/openvino/c/openvino.h" ]] ||
     [[ -f "$OPENVINO_ROOT/runtime/include/openvino/c/openvino.h" ]]; then
    report ok "openvino: $OPENVINO_ROOT"
  else
    report missing "openvino: openvino/c/openvino.h not found under OPENVINO_ROOT"
  fi
else
  report skip "openvino: set OPENVINO_ROOT (pip install openvino, then its site-packages/openvino)"
fi

if [[ -n "${NCNN_ROOT:-}" ]]; then
  if [[ -f "$NCNN_ROOT/include/ncnn/net.h" ]]; then
    report ok "ncnn: $NCNN_ROOT"
  else
    report missing "ncnn: include/ncnn/net.h not found under NCNN_ROOT"
  fi
else
  report skip "ncnn: set NCNN_ROOT (an NCNN install built with NCNN_VULKAN=ON)"
fi

system_trt=""
for header in /usr/include/x86_64-linux-gnu/NvInfer.h /usr/include/aarch64-linux-gnu/NvInfer.h /usr/include/NvInfer.h; do
  [[ -f "$header" ]] && system_trt="$header" && break
done
if [[ -n "${TENSORRT_ROOT:-}" ]]; then
  if [[ -f "$TENSORRT_ROOT/include/NvInfer.h" ]]; then
    report ok "tensorrt: $TENSORRT_ROOT"
  else
    report missing "tensorrt: NvInfer.h not found under TENSORRT_ROOT"
  fi
elif [[ -n "$system_trt" ]]; then
  report ok "tensorrt: system install ($system_trt)"
else
  report skip "tensorrt: set TENSORRT_ROOT (TensorRT with headers, installed by you)"
fi

llama_root="${UAIRT_LLAMACPP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/llama.cpp}"
if [[ -f "$llama_root/include/llama.h" ]]; then
  report ok "llamacpp: $llama_root"
else
  report skip "llamacpp: run git submodule update --init third_party/llama.cpp (or set UAIRT_LLAMACPP_ROOT)"
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
