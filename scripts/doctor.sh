#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reports the target device, build prerequisites and which backends can be built here.
# Output lines are "<status> <item>: <detail>" with status ok | missing | skip | info.
# Exits non-zero only when core prerequisites are missing.
set -uo pipefail

core_missing=0

report() { printf '%-8s %s\n' "$1" "$2"; }

have() { command -v "$1" >/dev/null 2>&1; }

windows=0
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) windows=1 ;; esac

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
chip=""
npu=""
if [[ $windows == 1 ]]; then
  chip="$(ps '(Get-CimInstance Win32_Processor | Select-Object -First 1).Name')"
  npu="$(ps "Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object { \$_.FriendlyName -like '*NPU*' -and \$_.Status -eq 'OK' } | Select-Object -First 1 -ExpandProperty FriendlyName")"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
else
  chip="$(tr -d '\0' < /sys/firmware/devicetree/base/model 2>/dev/null || true)"
  [[ -z "$chip" ]] && chip="$(grep -m1 -i 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
  for node in /dev/fastrpc-* /dev/adsprpc-smd* /dev/accel/accel*; do
    [[ -e "$node" ]] && npu="$node" && break
  done
fi
[[ -n "$chip" ]] && report info "chip: $chip"
if [[ -n "$npu" ]]; then report ok "npu: $npu"; else report skip "npu: none detected"; fi
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
     && vs="$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property displayName 2>/dev/null)" && [[ -n "$vs" ]]; then
  report ok "cc: $vs (use: cmake -G \"Visual Studio 17 2022\")"
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

if [[ -n "${TENSORRT_ROOT:-}" ]]; then
  if [[ -f "$TENSORRT_ROOT/include/NvInfer.h" ]]; then
    report ok "tensorrt: $TENSORRT_ROOT"
  else
    report missing "tensorrt: NvInfer.h not found under TENSORRT_ROOT"
  fi
else
  report skip "tensorrt: set TENSORRT_ROOT"
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
