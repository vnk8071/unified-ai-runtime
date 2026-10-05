#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reports build prerequisites and which backends can be built here.
# Output lines are "<status> <item>: <detail>" with status ok | missing | skip.
# Exits non-zero only when core prerequisites are missing.
set -uo pipefail

core_missing=0

report() { printf '%-8s %s\n' "$1" "$2"; }

have() { command -v "$1" >/dev/null 2>&1; }

echo "== platform"
report ok "os: $(uname -s) $(uname -m)"

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

if [[ -n "${QNN_SDK_ROOT:-}" ]]; then
  if [[ -f "$QNN_SDK_ROOT/include/QNN/QnnInterface.h" ]]; then
    report ok "qnn: $QNN_SDK_ROOT"
  else
    report missing "qnn: QNN_SDK_ROOT set but include/QNN/QnnInterface.h not found"
  fi
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
