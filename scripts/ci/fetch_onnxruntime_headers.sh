#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Puts ONNX Runtime's headers in .ort/include for the wheel build. The ONNX Runtime plugin needs only the header, not the
# library, which it loads at run time; the header is the same on every platform, so one small archive serves them all.
# The plugin asks for API 22, so it works with any ONNX Runtime 1.22 or newer.
set -euo pipefail
version="${1:-1.22.0}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="$root/.ort"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
rm -rf "$dest" && mkdir -p "$dest"
# Naming just the include directory works with GNU tar and with the BSD tar on macOS and Windows, and skips lib/, whose symlinks
# Windows cannot create.
archive="onnxruntime-linux-x64-${version}"
curl -fsSL --retry 3 "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive}.tgz" \
  | tar -xz -C "$work" "${archive}/include"
cp -R "$work/${archive}/include" "$dest/include"
test -f "$dest/include/onnxruntime_c_api.h" || { echo "onnxruntime_c_api.h not found in the archive" >&2; exit 1; }
echo "ONNX Runtime ${version} headers in $dest/include"
