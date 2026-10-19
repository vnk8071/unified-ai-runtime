#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails if vendor SDK files or proprietary headers were copied into the repo.
# See docs/LICENSING.md.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# llama.cpp and MNN are pinned submodules: the repository holds a pointer to each, not their files, and MNN's own tree carries Khronos, Apple
# (Core ML protobuf definitions) and Qualcomm-credited sources of its own. The other submodules (ncnn, whisper.cpp) pass the scan as they are.
exclude=(--exclude-dir=.git --exclude-dir=build --exclude-dir=docs --exclude-dir=llama.cpp --exclude-dir=MNN
         --exclude=check_no_vendor_files.sh --exclude=AGENTS.md)

status=0

if grep -rIl "${exclude[@]}" \
    -e "Confidential and Proprietary" \
    -e "Qualcomm Technologies, Inc" \
    -e "Apple Inc. All rights reserved" . ; then
  echo "error: files above look like proprietary vendor sources" >&2
  status=1
fi

if find . -path ./.git -prune -o -path ./build -prune -o -path ./third_party/llama.cpp -prune -o -path ./third_party/MNN -prune -o \
    \( -name 'libQnn*' -o -name 'libSNPE*' -o -name 'libnvinfer*' \
       -o -name '*.mlmodelc' -o -name 'QnnInterface.h' -o -name 'NvInfer*.h' \) \
    -print | grep .; then
  echo "error: vendor SDK files must not be committed" >&2
  status=1
fi

exit "$status"
