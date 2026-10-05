#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails if vendor SDK files or proprietary headers were copied into the repo.
# See docs/LICENSING.md.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

exclude=(--exclude-dir=.git --exclude-dir=build --exclude-dir=docs
         --exclude=check_no_vendor_files.sh --exclude=AGENTS.md)

status=0

if grep -rIl "${exclude[@]}" \
    -e "Confidential and Proprietary" \
    -e "Qualcomm Technologies, Inc" \
    -e "Apple Inc. All rights reserved" . ; then
  echo "error: files above look like proprietary vendor sources" >&2
  status=1
fi

if find . -path ./.git -prune -o -path ./build -prune -o \
    \( -name 'libQnn*' -o -name 'libSNPE*' -o -name 'libnvinfer*' \
       -o -name '*.mlmodelc' -o -name 'QnnInterface.h' -o -name 'NvInfer*.h' \) \
    -print | grep .; then
  echo "error: vendor SDK files must not be committed" >&2
  status=1
fi

exit "$status"
