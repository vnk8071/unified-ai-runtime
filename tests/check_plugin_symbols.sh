#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fails if a plugin exports a llama, ggml or gguf symbol, or does not export uairt_backend_get_api.
set -euo pipefail

plugin="$1"
if [[ "$(uname -s)" == "Darwin" ]]; then
  symbols="$(nm -gU "$plugin" | awk '{print $3}')"
else
  symbols="$(nm -gD --defined-only "$plugin" | awk '{print $3}')"
fi

if grep -E '(^|_)(llama|ggml|gguf)_' <<<"$symbols"; then
  echo "error: the plugin exports llama.cpp symbols (above); they must stay hidden" >&2
  exit 1
fi
if ! grep -q 'uairt_backend_get_api' <<<"$symbols"; then
  echo "error: uairt_backend_get_api is not exported" >&2
  exit 1
fi
