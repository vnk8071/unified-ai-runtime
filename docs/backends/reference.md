<!-- SPDX-License-Identifier: Apache-2.0 -->
# Reference backend

Source: `src/backends/reference.c`. A test-only backend with one float32 `[1, 4]` input copied
to one float32 `[1, 4]` output. It needs no SDK and is built with the core library, and also as a
loadable plugin that checks the plugin ABI.

No setup is needed: `cmake -S . -B build && cmake --build build && ctest --test-dir build`.
To write a real backend, start from [../writing-a-backend.md](../writing-a-backend.md).
