<!-- SPDX-License-Identifier: Apache-2.0 -->
# Writing a backend

A backend is a shared library (or a static object) that fills in a `uairt_backend_api`
table from `<uairt/uairt_backend.h>`. `src/backends/reference.c` is the smallest working
example; `onnxruntime.c`, `qnn.c` and `coreml.m` are real ones.

## The contract

A plugin exports one symbol:

```c
const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host);
```

The host passes a small function table (`set_error`). The plugin returns a static
`uairt_backend_api` with:

| Field | Purpose |
|---|---|
| `struct_size`, `abi_version` | `sizeof(uairt_backend_api)` and `UAIRT_BACKEND_ABI_VERSION` |
| `name` | the name users pass to `uairt_engine_create` |
| `supported_domains` | bit set of memory domains the backend accepts |
| `create_engine`, `destroy_engine` | parse options, open the vendor runtime |
| `load_model`, `destroy_model` | build the vendor's model object from `uairt_model_source` |
| `num_inputs`, `num_outputs`, `input_info`, `output_info` | describe I/O: dtype, shape, quantization, name |
| `run` | execute; blocking |
| `alloc_buffer`, `free_buffer` | optional, both or neither: zero-copy buffers |

Return `uairt_status` codes (see `docs/api.md`) and call `host->set_error(message)` before
returning an error so users get a useful `uairt_last_error()`.

## Rules

1. **Plain C boundary.** No exceptions, C++ types or STL across the table. Catch
   everything inside the plugin. Objective-C and C++ implementations are fine behind it.
2. **Do not link the core library.** Plugins talk to the host only through the tables.
   If you need `uairt_dtype_size` or similar, write a local copy. Check with
   `nm -u libuairt_backend_x.so | grep uairt` (should print nothing).
3. **The core validates first.** Before `run`, the core has checked counts, dtypes, shapes,
   buffer sizes and memory domains against what `input_info`/`output_info` returned, so
   `run` does not repeat it. It still must not read or write beyond `nbytes`.
4. **Report what the model really is.** Native dtypes, shapes and quantization. Reject
   what you cannot represent with `UAIRT_ERR_UNSUPPORTED` and a message naming the
   tensor. Do not convert silently.
5. **Do not retain caller pointers** (`uairt_tensor`, `uairt_model_source`) after the call
   returns.
6. **Artifacts are tied to devices.** Return `UAIRT_ERR_INCOMPATIBLE_MODEL` with the
   vendor's message when a model does not match the SDK or hardware. Do not fall back
   quietly.
7. **No vendor files in the repo.** See `docs/LICENSING.md`. Build against a path the
   user provides and load vendor libraries at run time where practical.
8. **Strict C.** Build with `-std=c11 -Wall -Wextra -Wpedantic -Werror`. Define
   `_POSIX_C_SOURCE 200809L` before includes if you use `strdup`, `setenv` or similar.

## Steps

1. Copy `src/backends/reference.c` and rename the entry points.
2. Add the build option and module target to `CMakeLists.txt` next to the existing
   backends (an `UAIRT_BUILD_<NAME>` option, find the SDK from a `<NAME>_ROOT` variable).
3. Write a test that loads the plugin, runs a small model and checks outputs against the
   vendor's own tool or Python package, in the style of `tests/compare_with_*.py`. Make
   sure the check is not trivially satisfied (for example, outputs that are all zero).
4. Test the error paths: missing file, wrong format, unknown option, missing SDK.
5. Run it under AddressSanitizer and UBSan, and LeakSanitizer where available.
6. Document options, model formats, limits and the tested SDK version in `docs/api.md`
   and `docs/design.md`, including anything you could not verify.

## Memory domains and zero-copy

`supported_domains` advertises what `run` accepts. To support DMABUF, implement
`alloc_buffer`/`free_buffer` and register the buffer with the vendor runtime on first
use; see the registration cache in `qnn.c`. To support `UAIRT_MEM_PINNED`, advertise it, allocate page-locked
memory in `alloc_buffer` (return the same pointer as handle and data, fd -1) and reject other domains; see
`tensorrt.cpp`. Domains the vendor needs but UAIRT does not
define yet (CUDA device pointers, IOSurface) are added as new bits without changing
existing ones.
