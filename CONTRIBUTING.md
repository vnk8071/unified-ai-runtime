<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing

## Build and test

```bash
scripts/doctor.sh
cmake -S . -B build -DCMAKE_C_FLAGS="-Wall -Wextra -Wpedantic -Werror"
cmake --build build && ctest --test-dir build --output-on-failure
```

Backends are opt-in (`-DUAIRT_BUILD_ONNXRUNTIME=ON`, `-DUAIRT_BUILD_QNN=ON`,
`-DUAIRT_BUILD_COREML=ON`) and need the vendor SDK installed by you; see
`docs/agents/setup.md`. Changes to a backend should be run on real hardware, with the
comparison test for that backend, and under the sanitizers.

## Rules that matter

- **No vendor files.** Never commit SDK headers, libraries, models or text copied from
  them (QAIRT, Apple, NVIDIA). `scripts/check_no_vendor_files.sh` runs under `ctest`.
  Read `docs/LICENSING.md` first.
- **ABI.** Public headers are an ABI. Append only; never reorder or resize existing
  structs, and give new structs a `struct_size`.
- **Plain C boundary.** No C++ types or exceptions across the plugin table.
- **Say what you did not verify.** If something was untested on real hardware, or a test
  only proves something weak, write that in the PR and in `docs/`.
- **Style.** Match the surrounding code. Minimal comments, only for constraints that are
  not obvious. Public symbols are `uairt_` / `UAIRT_`.

## Adding a backend

See `docs/writing-a-backend.md`.

## License

Contributions are accepted under Apache-2.0, the license of this repository. Add
`SPDX-License-Identifier: Apache-2.0` to new files.
