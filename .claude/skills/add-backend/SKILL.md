---
name: add-backend
description: >-
  Add a new inference runtime as a UAIRT backend plugin (for example another engine such as MNN or Paddle Lite was added), including the
  CMake option, tests, comparison with the vendor tool, docs and the repository's licensing checks. Use when the user asks to support,
  cover or wrap another runtime, or to extend a backend with a new device or option.
---

# Add a backend

Read `AGENTS.md`, `docs/writing-a-backend.md` (the plugin contract and rules) and `docs/LICENSING.md` first. Look at the two most recent
backends, `src/backends/mnn.cpp` and `paddlelite.cpp`, as worked examples.

**Decide before coding.** Is it a thin portability layer UAIRT should wrap (a runtime that loads a model file and runs it), or a compiler
or a server? Wrap runtimes; do not reimplement conversion. Is the licence compatible, and can it be a pinned submodule (open source) or must
the user install it (vendor SDK)? Never commit vendor files or download an SDK or accept a licence for the user.

Checklist, in this order, running `ctest` after each step:

1. `src/backends/<name>.c|cpp` from `reference.c`: report native dtypes and shapes, refuse what cannot be represented with
   `UAIRT_ERR_UNSUPPORTED`, return `UAIRT_ERR_INCOMPATIBLE_MODEL` for a bad model, `UAIRT_ERR_BACKEND_UNAVAILABLE` for a device that is not
   there (never fall back to the CPU quietly). Number inputs and outputs in a stable order (a name-keyed map is not the model's order).
2. The CMake block (`UAIRT_BUILD_<NAME>`, `<NAME>_ROOT`, a submodule under `third_party/` if open source). Link third-party code
   statically with hidden symbols so it cannot clash with the application's copy; plugins never link the core
   (`nm -u` shows no `uairt_` symbol).
3. `tests/test_<name>.c` (error paths, tensor names and order, a run, options) with a tiny model in `tests/data/`.
4. `tests/compare_with_<name>.py`: the same inputs through UAIRT and through the vendor's own API, identical or within a stated
   tolerance, with a guard that the output is not constant. Run it with every engine option, alone and together.
5. Run on real hardware and under the sanitizers; note exactly what was run where.
6. Docs: `docs/backends/<name>.md` (install, build, options, behaviour, limits, what was **not** run), a row in `README.md` and
   `docs/api.md`, `CHANGELOG.md`, `scripts/doctor.sh`, `scripts/vendors.tsv`, `bindings/python/uairt/_auto.py` (file extension to backend),
   `scripts/export_model.py` / `bench_model.py` if a converter exists.
7. A submodule needs an exclusion in `scripts/check_no_vendor_files.sh` like `llama.cpp` and `MNN`; run it and read its real exit status
   (not `tail`'s).

Report "core verified" and "backend verified" separately; a passing `ctest` without the vendor runtime proves only the core.
