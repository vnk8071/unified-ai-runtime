---
name: release-uairt
description: >-
  Prepare a UAIRT release: version bumps, changelog, tag, the wheels workflow and PyPI. Use when the user asks to release, publish, tag, bump
  the version, or check that the repository is ready to release. Publishing is outward-facing and cannot be undone; this skill prepares
  everything and stops for the user's confirmation before a tag is pushed.
---

# Release UAIRT

Read `docs/releasing.md` first. A published PyPI version can never be replaced or reused.

1. **Check readiness without publishing:** `git status` clean on `main`, `ctest` passes, the Python tests pass in a clean virtual
   environment (an old editable `uairt` install shadows the repo: uninstall it or use `PYTHONPATH=bindings/python` and `UAIRT_LIBRARY`),
   `scripts/check_no_vendor_files.sh` exits 0, and `scripts/check_references.py` finds no dead links you introduced.
2. **Version:** set it in **both** `pyproject.toml` and `include/uairt/uairt.h` (`UAIRT_VERSION_*`), plus `CITATION.cff` and the README status
   line; a test and the workflow fail if the first two disagree. Update `CHANGELOG.md` with what was added, changed and not verified.
3. **Dry run:** start the `wheels` workflow by hand with `publish: none` or `testpypi`, read the results for every platform, and try the
   TestPyPI install command from `docs/releasing.md`.
4. **Stop and ask** before pushing a `v*` tag or approving the `pypi` environment. Say what will be published (version, platforms,
   what is not in the wheels: QNN, TensorRT, CoreML, OpenVINO, NCNN, MNN, Paddle Lite, TFLite plugins) and what was not tested.
5. After a release, confirm the PyPI page and `pip install` in a clean environment, and update docs that name the published version.

Never put tokens in the repository; publishing uses trusted publishing. A bad release is fixed with a new version and a yank.
