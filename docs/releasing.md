<!-- SPDX-License-Identifier: Apache-2.0 -->
# Releasing the Python package

The `uairt` package on PyPI is a set of platform wheels plus a source distribution, built and published by
`.github/workflows/wheels.yml`. Version 0.1.0 is on https://pypi.org/project/uairt/. A release needs the one-time setup below and either a pushed `v*` tag or a manual
run of the workflow with `publish` set to `testpypi` or `pypi`.

## What is published

- **Wheels**, one per platform, named `uairt-<version>-py3-none-<platform>.whl`. The binding uses `ctypes`, so a wheel is not
  tied to a Python version: one wheel serves every Python 3.9 and later on its platform. Built: Linux x86_64 and aarch64
  (manylinux 2.28), macOS arm64 (11 and later), Windows x64, and Windows ARM64 once an ARM64 runner is enabled (see below). Not
  built: musllinux, 32-bit, and macOS on Intel.
- **A source distribution**, which `pip` builds on any other platform. It holds the core library and the reference backend; it
  has no ONNX Runtime plugin unless the ONNX Runtime headers are available (see "Building locally").
- **Inside a wheel:** the Python package, the native core library (`libuairt`), and the ONNX Runtime plugin in `uairt/plugins/`. The
  plugin loads ONNX Runtime when it loads, so it uses whichever release is installed (1.22 or newer): `pip install
  uairt[onnxruntime]` pulls in the `onnxruntime` wheel, and the package finds its library on its own.
- **Not inside a wheel:** QNN, TensorRT, CoreML, OpenVINO, NCNN and TFLite. QNN, TensorRT and CoreML need vendor SDKs that cannot
  be redistributed; build those plugins from source (see `docs/backends/`) and put them on `UAIRT_PLUGIN_PATH`.

## One-time setup (yours, on PyPI and GitHub)

The workflow publishes with **trusted publishing**: PyPI trusts this repository's workflow, so no API token is stored anywhere.
Package names are first-come: `uairt` was free when this was written, and it is yours only after the first upload.

1. Create accounts on https://pypi.org and https://test.pypi.org (they are separate) and turn on two-factor authentication.
2. On each site, add a **pending trusted publisher** (PyPI: your account, then Publishing; it reserves the name for the first upload):
   - PyPI project name `uairt`, owner `vnk8071`, repository `unified-ai-runtime`, workflow `wheels.yml`, environment `pypi`.
   - TestPyPI: the same, with environment `testpypi`.
3. In the GitHub repository's Settings, Environments, create `testpypi` and `pypi`. On `pypi`, add yourself as a **required
   reviewer**: the workflow then stops and waits for your approval before anything reaches PyPI.
4. Optional: when an ARM64 Windows runner is available to the repository (`windows-11-arm`), create the repository variable
   `ENABLE_WINDOWS_ARM64_CI` with the value `true`; the ARM64 wheel is then built, tested and published too.
5. The package metadata links to the repository (`project.urls` in `pyproject.toml`). If the repository is private, make it public
   or change that link before the first release: PyPI pages are public.

## Release steps

1. Pick the version. Change it in **both** `pyproject.toml` (`version`) and `include/uairt/uairt.h` (`UAIRT_VERSION_*`); a test
   and the workflow fail if they disagree. A pre-release such as `0.1.0rc1` goes in `pyproject.toml` only, with the numbers
   matching the header. Update `CHANGELOG.md`.
2. Commit, then tag and push the tag: `git tag v0.1.0 && git push origin v0.1.0`.
3. The workflow checks the versions, builds the source distribution and the wheels, installs each wheel in a clean environment
   and runs the Python tests against it, and uploads everything to **TestPyPI**.
4. Look at the TestPyPI page and try it:
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "uairt[onnxruntime]"`.
5. Approve the `pypi` environment in the workflow run. The same files then go to PyPI.

For a dry run without a tag, start the workflow by hand (Actions, wheels, Run workflow) and choose `testpypi`; choosing `none`
only builds and tests.

An uploaded version can never be replaced or reused, even if you delete it. Fix a bad release by publishing a new version, and
"yank" the bad one on the PyPI project page so installers skip it.

## Building locally

```bash
pip install build
python -m build --wheel .                       # the core and the reference backend only
```

To include the ONNX Runtime plugin, fetch its headers (the plugin needs the header, not the library) and ask for it:

```bash
scripts/ci/fetch_onnxruntime_headers.sh         # puts ONNX Runtime 1.22 headers in .ort/
UAIRT_WHEEL_ONNXRUNTIME=ON python -m build --wheel .
```

`pipx run cibuildwheel` builds and tests the wheel the way CI does, in a clean environment.

## Verified so far

Built and tested on a Windows 11 ARM64 laptop: the wheel installed in a clean virtual environment runs an ONNX model through
`AutoModel` with the pip `onnxruntime` (1.30), the Python tests pass against the installed wheel, `twine check` passes, and a wheel
builds from the source distribution alone. The Linux, macOS and Windows x64 wheels are built by CI; they have not been run until
the workflow has run on them.
