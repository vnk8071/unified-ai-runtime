---
name: test-on-colab-gpu
description: >-
  Write, fix and review the Google Colab notebooks under examples/notebooks that build and test UAIRT on an NVIDIA GPU (TensorRT, CUDA,
  ONNX Runtime providers, llama.cpp and whisper.cpp with CUDA). Use when the user asks for a Colab notebook, to test CUDA or TensorRT, or
  gives you a notebook with saved outputs to check.
---

# Colab GPU notebooks

You cannot run Colab yourself: the user runs the notebook and the saved outputs come back in the file. Be explicit about that.

**Writing a cell**
- Use a build folder of its own (for example `/content/uairt-<purpose>-build`): a runtime shared with another notebook may already hold a folder configured
  with another generator, and CMake refuses to reuse it. Refresh an existing clone instead of trusting it.
- Create every directory a library will write into (the TensorRT provider's engine cache does not create its folder).
- Make each check record a PASS, FAIL or SKIP row with a detail, and end with a summary table, so one failure does not hide the rest.
- Prove the GPU was used: GPU memory with the model loaded must exceed the CPU run's (a CUDA build may create its context even on a CPU
  run, so compare the two runs, not an idle GPU), and use `cpu_fallback=0` for ONNX Runtime providers.
- Pin versions that matter (TensorRT 10.9.0.34), check downloaded models against published hashes, and say what the notebook installs.
- Do not claim a result the notebook has not produced. Mark sections without saved output as not yet run.

**Checking a notebook the user ran**
Parse the `.ipynb` JSON and read each code cell's `outputs` and `execution_count`, stripping ANSI escapes. Report per cell what passed and
the exact error of the first failure; a cell after a failure has no output because it did not run. Find the cause in the code or the
environment (a stale folder from another notebook, a missing directory, a CUDA mismatch), patch the saved notebook in place so the user
keeps their runtime state, clear the failed cell's output, and say which cell to re-run from.

Before committing a notebook with outputs, check it contains no local paths or tokens. Commit only when asked.
