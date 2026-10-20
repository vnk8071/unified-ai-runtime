---
name: write-uairt-app
description: >-
  Write, build and run a program that uses UAIRT (the C API, the C++ wrapper, the Python binding or the Rust crate) to run a model: image
  classification or detection, speech to text, chat with a language model, or a custom pipeline. Use when the user asks for an app, script,
  demo or snippet that loads a model through UAIRT, or asks how to prepare a model's inputs and decode its outputs.
---

# Write an application that uses UAIRT

1. Read `AGENTS.md`, `docs/agents/write-app.md`, and `docs/api.md`. Do not invent API names: copy the call order from the nearest
   example in `examples/` (`quickstart.c`, `run_model.c`, `examples/python/auto_model.py`, `llm_generate.py`, `examples/cpp`,
   `examples/rust`).
2. Ask the model what it expects instead of guessing: `build/run_model --info <plugin> <backend> <model>`, or
   `python examples/python/inspect_model.py <model>`, which also prints a first guess at the pre- and post-processing.
3. Write the pre- and post-processing from the model's own documentation (`docs/agents/pre-post-processing.md`): image layout
   and normalisation, letterboxing and NMS for detectors, chat templates and sampling for language models, quantization scale and
   zero point for quantized tensors. UAIRT never converts dtypes silently.
4. Language models: use `libuairt_llm` (`uairt_llm_generate` and `uairt_llm_sample` from `uairt/uairt_llm.h`, or `generate()` / `complete()` in Python) for sampling and streaming; it stops
   at the model's end-of-generation token unless told not to.
5. Build and run it on the real device with a real input and show the real output. Compare with the framework the model came from when a
   reference exists. Say what you did not verify.

Rules: no C++ types, exceptions or STL across the C ABI; check every `uairt_status`; keep buffers alive until the call returns; do not
write machine-specific paths into examples.
