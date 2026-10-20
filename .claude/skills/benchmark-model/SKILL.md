---
name: benchmark-model
description: >-
  Find which UAIRT backend, and which of its options, is fastest and still accurate for a given model on this machine. Use when the user
  asks which backend to use, wants backends compared, asks "how fast", or wants a benchmark of a PyTorch or ONNX model across ONNX Runtime,
  CoreML, ExecuTorch, OpenVINO, NCNN and MNN, or of model files they already have for QNN, TensorRT or TFLite.
---

# Benchmark a model across backends

1. Read `AGENTS.md` and `docs/agents/benchmark-model.md`.
2. Build UAIRT shared with the backends you want compared, then run `scripts/bench_model.py --list`: a variant whose plugin is not
   built is skipped, so fix that before measuring.
3. Run it on the model (`torchvision:resnet18 --shape 1,3,224,224`, `model.onnx`, or `--artifact qnn=model.dlc,device=npu`). It
   exports to each target, runs every backend and option variant in its own process, compares with the original, and times with
   the core's own run statistics.
4. Read the verdict as two lines: the fastest result that is **exact** (float32 against float32) and the fastest overall, which
   may be a **reduced-precision** run (half-precision GPU or Neural Engine). Never present a reduced row as the fast exact one.
5. Report: machine, model, shape, input (random, so it says nothing about task accuracy), runs, the table, and what was not
   measured (memory, power, other thread counts, other machines). Two runs a few minutes apart should agree within roughly 5%;
   if they do not, say the machine was busy.

A `FAILED` row is information, not noise: constant or NaN output usually means an export or option problem. A backend reported
as unavailable did not silently run on the CPU.
