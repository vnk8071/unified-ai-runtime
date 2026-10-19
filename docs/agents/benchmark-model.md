# Which backend is fastest for this model on this machine

Goal: the user has a model and a machine and asks which UAIRT backend, and which of its options, to use. You measure instead of guessing:
`scripts/bench_model.py` runs the model on every backend that can load it, checks each result against the original, and ranks them. The rules in
`AGENTS.md` apply, in particular: report what you measured, say what you did not.

## 1. Run it

```bash
scripts/bench_model.py torchvision:resnet18 --shape 1,3,224,224           # a PyTorch source (random weights unless --state-dict is given)
scripts/bench_model.py my_model.py:build --shape 1,3,640,640 --state-dict weights.pt
scripts/bench_model.py model.onnx                                         # an ONNX file with fixed input shapes
scripts/bench_model.py model.onnx --targets onnx openvino --runs 100 --json result.json
scripts/bench_model.py model.onnx --artifact qnn=model.dlc,device=npu --artifact tensorrt=model.engine      # files you already have
scripts/bench_model.py --list                                              # the variants, and whether each backend's plugin is found
```

It needs what `scripts/export_model.py` needs (the converter packages it names, in a throw-away virtual environment; UAIRT built shared with the
backend plugins on `UAIRT_PLUGIN_PATH`, `UAIRT_LIBRARY`), because it exports to each target with that tool, which also does the conversion
checks of [export-model.md](export-model.md). Run `--list` first: a variant whose plugin is not built is skipped, so build the backends you want
compared before you run it.

What it does: exports to each target (onnx, coreml, executorch, openvino, ncnn, mnn), then runs every variant (CoreML's four compute-unit settings,
OpenVINO and NCNN and MNN at full and at half precision, MNN on Metal, and so on) **in its own process**, because the native libraries of
different runtimes can deadlock in one process and a crash then costs one row. Each run loads the file through UAIRT, runs once and compares the raw
outputs with the reference (PyTorch, or ONNX Runtime for an ONNX source) on the same random input, warms up, and times the runs.

## 2. Read the result

```
| Backend and options              | Load ms | Median ms | p90 ms | Min ms | Max diff | Accuracy |
| coreml compute_units=cpu_and_gpu |   155.6 |     1.800 |  2.304 |  1.672 | 2.03e-06 | ok       |
| mnn device=metal                 |   491.2 |     1.626 |  1.860 |  1.512 |  0.00412 | reduced  |
| onnxruntime                      |    54.0 |    13.293 | 14.202 | 11.822 | 5.25e-06 | ok       |
Fastest accurate: coreml compute_units=cpu_and_gpu, 1.800 ms median.
Fastest overall: mnn device=metal, 1.626 ms median, at reduced precision (max diff 0.00412); check the task result before using it.
```

- **Latency** is the time of the backend's run call, measured by the UAIRT core (`model.run_stats()`), so it leaves out Python and the copy into
  NumPy. **Load** is the engine plus the model load, and includes compiling where a backend does that (CoreML, QNN).
- **Accuracy** is the largest absolute difference from the reference. `ok` is float32 against float32 (within `--atol`, 1e-4, plus 0.02% of the
  output's magnitude). `reduced` is within `--max-rel` (3%) of the magnitude: a half-precision backend or a GPU lands here, and may be the right
  choice, but only after the task result is checked. `FAILED` is beyond that, or NaN, or a constant output; do not use that row.
- The verdict names the fastest `ok` row and, if a `reduced` row is faster, that too with its difference. Report both, not only the faster one.

Run here, on an Apple M5 (macOS arm64), ResNet-18 with random weights at 1x3x224x224, 50 runs (13 variants of six exports; the table above is a
sample of it). It put CoreML on the GPU first among the exact ones at 1.8 ms and ONNX Runtime at 13.3 ms; MNN on Metal at half precision was
fastest overall at 1.6 ms (4e-3 off); `coreml cpu_and_ne` was no faster than `cpu_only`, probably because the model is exported with float32
compute, which the Neural Engine does not run (an fp16 export would; not tried through this tool). Two runs a few minutes apart agreed to about 5%.

## 3. What this does not tell you

- **Task accuracy.** The input is random, so a difference is a numeric one. For a quantized or half-precision candidate, compare the real task
  result (class, boxes, tokens) on real inputs before choosing it.
- **Other machines, other thermal states.** One machine, one moment, default thread counts. Say which machine and what else was running; a laptop on
  battery or a phone that is hot gives different numbers. Re-run on the device that will ship.
- **Memory and power** are not measured.
- **Batch size and shapes** are the ones you give; the winner for 1 image can lose for 32.
- **Backends without an export route here** (TensorRT, QNN, TFLite, llama.cpp) appear only if you pass a file with `--artifact backend=path[,option=value]`; the
  file's input shapes must match the reference's. NPU and GPU variants of those need the device and the vendor SDK ([setup.md](setup.md)).
- A variant that is **unavailable** (no such device) or **skipped** (plugin not found) says so in its row; it is never replaced by the CPU.

## 4. Report it

Give the user the table (or `--json result.json`), the two verdict lines, the machine line, and the section 3 caveats that apply. If the fastest
row is `reduced`, say that the task accuracy still has to be checked. Do not claim a backend is fastest "in general": it is fastest for this model,
at this shape, on this machine.
