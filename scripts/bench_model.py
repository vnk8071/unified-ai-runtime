#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark one model on every UAIRT backend that can run it, and say which is fastest among those that are accurate.

    scripts/bench_model.py torchvision:resnet18 --shape 1,3,224,224
    scripts/bench_model.py model.onnx --targets onnx openvino ncnn --runs 100 --json result.json
    scripts/bench_model.py model.onnx --artifact qnn=model.dlc,device=npu --artifact tflite=model.tflite

SOURCE is what scripts/export_model.py takes: `torchvision:<name>`, `<file.py or module>:<factory>` (needs --shape) or an `.onnx` file. The model
is exported to each target (onnx, coreml, executorch, openvino, ncnn, mnn) with export_model.py, then every backend and option variant below runs
in its own process: it loads the file through UAIRT, runs once and compares the raw outputs with the original (PyTorch, or ONNX Runtime for an ONNX
source) on the same random input, warms up, and times the runs. A backend whose plugin is not built, or a device that is not there, is reported as
skipped or unavailable, never silently replaced by the CPU. `--artifact backend=path[,key=value...]` adds a file you already have (a QNN `.dlc`, a
TensorRT engine, a TFLite file), compared with the same reference; its input shapes must match the reference's.

Latency is the time of the backend's run call measured by the UAIRT core (`model.run_stats()`), so it excludes Python and the copy into NumPy; load
time is the engine plus model load. Accuracy is the largest absolute difference of the outputs from the reference, called `ok` within `--atol` plus
0.02% of the output's magnitude (float32 against float32), `reduced` within `--max-rel` of the magnitude (a half-precision backend or device lands
here) and `FAILED` beyond that. Memory is not measured. The input is random, so this says nothing about task accuracy: for a quantized model compare the task
result as docs/agents/export-model.md says.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_model as em  # noqa: E402

# target -> [(backend, engine options)]: the variants worth comparing. Half-precision and GPU variants sit next to the exact ones.
VARIANTS: Dict[str, List[Tuple[str, Dict[str, str]]]] = {
    "onnx": [("onnxruntime", {})] + [
        ("onnxruntime", {"execution_provider": "CoreML", "ep_option.MLComputeUnits": u, "cpu_fallback": "0"})
        for u in ("CPUAndGPU", "CPUAndNeuralEngine", "ALL")],     # ONNX Runtime's CoreML provider; refused, not skipped, where it is not built in
    "coreml": [("coreml", {"compute_units": u}) for u in ("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all")],
    "executorch": [("executorch", {})],
    "openvino": [("openvino", {"inference_precision": "f32"}), ("openvino", {"inference_precision": "f16"})],
    "ncnn": [("ncnn", {"fp16": "false"}), ("ncnn", {"fp16": "true"})],
    "mnn": [("mnn", {"precision": "high"}), ("mnn", {"precision": "low"}), ("mnn", {"device": "metal"})],
}
# options that every variant of a backend needs besides the ones above (NCNN's input shapes depend on the model)
TIMEOUT_SECONDS = 300


# ---- the worker: one backend, one option set, one process ---------------------------------------------------------------------------

def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_one(path: str, backend: str, options: Dict[str, str], data_path: str, runs: int, warmup: int, max_seconds: float) -> Dict:
    """Loads `path` through UAIRT, checks the outputs against the reference stored in `data_path`, and times the runs. Returns a result dict."""
    import uairt  # a worker is the only process that loads backend plugins

    data = np.load(data_path)
    inputs = [data[f"input{i}"] for i in range(len([k for k in data.files if k.startswith("input")]))]
    reference = [data[f"reference{i}"] for i in range(len([k for k in data.files if k.startswith("reference")]))]
    result: Dict = {"backend": backend, "options": options}
    started = time.perf_counter()
    try:
        model = uairt.AutoModel.from_file(path, backend=backend, options=options)
    except uairt.UairtError as error:
        kind = "unavailable" if isinstance(error, (uairt.BackendUnavailable, uairt.Unsupported)) else "error"
        return {**result, "status": kind, "error": str(error)[:300]}
    with model:
        result["load_ms"] = (time.perf_counter() - started) * 1000.0
        try:
            cast = em.align_inputs(model, inputs)
            first = [np.asarray(x) for x in model.run(*cast)]
        except (uairt.UairtError, em.ExportError) as error:
            return {**result, "status": "error", "error": str(error)[:300]}
        diffs, scales = [], []
        for ref, got in em.match_outputs(reference, first):
            r64, g64 = ref.astype(np.float64), got.astype(np.float64)
            diffs.append(float(np.max(np.abs(r64 - g64))) if ref.size else 0.0)
            scales.append(float(np.max(np.abs(r64))) if ref.size else 0.0)
            if not np.all(np.isfinite(g64)) or (np.unique(g64).size <= 1 and ref.size > 1):
                diffs[-1] = float("inf")
        result["max_diff"] = max(diffs)
        result["scale"] = max(scales)

        for _ in range(warmup):
            model.run(*cast)
        has_stats = True
        try:
            model.reset_run_stats()
        except uairt.UairtError:
            has_stats = False
        times: List[float] = []
        deadline = time.perf_counter() + max_seconds
        for _ in range(runs):
            t0 = time.perf_counter()
            model.run(*cast)
            wall = (time.perf_counter() - t0) * 1000.0
            times.append(model.run_stats().last_ms if has_stats else wall)
            if time.perf_counter() > deadline:
                break
        result.update(status="ok", runs=len(times), median_ms=percentile(times, 50), p90_ms=percentile(times, 90),
                      min_ms=float(min(times)), timing="core" if has_stats else "wall")
    return result


# ---- the driver --------------------------------------------------------------------------------------------------------------------

def classify(result: Dict, atol: float, max_rel: float) -> str:
    """ok / reduced / FAILED from the largest difference and the output's magnitude."""
    diff, scale = result.get("max_diff"), result.get("scale", 0.0)
    if diff is None:
        return "n/a"
    if not np.isfinite(diff):
        return "FAILED"
    if diff <= atol + 2e-4 * scale:
        return "ok"
    if diff <= max_rel * max(scale, 1e-12):
        return "reduced"
    return "FAILED"


def label(result: Dict) -> str:
    options = ",".join(f"{k}={v}" for k, v in result.get("options", {}).items() if k != "input_shapes")
    return f"{result['backend']} {options}".strip()


def render_table(results: List[Dict]) -> str:
    """The results as a Markdown table, fastest first among the rows that ran; rows that did not run follow."""
    ran = sorted((r for r in results if r.get("status") == "ok"), key=lambda r: r["median_ms"])
    rest = [r for r in results if r.get("status") != "ok"]
    lines = ["| Backend and options | Load ms | Median ms | p90 ms | Min ms | Max diff | Accuracy |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in ran:
        lines.append(f"| {label(r)} | {r['load_ms']:.1f} | {r['median_ms']:.3f} | {r['p90_ms']:.3f} | {r['min_ms']:.3f} | "
                     f"{r['max_diff']:.3g} | {r['accuracy']} |")
    for r in rest:
        reason = r.get("error", r.get("status", "")).replace("|", "/").replace("\n", " ")
        lines.append(f"| {label(r)} | | | | | | {r.get('status')}: {reason[:90]} |")
    return "\n".join(lines)


def verdict(results: List[Dict]) -> List[str]:
    ran = [r for r in results if r.get("status") == "ok" and r["accuracy"] != "FAILED"]
    lines = []
    exact = sorted((r for r in ran if r["accuracy"] == "ok"), key=lambda r: r["median_ms"])
    reduced = sorted((r for r in ran if r["accuracy"] == "reduced"), key=lambda r: r["median_ms"])
    if exact:
        lines.append(f"Fastest accurate: {label(exact[0])}, {exact[0]['median_ms']:.3f} ms median.")
    else:
        lines.append("No backend ran within the accuracy tolerance.")
    if reduced and (not exact or reduced[0]["median_ms"] < exact[0]["median_ms"]):
        lines.append(f"Fastest overall: {label(reduced[0])}, {reduced[0]['median_ms']:.3f} ms median, at reduced precision "
                     f"(max diff {reduced[0]['max_diff']:.3g}); check the task result before using it.")
    return lines


def environment() -> Dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform == "darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info["cpu"] = chip or platform.processor()
    else:
        info["cpu"] = platform.processor() or platform.machine()
    try:
        import uairt
        info["uairt"] = uairt.version()
        info["backends"] = sorted(name for name, s in uairt.available_backends().items() if s["registered"] or s["plugin"])
    except Exception as error:  # reported, not fatal: the workers will say what is wrong
        info["uairt"] = f"unavailable ({error})"
        info["backends"] = []
    return info


def parse_artifact(text: str) -> Tuple[str, str, Dict[str, str]]:
    backend, _, rest = text.partition("=")
    if not backend or not rest:
        raise em.ExportError(f"--artifact wants backend=path[,key=value...], got '{text}'")
    path, *pairs = rest.split(",")
    options = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not key or not _:
            raise em.ExportError(f"--artifact option '{pair}' is not key=value")
        options[key] = value
    return backend, path, options


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="torchvision:<name>, <file.py|module>:<factory>, or an .onnx file")
    parser.add_argument("--targets", nargs="+", choices=sorted(VARIANTS), help="export targets to try (default: all of them)")
    parser.add_argument("--shape", action="append", help="input shape such as 1,3,224,224 (needed for a PyTorch source)")
    parser.add_argument("--dtype", action="append", help="input dtype (default float32), one per --shape")
    parser.add_argument("--artifact", action="append", default=[], metavar="BACKEND=PATH[,K=V...]", help="a file you already have")
    parser.add_argument("--state-dict", help="weights for a PyTorch source (loaded with weights_only=True)")
    parser.add_argument("--runs", type=int, default=50, help="timed runs per variant (default 50)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=20.0, help="stop timing a variant after this long")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-rel", type=float, default=3e-2, help="largest difference, as a fraction of the output's magnitude, still reported as reduced precision")
    parser.add_argument("--out", help="directory for the exported files (default: a temporary one)")
    parser.add_argument("--json", help="also write the results to this file")
    parser.add_argument("--list", action="store_true", help="list the variants and whether their backend is available, then exit")
    # internal: one backend, one option set, in this process
    parser.add_argument("--run-one", nargs=2, metavar=("FILE", "BACKEND"), help=argparse.SUPPRESS)
    parser.add_argument("--options-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("--data", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.run_one:
        outcome = run_one(args.run_one[0], args.run_one[1], json.loads(args.options_json), args.data, args.runs, args.warmup, args.max_seconds)
        print(json.dumps(outcome))
        return 0

    env = environment()
    if args.list:
        print(f"UAIRT {env['uairt']} on {env['cpu']}; backends found: {', '.join(env['backends']) or 'none'}")
        for target in sorted(VARIANTS):
            for backend, options in VARIANTS[target]:
                available = "available" if backend in env["backends"] else "plugin not found"
                print(f"  {target:11} {backend} {','.join(f'{k}={v}' for k, v in options.items()):28} {available}")
        return 0
    if not args.source and not args.artifact:
        parser.error("give a SOURCE or at least one --artifact")

    try:
        artifacts = [parse_artifact(a) for a in args.artifact]
        source = args.source
        is_onnx = bool(source) and source.endswith(".onnx")
        model = None
        onnx_path = Path(source) if is_onnx else None
        if is_onnx:
            shapes, dtypes = em.onnx_input_info(onnx_path)
        elif source:
            if not args.shape:
                raise em.ExportError("a PyTorch source needs --shape (the shape you will really feed)")
            shapes = em.parse_shapes(args.shape)
            dtypes = args.dtype or ["float32"] * len(shapes)
            model = em.load_torch_model(source, args.state_dict)
        else:  # only artifacts: the reference is the first artifact's input signature, run by ONNX Runtime
            raise em.ExportError("--artifact alone has no reference to compare with: give a SOURCE as well")
        inputs = em.make_inputs(shapes, dtypes)
        reference = em.reference_outputs(source, model, onnx_path if model is None else None, inputs)
    except em.ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    workdir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="uairt-bench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    data_path = workdir / "reference.npz"
    np.savez(data_path, **{f"input{i}": x for i, x in enumerate(inputs)}, **{f"reference{i}": x for i, x in enumerate(reference)})
    name = Path(source).stem if is_onnx else source.split(":")[-1]
    targets = args.targets or sorted(VARIANTS)
    results: List[Dict] = []

    def measure(path: str, backend: str, options: Dict[str, str]) -> Dict:
        if backend not in env["backends"]:
            return {"backend": backend, "options": options, "status": "skipped", "error": "plugin not found (UAIRT_PLUGIN_PATH)"}
        command = [sys.executable, str(Path(__file__).resolve()), "--run-one", path, backend, "--options-json", json.dumps(options),
                   "--data", str(data_path), "--runs", str(args.runs), "--warmup", str(args.warmup), "--max-seconds", str(args.max_seconds)]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return {"backend": backend, "options": options, "status": "error", "error": f"timed out after {TIMEOUT_SECONDS} s"}
        line = next((l for l in reversed(done.stdout.splitlines()) if l.startswith("{")), None)
        if done.returncode != 0 or line is None:
            tail = (done.stderr.strip().splitlines() or ["no output"])[-1]
            return {"backend": backend, "options": options, "status": "error", "error": f"crashed (status {done.returncode}): {tail[:200]}"}
        outcome = json.loads(line)
        outcome["accuracy"] = classify(outcome, args.atol, args.max_rel)
        return outcome

    for target in targets:
        print(f"[{target}] exporting...", file=sys.stderr, flush=True)
        command = [sys.executable, str(HERE / "export_model.py"), source, "--to", target, "--out", str(workdir), "--no-verify", "--name", name]
        for s in args.shape or []:
            command += ["--shape", s]
        for d in args.dtype or []:
            command += ["--dtype", d]
        if args.state_dict:
            command += ["--state-dict", args.state_dict]
        done = subprocess.run(command, capture_output=True, text=True)
        wrote = next((l.split(" wrote ", 1)[1].strip() for l in done.stdout.splitlines() if l.startswith(f"[{target}] wrote ")), None)
        if done.returncode != 0 or wrote is None:
            reason = (done.stderr.strip().splitlines() or ["export failed"])[-1]
            for backend, options in VARIANTS[target]:
                results.append({"backend": backend, "options": options, "status": "error", "error": f"export to {target} failed: {reason[:160]}"})
            continue
        for backend, options in VARIANTS[target]:
            extra = {}
            if backend == "ncnn":  # NCNN's shapes have no batch dimension; the files are read as given
                extra = {"input_shapes": ";".join("x".join(str(d) for d in (s[1:] if len(s) > 1 and s[0] == 1 else s)) for s in shapes)}
            print(f"[{target}] running {backend} {options}...", file=sys.stderr, flush=True)
            results.append(measure(wrote, backend, {**options, **extra}))
    for backend, path, options in artifacts:
        print(f"[artifact] running {backend} on {path}...", file=sys.stderr, flush=True)
        results.append(measure(path, backend, options))

    print(f"\nModel: {source or 'artifacts'}, inputs {shapes}, {args.runs} runs after {args.warmup} warm-up runs per variant")
    print(f"Machine: {env['cpu']} ({env['platform']}); UAIRT {env['uairt']}\n")
    print(render_table(results))
    print()
    for line in verdict(results):
        print(line)
    if args.json:
        Path(args.json).write_text(json.dumps({"environment": env, "source": source, "shapes": [list(s) for s in shapes], "results": results}, indent=2))
        print(f"\nResults written to {args.json}")
    return 0 if any(r.get("status") == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
