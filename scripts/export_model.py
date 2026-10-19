#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a PyTorch model (or an ONNX file) to the format a UAIRT backend loads, then check it through UAIRT against the original.

    scripts/export_model.py torchvision:resnet18 --shape 1,3,224,224 --to onnx coreml executorch --out exported
    scripts/export_model.py my_model.py:build --shape 1,3,640,640 --to onnx --state-dict weights.pt
    scripts/export_model.py model.onnx --to openvino ncnn

SOURCE is `torchvision:<name>` (random weights unless --state-dict is given), `<file.py or module>:<factory>` (a function returning an
nn.Module), or an `.onnx` file. Targets: onnx (onnxruntime backend), coreml (coreml), executorch (executorch, XNNPACK), openvino (openvino, from
ONNX), ncnn (ncnn, through pnnx), mnn (mnn, through MNN's converter, from ONNX). Each export writes `<out>/<name>.<ext>` and, unless --no-verify, runs the reference (PyTorch, or ONNX
Runtime for an ONNX source) and the converted file through UAIRT on the same random input and compares the raw outputs.

What the converters need is not installed by this script: the packages it names when one is missing (torch, onnx, coremltools, executorch,
openvino, pnnx, MNN, onnxruntime) go in a throw-away virtual environment, and UAIRT's Python binding needs a shared build of the core and the
backend plugins (UAIRT_LIBRARY, UAIRT_PLUGIN_PATH; see bindings/python/README.md). Exit status is 0 only if every export and check passed.
"""
import argparse
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# target -> (UAIRT backend, file extension of what UAIRT loads)
TARGETS: Dict[str, Tuple[str, str]] = {
    "onnx": ("onnxruntime", ".onnx"),
    "coreml": ("coreml", ".mlpackage"),
    "executorch": ("executorch", ".pte"),
    "openvino": ("openvino", ".xml"),
    "ncnn": ("ncnn", ".ncnn.param"),
    "mnn": ("mnn", ".mnn"),
}


class ExportError(Exception):
    """A step that cannot continue; the message says what to install or change."""


def need(module: str, why: str, pip: Optional[str] = None):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ExportError(f"{why} needs the Python package '{module}' ({error}); install {pip or module} into a virtual environment") from error


# ---- the source model ---------------------------------------------------------------------------------------------------------

def load_torch_model(spec: str, state_dict: Optional[str]):
    torch = need("torch", "a PyTorch source")
    torch.manual_seed(0)
    where, _, name = spec.partition(":")
    if not name:
        raise ExportError(f"SOURCE '{spec}' must be torchvision:<name>, <file.py or module>:<factory>, or an .onnx file")
    if where == "torchvision":
        torchvision = need("torchvision", "torchvision:<name>")
        factory = getattr(torchvision.models, name, None)
        if factory is None:
            raise ExportError(f"torchvision.models has no '{name}'")
        model = factory(weights=None)
    else:
        if where.endswith(".py"):
            module_spec = importlib.util.spec_from_file_location(Path(where).stem, where)
            module = importlib.util.module_from_spec(module_spec)
            sys.path.insert(0, str(Path(where).resolve().parent))
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(where)
        model = getattr(module, name)()
    if state_dict:
        # weights_only: a checkpoint is a pickle, and an untrusted one can run code when loaded the old way
        model.load_state_dict(torch.load(state_dict, map_location="cpu", weights_only=True))
    return model.eval()


def parse_shapes(texts: Sequence[str]) -> List[Tuple[int, ...]]:
    try:
        return [tuple(int(d) for d in text.split(",")) for text in texts]
    except ValueError as error:
        raise ExportError(f"--shape wants comma-separated integers such as 1,3,224,224 ({error})") from error


def make_inputs(shapes: Sequence[Tuple[int, ...]], dtypes: Sequence[str]) -> List[np.ndarray]:
    """Deterministic random inputs: standard normal for floats, small non-negative integers for integer dtypes (token ids, indices)."""
    rng = np.random.default_rng(0)
    inputs = []
    for shape, dtype in zip(shapes, dtypes):
        if np.issubdtype(np.dtype(dtype), np.floating):
            inputs.append(rng.standard_normal(shape).astype(dtype))
        else:
            inputs.append(rng.integers(0, 10, shape).astype(dtype))
    return inputs


def flat(outputs) -> list:
    if isinstance(outputs, (list, tuple)):
        return [x for item in outputs for x in flat(item)]
    return [outputs]


# ---- the exporters: each writes the file UAIRT loads and returns its path ---------------------------------------------------------

def export_onnx(model, inputs, out: Path, opset: int) -> Path:
    torch = need("torch", "the ONNX export")
    onnx = need("onnx", "checking the ONNX file")
    example = [torch.from_numpy(x) for x in inputs]
    with torch.no_grad():
        count = len(flat(model(*example)))
    kwargs = dict(opset_version=opset, input_names=[f"input{i}" for i in range(len(example))],
                  output_names=[f"output{i}" for i in range(count)])
    try:  # torch 2.5 and later: the TorchScript-based exporter needs no extra package and keeps the graph simple
        torch.onnx.export(model, tuple(example), str(out), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, tuple(example), str(out), **kwargs)
    onnx.checker.check_model(onnx.load(str(out)))
    return out


def export_coreml(model, inputs, out: Path) -> Path:
    torch = need("torch", "the CoreML export")
    ct = need("coremltools", "the CoreML export")
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(torch.from_numpy(x) for x in inputs))
    tensor_types = [ct.TensorType(name=f"input{i}", shape=tuple(x.shape)) for i, x in enumerate(inputs)]
    # float32 compute so the check against PyTorch is tight; choose float16 for speed after checking accuracy
    ml = ct.convert(traced, inputs=tensor_types, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                    minimum_deployment_target=ct.target.macOS13)
    ml.save(str(out))
    return out


def export_executorch(model, inputs, out: Path, delegate: str) -> Path:
    torch = need("torch", "the ExecuTorch export")
    exir = need("executorch.exir", "the ExecuTorch export", "executorch")
    partitioners = []
    if delegate == "xnnpack":
        xnnpack = need("executorch.backends.xnnpack.partition.xnnpack_partitioner", "the XNNPACK delegate", "executorch")
        partitioners = [xnnpack.XnnpackPartitioner()]
    elif delegate != "none":
        raise ExportError(f"--delegate must be xnnpack or none, not '{delegate}'")
    program = torch.export.export(model, tuple(torch.from_numpy(x) for x in inputs))
    lowered = exir.to_edge_transform_and_lower(program, partitioner=partitioners).to_executorch()
    out.write_bytes(lowered.buffer)
    return out


def export_openvino(onnx_path: Path, out: Path) -> Path:
    openvino = need("openvino", "the OpenVINO conversion")
    # float32 weights (no fp16 compression) so the check is tight; OpenVINO's own `ovc` does the same with --compress_to_fp16=False
    openvino.save_model(openvino.convert_model(str(onnx_path)), str(out), compress_to_fp16=False)
    return out


def export_ncnn(model, inputs, out: Path) -> Path:
    torch = need("torch", "the NCNN export")
    pnnx = need("pnnx", "the NCNN export")
    stem = out.with_suffix("").with_suffix("")  # <out>/<name>.ncnn.param -> <out>/<name>
    try:
        # fp16=False: pnnx stores the weights as float16 by default, which alone moves a ResNet-18's logits by 0.02; use True for a smaller file
        # once the accuracy is checked
        pnnx.export(model, str(stem) + ".pt", tuple(torch.from_numpy(x) for x in inputs), fp16=False)
    except Exception as error:
        # pnnx converts first and then runs a Python file it generated to compare outputs; that file can fail to parse (seen with
        # torch 2.14 and pnnx 20260526) after the NCNN files are written. The check through UAIRT below is the real test.
        if not (out.exists() and out.with_suffix(".bin").exists()):
            raise ExportError(f"pnnx failed before writing {out}: {error}") from error
        print(f"[ncnn] note: pnnx's own self-check failed ({type(error).__name__}: {error}); the files were written, checking them through UAIRT")
    if not out.exists():
        raise ExportError(f"pnnx did not write {out}; look at its output next to it")
    return out


def export_ncnn_from_onnx(onnx_path: Path, shapes: Sequence[Tuple[int, ...]], out: Path) -> Path:
    pnnx = need("pnnx", "the NCNN conversion of an ONNX file")
    binary = Path(pnnx.EXEC_PATH) if getattr(pnnx, "EXEC_PATH", None) else None
    if binary is None or not binary.exists():
        raise ExportError("could not find the pnnx executable that the pnnx package installs")
    stem = out.with_suffix("").with_suffix("")
    local = stem.with_suffix(".onnx")          # pnnx writes next to its input
    if local.resolve() != onnx_path.resolve():
        local.write_bytes(onnx_path.read_bytes())
    shape_text = ",".join("[" + ",".join(str(d) for d in s) + "]" for s in shapes)
    result = subprocess.run([str(binary), str(local), f"inputshape={shape_text}", "fp16=0"], capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise ExportError(f"pnnx failed on {onnx_path} (status {result.returncode}): {result.stderr.strip()[-300:] or result.stdout.strip()[-300:]}")
    return out


def export_mnn_from_onnx(onnx_path: Path, out: Path) -> Path:
    need("MNN", "the MNN conversion", "MNN")
    # MNN's converter is a module of the pip package; it was seen installing and replacing packages (protobuf) on its first run
    result = subprocess.run([sys.executable, "-m", "MNN.tools.mnnconvert", "-f", "ONNX", "--modelFile", str(onnx_path),
                             "--MNNModel", str(out), "--bizCode", "uairt"], capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise ExportError(f"mnnconvert failed on {onnx_path} (status {result.returncode}): {(result.stderr or result.stdout).strip()[-300:]}")
    return out


# ---- the check: reference vs the converted file through UAIRT ----------------------------------------------------------------------

def reference_outputs(source: str, model, onnx_path: Optional[Path], inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
    if onnx_path is not None and model is None:
        ort = need("onnxruntime", "the reference run of an ONNX source", "onnxruntime")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        return [np.asarray(x) for x in session.run(None, dict(zip(names, inputs)))]
    torch = need("torch", "the reference run")
    with torch.no_grad():
        return [x.detach().cpu().numpy() for x in flat(model(*[torch.from_numpy(i) for i in inputs]))]


def run_uairt(path: Path, backend: str, inputs: Sequence[np.ndarray], options: Dict[str, str]) -> List[np.ndarray]:
    uairt = need("uairt", "the check through UAIRT", "the uairt package, or set PYTHONPATH=bindings/python and UAIRT_LIBRARY (shared build)")
    with uairt.AutoModel.from_file(str(path), backend=backend, options=options) as model:
        if len(model.inputs) != len(inputs):
            raise ExportError(f"the converted model has {len(model.inputs)} inputs, the reference {len(inputs)}")
        cast = []
        for info, value in zip(model.inputs, inputs):
            if tuple(info.shape) != tuple(value.shape):
                # NCNN has no batch dimension: a model that differs only by size-1 dimensions gets the same data reshaped
                if tuple(d for d in info.shape if d != 1) != tuple(d for d in value.shape if d != 1):
                    raise ExportError(f"input '{info.name}': the converted model wants shape {tuple(info.shape)}, the reference ran with {tuple(value.shape)}")
                value = value.reshape(info.shape)
            cast.append(value.astype(info.dtype))
        return [np.asarray(x) for x in model.run(*cast)]


def match_outputs(reference: Sequence[np.ndarray], got: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Pair outputs by position, or by shape when the backend orders them differently (CoreML sorts outputs by name). An output that
    differs only by dropped size-1 dimensions (NCNN has no batch dimension) is reshaped to the reference's shape."""
    if len(reference) != len(got):
        raise ExportError(f"the converted model has {len(got)} outputs, the reference {len(reference)}")

    def squeezed(a):
        return tuple(d for d in a.shape if d != 1)

    def same(r, g):
        return r.shape == g.shape or squeezed(r) == squeezed(g)

    ordered = list(got) if all(same(r, g) for r, g in zip(reference, got)) else None
    if ordered is None:
        remaining, ordered = list(got), []
        for r in reference:
            index = next((i for i, g in enumerate(remaining) if same(r, g)), None)
            if index is None:
                raise ExportError(f"no output of the converted model has the reference's shape {r.shape}; shapes are {[g.shape for g in got]}")
            ordered.append(remaining.pop(index))
    return [(r, g.reshape(r.shape)) for r, g in zip(reference, ordered)]


def compare(reference: Sequence[np.ndarray], got: Sequence[np.ndarray], atol: float) -> Tuple[bool, List[str]]:
    lines, ok = [], True
    for i, (r, g) in enumerate(match_outputs(reference, got)):
        r64, g64 = r.astype(np.float64), g.astype(np.float64)
        diff = float(np.max(np.abs(r64 - g64))) if r.size else 0.0
        scale = float(np.max(np.abs(r64))) if r.size else 0.0
        constant = np.unique(g64).size <= 1 and r.size > 1
        passed = bool(np.isfinite(diff) and diff <= atol + 2e-3 * scale and not constant)
        ok = ok and passed
        lines.append(f"output {i} {tuple(r.shape)}: max |diff| {diff:.3g} (reference magnitude {scale:.3g}) "
                     f"{'ok' if passed else 'FAILED' + (' (the converted output is constant)' if constant else '')}")
    return ok, lines


# ---- driver -------------------------------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="torchvision:<name>, <file.py|module>:<factory>, or an .onnx file")
    parser.add_argument("--to", nargs="+", required=True, choices=sorted(TARGETS), help="target format(s)")
    parser.add_argument("--shape", action="append", help="input shape such as 1,3,224,224; repeat for several inputs (needed for a PyTorch source)")
    parser.add_argument("--dtype", action="append", help="input dtype (default float32); repeat per input, e.g. int64 for token ids")
    parser.add_argument("--out", default="exported", help="output directory")
    parser.add_argument("--name", help="file name stem (default: from SOURCE)")
    parser.add_argument("--state-dict", help="a state dict file to load into the PyTorch model (loaded with weights_only=True)")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--delegate", default="xnnpack", help="ExecuTorch delegate: xnnpack or none")
    parser.add_argument("--atol", type=float, default=1e-4, help="absolute tolerance of the check (plus 0.2%% of the output's magnitude, for backends that compute in half precision)")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="backend option for the check, e.g. input_shapes=1x3x224x224 for ncnn")
    parser.add_argument("--no-verify", action="store_true", help="export only")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    if len(args.to) > 1:
        # One process per target: ONNX Runtime, OpenVINO and PyTorch each bring native libraries (abseil among them) that deadlock when
        # they meet in one process (seen: OpenVINO blocking in a mutex after an ONNX Runtime check), and a converter that crashes
        # then costs only its own target.
        rest, skipping = [], False
        for token in argv:
            if token == "--to":
                skipping = True
            elif skipping and token.startswith("-"):
                skipping = False
            if not skipping:
                rest.append(token)
        codes = []
        for target in args.to:
            print(f"--- {target}", flush=True)
            codes.append(subprocess.call([sys.executable, str(Path(__file__).resolve()), *rest, "--to", target]))
        print("\nsummary: " + "; ".join(f"{t}: {'ok' if c == 0 else 'FAILED'}" for t, c in zip(args.to, codes)))
        return 1 if any(codes) else 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    options = dict(item.split("=", 1) for item in args.option)
    is_onnx_source = args.source.endswith(".onnx")
    name = args.name or (Path(args.source).stem if is_onnx_source else args.source.split(":")[-1])

    model = None
    onnx_path: Optional[Path] = Path(args.source) if is_onnx_source else None
    try:
        if is_onnx_source:
            unsupported = [t for t in args.to if t not in ("onnx", "openvino", "ncnn", "mnn")]
            if unsupported:
                raise ExportError(f"an ONNX file converts to openvino, ncnn and mnn here; {', '.join(unsupported)} needs the PyTorch model (CoreML and ExecuTorch do not read ONNX)")
            onnx = need("onnx", "reading the ONNX file")
            graph = onnx.load(str(onnx_path)).graph
            initializers = {i.name for i in graph.initializer}
            real_inputs = [i for i in graph.input if i.name not in initializers]
            shapes = [tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in real_inputs]
            if any(d <= 0 for s in shapes for d in s):
                raise ExportError(f"the ONNX inputs {shapes} are not fixed: re-export with a static shape, or fix them (see docs/agents/export-model.md)")
            elem = {1: "float32", 7: "int64", 6: "int32", 10: "float16", 2: "uint8"}
            dtypes = [elem.get(i.type.tensor_type.elem_type, "float32") for i in real_inputs]
        else:
            if not args.shape:
                raise ExportError("a PyTorch source needs --shape (the shape you will really feed; it is not guessed)")
            shapes = parse_shapes(args.shape)
            dtypes = (args.dtype or ["float32"] * len(shapes))
            if len(dtypes) != len(shapes):
                raise ExportError("give one --dtype per --shape")
            model = load_torch_model(args.source, args.state_dict)
        inputs = make_inputs(shapes, dtypes)
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    failures = 0
    reference = None
    results = []
    for target in args.to:
        backend, extension = TARGETS[target]
        path = out_dir / f"{name}{extension}"
        try:
            if target == "onnx":
                path = export_onnx(model, inputs, path, args.opset) if model is not None else onnx_path
                if model is not None:
                    onnx_path = path
            elif target == "coreml":
                path = export_coreml(model, inputs, path)
            elif target == "executorch":
                path = export_executorch(model, inputs, path, args.delegate)
            elif target == "ncnn":
                path = export_ncnn(model, inputs, path) if model is not None else export_ncnn_from_onnx(onnx_path, shapes, path)
                # NCNN shapes have no batch dimension, and its default on an Arm CPU is half precision (the check wants float32)
                per_input = ";".join("x".join(str(d) for d in (s[1:] if len(s) > 1 and s[0] == 1 else s)) for s in shapes)
                options = {"input_shapes": per_input, "fp16": "false", **options}
            elif target == "mnn":
                if onnx_path is None:  # a PyTorch source goes through ONNX first
                    onnx_path = export_onnx(model, inputs, out_dir / f"{name}.onnx", args.opset)
                path = export_mnn_from_onnx(onnx_path, path)
                options = {"precision": "high", **options}  # float32 compute for the check
            elif target == "openvino":
                options = {"inference_precision": "f32", **options}  # OpenVINO's default on an Arm CPU is half precision
                if onnx_path is None:  # a PyTorch source goes through ONNX first
                    onnx_path = export_onnx(model, inputs, out_dir / f"{name}.onnx", args.opset)
                path = export_openvino(onnx_path, path)
            print(f"[{target}] wrote {path}")
            if args.no_verify:
                results.append((target, "exported, not checked"))
                continue
            if reference is None:
                reference = reference_outputs(args.source, model, onnx_path if model is None else None, inputs)
            got = run_uairt(path, backend, inputs, options)
            ok, lines = compare(reference, got, args.atol)
            for line in lines:
                print(f"[{target}]   {line}")
            results.append((target, "ok" if ok else "FAILED the check"))
            failures += 0 if ok else 1
        except ExportError as error:
            print(f"[{target}] error: {error}", file=sys.stderr)
            results.append((target, "error"))
            failures += 1
        except Exception as error:  # a converter's own failure: report it with the target, keep going with the others
            print(f"[{target}] {type(error).__name__}: {error}", file=sys.stderr)
            results.append((target, "error"))
            failures += 1

    print("\nsummary: " + "; ".join(f"{t}: {r}" for t, r in results))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
