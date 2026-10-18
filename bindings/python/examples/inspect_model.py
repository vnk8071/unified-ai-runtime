# SPDX-License-Identifier: Apache-2.0
"""Show a model's inputs and outputs, and a first guess at the pre- and post-processing it needs.

    python inspect_model.py model.onnx [--device cpu|gpu|npu] [--backend NAME] [--option key=value ...]
    python inspect_model.py yolov8n_ncnn_model --option input_shapes=3x640x640
    python inspect_model.py model.pte --backend executorch

The table is fact: it is what the backend reports for the loaded model. The "Looks like" lines are a heuristic from names, dtypes
and shapes, and a hint of where to read next in docs/agents/pre-post-processing.md; confirm every one against the model card or the
original framework before relying on it. Needs the native library and the backend plugin (see bindings/python/README.md).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GUIDE = "docs/agents/pre-post-processing.md"
TEXT_INPUT_NAMES = ("input_ids", "input_id", "token_ids", "tokens", "ids")


def describe(tensor) -> str:
    """One line for a tensor: name, dtype, shape, and its quantization when it has one."""
    quant = f"  quantized: scale={tensor.quant_scale:g} zero_point={tensor.quant_zero_point}" if tensor.quant_scale else ""
    return f"{tensor.name or '(unnamed)':<28} {str(tensor.dtype):<8} {tuple(tensor.shape)}{quant}"


def _is_float(dtype) -> bool:
    return str(dtype).startswith(("float", "bfloat"))


def _is_int(dtype) -> bool:
    return str(dtype).startswith(("int", "uint"))


def _image_layout(shape):
    """('NCHW' | 'NHWC', height, width) for a 4-D shape that looks like an image batch, else None."""
    if len(shape) != 4:
        return None
    if shape[1] in (1, 3, 4) and shape[2] >= 16 and shape[3] >= 16:
        return "NCHW", shape[2], shape[3]
    if shape[3] in (1, 3, 4) and shape[1] >= 16 and shape[2] >= 16:
        return "NHWC", shape[1], shape[2]
    return None


def _image_outputs(outputs, input_size):
    notes = []
    names = [o.name.lower() for o in outputs]
    if len(outputs) >= 2 and any("box" in n for n in names) and any("score" in n or "class" in n for n in names):
        return [f"a detector with the box decoding and NMS built in (SSD style): read boxes, scores and classes from the outputs directly ({GUIDE} section 5)"]
    for out in outputs:
        shape = tuple(out.shape)
        if len(shape) == 2 and shape[1] >= 2:
            notes.append(f"output '{out.name}': a classifier over {shape[1]} classes, or an embedding; softmax + top-k ({GUIDE} section 5); labels come from the model card")
        elif len(shape) == 3 and shape[2] > shape[1] >= 5 and shape[2] >= 100:
            notes.append(f"output '{out.name}': a YOLO-style head, {shape[1]} features x {shape[2]} anchors (4 box values + {shape[1] - 4} class scores, or 5 + {shape[1] - 5} with objectness); decode_yolo + NMS ({GUIDE} section 5)")
        elif len(shape) == 3 and shape[1] > shape[2] >= 5 and shape[1] >= 100:
            notes.append(f"output '{out.name}': a YOLO-style head, {shape[1]} anchors x {shape[2]} features; decode_yolo + NMS ({GUIDE} section 5)")
        elif len(shape) == 3 and shape[1] < 1000:
            notes.append(f"output '{out.name}': {shape[1]} tokens of {shape[2]} features, probably a vision transformer's features; pool or take the class token ({GUIDE} section 5)")
        elif len(shape) == 4 and shape[2] >= 8 and shape[3] >= 8:
            notes.append(f"output '{out.name}': a dense map of {shape[1]} channel(s) at {shape[2]}x{shape[3]}: segmentation (argmax over channels), depth (1 channel) or heatmaps; upsample to the image ({GUIDE} section 5)")
    return notes


def guess(inputs, outputs, backend=None) -> list:
    """Lines saying what the model looks like and what to read next. Works on any objects with name, dtype, shape and quant_scale."""
    if not inputs and not outputs:
        return [f"a language model or a server model (no tensors): tokenize, chat template and decoding are yours ({GUIDE} sections 6 and 7)"]
    notes = []
    names = [i.name.lower() for i in inputs]
    first = inputs[0] if inputs else None

    if any(("past" in n or "cache" in n or "key_values" in n) for n in names):
        notes.append(f"a decoder with a KV cache as inputs: those have dynamic shapes, which UAIRT does not support; use a GGUF model with the llamacpp backend ({GUIDE} section 7)")
    text_inputs = [i for i in inputs if _is_int(i.dtype) and len(i.shape) == 2 and (i.name.lower() in TEXT_INPUT_NAMES or "input_ids" in i.name.lower())]
    if text_inputs:
        length = text_inputs[0].shape[1]
        notes.append(f"a text model: tokenize with the tokenizer published with the model, pad or truncate to exactly {length} tokens, "
                     f"and give attention_mask / token_type_ids when the model has them ({GUIDE} section 6)")
        for out in outputs:
            n, shape = out.name.lower(), tuple(out.shape)
            if "start" in n or "end" in n:
                notes.append(f"output '{out.name}': extractive question answering logits, argmax of start and end ({GUIDE} section 5)")
            elif len(shape) == 3 and shape[2] >= 20000:
                notes.append(f"output '{out.name}': next-token logits over {shape[2]} tokens: a language model head; a GGUF with llamacpp is the supported route ({GUIDE} section 7)")
            elif len(shape) == 3:
                notes.append(f"output '{out.name}': {shape[1]} tokens x {shape[2]} features: embeddings (mean_pool + l2_normalize) or token classification (argmax over {shape[2]} labels) ({GUIDE} section 5)")
            elif len(shape) == 2 and ("pool" in n or "embed" in n):
                notes.append(f"output '{out.name}': a pooled vector of {shape[1]} features per sentence: use it as the embedding, l2_normalize ({GUIDE} section 5)")
            elif len(shape) == 2 and "logit" in n:
                notes.append(f"output '{out.name}': {shape[1]} logits: sequence classification, softmax + argmax; labels from the model's config ({GUIDE} section 5)")
            elif len(shape) == 2:
                notes.append(f"output '{out.name}': a [1, {shape[1]}] vector: classification logits or a pooled embedding, depending on the model; read its card ({GUIDE} section 5)")
        return notes

    if first is not None and len(first.shape) == 4:
        layout = _image_layout(tuple(first.shape))
        if layout:
            name, height, width = layout
            if first.quant_scale:
                how = "integers with a scale and zero point: quantize the preprocessed pixels"
            elif _is_int(first.dtype):
                how = f"{first.dtype}: raw 0-255 pixels as they are, unless the model card says otherwise"
            else:
                how = f"{first.dtype}: scaled to 0-1 or -1..1, or normalized with a mean and std per channel; the model card says which"
            notes.append(f"an image model, layout {name}, {height}x{width}, input is {how} ({GUIDE} section 4)")
            notes.append(f"the shape is static: resize, crop or letterbox to exactly {height}x{width}, in the colour order and range the model was trained with")
            notes.extend(_image_outputs(outputs, (height, width)))
            return notes

    if first is not None and _is_float(first.dtype) and len(first.shape) == 1 and first.shape[0] >= 1600:
        notes.append(f"a waveform of {first.shape[0]} samples (mono, usually 16 kHz, values in [-1, 1]; samples / sample rate = seconds): "
                     f"decode, resample, pad or trim to exactly that length ({GUIDE} section 4)")
    elif first is not None and _is_float(first.dtype) and len(first.shape) == 2 and first.shape[1] >= 1600:
        notes.append(f"a waveform of {first.shape[1]} samples (see above): decode, resample, pad or trim to exactly that length ({GUIDE} section 4)")
    elif first is not None and _is_float(first.dtype) and len(first.shape) == 3 and first.shape[1] in (40, 64, 80, 128):
        notes.append(f"log-mel features, {first.shape[1]} mel bins x {first.shape[2]} frames (Whisper-style: 80 x 3000); the model's own feature extractor computes them ({GUIDE} section 4)")
    for out in outputs:
        if str(out.dtype) == "uint8" and out.name.lower() in ("text", "transcript"):
            notes.append(f"output '{out.name}': UTF-8 text bytes, zero padded; the length is in the next output (this is the whispercpp convention, docs/backends/whispercpp.md)")
        elif len(out.shape) == 3 and first is not None and len(first.shape) <= 2 and out.shape[2] >= 20:
            notes.append(f"output '{out.name}': {out.shape[1]} steps x {out.shape[2]} symbols: CTC logits if this is speech recognition; ctc_greedy then the model's vocabulary ({GUIDE} section 5)")
    if any(i.quant_scale for i in inputs) or any(o.quant_scale for o in outputs):
        notes.append(f"quantized tensors: integers in, integers out; quantize inputs and dequantize outputs with the printed scale and zero point ({GUIDE} section 4)")
    return notes or [f"no known pattern: read the model card and the original framework's pipeline ({GUIDE} section 2)"]


def find_shared_builds(root: Path) -> list:
    """Shared builds of libuairt under root (build*/libuairt.dylib, .so, or uairt.dll), newest first."""
    patterns = ("build*/libuairt.dylib", "build*/libuairt.so", "build*/Release/uairt.dll", "build*/Debug/uairt.dll")
    found = [path for pattern in patterns for path in root.glob(pattern)]
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def setup_help(root: Path, error=None) -> str:
    """What to do when the Python binding cannot be loaded: the exact commands, using an existing shared build when there is one."""
    lines = []
    if error is not None:
        lines.append(f"cannot load the uairt Python binding: {error}")
    missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
    if missing and missing != "uairt" and not missing.startswith("uairt."):
        # the binding was found; one of its own dependencies is not installed, which more build steps cannot fix
        lines += [f"The binding itself was found; it needs the Python package '{missing}', which this interpreter ({sys.executable}) does not have:",
                  f"  {sys.executable} -m pip install {missing}      # or: uv pip install --python {sys.executable} {missing}",
                  "Ask before installing into the user's own environment; a throwaway virtual environment is fine."]
        return "\n".join(lines)
    shared = find_shared_builds(root)
    if shared:
        lines += ["A shared build exists. Point the binding at it, then run this script again (the plugins next to libuairt are found by themselves):",
                  f"  export PYTHONPATH={root}/bindings/python UAIRT_LIBRARY={shared[0]}"]
    else:
        lines += ["The binding needs a shared build of the library plus the plugin for your backend. From the repository root:",
                  "  cmake -S . -B build-shared -DBUILD_SHARED_LIBS=ON -DUAIRT_BUILD_<BACKEND>=ON && cmake --build build-shared",
                  "  export PYTHONPATH=$PWD/bindings/python UAIRT_LIBRARY=$PWD/build-shared/libuairt.dylib   # libuairt.so on Linux, uairt.dll on Windows",
                  "(<BACKEND> is ONNXRUNTIME, LLAMACPP, NCNN, ...: see docs/backends/.)"]
    lines.append("For a quick look without Python: build/run_model --info <plugin> <backend> <model>   (a language model shows no tensors)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--device", default=None, help="cpu, gpu, npu, ...; default: the backend's own")
    parser.add_argument("--backend", default=None, help="force a backend instead of choosing by file type")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="engine option, repeatable")
    args = parser.parse_args(argv)
    options = dict(item.split("=", 1) for item in args.option)

    try:
        import uairt
    except (ImportError, OSError) as error:        # no PYTHONPATH, or the shared library was not found
        print(setup_help(REPO_ROOT, error), file=sys.stderr)
        return 2
    try:
        model = uairt.AutoModel.from_file(args.model, device=args.device, backend=args.backend, options=options or None)
    except uairt.UairtError as error:
        print(f"cannot load {args.model}: {error}", file=sys.stderr)
        print("If it says no usable plugin was found, build that backend (-DUAIRT_BUILD_<BACKEND>=ON, see docs/backends/) and keep it next to libuairt;\n"
              "a .bin file goes to QNN unless you pass --backend, and NCNN needs --option input_shapes=....", file=sys.stderr)
        return 1
    with model:
        print(f"backend: {model.backend}   options: {model.options}")
        print("inputs:")
        for tensor in model.inputs:
            print("  " + describe(tensor))
        print("outputs:")
        for tensor in model.outputs:
            print("  " + describe(tensor))
        if not model.inputs and not model.outputs:
            try:
                print(f"vocabulary: {model.vocab_size} tokens")
            except Exception:
                pass
        print("Looks like (a heuristic, confirm it):")
        for line in guess(model.inputs, model.outputs, model.backend):
            print("  - " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
