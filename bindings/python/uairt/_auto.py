# SPDX-License-Identifier: Apache-2.0
"""AutoModel: choose the backend from the model file and the device, then load the model.

It is a convenience layer over Engine and Model, and it is strict. The first candidate backend that is available
(registered, or with a plugin that can be found) and supports the requested device is used; if loading it fails, that
error is raised. There is no fallback to another runtime or to the CPU, and `plan.skipped` says why other candidates
were passed over. Pass `backend=` to force one.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import _plugins
from ._core import (Buffer, Engine, InvalidArgument, Model, NotFound, Session, TensorInfo, Unsupported, backends)
from ._generate import generate as _generate

# File suffix -> backends that can load it, most preferred first.
FORMATS: Dict[str, List[str]] = {
    ".dlc": ["qnn"],
    ".bin": ["qnn"],  # a QNN context binary; NCNN's .bin is found next to its .param
    ".onnx": ["onnxruntime", "openvino"],
    ".xml": ["openvino"],
    ".param": ["ncnn"],
    ".engine": ["tensorrt"],
    ".plan": ["tensorrt"],
    ".tflite": ["tflite"],
    ".mlmodelc": ["coreml"],
    ".mlpackage": ["coreml"],
    ".mlmodel": ["coreml"],
    ".gguf": ["llamacpp"],
}

_KINDS = ("cpu", "gpu", "npu", "cuda", "tensorrt", "vulkan")
_DEVICE = re.compile(r"^(?P<kind>[a-z]+)(?::(?P<index>\d+))?$")


class _Skip(Exception):
    """A backend cannot serve the requested device."""


def parse_device(device: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    """"gpu:1" -> ("gpu", 1). None, "" and "auto" mean the backend's own default: (None, None)."""
    if device is None or str(device).strip().lower() in ("", "auto"):
        return None, None
    match = _DEVICE.match(str(device).strip().lower())
    if not match or match.group("kind") not in _KINDS:
        raise InvalidArgument(1, f"device must be one of {', '.join(_KINDS)}, optionally with an index "
                                 f"such as 'gpu:1', or 'auto'; got {device!r}")
    index = match.group("index")
    return match.group("kind"), int(index) if index is not None else None


def _no_index(name: str, index: Optional[int]) -> None:
    if index is not None:
        raise _Skip(f"{name} does not take a device index here; pass backend options instead")


def _qnn(kind, index):
    _no_index("qnn", index)
    if kind in (None, "npu"):
        return {"device": "npu"}  # QNN has no default device, and a .dlc or context binary targets the NPU
    if kind in ("cpu", "gpu"):
        return {"device": kind}
    raise _Skip(f"qnn runs on cpu, gpu or npu, not {kind}")


def _onnxruntime(kind, index):
    if kind is None:
        return {}
    if kind == "cpu":
        _no_index("onnxruntime on the cpu", index)
        return {"execution_provider": "cpu"}
    if kind in ("gpu", "cuda", "tensorrt"):
        options = {"execution_provider": "tensorrt" if kind == "tensorrt" else "cuda"}
        if index is not None:
            options["ep_option.device_id"] = str(index)
        return options
    raise _Skip(f"onnxruntime has no built-in {kind} provider; for an NPU use a plugin execution provider "
                "(backend='onnxruntime' with options ep_library and ep_name)")


def _openvino(kind, index):
    if kind is None:
        return {}
    if kind == "cpu":
        _no_index("openvino on the cpu", index)
        return {"device": "CPU"}
    if kind == "gpu":
        return {"device": "GPU" if index is None else f"GPU.{index}"}
    if kind == "npu":
        _no_index("openvino on the npu", index)
        return {"device": "NPU"}
    raise _Skip(f"openvino has no {kind} device (it runs on Intel cpu, gpu and npu)")


def _ncnn(kind, index):
    if kind is None:
        return {}
    if kind == "cpu":
        _no_index("ncnn on the cpu", index)
        return {"device": "cpu"}
    if kind in ("gpu", "vulkan"):
        return {"device": "vulkan", "vulkan_device": str(index or 0)}
    raise _Skip(f"ncnn runs on the cpu or a Vulkan gpu, not {kind}")


def _tensorrt(kind, index):
    if kind in (None, "gpu", "cuda", "tensorrt"):
        return {} if index is None else {"device": str(index)}
    raise _Skip(f"tensorrt runs on an NVIDIA gpu, not {kind}")


def _tflite(kind, index):
    if kind in (None, "cpu"):
        _no_index("tflite", index)
        return {}
    raise _Skip(f"tflite on {kind} needs a delegate: pass backend='tflite' with options such as delegate='qnn', "
                "delegate_library and backend_library")


def _coreml(kind, index):
    _no_index("coreml", index)
    units = {None: None, "cpu": "cpu_only", "gpu": "cpu_and_gpu", "npu": "cpu_and_ne"}
    if kind not in units:
        raise _Skip(f"coreml runs on the cpu, gpu or neural engine, not {kind}")
    return {} if units[kind] is None else {"compute_units": units[kind]}


def _llamacpp(kind, index):
    _no_index("llamacpp", index)
    layers = {None: None, "cpu": "0", "gpu": "99"}
    if kind not in layers:
        raise _Skip(f"llamacpp runs on the cpu or gpu, not {kind}")
    return {} if layers[kind] is None else {"n_gpu_layers": layers[kind]}


_MAPPERS: Dict[str, Callable] = {
    "qnn": _qnn, "onnxruntime": _onnxruntime, "openvino": _openvino, "ncnn": _ncnn, "tensorrt": _tensorrt,
    "tflite": _tflite, "coreml": _coreml, "llamacpp": _llamacpp,
}


def candidates_for(path) -> List[str]:
    """Backends that can load this file, most preferred first. A directory is judged by what it holds, so an
    ultralytics `_openvino_model` or `_ncnn_model` export works."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix in FORMATS:
        return list(FORMATS[suffix])
    if target.is_dir():
        held = {entry.suffix.lower() for entry in target.iterdir()}
        for inside in (".xml", ".param"):
            if inside in held:
                return list(FORMATS[inside])
    raise Unsupported(3, f"cannot tell which backend runs '{path}' ({suffix or 'no extension'}); known formats: "
                         f"{', '.join(sorted(FORMATS))}; pass backend= to choose")


@dataclass(frozen=True)
class Plan:
    """What AutoModel decided, before anything is loaded."""
    backend: str
    device: Optional[str]  # as requested; None means the backend's default
    options: Dict[str, str]  # engine options: the device mapping, then yours (yours win)
    skipped: Dict[str, str] = field(default_factory=dict)  # candidates passed over, and why
    forced: bool = False  # True when backend= was given


def resolve(path, device: Optional[str] = None, backend: Optional[str] = None,
            options: Optional[Dict[str, object]] = None,
            available: Optional[Callable[[str], bool]] = None) -> Plan:
    """Chooses the backend and engine options for a model file and device without loading anything.
    `available` replaces the plugin check, which lets the choice be tested without plugins."""
    kind, index = parse_device(device)
    check = available or _plugins.is_available
    user = {str(key): str(value) for key, value in (options or {}).items()}
    names = [backend] if backend else candidates_for(path)
    skipped: Dict[str, str] = {}
    device_reasons = 0
    for name in names:
        mapper = _MAPPERS.get(name)
        try:
            if mapper is None:
                if kind is not None:
                    raise _Skip(f"no device mapping is known for backend '{name}'; pass options instead")
                mapped: Dict[str, str] = {}
            else:
                mapped = mapper(kind, index)
        except _Skip as why:
            skipped[name] = str(why)
            device_reasons += 1
            continue
        if not check(name):
            skipped[name] = ("no plugin found (registered: " + (", ".join(backends()) or "none")
                             + "; searched: " + (", ".join(str(d) for d in _plugins.plugin_dirs()) or "nothing") + ")")
            continue
        return Plan(name, device, {**mapped, **user}, skipped, bool(backend))
    wanted = f"device '{device}'" if device else "the default device"
    message = f"cannot run '{path}' on {wanted}: " + "; ".join(f"{name}: {why}" for name, why in skipped.items())
    raise (Unsupported(3, message) if device_reasons == len(skipped) else NotFound(2, message))


def available_backends() -> Dict[str, Dict[str, object]]:
    """Every backend name the package knows or has found, with whether it is registered and where its plugin is."""
    plugins = _plugins.discover_plugins()
    known = set(_MAPPERS) | {name for names in FORMATS.values() for name in names} | set(plugins) | set(backends())
    registered = set(backends())
    return {name: {"registered": name in registered, "plugin": plugins.get(name)} for name in sorted(known)}


class AutoModel:
    """A model loaded through the backend `resolve` chose. It owns its engine and releases both on close.

        model = AutoModel.from_file("yolov8n.onnx", device="gpu")
        print(model.backend, model.options)      # which runtime ran it, and how it was configured
        (y,) = model.run(x)
    """

    def __init__(self, engine: Engine, model: Model, plan: Plan, path: str):
        self.engine = engine
        self.model = model
        self.plan = plan
        self.path = path

    @classmethod
    def resolve(cls, path, *, device: Optional[str] = None, backend: Optional[str] = None,
                options: Optional[Dict[str, object]] = None, **option_kwargs) -> Plan:
        """The plan from_file would follow, without loading the model."""
        return resolve(path, device, backend, {**(options or {}), **option_kwargs})

    @classmethod
    def from_file(cls, path, *, device: Optional[str] = None, backend: Optional[str] = None,
                  options: Optional[Dict[str, object]] = None, format: Optional[str] = None,
                  **option_kwargs) -> "AutoModel":
        """Loads a model file. `device` is cpu, gpu, npu, cuda, tensorrt or vulkan (optionally 'gpu:1'), or None for the
        backend's default; `options` and keyword arguments go to the backend and override the device mapping."""
        plan = cls.resolve(path, device=device, backend=backend, options=options, **option_kwargs)
        engine = Engine(plan.backend, plan.options)  # loads the backend's plugin if it is not registered yet
        try:
            model = engine.load_model(str(path), format=format)
        except BaseException:
            engine.close()
            raise
        return cls(engine, model, plan, str(path))

    @property
    def backend(self) -> str:
        return self.plan.backend

    @property
    def device(self) -> Optional[str]:
        return self.plan.device

    @property
    def options(self) -> Dict[str, str]:
        return self.plan.options

    @property
    def inputs(self) -> List[TensorInfo]:
        return self.model.inputs

    @property
    def outputs(self) -> List[TensorInfo]:
        return self.model.outputs

    def run(self, *inputs: np.ndarray, out: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, ...]:
        return self.model.run(*inputs, out=out)

    def run_buffers(self, inputs: Sequence[Buffer], outputs: Sequence[Buffer]) -> None:
        self.model.run_buffers(inputs, outputs)

    def alloc_buffer(self, nbytes: int, domain: str = "host") -> Buffer:
        return self.engine.alloc_buffer(nbytes, domain)

    @property
    def vocab_size(self) -> int:
        return self.model.vocab_size

    def tokenize(self, text: str, add_special: bool = True) -> np.ndarray:
        return self.model.tokenize(text, add_special)

    def detokenize(self, tokens) -> str:
        return self.model.detokenize(tokens)

    def detokenize_bytes(self, tokens) -> bytes:
        return self.model.detokenize_bytes(tokens)

    def session(self, options: Optional[Dict[str, object]] = None, **kwargs) -> Session:
        return self.model.session(options, **kwargs)

    def generate(self, prompt: str, **kwargs):
        """Yields text pieces; see uairt.generate."""
        return _generate(self, prompt, **kwargs)

    def close(self) -> None:
        self.model.close()
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"AutoModel({self.path!r}, backend={self.backend!r}, device={self.device!r})"


def load(path, **kwargs) -> AutoModel:
    """Shorthand for AutoModel.from_file."""
    return AutoModel.from_file(path, **kwargs)
