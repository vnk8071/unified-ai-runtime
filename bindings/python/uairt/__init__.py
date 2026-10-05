# SPDX-License-Identifier: Apache-2.0
"""Python binding for the Unified AI Runtime.

    import uairt
    with uairt.AutoModel.from_file("model.onnx", device="gpu") as model:     # backend chosen from the file and device
        (y,) = model.run(x)

    with uairt.Engine("onnxruntime") as engine, engine.load_model("model.onnx") as model:   # or choose it yourself
        (y,) = model.run(x)

Backend plugins are found on UAIRT_PLUGIN_PATH, in the package's plugins/ directory and next to libuairt, and loaded
when an engine asks for them; uairt.load_backend_library(path) still loads one by path.
"""
from ._core import (  # noqa: F401
    Buffer, Engine, Model, Session, TensorInfo, UairtError, backends, load_backend_library, version,
    InvalidArgument, NotFound, Unsupported, BackendUnavailable, IncompatibleModel, OutOfMemory, RuntimeFailure,
    VersionMismatch, IoFailure,
)
from ._auto import AutoModel, Plan, available_backends, load, resolve  # noqa: F401
from ._generate import generate  # noqa: F401
from ._plugins import discover_plugins, ensure_backend, find_plugin, plugin_dirs  # noqa: F401

__all__ = [
    "Buffer", "Engine", "Model", "Session", "generate", "TensorInfo", "UairtError", "backends", "load_backend_library", "version",
    "AutoModel", "Plan", "available_backends", "load", "resolve", "discover_plugins", "ensure_backend", "find_plugin",
    "plugin_dirs",
]
