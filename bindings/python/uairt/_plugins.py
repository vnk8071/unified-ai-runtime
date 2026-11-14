# SPDX-License-Identifier: Apache-2.0
"""Finds backend plugins (libuairt_backend_<name>.so, .dylib or .dll) and loads them on demand."""
import importlib.util
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import _lib
from ._core import NotFound, backends, load_backend_library

PREFIX = "libuairt_backend_"
SUFFIXES = (".so", ".dylib", ".dll")


def plugin_dirs() -> List[Path]:
    """Directories searched for plugins, in order: UAIRT_PLUGIN_PATH, the package's plugins/, the directory of the
    loaded libuairt and its uairt/ subdirectory (the install layout lib/uairt)."""
    dirs: List[Path] = [Path(entry) for entry in os.environ.get("UAIRT_PLUGIN_PATH", "").split(os.pathsep) if entry]
    dirs.append(Path(__file__).resolve().parent / "plugins")
    if _lib.library_path:
        dirs.extend([_lib.library_path.parent, _lib.library_path.parent / "uairt"])
    seen, unique = set(), []
    for directory in dirs:
        if directory not in seen:
            seen.add(directory)
            unique.append(directory)
    return unique


def discover_plugins() -> Dict[str, Path]:
    """Plugins found in the search directories, by backend name. The first directory wins."""
    found: Dict[str, Path] = {}
    for directory in plugin_dirs():
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith(PREFIX) and entry.suffix in SUFFIXES:
                found.setdefault(entry.name[len(PREFIX):-len(entry.suffix)], entry)
    return found


def find_plugin(backend: str) -> Optional[Path]:
    return discover_plugins().get(backend)


def is_available(backend: str) -> bool:
    """True when the backend is registered or a plugin for it can be found. It does not check that the vendor runtime
    the plugin needs is installed; loading the plugin reports that."""
    return backend in backends() or find_plugin(backend) is not None


def _onnxruntime_library() -> Optional[Path]:
    """The library inside a pip-installed onnxruntime, found without importing it."""
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        capi = Path(location) / "capi"
        for pattern in ("onnxruntime.dll", "libonnxruntime.so", "libonnxruntime.so.*", "libonnxruntime.dylib",
                        "libonnxruntime.*.dylib"):
            for entry in sorted(capi.glob(pattern)):
                return entry
    return None


# Vendor runtimes that ship as Python packages: where a plugin can find them, set before the plugin loads.
_RUNTIME_LOCATORS: Dict[str, Callable[[], Optional[Path]]] = {"onnxruntime": _onnxruntime_library}
_RUNTIME_VARIABLES = {"onnxruntime": "UAIRT_ONNXRUNTIME_LIBRARY"}

# os.add_dll_directory() handles must stay alive for the process, or Windows drops the directory again.
_DLL_DIRECTORIES: List[object] = []


def _add_dll_directory(library: Path) -> None:
    """Makes a plugin's bundled dependencies findable on Windows."""
    if os.name != "nt":
        return
    try:
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(library.parent)))
    except (AttributeError, OSError):
        os.environ["PATH"] = str(library.parent) + os.pathsep + os.environ.get("PATH", "")


def onnxruntime_qnn_plugin() -> Optional[Tuple[str, Path]]:
    """The QNN execution-provider plugin from the `onnxruntime-qnn` package, as (ep_name, library), or None.

    The wheel bundles the QNN libraries next to the plugin. On Windows that directory is added to the DLL search path,
    so QnnHtp.dll and its dependencies are found when ONNX Runtime registers the provider.
    """
    if importlib.util.find_spec("onnxruntime_qnn") is None:
        return None
    try:
        import onnxruntime_qnn
    except ImportError:
        return None
    library = Path(onnxruntime_qnn.get_library_path())
    if not library.is_file():
        return None
    _add_dll_directory(library)
    return onnxruntime_qnn.get_ep_name(), library


# Plugin execution providers with a Python package that ships their library. The key is the name passed to ONNX Runtime.
_EP_PACKAGES = {
    "QNNExecutionProvider": "onnxruntime_qnn",
    "MLXExecutionProvider": "onnxruntime_ep_mlx",
    "WebGpuExecutionProvider": "onnxruntime_ep_webgpu",
}


def plugin_library_for_ep(ep_name: str) -> Optional[Path]:
    """The library that provides a named plugin execution provider, when a known package ships it; else None."""
    package = _EP_PACKAGES.get(ep_name)
    if package is None or importlib.util.find_spec(package) is None:
        return None
    module = importlib.import_module(package)
    getter = getattr(module, "get_library_path", None) or getattr(module, "library_path", None)
    if getter is None:
        return None
    library = Path(getter() if callable(getter) else getter)
    if not library.is_file():
        return None
    _add_dll_directory(library)
    return library


def ensure_backend(backend: str) -> Optional[Path]:
    """Registers the backend by loading its plugin if needed. Returns the plugin path, or None if it was already
    registered. Raises NotFound when there is no plugin, and the loader's error when the plugin cannot load."""
    if backend in backends():
        return None
    path = find_plugin(backend)
    if path is None:
        searched = ", ".join(str(d) for d in plugin_dirs()) or "nothing"
        raise NotFound(2, f"backend '{backend}' is not registered and no plugin was found "
                          f"(searched {searched}; set UAIRT_PLUGIN_PATH or call load_backend_library)")
    variable = _RUNTIME_VARIABLES.get(backend)
    if variable and not os.environ.get(variable):
        located = _RUNTIME_LOCATORS[backend]()
        if located is not None:
            os.environ[variable] = str(located)  # the plugin loads its runtime from here, so a pip install is used
    load_backend_library(str(path))
    if backend not in backends():
        raise NotFound(2, f"{path} did not register a backend named '{backend}' (it registered: "
                          f"{', '.join(backends())})")
    return path
