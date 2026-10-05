# SPDX-License-Identifier: Apache-2.0
"""Finds backend plugins (libuairt_backend_<name>.so, .dylib or .dll) and loads them on demand."""
import os
from pathlib import Path
from typing import Dict, List, Optional

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
    load_backend_library(str(path))
    if backend not in backends():
        raise NotFound(2, f"{path} did not register a backend named '{backend}' (it registered: "
                          f"{', '.join(backends())})")
    return path
