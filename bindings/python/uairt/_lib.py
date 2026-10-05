# SPDX-License-Identifier: Apache-2.0
"""Loads libuairt and declares the C API with ctypes. Layouts must match include/uairt/uairt.h."""
import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

UAIRT_MAX_RANK = 8

STATUS_NAMES = {
    0: "ok", 1: "invalid argument", 2: "not found", 3: "unsupported", 4: "backend unavailable",
    5: "incompatible model", 6: "out of memory", 7: "runtime error", 8: "version mismatch", 9: "io error",
}

MEM_HOST = 1 << 0
MEM_DMABUF = 1 << 1
MEM_PINNED = 1 << 2


class Tensor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("dtype", ctypes.c_int32),
        ("rank", ctypes.c_uint32),
        ("dims", ctypes.c_int64 * UAIRT_MAX_RANK),
        ("quant_scale", ctypes.c_float),
        ("quant_zero_point", ctypes.c_int32),
        ("domain", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("dmabuf_fd", ctypes.c_int32),
        ("nbytes", ctypes.c_size_t),
        ("name", ctypes.c_char_p),
    ]


class Option(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_char_p)]


class ModelSource(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("format", ctypes.c_char_p),
    ]


def _candidates():
    names = {"darwin": ["libuairt.dylib"], "win32": ["uairt.dll"]}.get(sys.platform, ["libuairt.so"])
    env = os.environ.get("UAIRT_LIBRARY")
    if env:
        yield Path(env)
    here = Path(__file__).resolve().parent
    for name in names:
        yield here / name
    repo = here.parents[2]
    for build in sorted(repo.glob("build*")):
        for name in names:
            yield build / name
    found = ctypes.util.find_library("uairt")
    if found:
        yield Path(found)


library_path = None  # where libuairt was loaded from, when that is a file; plugin discovery starts there


def load():
    global library_path
    tried = []
    for path in _candidates():
        tried.append(str(path))
        if path.exists() or not path.is_absolute():
            try:
                handle = ctypes.CDLL(str(path))
            except OSError:
                continue
            library_path = path.resolve() if path.exists() else None
            return handle
    raise OSError(
        "cannot find libuairt; build it with -DBUILD_SHARED_LIBS=ON and set UAIRT_LIBRARY to its path "
        f"(tried: {', '.join(tried) or 'nothing'})")


lib = load()

c_void_pp = ctypes.POINTER(ctypes.c_void_p)


def _declare(name, restype, *argtypes):
    function = getattr(lib, name)
    function.restype = restype
    function.argtypes = list(argtypes)
    return function


version_string = _declare("uairt_version_string", ctypes.c_char_p)
status_string = _declare("uairt_status_string", ctypes.c_char_p, ctypes.c_int32)
last_error = _declare("uairt_last_error", ctypes.c_char_p)
dtype_size = _declare("uairt_dtype_size", ctypes.c_size_t, ctypes.c_int32)
tensor_init = _declare("uairt_tensor_init", None, ctypes.POINTER(Tensor))
backend_count = _declare("uairt_backend_count", ctypes.c_size_t)
backend_name = _declare("uairt_backend_name", ctypes.c_char_p, ctypes.c_size_t)
load_backend_library = _declare("uairt_load_backend_library", ctypes.c_int32, ctypes.c_char_p)
engine_create = _declare("uairt_engine_create", ctypes.c_int32, ctypes.c_char_p, ctypes.POINTER(Option),
                         ctypes.c_size_t, c_void_pp)
engine_destroy = _declare("uairt_engine_destroy", None, ctypes.c_void_p)
model_load = _declare("uairt_model_load", ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ModelSource), c_void_pp)
model_destroy = _declare("uairt_model_destroy", None, ctypes.c_void_p)
model_num_inputs = _declare("uairt_model_num_inputs", ctypes.c_size_t, ctypes.c_void_p)
model_num_outputs = _declare("uairt_model_num_outputs", ctypes.c_size_t, ctypes.c_void_p)
model_input_info = _declare("uairt_model_input_info", ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.POINTER(Tensor))
model_output_info = _declare("uairt_model_output_info", ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.POINTER(Tensor))
model_run = _declare("uairt_model_run", ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(Tensor), ctypes.c_size_t,
                     ctypes.POINTER(Tensor), ctypes.c_size_t)
buffer_alloc = _declare("uairt_buffer_alloc", ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                        c_void_pp)
buffer_free = _declare("uairt_buffer_free", None, ctypes.c_void_p)
buffer_data = _declare("uairt_buffer_data", ctypes.c_void_p, ctypes.c_void_p)
buffer_fd = _declare("uairt_buffer_fd", ctypes.c_int32, ctypes.c_void_p)
buffer_size = _declare("uairt_buffer_size", ctypes.c_size_t, ctypes.c_void_p)
tensor_use_buffer = _declare("uairt_tensor_use_buffer", None, ctypes.POINTER(Tensor), ctypes.c_void_p)
