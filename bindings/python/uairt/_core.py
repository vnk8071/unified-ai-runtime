# SPDX-License-Identifier: Apache-2.0
import ctypes
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import _lib

_DTYPES = {
    1: np.float32, 2: np.float16, 4: np.int8, 5: np.uint8, 6: np.int16, 7: np.int32, 8: np.int64,
    9: np.bool_, 10: np.uint16,
}


class UairtError(RuntimeError):
    """A UAIRT call failed. `status` is the C status code; `message` is uairt_last_error()."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.status_name = _lib.STATUS_NAMES.get(status, "unknown status")
        self.message = message
        super().__init__(f"{self.status_name}: {message}" if message else self.status_name)


def _subclass(name, status):
    cls = type(name, (UairtError,), {"STATUS": status})
    return cls


InvalidArgument = _subclass("InvalidArgument", 1)
NotFound = _subclass("NotFound", 2)
Unsupported = _subclass("Unsupported", 3)
BackendUnavailable = _subclass("BackendUnavailable", 4)
IncompatibleModel = _subclass("IncompatibleModel", 5)
OutOfMemory = _subclass("OutOfMemory", 6)
RuntimeFailure = _subclass("RuntimeFailure", 7)
VersionMismatch = _subclass("VersionMismatch", 8)
IoFailure = _subclass("IoFailure", 9)
_BY_STATUS = {c.STATUS: c for c in (InvalidArgument, NotFound, Unsupported, BackendUnavailable, IncompatibleModel,
                                     OutOfMemory, RuntimeFailure, VersionMismatch, IoFailure)}


def _check(status: int):
    if status != 0:
        message = (_lib.last_error() or b"").decode(errors="replace")
        raise _BY_STATUS.get(status, UairtError)(status, message)


def version() -> str:
    return _lib.version_string().decode()


def backends() -> List[str]:
    return [_lib.backend_name(i).decode() for i in range(_lib.backend_count())]


def load_backend_library(path: str) -> None:
    """Loads a backend plugin and registers it. Plugins stay loaded for the process lifetime."""
    _check(_lib.load_backend_library(str(path).encode()))


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    quant_scale: float
    quant_zero_point: int

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    def dequantize(self, values: np.ndarray) -> np.ndarray:
        """Real values from stored values: (q - zero_point) * scale. Needs a quantized tensor."""
        if not self.quant_scale:
            raise ValueError(f"tensor '{self.name}' is not quantized")
        return (values.astype(np.float32) - self.quant_zero_point) * self.quant_scale


def _info(tensor: _lib.Tensor) -> TensorInfo:
    dtype = _DTYPES.get(tensor.dtype)
    if dtype is None:
        raise Unsupported(3, f"tensor dtype code {tensor.dtype} has no NumPy equivalent")
    return TensorInfo(
        name=(tensor.name or b"").decode(), dtype=np.dtype(dtype),
        shape=tuple(tensor.dims[i] for i in range(tensor.rank)),
        quant_scale=tensor.quant_scale, quant_zero_point=tensor.quant_zero_point)


def _new_tensor() -> _lib.Tensor:
    tensor = _lib.Tensor()
    _lib.tensor_init(ctypes.byref(tensor))
    return tensor


class Buffer:
    """A zero-copy buffer allocated by a backend. Free it after the models that used it."""

    def __init__(self, engine: "Engine", nbytes: int, domain: str = "host"):
        domains = {"host": _lib.MEM_HOST, "dmabuf": _lib.MEM_DMABUF, "pinned": _lib.MEM_PINNED}
        if domain not in domains:
            raise InvalidArgument(1, f"domain must be 'host', 'dmabuf' or 'pinned', got {domain!r}")
        self._engine = engine
        handle = ctypes.c_void_p()
        _check(_lib.buffer_alloc(engine._handle, nbytes, domains[domain], ctypes.byref(handle)))
        self._handle = handle.value
        self.domain = domain

    @property
    def nbytes(self) -> int:
        return _lib.buffer_size(self._handle)

    @property
    def fd(self) -> int:
        return _lib.buffer_fd(self._handle)

    def array(self, dtype, shape) -> np.ndarray:
        """A NumPy view of the buffer (no copy). Valid until the buffer is freed."""
        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        if count * dtype.itemsize > self.nbytes:
            raise InvalidArgument(1, "the requested view is larger than the buffer")
        pointer = ctypes.cast(_lib.buffer_data(self._handle), ctypes.POINTER(ctypes.c_char))
        return np.ctypeslib.as_array(pointer, shape=(self.nbytes,))[: count * dtype.itemsize].view(dtype).reshape(shape)

    def close(self) -> None:
        if self._handle:
            _lib.buffer_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Model:
    def __init__(self, engine: "Engine", handle: int):
        self._engine = engine  # keeps the engine alive for the model's lifetime
        self._handle = handle
        self.inputs: List[TensorInfo] = []
        self.outputs: List[TensorInfo] = []
        for count, getter, target in ((_lib.model_num_inputs, _lib.model_input_info, self.inputs),
                                      (_lib.model_num_outputs, _lib.model_output_info, self.outputs)):
            for i in range(count(handle)):
                tensor = _lib.Tensor()
                _check(getter(handle, i, ctypes.byref(tensor)))
                target.append(_info(tensor))

    def run(self, *inputs: np.ndarray, out: Optional[Sequence[np.ndarray]] = None) -> Tuple[np.ndarray, ...]:
        """Runs the model. Inputs must match the model's dtype and shape exactly. Returns the outputs."""
        self._ensure_open()
        if len(inputs) != len(self.inputs):
            raise InvalidArgument(1, f"model takes {len(self.inputs)} inputs, got {len(inputs)}")
        arrays = []
        for array, info in zip(inputs, self.inputs):
            array = np.asarray(array)
            if array.dtype != info.dtype or tuple(array.shape) != info.shape:
                raise InvalidArgument(1, f"input '{info.name}' needs {info.dtype} {info.shape}, "
                                         f"got {array.dtype} {tuple(array.shape)}")
            arrays.append(np.ascontiguousarray(array))
        if out is None:
            out = [np.empty(info.shape, info.dtype) for info in self.outputs]
        elif len(out) != len(self.outputs):
            raise InvalidArgument(1, f"model has {len(self.outputs)} outputs, got {len(out)} buffers")
        for array, info in zip(out, self.outputs):
            if array.dtype != info.dtype or tuple(array.shape) != info.shape or not array.flags["C_CONTIGUOUS"] \
                    or not array.flags["WRITEABLE"]:
                raise InvalidArgument(1, f"output '{info.name}' needs a writeable contiguous {info.dtype} "
                                         f"{info.shape} array")
        in_tensors = (_lib.Tensor * max(len(arrays), 1))()
        out_tensors = (_lib.Tensor * max(len(out), 1))()
        for tensors, source in ((in_tensors, arrays), (out_tensors, out)):
            for i, array in enumerate(source):
                _lib.tensor_init(ctypes.byref(tensors[i]))
                tensors[i].data = array.ctypes.data
                tensors[i].nbytes = array.nbytes
        for tensors, infos in ((in_tensors, self.inputs), (out_tensors, self.outputs)):
            for i, info in enumerate(infos):
                tensors[i].dtype = {v: k for k, v in _DTYPES.items()}[info.dtype.type]
                tensors[i].rank = len(info.shape)
                for d, dim in enumerate(info.shape):
                    tensors[i].dims[d] = dim
        _check(_lib.model_run(self._handle, in_tensors, len(arrays), out_tensors, len(out)))
        return tuple(out)

    def run_buffers(self, inputs: Sequence[Buffer], outputs: Sequence[Buffer]) -> None:
        """Runs with zero-copy buffers (for example DMABUF) bound to the model's inputs and outputs."""
        self._ensure_open()
        if len(inputs) != len(self.inputs) or len(outputs) != len(self.outputs):
            raise InvalidArgument(1, "one buffer is needed per model input and output")
        in_tensors = (_lib.Tensor * max(len(inputs), 1))()
        out_tensors = (_lib.Tensor * max(len(outputs), 1))()
        for tensors, buffers, getter, count in ((in_tensors, inputs, _lib.model_input_info, len(inputs)),
                                                (out_tensors, outputs, _lib.model_output_info, len(outputs))):
            for i in range(count):
                _check(getter(self._handle, i, ctypes.byref(tensors[i])))
                _lib.tensor_use_buffer(ctypes.byref(tensors[i]), buffers[i]._handle)
        _check(_lib.model_run(self._handle, in_tensors, len(inputs), out_tensors, len(outputs)))

    def _ensure_open(self):
        if not self._handle:
            raise InvalidArgument(1, "the model is closed")

    def close(self) -> None:
        if self._handle:
            _lib.model_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Engine:
    """One backend configuration. Options are passed to the backend as strings."""

    def __init__(self, backend: str, options: Optional[Dict[str, object]] = None, **kwargs):
        from ._plugins import ensure_backend  # a plugin that is not registered yet is loaded from the search path

        ensure_backend(backend)
        merged = dict(options or {})
        merged.update(kwargs)
        array = (_lib.Option * max(len(merged), 1))()
        keep = []
        for i, (key, value) in enumerate(merged.items()):
            k, v = str(key).encode(), str(value).encode()
            keep.extend((k, v))
            array[i].key, array[i].value = k, v
        handle = ctypes.c_void_p()
        _check(_lib.engine_create(backend.encode(), array, len(merged), ctypes.byref(handle)))
        self._handle = handle.value
        self.backend = backend

    def load_model(self, path: Optional[str] = None, *, data: Optional[bytes] = None,
                   format: Optional[str] = None) -> Model:
        """Loads a model from a file path or from bytes."""
        if (path is None) == (data is None):
            raise InvalidArgument(1, "give exactly one of path or data")
        if not self._handle:
            raise InvalidArgument(1, "the engine is closed")
        source = _lib.ModelSource()
        source.struct_size = ctypes.sizeof(_lib.ModelSource)
        buffer = None
        if path is not None:
            source.path = str(path).encode()
        else:
            buffer = ctypes.create_string_buffer(bytes(data), len(data))
            source.data = ctypes.cast(buffer, ctypes.c_void_p).value
            source.size = len(data)
        if format:
            source.format = format.encode()
        handle = ctypes.c_void_p()
        _check(_lib.model_load(self._handle, ctypes.byref(source), ctypes.byref(handle)))
        return Model(self, handle.value)

    def alloc_buffer(self, nbytes: int, domain: str = "host") -> Buffer:
        return Buffer(self, nbytes, domain)

    def close(self) -> None:
        if self._handle:
            _lib.engine_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
