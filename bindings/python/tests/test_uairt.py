# SPDX-License-Identifier: Apache-2.0
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

import uairt
from uairt import _lib

REPO = Path(__file__).resolve().parents[3]
ORT_PLUGIN = os.environ.get("UAIRT_TEST_ORT_PLUGIN")
ADD_MUL = REPO / "tests" / "data" / "add_mul.onnx"


def test_struct_layouts_match_the_c_header():
    tensor = _lib.Tensor()
    _lib.tensor_init(ctypes.byref(tensor))
    assert tensor.struct_size == ctypes.sizeof(_lib.Tensor)
    assert tensor.dmabuf_fd == -1 and tensor.domain == _lib.MEM_HOST


def test_version_and_backends():
    assert uairt.version().count(".") == 2
    assert "reference" in uairt.backends()


def test_reference_roundtrip():
    with uairt.Engine("reference") as engine, engine.load_model("unused") as model:
        assert [i.shape for i in model.inputs] == [(1, 4)]
        assert model.inputs[0].dtype == np.float32
        x = np.array([[1, 2, 3, 4]], np.float32)
        (y,) = model.run(x)
        np.testing.assert_array_equal(y, x)
        out = np.zeros((1, 4), np.float32)
        (z,) = model.run(x * 2, out=[out])
        assert z is out
        np.testing.assert_array_equal(out, x * 2)


def test_run_validates_inputs_without_converting():
    with uairt.Engine("reference") as engine, engine.load_model("unused") as model:
        with pytest.raises(uairt.InvalidArgument):
            model.run(np.zeros((1, 4), np.float64))
        with pytest.raises(uairt.InvalidArgument):
            model.run(np.zeros((1, 5), np.float32))
        with pytest.raises(uairt.InvalidArgument):
            model.run()
        non_contiguous = np.arange(8, dtype=np.float32).reshape(1, 8)[:, ::2]
        assert not non_contiguous.flags["C_CONTIGUOUS"]
        (y,) = model.run(non_contiguous)  # copied to a contiguous array first
        np.testing.assert_array_equal(y, [[0, 2, 4, 6]])


def test_errors_carry_status_and_message():
    with pytest.raises(uairt.NotFound) as info:
        uairt.Engine("no_such_backend")
    assert info.value.status == 2 and "no_such_backend" in info.value.message
    with pytest.raises(uairt.UairtError):
        uairt.load_backend_library("/no/such/plugin.so")


def test_lifetimes_and_double_close():
    engine = uairt.Engine("reference")
    model = engine.load_model("unused")
    del engine  # the model keeps the engine alive
    model.run(np.zeros((1, 4), np.float32))
    model.close()
    model.close()
    with pytest.raises(uairt.InvalidArgument):
        model.run(np.zeros((1, 4), np.float32))


def test_host_buffers_and_unsupported_dmabuf():
    with uairt.Engine("reference") as engine, engine.load_model("unused") as model:
        with engine.alloc_buffer(16) as source, engine.alloc_buffer(16) as dest:
            source.array(np.float32, (1, 4))[:] = [[5, 6, 7, 8]]
            model.run_buffers([source], [dest])
            np.testing.assert_array_equal(dest.array(np.float32, (1, 4)), [[5, 6, 7, 8]])
            with pytest.raises(uairt.InvalidArgument):
                source.array(np.float32, (1, 5))
        with pytest.raises(uairt.Unsupported):
            engine.alloc_buffer(16, "dmabuf")
        with pytest.raises(uairt.InvalidArgument):
            engine.alloc_buffer(16, "gpu")


def test_options_are_passed_as_strings():
    # The reference backend ignores options; this checks that mixed option sources are accepted.
    with uairt.Engine("reference", options={"threads": 2}, mode="fast"):
        pass


@pytest.mark.skipif(not ORT_PLUGIN or not ADD_MUL.exists(), reason="set UAIRT_TEST_ORT_PLUGIN to run")
def test_onnxruntime_two_inputs_two_outputs():
    uairt.load_backend_library(ORT_PLUGIN)
    with uairt.Engine("onnxruntime", intra_op_threads=1) as engine, engine.load_model(str(ADD_MUL)) as model:
        assert [i.name for i in model.inputs] == ["x", "y"]
        assert [o.name for o in model.outputs] == ["sum", "product"]
        x = np.array([[1, 2, 3, 4]], np.float32)
        y = np.array([[10, 20, 30, 40]], np.float32)
        total, product = model.run(x, y)
        np.testing.assert_array_equal(total, x + y)
        np.testing.assert_array_equal(product, x * y)
    with uairt.Engine("onnxruntime") as engine:
        with engine.load_model(data=ADD_MUL.read_bytes()) as model:
            assert len(model.outputs) == 2
        with pytest.raises(uairt.IoFailure):
            engine.load_model("/no/such/model.onnx")
        with pytest.raises(uairt.InvalidArgument):
            uairt.Engine("onnxruntime", no_such_option=1)
