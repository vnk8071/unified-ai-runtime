# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path

import numpy as np
import pytest

import uairt
from uairt import _lib

ALL = lambda name: True  # noqa: E731
NONE = lambda name: False  # noqa: E731


def plan(path, available=ALL, **kwargs):
    return uairt.resolve(path, available=available, **kwargs)


@pytest.mark.parametrize("path, backend", [
    ("m.dlc", "qnn"), ("m.bin", "qnn"), ("m.onnx", "onnxruntime"), ("M.ONNX", "onnxruntime"), ("m.xml", "openvino"),
    ("m.param", "ncnn"), ("m.engine", "tensorrt"), ("m.plan", "tensorrt"), ("m.tflite", "tflite"),
    ("m.mlmodelc", "coreml"), ("m.mlpackage", "coreml"), ("m.mlmodel", "coreml"),
])
def test_backend_comes_from_the_file_suffix(path, backend):
    assert plan(path).backend == backend


def test_unknown_suffix_is_unsupported_and_says_how_to_choose():
    with pytest.raises(uairt.Unsupported, match="pass backend="):
        plan("model.safetensors")


def test_directories_are_judged_by_what_they_hold(tmp_path):
    openvino = tmp_path / "yolov8n_openvino_model"
    openvino.mkdir()
    (openvino / "yolov8n.xml").write_text("<net/>")
    (openvino / "yolov8n.bin").write_bytes(b"0")
    ncnn = tmp_path / "yolov8n_ncnn_model"
    ncnn.mkdir()
    (ncnn / "model.ncnn.param").write_text("7767517")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert plan(str(openvino)).backend == "openvino"
    assert plan(str(ncnn)).backend == "ncnn"
    with pytest.raises(uairt.Unsupported):
        plan(str(empty))


@pytest.mark.parametrize("path, device, options", [
    ("m.dlc", None, {"device": "npu"}),
    ("m.dlc", "npu", {"device": "npu"}),
    ("m.dlc", "cpu", {"device": "cpu"}),
    ("m.dlc", "gpu", {"device": "gpu"}),
    ("m.onnx", None, {}),
    ("m.onnx", "cpu", {"execution_provider": "cpu"}),
    ("m.onnx", "gpu", {"execution_provider": "cuda"}),
    ("m.onnx", "cuda:1", {"execution_provider": "cuda", "ep_option.device_id": "1"}),
    ("m.onnx", "tensorrt", {"execution_provider": "tensorrt"}),
    ("m.xml", None, {}),
    ("m.xml", "cpu", {"device": "CPU"}),
    ("m.xml", "gpu", {"device": "GPU"}),
    ("m.xml", "gpu:1", {"device": "GPU.1"}),
    ("m.xml", "npu", {"device": "NPU"}),
    ("m.param", "cpu", {"device": "cpu"}),
    ("m.param", "vulkan", {"device": "vulkan", "vulkan_device": "0"}),
    ("m.param", "gpu:2", {"device": "vulkan", "vulkan_device": "2"}),
    ("m.engine", "gpu", {}),
    ("m.engine", "cuda:1", {"device": "1"}),
    ("m.mlmodelc", "npu", {"compute_units": "cpu_and_ne"}),
    ("m.mlmodelc", "cpu", {"compute_units": "cpu_only"}),
    ("m.tflite", "cpu", {}),
])
def test_device_maps_to_backend_options(path, device, options):
    assert plan(path, device=device).options == options


@pytest.mark.parametrize("path, device, backend, why", [
    ("m.engine", "cpu", None, "NVIDIA gpu"),
    ("m.dlc", "cuda", None, "cpu, gpu or npu"),
    ("m.tflite", "npu", None, "delegate"),
    ("m.onnx", "npu", "onnxruntime", "execution provider"),
    ("m.param", "npu", None, "Vulkan"),
    ("m.xml", "cuda", None, "no cuda"),
    ("m.mlmodelc", "gpu:1", None, "index"),
])
def test_a_device_the_backend_cannot_serve_is_an_error_not_a_fallback(path, device, backend, why):
    with pytest.raises(uairt.Unsupported, match=why):
        plan(path, device=device, backend=backend)


def test_candidates_are_tried_in_order_and_the_reasons_are_kept():
    both = plan("m.onnx", device="gpu")
    assert both.backend == "onnxruntime" and both.skipped == {}
    only_openvino = plan("m.onnx", device="gpu", available=lambda name: name == "openvino")
    assert only_openvino.backend == "openvino" and only_openvino.options == {"device": "GPU"}
    assert "no plugin found" in only_openvino.skipped["onnxruntime"]
    npu = plan("m.onnx", device="npu")
    assert npu.backend == "openvino" and "execution provider" in npu.skipped["onnxruntime"]


def test_nothing_available_is_not_found_and_lists_the_search():
    with pytest.raises(uairt.NotFound, match="no plugin found") as caught:
        plan("m.onnx", available=NONE)
    assert "onnxruntime" in str(caught.value) and "openvino" in str(caught.value)


def test_your_options_override_the_device_mapping():
    chosen = plan("m.onnx", device="gpu", options={"execution_provider": "tensorrt", "intra_op_threads": 4})
    assert chosen.options == {"execution_provider": "tensorrt", "intra_op_threads": "4"}


def test_forcing_a_backend_skips_the_format_table():
    forced = plan("whatever.bin", backend="ncnn", device="cpu")
    assert forced.backend == "ncnn" and forced.forced
    with pytest.raises(uairt.Unsupported, match="no device mapping"):
        plan("x", backend="reference", device="cpu")
    assert plan("x", backend="reference").options == {}


@pytest.mark.parametrize("device, expected", [
    (None, (None, None)), ("", (None, None)), ("auto", (None, None)), ("GPU", ("gpu", None)),
    ("gpu:1", ("gpu", 1)), (" NPU ", ("npu", None)), ("vulkan:3", ("vulkan", 3)),
])
def test_parse_device(device, expected):
    assert uairt._auto.parse_device(device) == expected


@pytest.mark.parametrize("device", ["tpu", "gpu:", "gpu:x", "cuda 1"])
def test_parse_device_rejects_unknown_names(device):
    with pytest.raises(uairt.InvalidArgument):
        uairt._auto.parse_device(device)


# The tests below use the real native library with the reference backend.

def test_auto_model_runs_and_reports_its_choice():
    with uairt.AutoModel.from_file("unused", backend="reference") as model:
        assert model.backend == "reference" and model.device is None and model.options == {}
        assert [i.shape for i in model.inputs] == [(1, 4)]
        x = np.arange(4, dtype=np.float32).reshape(1, 4)
        (y,) = model.run(x)
        np.testing.assert_array_equal(y, x)
        assert "reference" in repr(model)
        with model.alloc_buffer(16) as source, model.alloc_buffer(16) as dest:
            source.array(np.float32, (1, 4))[...] = x
            model.run_buffers([source], [dest])
            np.testing.assert_array_equal(dest.array(np.float32, (1, 4)), x)
    assert uairt.load("unused", backend="reference").backend == "reference"


def test_auto_model_resolve_plans_without_loading_anything():
    chosen = uairt.AutoModel.resolve("unused", backend="reference", threads=2, options={"a": 1})
    assert chosen.backend == "reference" and chosen.options == {"a": "1", "threads": "2"}
    with pytest.raises(uairt.NotFound):  # a plan fails the way from_file would: there is no onnxruntime plugin here
        uairt.AutoModel.resolve("m.onnx", device="cpu", backend="onnxruntime")


def test_a_missing_plugin_is_a_clear_error():
    with pytest.raises(uairt.NotFound, match="UAIRT_PLUGIN_PATH") as caught:
        uairt.Engine("no_such_backend")
    assert "no plugin was found" in str(caught.value)
    with pytest.raises(uairt.NotFound, match="no plugin found"):
        uairt.AutoModel.from_file("model.onnx")


def test_available_backends_lists_the_registered_one():
    status = uairt.available_backends()
    assert status["reference"]["registered"] is True
    assert status["onnxruntime"]["registered"] is False or status["onnxruntime"]["plugin"] is not None


def test_plugins_are_found_on_the_search_path(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "libuairt_backend_foo.so").write_bytes(b"")
    (second / "libuairt_backend_foo.dll").write_bytes(b"")
    (second / "libuairt_backend_bar.dylib").write_bytes(b"")
    (second / "unrelated.so").write_bytes(b"")
    monkeypatch.setenv("UAIRT_PLUGIN_PATH", f"{first}{__import__('os').pathsep}{second}")
    found = uairt.discover_plugins()
    assert found["foo"] == first / "libuairt_backend_foo.so"  # the first directory wins
    assert found["bar"] == second / "libuairt_backend_bar.dylib"
    assert "unrelated" not in found and uairt.find_plugin("missing") is None


def _reference_plugin():
    if _lib.library_path is None:
        return None
    for entry in _lib.library_path.parent.glob("uairt_reference_plugin*"):
        if entry.suffix in (".so", ".dylib", ".dll"):
            return entry
    return None


def test_an_engine_loads_its_plugin_from_the_search_path(tmp_path, monkeypatch):
    plugin = _reference_plugin()
    if plugin is None:
        pytest.skip("the reference plugin is not built next to libuairt")
    # The plugin registers itself as 'reference-plugin', and discovery names plugins by file name.
    shutil.copy(plugin, tmp_path / f"libuairt_backend_reference-plugin{plugin.suffix}")
    monkeypatch.setenv("UAIRT_PLUGIN_PATH", str(tmp_path))
    assert "reference-plugin" not in uairt.backends()
    assert uairt.ensure_backend("reference-plugin") is not None
    assert uairt.ensure_backend("reference-plugin") is None  # already registered
    with uairt.Engine("reference-plugin") as engine, engine.load_model("unused") as model:
        (y,) = model.run(np.ones((1, 4), np.float32))
        assert y.shape == (1, 4)
    with uairt.AutoModel.from_file("unused", backend="reference-plugin") as model:
        assert model.backend == "reference-plugin"
