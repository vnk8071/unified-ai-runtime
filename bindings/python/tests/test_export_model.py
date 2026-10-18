# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/export_model.py. The logic that needs no ML framework always runs; one end-to-end export needs torch, onnx and the
ONNX Runtime backend, and is skipped without them."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "export_model.py"
spec = importlib.util.spec_from_file_location("export_model", SCRIPT)
export_model = importlib.util.module_from_spec(spec)
sys.modules["export_model"] = export_model
spec.loader.exec_module(export_model)


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_parse_shapes_and_a_clear_error():
    assert export_model.parse_shapes(["1,3,224,224", "1,8"]) == [(1, 3, 224, 224), (1, 8)]
    with pytest.raises(export_model.ExportError, match="comma-separated integers"):
        export_model.parse_shapes(["1,x,3"])


def test_inputs_are_deterministic_and_typed():
    a = export_model.make_inputs([(2, 3), (4,)], ["float32", "int64"])
    b = export_model.make_inputs([(2, 3), (4,)], ["float32", "int64"])
    assert a[0].dtype == np.float32 and a[1].dtype == np.int64
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert a[1].min() >= 0 and a[1].max() < 10          # token ids and indices stay small and valid


def test_need_names_the_package_to_install():
    with pytest.raises(export_model.ExportError, match="install some-pip-name"):
        export_model.need("definitely_not_a_module_xyz", "a test", "some-pip-name")


def test_match_outputs_by_position_by_shape_and_without_batch_dimension():
    r0, r1 = np.zeros((1, 10)), np.zeros((1, 4, 4))
    assert [g.shape for _, g in export_model.match_outputs([r0, r1], [np.ones((1, 10)), np.ones((1, 4, 4))])] == [(1, 10), (1, 4, 4)]
    swapped = export_model.match_outputs([r0, r1], [np.ones((1, 4, 4)), np.ones((1, 10))])   # a backend that sorts outputs by name
    assert [g.shape for _, g in swapped] == [(1, 10), (1, 4, 4)]
    squeezed = export_model.match_outputs([r0], [np.arange(10.0)])                           # NCNN has no batch dimension
    assert squeezed[0][1].shape == (1, 10)
    with pytest.raises(export_model.ExportError, match="no output"):
        export_model.match_outputs([r0], [np.ones((3, 3))])
    with pytest.raises(export_model.ExportError, match="outputs"):
        export_model.match_outputs([r0, r1], [np.ones((1, 10))])


def test_compare_accepts_small_differences_and_catches_wrong_or_constant_outputs():
    rng = np.random.default_rng(0)
    reference = [rng.standard_normal((1, 100)).astype(np.float32)]
    ok, _ = export_model.compare(reference, [reference[0] + 1e-6], atol=1e-4)
    assert ok
    ok, lines = export_model.compare(reference, [reference[0] + 0.5], atol=1e-4)
    assert not ok and "FAILED" in lines[0]
    ok, lines = export_model.compare(reference, [np.zeros((1, 100), np.float32)], atol=1e9)
    assert not ok and "constant" in lines[0]                                 # a converted model that outputs one value proves nothing
    ok, _ = export_model.compare(reference, [np.full((1, 100), np.nan, np.float32)], atol=1e9)
    assert not ok


def test_a_pytorch_source_needs_a_shape_and_an_onnx_source_cannot_go_to_coreml():
    result = run_script("torchvision:resnet18", "--to", "onnx")
    assert result.returncode == 2 and "--shape" in result.stderr
    result = run_script("model.onnx", "--to", "coreml")
    assert result.returncode == 2 and "needs the PyTorch model" in result.stderr


def have_end_to_end():
    for module in ("torch", "onnx", "uairt"):
        if importlib.util.find_spec(module) is None:
            return False
    import uairt
    return bool(uairt.available_backends().get("onnxruntime", {}).get("plugin") or uairt.available_backends().get("onnxruntime", {}).get("registered"))


def test_end_to_end_pytorch_to_onnx_checked_through_uairt(tmp_path):
    if not have_end_to_end():
        pytest.skip("needs torch, onnx, and UAIRT with the onnxruntime backend")
    source = tmp_path / "tiny.py"
    source.write_text("import torch\nfrom torch import nn\n\ndef build():\n    torch.manual_seed(0)\n"
                      "    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)).eval()\n")
    result = run_script(f"{source}:build", "--shape", "1,3,16,16", "--to", "onnx", "--out", str(tmp_path / "out"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "summary: onnx: ok" in result.stdout and (tmp_path / "out" / "build.onnx").exists()
