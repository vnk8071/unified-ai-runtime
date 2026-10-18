# SPDX-License-Identifier: Apache-2.0
"""Tests for examples/inspect_model.py: the guessing logic only, on stand-in tensors (no native library, no model)."""
import importlib.util
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "python"
spec = importlib.util.spec_from_file_location("inspect_model", EXAMPLES / "inspect_model.py")
inspect_model = importlib.util.module_from_spec(spec)
sys.modules["inspect_model"] = inspect_model
spec.loader.exec_module(inspect_model)

Tensor = namedtuple("Tensor", "name dtype shape quant_scale quant_zero_point")


def t(name, shape, dtype="float32", scale=0.0, zero=0):
    return Tensor(name, np.dtype(dtype), tuple(shape), scale, zero)


def said(inputs, outputs):
    return " | ".join(inspect_model.guess(inputs, outputs))


def test_no_tensors_means_a_language_model():
    assert "language model" in said([], [])


def test_image_classifier_nchw():
    text = said([t("input", (1, 3, 224, 224))], [t("logits", (1, 1000))])
    assert "NCHW" in text and "224x224" in text and "1000 classes" in text


def test_float_and_integer_image_inputs_are_described_differently():
    floats = said([t("input", (1, 3, 224, 224))], [])
    assert "mean and std" in floats and "raw 0-255" not in floats
    ints = said([t("input", (1, 224, 224, 3), "uint8")], [])
    assert "raw 0-255 pixels" in ints


def test_image_input_nhwc_and_quantized():
    text = said([t("images", (1, 224, 224, 3), "uint8", 0.0078, 128)], [t("probs", (1, 1001), "uint8", 0.0039, 0)])
    assert "NHWC" in text and "quantize the preprocessed pixels" in text


def test_yolo_heads_both_orientations():
    v8 = said([t("images", (1, 3, 640, 640))], [t("output0", (1, 84, 8400))])
    assert "YOLO-style head" in v8 and "84 features x 8400 anchors" in v8 and "80 class scores" in v8
    v5 = said([t("images", (1, 3, 640, 640))], [t("output0", (1, 25200, 85))])
    assert "25200 anchors x 85 features" in v5


def test_ssd_style_detector_with_nms_built_in():
    text = said([t("image", (1, 300, 300, 3), "uint8")],
                [t("detection_boxes", (1, 10, 4)), t("detection_classes", (1, 10)), t("detection_scores", (1, 10))])
    assert "NMS built in" in text and "YOLO" not in text


def test_segmentation_map():
    assert "segmentation" in said([t("input", (1, 3, 512, 512))], [t("out", (1, 21, 128, 128))])


def test_text_encoder_with_embeddings_and_classification_outputs():
    inputs = [t("input_ids", (1, 128), "int64"), t("attention_mask", (1, 128), "int64")]
    text = said(inputs, [t("last_hidden_state", (1, 128, 768)), t("logits", (1, 2))])
    assert "exactly 128 tokens" in text and "mean_pool" in text and "2 logits" in text
    pooled = said(inputs, [t("pooler_output", (1, 768))])
    assert "pooled vector of 768" in pooled and "logits" not in pooled
    unclear = said(inputs, [t("output", (1, 3))])
    assert "classification logits or a pooled embedding" in unclear


def test_question_answering_outputs():
    inputs = [t("input_ids", (1, 384), "int64")]
    assert "question answering" in said(inputs, [t("start_logits", (1, 384)), t("end_logits", (1, 384))])


def test_kv_cache_decoder_is_called_unsupported():
    inputs = [t("input_ids", (1, 1), "int64"), t("past_key_values.0.key", (1, 8, 64, 64))]
    text = said(inputs, [t("logits", (1, 1, 32000))])
    assert "KV cache" in text and "GGUF" in text


def test_waveform_and_log_mel_inputs():
    assert "waveform of 160000 samples" in said([t("audio", (160000,))], [t("out", (1, 10))])
    assert "waveform of 16000 samples" in said([t("input_values", (1, 16000))], [t("logits", (1, 49, 32))])
    assert "CTC" in said([t("input_values", (1, 16000))], [t("logits", (1, 49, 32))])
    assert "80 mel bins x 3000 frames" in said([t("input_features", (1, 80, 3000))], [t("out", (1, 1500, 384))])


def test_whisper_cpp_convention():
    text = said([t("audio", (176000,))], [t("text", (4096,), "uint8"), t("text_length", (1,), "int32")])
    assert "waveform of 176000" in text and "UTF-8 text bytes" in text


def test_unknown_pattern_falls_back_to_reading_the_model_card():
    text = said([t("x", (1, 4)), t("y", (1, 4))], [t("sum", (1, 4)), t("product", (1, 4))])
    assert text.startswith("no known pattern") and "model card" in text


def test_describe_shows_quantization_only_when_present():
    assert "quantized" not in inspect_model.describe(t("x", (1, 4)))
    assert "scale=0.5 zero_point=3" in inspect_model.describe(t("q", (1, 4), "int8", 0.5, 3))


# ---- what happens when the binding or the model cannot be loaded (found when an agent ran the script in a fresh checkout) ----

import os
import subprocess
import time

SCRIPT = EXAMPLES / "inspect_model.py"


def _touch(path, age):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))


def test_setup_help_points_at_the_newest_shared_build(tmp_path):
    _touch(tmp_path / "build-old" / "libuairt.dylib", 1000)
    _touch(tmp_path / "build-new" / "libuairt.dylib", 10)
    text = inspect_model.setup_help(tmp_path, ImportError("No module named 'uairt'"))
    assert "cannot load the uairt Python binding: No module named 'uairt'" in text
    assert f"UAIRT_LIBRARY={tmp_path / 'build-new' / 'libuairt.dylib'}" in text and "build-old" not in text
    assert f"PYTHONPATH={tmp_path}/bindings/python" in text and "run_model --info" in text


def test_setup_help_gives_the_build_commands_when_there_is_no_shared_build(tmp_path):
    (tmp_path / "build").mkdir()                                            # a static build has no libuairt shared library
    text = inspect_model.setup_help(tmp_path)
    assert "-DBUILD_SHARED_LIBS=ON" in text and "-DUAIRT_BUILD_<BACKEND>=ON" in text and "UAIRT_LIBRARY=" in text
    assert "cannot load" not in text


def test_a_missing_dependency_is_named_and_not_blamed_on_the_build(tmp_path):
    _touch(tmp_path / "build" / "libuairt.dylib", 10)                       # a build exists, so the old text would have said "point at it"
    error = ModuleNotFoundError("No module named 'numpy'", name="numpy")
    text = inspect_model.setup_help(tmp_path, error)
    assert "needs the Python package 'numpy'" in text and "pip install numpy" in text and "throwaway virtual environment" in text
    assert "A shared build exists" not in text and "cmake" not in text
    # the binding module itself missing still gets the build instructions
    assert "A shared build exists" in inspect_model.setup_help(tmp_path, ModuleNotFoundError("No module named 'uairt'", name="uairt"))


def _run_script(*args, env=None):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True, env=env)


def test_missing_binding_prints_setup_steps_not_a_traceback():
    probe = subprocess.run([sys.executable, "-I", "-c", "import uairt"], capture_output=True)
    if probe.returncode == 0:
        pytest.skip("uairt is installed in this interpreter, so the import cannot be made to fail")
    result = _run_script("model.onnx")
    assert result.returncode == 2
    assert "cannot load the uairt Python binding" in result.stderr and "Traceback" not in result.stderr
    assert "UAIRT_LIBRARY" in result.stderr


def test_a_model_that_cannot_be_loaded_gets_a_clear_message_and_a_hint(tmp_path):
    fake = tmp_path / "uairt"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "class UairtError(Exception):\n    pass\n\n"
        "class AutoModel:\n    @staticmethod\n    def from_file(path, **kwargs):\n        raise UairtError('no usable plugin for qnn')\n")
    # -I ignores PYTHONPATH, so put the stand-in next to a copy of the script instead
    copy = tmp_path / "inspect_copy.py"
    copy.write_text(SCRIPT.read_text())
    result = subprocess.run([sys.executable, str(copy), "weights.bin"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert "cannot load weights.bin: no usable plugin for qnn" in result.stderr
    assert "--backend" in result.stderr and "input_shapes" in result.stderr
