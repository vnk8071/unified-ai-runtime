# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/bench_model.py. The logic that needs no ML framework always runs; one end-to-end benchmark needs onnx, onnxruntime and
the ONNX Runtime backend, and is skipped without them."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("bench_model", SCRIPTS / "bench_model.py")
bench = importlib.util.module_from_spec(spec)
sys.modules["bench_model"] = bench
spec.loader.exec_module(bench)


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPTS / "bench_model.py"), *args], capture_output=True, text=True)


def row(backend, median, diff, accuracy, **options):
    return {"backend": backend, "options": options, "status": "ok", "load_ms": 1.0, "median_ms": median, "p90_ms": median * 1.1,
            "min_ms": median * 0.9, "max_diff": diff, "scale": 5.0, "accuracy": accuracy}


def test_accuracy_classes_separate_float32_from_reduced_precision_and_failures():
    c = lambda diff, scale=5.0: bench.classify({"max_diff": diff, "scale": scale}, 1e-4, 3e-2)  # noqa: E731
    assert c(5e-6) == "ok" and c(1e-3) == "ok"            # float32 against float32
    assert c(4e-3) == "reduced" and c(0.048) == "reduced"  # half precision on logits of size 5
    assert c(0.5) == "FAILED" and c(float("inf")) == "FAILED"
    assert bench.classify({}, 1e-4, 3e-2) == "n/a"          # a variant that never produced an output


def test_table_is_fastest_first_and_unrun_variants_follow_with_their_reason():
    results = [row("slow", 9.0, 1e-6, "ok"), row("fast", 1.0, 1e-6, "ok"),
               {"backend": "tflite", "options": {}, "status": "skipped", "error": "plugin not found | (UAIRT_PLUGIN_PATH)"},
               {"backend": "coreml", "options": {"compute_units": "all"}, "status": "unavailable", "error": "no device"}]
    lines = bench.render_table(results).splitlines()
    assert [l.split("|")[1].strip() for l in lines[2:]] == ["fast", "slow", "tflite", "coreml compute_units=all"]
    assert "skipped: plugin not found / (UAIRT_PLUGIN_PATH)" in lines[4]       # a pipe in a reason must not break the table


def test_verdict_names_the_fastest_accurate_and_flags_a_faster_reduced_precision_one():
    exact, fast_reduced = row("onnxruntime", 5.0, 1e-6, "ok"), row("coreml", 2.0, 0.05, "reduced", compute_units="all")
    both = bench.verdict([exact, fast_reduced, row("broken", 0.1, 9.0, "FAILED")])
    assert "Fastest accurate: onnxruntime" in both[0] and "Fastest overall: coreml compute_units=all" in both[1] and "reduced precision" in both[1]
    assert len(bench.verdict([exact])) == 1                                     # nothing faster to warn about
    assert "No backend ran within the accuracy tolerance" in bench.verdict([row("x", 1.0, 9.0, "FAILED")])[0]


def test_label_hides_the_input_shapes_option():
    assert bench.label({"backend": "ncnn", "options": {"fp16": "true", "input_shapes": "3x224x224"}}) == "ncnn fp16=true"


def test_variants_use_only_targets_and_backends_that_export_model_knows():
    em = bench.em
    assert set(bench.VARIANTS) <= set(em.TARGETS)
    for target, variants in bench.VARIANTS.items():
        assert {backend for backend, _ in variants} == {em.TARGETS[target][0]}


def test_artifacts_parse_and_bad_ones_are_explained():
    assert bench.parse_artifact("qnn=model.dlc,device=npu,cache_dir=/tmp/c") == ("qnn", "model.dlc", {"device": "npu", "cache_dir": "/tmp/c"})
    assert bench.parse_artifact("tflite=m.tflite") == ("tflite", "m.tflite", {})
    for bad in ("nothing", "qnn=", "qnn=m.dlc,device"):
        with pytest.raises(bench.em.ExportError):
            bench.parse_artifact(bad)


def test_percentile_of_timings():
    assert bench.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert bench.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 90) == pytest.approx(4.6)


def test_command_line_errors_are_clear():
    result = run_script("torchvision:resnet18")                                  # a PyTorch source needs --shape
    assert result.returncode == 2 and "--shape" in result.stderr
    assert run_script().returncode == 2                                          # nothing to benchmark
    result = run_script("--artifact", "qnn=m.dlc")                               # artifacts need a reference to compare with
    assert result.returncode == 2 and "--artifact alone" in result.stderr


def have_end_to_end():
    for module in ("onnx", "onnxruntime", "uairt"):
        if importlib.util.find_spec(module) is None:
            return False
    import uairt
    return bool(uairt.available_backends().get("onnxruntime", {}).get("plugin") or uairt.available_backends().get("onnxruntime", {}).get("registered"))


def test_end_to_end_benchmark_of_an_onnx_file_writes_json(tmp_path):
    if not have_end_to_end():
        pytest.skip("needs onnx, onnxruntime, and UAIRT with the onnxruntime backend")
    model = Path(__file__).resolve().parents[3] / "tests" / "data" / "add_mul.onnx"
    result = run_script(str(model), "--targets", "onnx", "--runs", "10", "--out", str(tmp_path / "w"), "--json", str(tmp_path / "r.json"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Fastest accurate: onnxruntime" in result.stdout
    data = json.loads((tmp_path / "r.json").read_text())
    (entry,) = data["results"]
    assert entry["status"] == "ok" and entry["accuracy"] == "ok" and entry["runs"] == 10 and entry["median_ms"] > 0
