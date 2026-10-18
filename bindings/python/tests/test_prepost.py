# SPDX-License-Identifier: Apache-2.0
"""Tests for examples/prepost.py: NumPy only (the two image tests are skipped when Pillow is missing), no native library or model needed."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "python"
spec = importlib.util.spec_from_file_location("prepost", EXAMPLES / "prepost.py")
prepost = importlib.util.module_from_spec(spec)
sys.modules["prepost"] = prepost
spec.loader.exec_module(prepost)


def test_quantize_roundtrip_stays_within_half_a_step():
    x = np.linspace(-3.0, 3.0, 101, dtype=np.float32)
    scale, zero = 6.0 / 255.0, 128
    q = prepost.quantize(x, scale, zero, np.uint8)
    assert q.dtype == np.uint8
    assert np.abs(prepost.dequantize(q, scale, zero) - x).max() <= scale / 2 + 1e-6


def test_quantize_clips_to_the_dtype_range():
    q = prepost.quantize(np.array([-1e6, 1e6]), 0.5, 0, np.int8)
    assert q.tolist() == [-128, 127]


def test_softmax_is_stable_and_sums_to_one():
    p = prepost.softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-12 and p.argmax() == 1


def test_topk_orders_largest_first_and_handles_small_inputs():
    idx, val = prepost.topk(np.array([0.1, 0.7, 0.2, 0.7, 0.05]), 3)
    assert val.tolist() == [0.7, 0.7, 0.2] and idx.tolist() == [1, 3, 2]
    assert prepost.topk(np.array([3.0, 1.0]), 5)[0].tolist() == [0, 1]


def brute_force_nms(boxes, scores, threshold):
    """An independent O(n^2) version: keep a box unless a better kept box overlaps it by more than the threshold."""
    def iou(a, b):
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    kept = []
    for i in sorted(range(len(boxes)), key=lambda j: -scores[j]):
        if all(iou(boxes[i], boxes[k]) <= threshold for k in kept):
            kept.append(i)
    return kept


def test_nms_matches_a_brute_force_reference_on_random_boxes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        xy = rng.uniform(0, 100, (60, 2))
        wh = rng.uniform(5, 40, (60, 2))
        boxes = np.hstack([xy, xy + wh])
        scores = rng.permutation(60).astype(np.float64)        # distinct scores, so the order is unambiguous
        for threshold in (0.3, 0.5, 0.7):
            assert prepost.nms(boxes, scores, threshold).tolist() == brute_force_nms(boxes, scores, threshold)


def test_class_aware_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float64)
    assert prepost.nms(boxes, np.array([0.9, 0.8]), 0.5).tolist() == [0]
    assert sorted(prepost.nms(boxes, np.array([0.9, 0.8]), 0.5, classes=np.array([0, 1])).tolist()) == [0, 1]


def test_decode_yolo_v8_layout_and_v5_layout_agree():
    # two classes: anchor 0 is a strong class-1 box, anchor 1 is below the threshold, the other anchors are empty
    v8 = np.zeros((1, 6, 12), dtype=np.float32)                         # (1, 4 + classes, anchors)
    v8[0, :4, 0] = [50, 60, 20, 10]
    v8[0, 4:, 0] = [0.1, 0.9]
    v8[0, :4, 1] = [10, 10, 4, 4]
    v8[0, 4:, 1] = [0.05, 0.1]
    boxes, scores, classes = prepost.decode_yolo(v8, conf=0.25)
    assert boxes.tolist() == [[40.0, 55.0, 60.0, 65.0]] and classes.tolist() == [1]
    assert abs(float(scores[0]) - 0.9) < 1e-6

    v5 = np.zeros((1, 12, 7), dtype=np.float32)                          # (1, anchors, 5 + classes)
    v5[0, 0, :4] = [50, 60, 20, 10]
    v5[0, 0, 4] = 1.0
    v5[0, 0, 5:] = [0.1, 0.9]
    boxes5, scores5, classes5 = prepost.decode_yolo(v5, conf=0.25, objectness=True)
    assert boxes5.tolist() == boxes.tolist() and classes5.tolist() == [1]


def test_decode_yolo_returns_empty_arrays_when_nothing_passes():
    boxes, scores, classes = prepost.decode_yolo(np.zeros((1, 6, 40), dtype=np.float32), conf=0.5)
    assert len(boxes) == len(scores) == len(classes) == 0


def test_decode_yolo_explicit_orientation_and_a_clear_error_for_a_wrong_shape():
    ambiguous = np.zeros((6, 4), dtype=np.float32)                       # fewer anchors than features: the guess is wrong here
    ambiguous[:4, 0] = [10, 10, 4, 4]
    ambiguous[4:, 0] = [0.9, 0.1]
    boxes, _, classes = prepost.decode_yolo(ambiguous, channels_first=True)
    assert boxes.tolist() == [[8.0, 8.0, 12.0, 12.0]] and classes.tolist() == [0]
    with pytest.raises(ValueError, match="at least 5 features"):
        prepost.decode_yolo(np.zeros((1, 6, 4), dtype=np.float32))


def test_ctc_greedy_merges_repeats_but_keeps_a_repeat_after_a_blank():
    #            a  a  _  a  b  b  _  _  c
    steps = [1, 1, 0, 1, 2, 2, 0, 0, 3]
    logits = np.eye(4)[steps]
    assert prepost.ctc_greedy(logits, blank=0) == [1, 1, 2, 3]
    assert prepost.ctc_greedy(np.eye(4)[[0, 0, 0]]) == []


def test_mean_pool_ignores_padding_and_normalize_gives_unit_vectors():
    hidden = np.array([[[1.0, 1.0], [3.0, 5.0], [100.0, 100.0]]], dtype=np.float32)   # last token is padding
    pooled = prepost.mean_pool(hidden, np.array([[1, 1, 0]]))
    assert pooled.tolist() == [[2.0, 3.0]]
    assert abs(float(np.linalg.norm(prepost.l2_normalize(pooled)[0])) - 1.0) < 1e-6


def test_to_tensor_layout_scale_and_normalisation():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 255, 0, 51
    x = prepost.to_tensor(rgb)
    assert x.shape == (1, 3, 4, 6) and x.dtype == np.float32
    assert np.allclose(x[0, :, 0, 0], [1.0, 0.0, 0.2])
    nhwc = prepost.to_tensor(rgb, layout="NHWC", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert nhwc.shape == (1, 4, 6, 3) and np.allclose(nhwc[0, 0, 0], [1.0, -1.0, -0.6])
    assert np.allclose(prepost.to_tensor(rgb, bgr=True)[0, :, 0, 0], [0.2, 0.0, 1.0])
    with pytest.raises(ValueError):
        prepost.to_tensor(rgb, layout="CHWN")


needs_pillow = pytest.mark.skipif(importlib.util.find_spec("PIL") is None, reason="Pillow is needed for the image helpers")


@needs_pillow
def test_letterbox_and_undo_letterbox_round_trip_a_box():
    from PIL import Image
    image = Image.new("RGB", (200, 100), (10, 20, 30))
    canvas, scale, pad_x, pad_y = prepost.letterbox(image, 64)
    assert canvas.shape == (64, 64, 3) and scale == 0.32 and (pad_x, pad_y) == (0, 16)
    assert canvas[0, 0].tolist() == [114, 114, 114] and canvas[32, 32].tolist() == [10, 20, 30]   # padding, then image
    original = np.array([[50.0, 20.0, 150.0, 80.0]])
    letterboxed = original * scale + np.array([pad_x, pad_y, pad_x, pad_y])
    assert np.allclose(prepost.undo_letterbox(letterboxed, scale, pad_x, pad_y, 200, 100), original)
    clipped = prepost.undo_letterbox(np.array([[-5.0, -5.0, 999.0, 999.0]]), scale, pad_x, pad_y, 200, 100)
    assert clipped.tolist() == [[0.0, 0.0, 200.0, 100.0]]


@needs_pillow
def test_resize_center_crop_gives_the_requested_size_and_crops_the_middle():
    from PIL import Image
    image = Image.new("RGB", (300, 100), (0, 0, 0))
    image.paste((255, 255, 255), (130, 0, 170, 100))                    # a narrow white band in the middle
    crop = prepost.resize_center_crop(image, 64)
    assert crop.size == (64, 64)
    assert np.asarray(crop)[32, 32].tolist() == [255, 255, 255] and np.asarray(crop)[32, 0].tolist() != [255, 255, 255]
