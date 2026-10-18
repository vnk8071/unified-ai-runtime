# SPDX-License-Identifier: Apache-2.0
"""Pre- and post-processing helpers for the recipes in docs/agents/pre-post-processing.md.

UAIRT runs the tensors; everything around them is the application's. These functions are the parts that recur: they are NumPy
only (Pillow is imported where an image is read), they take the numbers that belong to the model (size, mean, std, scale,
thresholds) as arguments, and they carry no defaults that are really properties of a model. Take those numbers from the model
card or from the original framework's preprocessing config, never from this file.

    import prepost
    image = prepost.load_rgb("photo.jpg")
    x = prepost.to_tensor(prepost.resize_center_crop(image, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
"""
from typing import Optional, Sequence, Tuple

import numpy as np


# ---- images ---------------------------------------------------------------------------------------------------------------

def load_rgb(path):
    """A Pillow image in RGB order (decoders differ: OpenCV returns BGR, Pillow RGB)."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def resize_center_crop(image, size: int, crop_ratio: float = 0.875):
    """Resize the short side to size / crop_ratio, then crop the centre size x size. 0.875 is the usual ImageNet recipe
    (256 then 224); some models resize straight to size x size instead, so check the model card.

    The rounding follows torchvision's Resize(256) + CenterCrop(224) exactly (the long side is truncated, the crop offset is
    rounded), because a one-pixel difference is enough to change a prediction; another framework may round differently, so
    compare with the original pipeline on the same image."""
    from PIL import Image
    width, height = image.size
    short = round(size / crop_ratio)
    if width <= height:
        new_w, new_h = short, int(short * height / width)
    else:
        new_w, new_h = int(short * width / height), short
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    left, top = int(round((new_w - size) / 2.0)), int(round((new_h - size) / 2.0))
    return resized.crop((left, top, left + size, top + size))


def letterbox(image, size: int, fill: int = 114) -> Tuple[np.ndarray, float, int, int]:
    """Fit the image inside a size x size canvas keeping its aspect ratio. Returns (uint8 HxWx3 canvas, scale, pad_x, pad_y);
    keep the last three to map boxes back with undo_letterbox."""
    from PIL import Image
    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return np.asarray(canvas), scale, pad_x, pad_y


def to_tensor(rgb, *, layout: str = "NCHW", scale: float = 1.0 / 255.0,
              mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None,
              bgr: bool = False, dtype=np.float32) -> np.ndarray:
    """HxWx3 uint8 pixels to a model input: optional BGR swap, multiply by `scale`, then (x - mean) / std per channel
    (mean and std are in the scaled range, as torchvision gives them), layout NCHW or NHWC, a batch dimension of 1."""
    x = np.asarray(rgb, dtype=np.float32)
    if bgr:
        x = x[..., ::-1]
    x = x * scale
    if mean is not None:
        x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std if std is not None else (1.0, 1.0, 1.0), dtype=np.float32)
    if layout == "NCHW":
        x = x.transpose(2, 0, 1)
    elif layout != "NHWC":
        raise ValueError("layout must be NCHW or NHWC")
    return np.ascontiguousarray(x[None], dtype=dtype)


def quantize(x: np.ndarray, scale: float, zero_point: int, dtype) -> np.ndarray:
    """Real values to the integers a quantized input takes: round(x / scale) + zero_point, clipped to the dtype's range."""
    info = np.iinfo(dtype)
    return np.clip(np.round(np.asarray(x, dtype=np.float64) / scale) + zero_point, info.min, info.max).astype(dtype)


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """The inverse of quantize for an output: (q - zero_point) * scale. TensorInfo.dequantize does the same."""
    return (np.asarray(q, dtype=np.float32) - zero_point) * np.float32(scale)


# ---- classification, embeddings -------------------------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax. Skip it if the model's last layer already is one (outputs in [0, 1] summing to 1)."""
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def topk(scores: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """(indices, values) of the k largest entries of a 1-D array, largest first."""
    scores = np.asarray(scores).reshape(-1)
    k = min(k, scores.size)
    order = np.argpartition(-scores, k - 1)[:k]
    order = order[np.argsort(-scores[order], kind="stable")]
    return order, scores[order]


def mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Sentence embedding from token embeddings (B, L, H): the mean over the real tokens, ignoring padding."""
    mask = np.asarray(attention_mask, dtype=np.float32)[..., None]
    return (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


# ---- detection ------------------------------------------------------------------------------------------------------------

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45,
        classes: Optional[np.ndarray] = None) -> np.ndarray:
    """Greedy non-maximum suppression on xyxy boxes; returns the kept indices, best score first. With `classes`, boxes of
    different classes never suppress each other."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if classes is not None:
        boxes = boxes + (np.asarray(classes, dtype=np.float64) * (boxes.max() + 1.0))[:, None]
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    while order.size:
        best = order[0]
        kept.append(int(best))
        order = order[1:][box_iou(boxes[best], boxes[order[1:]]) <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def decode_yolo(output: np.ndarray, conf: float = 0.25, iou: float = 0.45, objectness: bool = False,
                channels_first: Optional[bool] = None):
    """Boxes from a YOLO head, in the pixel space of the model input.

    YOLOv8/v11 exports give (1, 4 + classes, anchors) and carry no objectness; YOLOv5 exports give (1, anchors, 5 + classes)
    and do (pass objectness=True). Rows are centre x, centre y, width, height, [objectness], class scores. Returns
    (xyxy boxes, scores, class ids) after class-aware NMS. Check the export: a head with NMS built in has a different layout.

    The orientation is guessed from the shape, because a real head has far more anchors (thousands) than features (tens):
    pass channels_first=True for (features, anchors) or False for (anchors, features) when the guess could be wrong."""
    out = np.asarray(output)
    out = out.reshape(out.shape[-2], out.shape[-1])
    if channels_first is None:
        channels_first = out.shape[0] < out.shape[1]
    if channels_first:                       # (features, anchors) -> (anchors, features)
        out = out.T
    if out.shape[1] < (6 if objectness else 5):
        raise ValueError(f"a YOLO head needs at least {6 if objectness else 5} features per anchor, got shape {out.shape}; "
                         "check objectness= and channels_first=")
    cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    if objectness:
        class_scores = out[:, 5:] * out[:, 4:5]
    else:
        class_scores = out[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(classes)), classes]
    keep = scores > conf
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)[keep]
    scores, classes = scores[keep], classes[keep]
    kept = nms(boxes, scores, iou, classes) if len(boxes) else np.zeros(0, dtype=np.int64)
    return boxes[kept], scores[kept], classes[kept]


def undo_letterbox(boxes: np.ndarray, scale: float, pad_x: int, pad_y: int, width: int, height: int) -> np.ndarray:
    """Map xyxy boxes from the letterboxed input back to the original image (width x height pixels), clipped to it."""
    mapped = (np.asarray(boxes, dtype=np.float64) - np.array([pad_x, pad_y, pad_x, pad_y])) / scale
    mapped[:, [0, 2]] = mapped[:, [0, 2]].clip(0, width)
    mapped[:, [1, 3]] = mapped[:, [1, 3]].clip(0, height)
    return mapped


# ---- speech ---------------------------------------------------------------------------------------------------------------

def ctc_greedy(logits: np.ndarray, blank: int = 0) -> list:
    """Greedy CTC decoding of (time, vocabulary) scores: best symbol per step, merge repeats, drop the blank. Returns token
    ids; turn them into text with the model's own vocabulary. The blank index is the model's (often 0, check its config)."""
    best = np.asarray(logits).argmax(axis=-1)
    tokens, previous = [], blank
    for token in best.tolist():
        if token != previous and token != blank:
            tokens.append(token)
        previous = token
    return tokens
