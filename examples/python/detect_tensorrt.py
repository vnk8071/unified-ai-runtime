# SPDX-License-Identifier: Apache-2.0
"""Object detection with a YOLOv8 TensorRT engine through the UAIRT TensorRT backend.

Takes an engine exported by ultralytics (`YOLO("yolov8n.pt").export(format="engine")`), runs it on an image with
pinned (page-locked) buffers, and prints the detections. Needs numpy and Pillow; no ultralytics or TensorRT Python
package is needed to run it.

    python detect_tensorrt.py <libuairt_backend_tensorrt.so> <model.engine> <image> [--plain] [--runs N]

Set UAIRT_LIBRARY to libuairt.so (built with -DBUILD_SHARED_LIBS=ON) and put bindings/python on PYTHONPATH.
"""
import argparse
import json
import statistics
import sys
import time

import numpy as np
from PIL import Image

import uairt

CONFIDENCE = 0.25
IOU = 0.7


def read_names(path):
    """Ultralytics prefixes the engine with a 4-byte length and a JSON block that holds the class names."""
    with open(path, "rb") as engine_file:
        head = engine_file.read(4)
        length = int.from_bytes(head, "little")
        try:
            return {int(k): v for k, v in json.loads(engine_file.read(length))["names"].items()}
        except (ValueError, KeyError):
            return {}


def letterbox(image, size):
    """Resize keeping the aspect ratio and pad to a square, like ultralytics. Returns CHW float32 in [0, 1]."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return array, scale, pad_x, pad_y


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(area, 1e-9)


def decode(output, scale, pad_x, pad_y, image_size):
    """YOLOv8 output is (1, 4 + classes, anchors): center x, y, width, height, then one score per class."""
    predictions = output[0].T
    scores = predictions[:, 4:]
    classes = scores.argmax(axis=1)
    confidences = scores.max(axis=1)
    keep = confidences > CONFIDENCE
    predictions, classes, confidences = predictions[keep], classes[keep], confidences[keep]
    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale, 0, image_size[0])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale, 0, image_size[1])
    detections = []
    for cls in np.unique(classes):  # per-class non-maximum suppression
        indices = np.flatnonzero(classes == cls)
        indices = indices[np.argsort(-confidences[indices])]
        while len(indices):
            best = indices[0]
            detections.append((int(cls), float(confidences[best]), boxes[best]))
            indices = indices[1:][iou(boxes[best], boxes[indices[1:]]) < IOU]
    return sorted(detections, key=lambda d: -d[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plugin")
    parser.add_argument("engine")
    parser.add_argument("image")
    parser.add_argument("--plain", action="store_true", help="use ordinary host arrays instead of pinned buffers")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    uairt.load_backend_library(args.plugin)
    names = read_names(args.engine)
    image = Image.open(args.image).convert("RGB")
    with uairt.Engine("tensorrt") as engine, engine.load_model(args.engine) as model:
        (inp,), (out,) = model.inputs, model.outputs
        x, scale, pad_x, pad_y = letterbox(image, inp.shape[2])
        x = np.ascontiguousarray(x[None])
        print(f"input {inp.name} {inp.shape} {inp.dtype}, output {out.name} {out.shape}; "
              f"{'plain host arrays' if args.plain else 'pinned buffers'}")

        latencies = []
        if args.plain:
            for _ in range(args.runs + 5):
                start = time.perf_counter()
                (y,) = model.run(x)
                latencies.append((time.perf_counter() - start) * 1e3)
        else:
            with engine.alloc_buffer(inp.nbytes, "pinned") as in_buf, engine.alloc_buffer(out.nbytes, "pinned") as out_buf:
                in_buf.array(np.float32, inp.shape)[...] = x
                for _ in range(args.runs + 5):
                    start = time.perf_counter()
                    model.run_buffers([in_buf], [out_buf])
                    latencies.append((time.perf_counter() - start) * 1e3)
                y = out_buf.array(np.float32, out.shape).copy()
        latencies = latencies[5:]

    detections = decode(y, scale, pad_x, pad_y, image.size)
    print(f"{len(detections)} detections:")
    for cls, confidence, box in detections:
        print(f"  {names.get(cls, cls):>12} {confidence:.2f}  box=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")
    print(f"latency_ms runs={len(latencies)} min={min(latencies):.2f} median={statistics.median(latencies):.2f}")


if __name__ == "__main__":
    main()
