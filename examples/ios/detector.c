/* SPDX-License-Identifier: Apache-2.0 */
#include "detector.h"

#include <stdlib.h>

const char* const kCocoNames[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

static int by_score_descending(const void* a, const void* b) {
  float x = ((const detection*)a)->score, y = ((const detection*)b)->score;
  return (x < y) - (x > y);
}

static float iou(const detection* a, const detection* b) {
  float x0 = a->x0 > b->x0 ? a->x0 : b->x0, y0 = a->y0 > b->y0 ? a->y0 : b->y0;
  float x1 = a->x1 < b->x1 ? a->x1 : b->x1, y1 = a->y1 < b->y1 ? a->y1 : b->y1;
  float w = x1 - x0, h = y1 - y0;
  if (w <= 0 || h <= 0) {
    return 0;
  }
  float inter = w * h;
  float union_area = (a->x1 - a->x0) * (a->y1 - a->y0) + (b->x1 - b->x0) * (b->y1 - b->y0) - inter;
  return union_area > 0 ? inter / union_area : 0;
}

size_t detector_decode(const float* scores, const float* boxes, size_t anchors, size_t classes, float confidence, float iou_threshold,
                       detection* out, size_t max_out) {
  detection* candidates = malloc(anchors * sizeof(detection));
  if (!candidates) {
    return 0;
  }
  size_t count = 0;
  for (size_t a = 0; a < anchors; ++a) {
    const float* row = scores + a * classes;
    size_t best = 0;
    for (size_t c = 1; c < classes; ++c) {
      best = row[c] > row[best] ? c : best;
    }
    if (row[best] > confidence) {
      const float* b = boxes + a * 4;
      candidates[count++] = (detection){b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2, row[best], (int)best};
    }
  }
  qsort(candidates, count, sizeof(detection), by_score_descending);
  size_t kept = 0;
  for (size_t i = 0; i < count && kept < max_out; ++i) {
    int suppressed = 0;
    for (size_t k = 0; k < kept && !suppressed; ++k) {
      suppressed = out[k].class_id == candidates[i].class_id && iou(&out[k], &candidates[i]) > iou_threshold;
    }
    if (!suppressed) {
      out[kept++] = candidates[i];
    }
  }
  free(candidates);
  return kept;
}
