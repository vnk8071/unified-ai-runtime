/* SPDX-License-Identifier: Apache-2.0 */
/* Post-processing for a YOLO-style detector network: class scores [anchors, classes] and boxes [anchors, 4] (centre x, centre y, width, height
 * as fractions of the image) to a short list of boxes: a confidence threshold, then class-aware greedy non-maximum suppression. Plain C. */
#ifndef UAIRT_IOS_DETECTOR_H
#define UAIRT_IOS_DETECTOR_H

#include <stddef.h>

typedef struct {
  float x0, y0, x1, y1; /* fractions of the image, top-left origin */
  float score;
  int class_id;
} detection;

/* Returns the number of detections written to `out` (at most `max_out`), best first. */
size_t detector_decode(const float* scores, const float* boxes, size_t anchors, size_t classes, float confidence, float iou_threshold,
                       detection* out, size_t max_out);

/* The 80 COCO class names, indexed by class id. */
extern const char* const kCocoNames[80];

#endif
