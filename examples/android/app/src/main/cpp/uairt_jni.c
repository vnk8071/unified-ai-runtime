/* SPDX-License-Identifier: Apache-2.0 */
/*
 * JNI layer of the example app: loads a quantized YOLOX detector through UAIRT's TFLite backend and runs it on a 640 x 640 RGB picture.
 * The model (AI Hub's YOLOX, w8a8) takes uint8 RGB and returns decoded boxes (x1, y1, x2, y2 in input pixels), scores and class ids,
 * all uint8 with quantization parameters; this file dequantizes them, applies a confidence threshold and class-aware NMS, and hands
 * Java a flat float array.
 */
#include <android/log.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <uairt/uairt.h>

#define TAG "UAIRT"
#define SIDE 640
#define MAX_RESULTS 30

static uairt_engine* g_engine;
static uairt_model* g_model;
static uairt_tensor g_in, g_out[3];
static int g_boxes = -1, g_scores = -1, g_classes = -1;
static uint8_t* g_input;
static uint8_t* g_output[3];

static size_t tensor_bytes(const uairt_tensor* tensor) {
  return (size_t)uairt_tensor_num_elements(tensor) * uairt_dtype_size(tensor->dtype);
}

static jstring text(JNIEnv* env, const char* message) {
  return (*env)->NewStringUTF(env, message);
}

static jstring error(JNIEnv* env, const char* what, uairt_status status) {
  char message[512];
  snprintf(message, sizeof(message), "%s: %s (%s)", what, uairt_status_string(status), uairt_last_error());
  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", message);
  return text(env, message);
}

JNIEXPORT jstring JNICALL Java_dev_uairt_example_Native_version(JNIEnv* env, jclass cls) {
  (void)cls;
  return text(env, uairt_version_string());
}

/* Returns "" on success, otherwise the error message. */
JNIEXPORT jstring JNICALL Java_dev_uairt_example_Native_load(JNIEnv* env, jclass cls, jstring plugin_dir, jstring model_path,
                                                             jint threads) {
  (void)cls;
  if (g_model) {
    return text(env, "");
  }
  const char* dir = (*env)->GetStringUTFChars(env, plugin_dir, NULL);
  const char* path = (*env)->GetStringUTFChars(env, model_path, NULL);
  char plugin[512], threads_text[16];
  snprintf(plugin, sizeof(plugin), "%s/libuairt_backend_tflite.so", dir);
  snprintf(threads_text, sizeof(threads_text), "%d", threads > 0 ? threads : 4);
  uairt_option options[1] = {{"num_threads", threads_text}};
  uairt_model_source source = {.struct_size = sizeof(source), .path = path};

  uairt_status status = uairt_load_backend_library(plugin);
  if (status == UAIRT_OK) status = uairt_engine_create("tflite", options, 1, &g_engine);
  if (status == UAIRT_OK) status = uairt_model_load(g_engine, &source, &g_model);
  jstring result = NULL;
  if (status != UAIRT_OK) {
    result = error(env, "load", status);
  } else {
    uairt_model_input_info(g_model, 0, &g_in);
    g_in.nbytes = tensor_bytes(&g_in);
    if (g_in.dtype != UAIRT_DTYPE_UINT8 || g_in.nbytes != (size_t)SIDE * SIDE * 3 || uairt_model_num_outputs(g_model) != 3) {
      result = text(env, "unexpected model: expected a uint8 640x640x3 input and three outputs");
    } else {
      g_input = malloc(g_in.nbytes);
      g_in.data = g_input;
      for (int i = 0; i < 3; ++i) {
        uairt_model_output_info(g_model, (size_t)i, &g_out[i]);
        g_out[i].nbytes = tensor_bytes(&g_out[i]);
        g_output[i] = malloc(g_out[i].nbytes);
        g_out[i].data = g_output[i];
        if (g_out[i].dtype != UAIRT_DTYPE_UINT8) {
          result = text(env, "unexpected model: outputs must be uint8");
        }
        if (strcmp(g_out[i].name, "boxes") == 0) g_boxes = i;
        if (strcmp(g_out[i].name, "scores") == 0) g_scores = i;
        if (strcmp(g_out[i].name, "class_idx") == 0) g_classes = i;
      }
      if (!result && (g_boxes < 0 || g_scores < 0 || g_classes < 0)) {
        result = text(env, "unexpected model: outputs must be named boxes, scores and class_idx");
      }
    }
  }
  if (!result) {
    __android_log_print(ANDROID_LOG_INFO, TAG, "model loaded: input scale %g, boxes scale %g zero point %d, scores scale %g",
                        g_in.quant_scale, g_out[g_boxes].quant_scale, g_out[g_boxes].quant_zero_point, g_out[g_scores].quant_scale);
    result = text(env, "");
  }
  (*env)->ReleaseStringUTFChars(env, plugin_dir, dir);
  (*env)->ReleaseStringUTFChars(env, model_path, path);
  return result;
}

typedef struct {
  float x0, y0, x1, y1, score;
  int class_id;
} candidate;

static int by_score(const void* a, const void* b) {
  float x = ((const candidate*)a)->score, y = ((const candidate*)b)->score;
  return (x < y) - (x > y);
}

static float overlap(const candidate* a, const candidate* b) {
  float x0 = a->x0 > b->x0 ? a->x0 : b->x0, y0 = a->y0 > b->y0 ? a->y0 : b->y0;
  float x1 = a->x1 < b->x1 ? a->x1 : b->x1, y1 = a->y1 < b->y1 ? a->y1 : b->y1;
  if (x1 <= x0 || y1 <= y0) return 0;
  float inter = (x1 - x0) * (y1 - y0);
  float total = (a->x1 - a->x0) * (a->y1 - a->y0) + (b->x1 - b->x0) * (b->y1 - b->y0) - inter;
  return total > 0 ? inter / total : 0;
}

/* pixels: SIDE * SIDE ARGB ints (Bitmap.getPixels). Returns [inference ms, n, then n x (x1, y1, x2, y2, score, class)] with boxes in
 * input pixels, or null on an error. */
JNIEXPORT jfloatArray JNICALL Java_dev_uairt_example_Native_detect(JNIEnv* env, jclass cls, jintArray pixels, jfloat confidence,
                                                                   jfloat iou) {
  (void)cls;
  if (!g_model || (*env)->GetArrayLength(env, pixels) != SIDE * SIDE) {
    return NULL;
  }
  jint* argb = (*env)->GetIntArrayElements(env, pixels, NULL);
  for (int i = 0; i < SIDE * SIDE; ++i) {
    g_input[3 * i] = (uint8_t)(argb[i] >> 16);
    g_input[3 * i + 1] = (uint8_t)(argb[i] >> 8);
    g_input[3 * i + 2] = (uint8_t)argb[i];
  }
  (*env)->ReleaseIntArrayElements(env, pixels, argb, JNI_ABORT);

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  uairt_status status = uairt_model_run(g_model, &g_in, 1, g_out, 3);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (status != UAIRT_OK) {
    error(env, "run", status);
    return NULL;
  }
  float ms = (float)((t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6);

  size_t anchors = g_out[g_scores].nbytes;
  candidate* all = malloc(anchors * sizeof(candidate));
  size_t count = 0;
  const uairt_tensor *b = &g_out[g_boxes], *s = &g_out[g_scores];
  for (size_t a = 0; a < anchors; ++a) {
    float score = ((float)g_output[g_scores][a] - (float)s->quant_zero_point) * s->quant_scale;
    if (score <= confidence) continue;
    const uint8_t* q = g_output[g_boxes] + 4 * a;
    float v[4];
    for (int k = 0; k < 4; ++k) v[k] = ((float)q[k] - (float)b->quant_zero_point) * b->quant_scale;
    all[count++] = (candidate){v[0], v[1], v[2], v[3], score, g_output[g_classes][a]};
  }
  qsort(all, count, sizeof(candidate), by_score);
  candidate kept[MAX_RESULTS];
  size_t n = 0;
  for (size_t i = 0; i < count && n < MAX_RESULTS; ++i) {
    int suppressed = 0;
    for (size_t k = 0; k < n && !suppressed; ++k) {
      suppressed = kept[k].class_id == all[i].class_id && overlap(&kept[k], &all[i]) > iou;
    }
    if (!suppressed) kept[n++] = all[i];
  }
  free(all);

  jfloatArray result = (*env)->NewFloatArray(env, (jsize)(2 + 6 * n));
  float* out = malloc((2 + 6 * n) * sizeof(float));
  out[0] = ms;
  out[1] = (float)n;
  for (size_t i = 0; i < n; ++i) {
    out[2 + 6 * i] = kept[i].x0;
    out[3 + 6 * i] = kept[i].y0;
    out[4 + 6 * i] = kept[i].x1;
    out[5 + 6 * i] = kept[i].y1;
    out[6 + 6 * i] = kept[i].score;
    out[7 + 6 * i] = (float)kept[i].class_id;
  }
  (*env)->SetFloatArrayRegion(env, result, 0, (jsize)(2 + 6 * n), out);
  free(out);
  return result;
}
