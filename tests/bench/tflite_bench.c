/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Times a TFLite model, optionally through the QNN TFLite delegate, so it can be
 * compared with UAIRT's QNN backend. Needs TFLite's C API headers and the delegate
 * header from your QAIRT SDK; neither is part of this repository.
 *
 *   tflite_bench [--delegate-backend htp] [--backend-lib libQnnHtp.so]
 *                [--skel-dir dir] [--precision quant|fp16]
 *                [--perf default|burst|high_performance|sustained|balanced|power_saver]
 *                <model.tflite> <input.bin> <out-prefix> <runs>
 *
 * Without --delegate-backend it runs TFLite's own CPU kernels. Outputs are written to
 * <out-prefix><index>.bin; the timing line goes to stderr.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "QnnTFLiteDelegate.h"
#include "tensorflow/lite/c/c_api.h"

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static int compare_doubles(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static void* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  *size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void* data = malloc(*size);
  if (data && fread(data, 1, *size, f) != *size) {
    free(data);
    data = NULL;
  }
  fclose(f);
  return data;
}

static void describe(const char* kind, int index, const TfLiteTensor* t) {
  TfLiteQuantizationParams q = TfLiteTensorQuantizationParams(t);
  printf("%s %d name=%s type=%d scale=%g zero_point=%d dims=", kind, index, TfLiteTensorName(t),
         (int)TfLiteTensorType(t), (double)q.scale, q.zero_point);
  for (int d = 0; d < TfLiteTensorNumDims(t); ++d) {
    printf("%s%d", d ? "x" : "", TfLiteTensorDim(t, d));
  }
  printf(" bytes=%zu\n", TfLiteTensorByteSize(t));
}

int main(int argc, char** argv) {
  const char *backend = NULL, *backend_lib = "", *skel_dir = "", *precision = "quant", *perf = "default";
  int arg = 1;
  for (; arg + 1 < argc && argv[arg][0] == '-'; arg += 2) {
    if (!strcmp(argv[arg], "--delegate-backend")) backend = argv[arg + 1];
    else if (!strcmp(argv[arg], "--backend-lib")) backend_lib = argv[arg + 1];
    else if (!strcmp(argv[arg], "--skel-dir")) skel_dir = argv[arg + 1];
    else if (!strcmp(argv[arg], "--precision")) precision = argv[arg + 1];
    else if (!strcmp(argv[arg], "--perf")) perf = argv[arg + 1];
    else { fprintf(stderr, "unknown option %s\n", argv[arg]); return 2; }
  }
  if (argc - arg != 4) {
    fprintf(stderr, "usage: %s [options] <model.tflite> <input.bin> <out-prefix> <runs>\n", argv[0]);
    return 2;
  }
  const char *model_path = argv[arg], *input_path = argv[arg + 1], *prefix = argv[arg + 2];
  int runs = atoi(argv[arg + 3]);

  double start = now_ms();
  TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteDelegate* delegate = NULL;
  if (backend) {
    TfLiteQnnDelegateOptions delegate_options = TfLiteQnnDelegateOptionsDefault();
    delegate_options.backend_type = !strcmp(backend, "htp") ? kHtpBackend : kGpuBackend;
    delegate_options.library_path = backend_lib;
    delegate_options.skel_library_dir = skel_dir;
    delegate_options.htp_options.precision = !strcmp(precision, "fp16") ? kHtpFp16 : kHtpQuantized;
    TfLiteQnnDelegateHtpPerformanceMode mode = kHtpDefault;
    if (!strcmp(perf, "burst")) mode = kHtpBurst;
    else if (!strcmp(perf, "high_performance")) mode = kHtpHighPerformance;
    else if (!strcmp(perf, "sustained")) mode = kHtpSustainedHighPerformance;
    else if (!strcmp(perf, "balanced")) mode = kHtpBalanced;
    else if (!strcmp(perf, "power_saver")) mode = kHtpPowerSaver;
    delegate_options.htp_options.performance_mode = mode;
    delegate = TfLiteQnnDelegateCreate(&delegate_options);
    if (!delegate) {
      fprintf(stderr, "TfLiteQnnDelegateCreate failed\n");
      return 1;
    }
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
  }
  TfLiteInterpreter* interpreter = model ? TfLiteInterpreterCreate(model, options) : NULL;
  if (!interpreter || TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    fprintf(stderr, "cannot create the interpreter for %s\n", model_path);
    return 1;
  }
  double init_ms = now_ms() - start;

  int n_in = TfLiteInterpreterGetInputTensorCount(interpreter);
  int n_out = TfLiteInterpreterGetOutputTensorCount(interpreter);
  for (int i = 0; i < n_in; ++i) describe("input", i, TfLiteInterpreterGetInputTensor(interpreter, i));
  for (int i = 0; i < n_out; ++i) describe("output", i, TfLiteInterpreterGetOutputTensor(interpreter, i));

  size_t size = 0;
  void* input = read_file(input_path, &size);
  TfLiteTensor* in0 = TfLiteInterpreterGetInputTensor(interpreter, 0);
  if (!input || size != TfLiteTensorByteSize(in0) ||
      TfLiteTensorCopyFromBuffer(in0, input, size) != kTfLiteOk) {
    fprintf(stderr, "input file does not match the model input (%zu bytes expected)\n",
            TfLiteTensorByteSize(in0));
    return 1;
  }

  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    fprintf(stderr, "invoke failed\n");
    return 1;
  }
  double* times = malloc((size_t)(runs ? runs : 1) * sizeof(*times));
  for (int i = 0; i < runs; ++i) {
    double t0 = now_ms();
    TfLiteInterpreterInvoke(interpreter);
    times[i] = now_ms() - t0;
  }
  for (int i = 0; i < n_out; ++i) {
    const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interpreter, i);
    size_t bytes = TfLiteTensorByteSize(t);
    void* buffer = malloc(bytes);
    TfLiteTensorCopyToBuffer(t, buffer, bytes);
    char path[1024];
    snprintf(path, sizeof(path), "%s%d.bin", prefix, i);
    FILE* f = fopen(path, "wb");
    fwrite(buffer, 1, bytes, f);
    fclose(f);
    free(buffer);
  }
  if (runs > 0) {
    qsort(times, (size_t)runs, sizeof(*times), compare_doubles);
    double sum = 0;
    for (int i = 0; i < runs; ++i) sum += times[i];
    fprintf(stderr, "init_ms=%.1f latency_ms runs=%d min=%.2f median=%.2f mean=%.2f\n", init_ms, runs,
            times[0], times[runs / 2], sum / runs);
  }
  TfLiteInterpreterDelete(interpreter);
  if (delegate) TfLiteQnnDelegateDelete(delegate);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  return 0;
}
