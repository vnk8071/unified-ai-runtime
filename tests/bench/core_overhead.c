/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Times uairt_model_run with zero-filled host buffers sized from the model. With the reference backend the plugin's work is a
 * 16-byte copy, so the number is the core's per-run overhead; with a real backend, run it before and after a core change and
 * compare. Built with the tests (bench_core_overhead), not run by ctest.
 *
 *   bench_core_overhead [--runs N] [--option key=value]... [<plugin>] <backend> <model>
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <uairt/uairt.h>

#define MAX_TENSORS 16
#define MAX_OPTIONS 16

static double now_ns(void) {
#if defined(_WIN32)
  LARGE_INTEGER frequency, counter;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart * 1e9 / (double)frequency.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}

static int compare_double(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

int main(int argc, char** argv) {
  long runs = 200000;
  int arg = 1;
  uairt_option options[MAX_OPTIONS];
  size_t num_options = 0;
  while (arg + 1 < argc && argv[arg][0] == '-') {
    if (strcmp(argv[arg], "--runs") == 0) {
      runs = atol(argv[arg + 1]);
    } else if (strcmp(argv[arg], "--option") == 0 && num_options < MAX_OPTIONS && strchr(argv[arg + 1], '=')) {
      char* pair = argv[arg + 1];
      char* equals = strchr(pair, '=');
      *equals = '\0';
      options[num_options].key = pair;
      options[num_options].value = equals + 1;
      ++num_options;
    } else {
      break;
    }
    arg += 2;
  }
  const char* plugin = argc - arg == 3 ? argv[arg++] : NULL;
  if (argc - arg != 2 || runs < 1000) {
    fprintf(stderr, "usage: %s [--runs N (>=1000)] [--option key=value]... [<plugin>] <backend> <model>\n", argv[0]);
    return 2;
  }
  if (plugin && uairt_load_backend_library(plugin) != UAIRT_OK) {
    fprintf(stderr, "load plugin: %s\n", uairt_last_error());
    return 1;
  }
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  uairt_model_source source = {.struct_size = sizeof(source), .path = argv[arg + 1]};
  if (uairt_engine_create(argv[arg], options, num_options, &engine) != UAIRT_OK ||
      uairt_model_load(engine, &source, &model) != UAIRT_OK) {
    fprintf(stderr, "%s\n", uairt_last_error());
    return 1;
  }
  size_t n_in = uairt_model_num_inputs(model), n_out = uairt_model_num_outputs(model);
  if (n_in > MAX_TENSORS || n_out > MAX_TENSORS) {
    fprintf(stderr, "more than %d tensors\n", MAX_TENSORS);
    return 1;
  }
  uairt_tensor in[MAX_TENSORS], out[MAX_TENSORS];
  for (size_t i = 0; i < n_in; ++i) {
    uairt_model_input_info(model, i, &in[i]);
    in[i].nbytes = (size_t)uairt_tensor_num_elements(&in[i]) * uairt_dtype_size(in[i].dtype);
    in[i].data = calloc(1, in[i].nbytes);
  }
  for (size_t i = 0; i < n_out; ++i) {
    uairt_model_output_info(model, i, &out[i]);
    out[i].nbytes = (size_t)uairt_tensor_num_elements(&out[i]) * uairt_dtype_size(out[i].dtype);
    out[i].data = calloc(1, out[i].nbytes);
  }

  enum { BATCH = 100 };
  long batches = runs / BATCH;
  double* samples = malloc((size_t)batches * sizeof(double));
  for (long warm = 0; warm < 1000; ++warm) {
    uairt_model_run(model, in, n_in, out, n_out);
  }
  for (long b = 0; b < batches; ++b) {
    double start = now_ns();
    for (int i = 0; i < BATCH; ++i) {
      if (uairt_model_run(model, in, n_in, out, n_out) != UAIRT_OK) {
        fprintf(stderr, "run: %s\n", uairt_last_error());
        return 1;
      }
    }
    samples[b] = (now_ns() - start) / BATCH;
  }
  qsort(samples, (size_t)batches, sizeof(double), compare_double);
  printf("%s: %zu in, %zu out, %ld runs: min %.0f ns, median %.0f ns per uairt_model_run\n", argv[arg], n_in, n_out,
         batches * BATCH, samples[0], samples[batches / 2]);
  return 0;
}
