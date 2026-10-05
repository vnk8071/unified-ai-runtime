/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Runs a model with raw input files and writes raw output files.
 *   run_model [--option key=value]... <plugin> <backend> <model> <out-prefix> <input.bin>...
 *   run_model [--option key=value]... --info <plugin> <backend> <model>
 * Outputs are written to <out-prefix><index>.bin. --info prints the model's I/O
 * and exits. --repeat N times N extra runs and prints min/median/mean latency.
 * --dmabuf allocates every input and output as a zero-copy DMABUF buffer, --pinned as a pinned (page-locked) one. --memory loads the model
 * file into memory and passes it as data instead of a path. The engine and
 * model creation time is printed to stderr as init_ms. Used by the compare_with_*.py scripts.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#define CLOCK_MONOTONIC 1
static int clock_gettime(int clock, struct timespec* ts) {
  LARGE_INTEGER freq, now;
  (void)clock;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  ts->tv_sec = (time_t)(now.QuadPart / freq.QuadPart);
  ts->tv_nsec = (long)((now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
  return 0;
}
#endif

#include <uairt/uairt.h>

#define MAX_OPTIONS 16

static void* read_file(const char* path, size_t* size) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  *size = (size_t)ftell(file);
  fseek(file, 0, SEEK_SET);
  void* data = malloc(*size ? *size : 1);
  if (data && fread(data, 1, *size, file) != *size) {
    free(data);
    data = NULL;
  }
  fclose(file);
  return data;
}

static int compare_doubles(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static int die(const char* what, uairt_status status) {
  fprintf(stderr, "%s: %s (%s)\n", what, uairt_status_string(status), uairt_last_error());
  return 1;
}

static void print_io(const char* kind, size_t index, const uairt_tensor* tensor) {
  printf("%s %zu name=%s dtype=%d scale=%g zero_point=%d dims=", kind, index,
         tensor->name ? tensor->name : "", tensor->dtype, tensor->quant_scale,
         tensor->quant_zero_point);
  for (uint32_t i = 0; i < tensor->rank; ++i) {
    printf("%s%lld", i ? "x" : "", (long long)tensor->dims[i]);
  }
  printf("\n");
}

int main(int argc, char** argv) {
  uairt_option options[MAX_OPTIONS];
  size_t num_options = 0;
  bool info_only = false;
  int repeat = 0;
  bool use_dmabuf = false;
  bool use_pinned = false;
  bool from_memory = false;
  int arg = 1;
  for (; arg < argc; ++arg) {
    if (strcmp(argv[arg], "--option") == 0 && arg + 1 < argc && num_options < MAX_OPTIONS) {
      char* pair = argv[++arg];
      char* equals = strchr(pair, '=');
      if (!equals) {
        fprintf(stderr, "--option expects key=value\n");
        return 2;
      }
      *equals = '\0';
      options[num_options].key = pair;
      options[num_options++].value = equals + 1;
    } else if (strcmp(argv[arg], "--repeat") == 0 && arg + 1 < argc) {
      repeat = atoi(argv[++arg]);
    } else if (strcmp(argv[arg], "--memory") == 0) {
      from_memory = true;
    } else if (strcmp(argv[arg], "--dmabuf") == 0) {
      use_dmabuf = true;
    } else if (strcmp(argv[arg], "--pinned") == 0) {
      use_pinned = true;
    } else if (strcmp(argv[arg], "--info") == 0) {
      info_only = true;
    } else {
      break;
    }
  }
  int positional = argc - arg;
  if (positional < (info_only ? 3 : 4)) {
    fprintf(stderr,
            "usage: %s [--option k=v]... [--info] <plugin> <backend> <model> [<out-prefix> <input.bin>...]\n",
            argv[0]);
    return 2;
  }
  const char *plugin = argv[arg], *backend = argv[arg + 1], *model_path = argv[arg + 2];
  const char* out_prefix = info_only ? NULL : argv[arg + 3];
  char** input_files = info_only ? NULL : argv + arg + 4;
  int num_input_files = info_only ? 0 : positional - 4;

  uairt_status status = uairt_load_backend_library(plugin);
  if (status != UAIRT_OK) {
    return die("load backend", status);
  }
  struct timespec init_start;
  clock_gettime(CLOCK_MONOTONIC, &init_start);
  uairt_engine* engine = NULL;
  status = uairt_engine_create(backend, options, num_options, &engine);
  if (status != UAIRT_OK) {
    return die("create engine", status);
  }
  uairt_model_source source = {.struct_size = sizeof(source), .path = model_path};
  void* model_bytes = NULL;
  if (from_memory) {
    size_t model_size = 0;
    model_bytes = read_file(model_path, &model_size);
    if (!model_bytes) {
      fprintf(stderr, "cannot read %s\n", model_path);
      return 1;
    }
    source.path = NULL;
    source.data = model_bytes;
    source.size = model_size;
  }
  uairt_model* model = NULL;
  status = uairt_model_load(engine, &source, &model);
  if (status != UAIRT_OK) {
    return die("load model", status);
  }
  struct timespec init_end;
  clock_gettime(CLOCK_MONOTONIC, &init_end);
  fprintf(stderr, "init_ms=%.1f\n", (double)(init_end.tv_sec - init_start.tv_sec) * 1e3 + (double)(init_end.tv_nsec - init_start.tv_nsec) / 1e6);

  size_t n_in = uairt_model_num_inputs(model);
  size_t n_out = uairt_model_num_outputs(model);
  uairt_tensor* in = calloc(n_in ? n_in : 1, sizeof(*in));
  uairt_tensor* out = calloc(n_out ? n_out : 1, sizeof(*out));
  for (size_t i = 0; i < n_in; ++i) {
    uairt_model_input_info(model, i, &in[i]);
    if (info_only) {
      print_io("input", i, &in[i]);
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    uairt_model_output_info(model, i, &out[i]);
    if (info_only) {
      print_io("output", i, &out[i]);
    }
  }
  if (info_only) {
    free(in);
    free(out);
    uairt_model_destroy(model);
    uairt_engine_destroy(engine);
    return 0;
  }

  if ((size_t)num_input_files != n_in) {
    fprintf(stderr, "model needs %zu input file(s), got %d\n", n_in, num_input_files);
    return 2;
  }
  const bool use_buffers = use_dmabuf || use_pinned;
  const uairt_memory_domain buffer_domain = use_pinned ? UAIRT_MEM_PINNED : UAIRT_MEM_DMABUF;
  uairt_buffer** in_buffers = calloc(n_in ? n_in : 1, sizeof(*in_buffers));
  uairt_buffer** out_buffers = calloc(n_out ? n_out : 1, sizeof(*out_buffers));
  for (size_t i = 0; i < n_in; ++i) {
    size_t size = 0;
    void* data = read_file(input_files[i], &size);
    if (!data) {
      fprintf(stderr, "cannot read %s\n", input_files[i]);
      return 1;
    }
    if (use_buffers) {
      status = uairt_buffer_alloc(engine, size, buffer_domain, &in_buffers[i]);
      if (status != UAIRT_OK) {
        return die("allocate input buffer", status);
      }
      memcpy(uairt_buffer_data(in_buffers[i]), data, size);
      free(data);
      uairt_tensor_use_buffer(&in[i], in_buffers[i]);
    } else {
      in[i].data = data;
      in[i].nbytes = size;
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    size_t size = (size_t)uairt_tensor_num_elements(&out[i]) * uairt_dtype_size(out[i].dtype);
    if (use_buffers) {
      status = uairt_buffer_alloc(engine, size, buffer_domain, &out_buffers[i]);
      if (status != UAIRT_OK) {
        return die("allocate output buffer", status);
      }
      uairt_tensor_use_buffer(&out[i], out_buffers[i]);
    } else {
      out[i].nbytes = size;
      out[i].data = malloc(size);
    }
  }

  status = uairt_model_run(model, in, n_in, out, n_out);
  if (status != UAIRT_OK) {
    return die("run", status);
  }
  if (repeat > 0) {
    double* times = malloc((size_t)repeat * sizeof(*times));
    for (int i = 0; i < repeat; ++i) {
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      status = uairt_model_run(model, in, n_in, out, n_out);
      clock_gettime(CLOCK_MONOTONIC, &end);
      if (status != UAIRT_OK) {
        return die("run", status);
      }
      times[i] = (double)(end.tv_sec - start.tv_sec) * 1e3 + (double)(end.tv_nsec - start.tv_nsec) / 1e6;
    }
    qsort(times, (size_t)repeat, sizeof(*times), compare_doubles);
    double sum = 0;
    for (int i = 0; i < repeat; ++i) {
      sum += times[i];
    }
    fprintf(stderr, "latency_ms runs=%d min=%.2f median=%.2f mean=%.2f\n", repeat, times[0],
            times[repeat / 2], sum / repeat);
    free(times);
  }
  for (size_t i = 0; i < n_out; ++i) {
    char path[1024];
    snprintf(path, sizeof(path), "%s%zu.bin", out_prefix, i);
    FILE* file = fopen(path, "wb");
    if (!file || fwrite(out[i].data, 1, out[i].nbytes, file) != out[i].nbytes) {
      fprintf(stderr, "cannot write %s\n", path);
      return 1;
    }
    fclose(file);
  }
  uairt_model_destroy(model);
  free(model_bytes);
  for (size_t i = 0; i < n_in; ++i) {
    if (in_buffers[i]) {
      uairt_buffer_free(in_buffers[i]);
    } else {
      free(in[i].data);
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    if (out_buffers[i]) {
      uairt_buffer_free(out_buffers[i]);
    } else {
      free(out[i].data);
    }
  }
  free(in_buffers);
  free(out_buffers);
  free(in);
  free(out);
  uairt_engine_destroy(engine);
  return 0;
}
