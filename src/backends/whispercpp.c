/* SPDX-License-Identifier: Apache-2.0 */
/*
 * whisper.cpp backend (speech to text), built as a loadable plugin that links the whisper library you installed.
 * v1 limits: a ggml .bin model from a path, greedy decoding, one input of fixed length. The tensor API has no strings and
 * shapes are static, so the model takes
 *   input 0  "audio"        float32 [n_samples]        mono PCM at 16 kHz in [-1, 1]
 * and returns
 *   output 0 "text"         uint8   [max_text_bytes]   the transcript as UTF-8, zero padded
 *   output 1 "text_length"  int32   [1]                bytes of text used
 * n_samples and max_text_bytes are engine options. A transcript that does not fit fails the run; it is never truncated.
 * A whisper context is not thread safe: do not run one model from two threads at once.
 */
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <whisper.h>

#include <uairt/uairt_backend.h>

#include "common.h"

#define OPTION_N_SAMPLES "n_samples"
#define OPTION_MAX_TEXT_BYTES "max_text_bytes"
#define OPTION_LANGUAGE "language"
#define OPTION_TRANSLATE "translate"
#define OPTION_THREADS "num_threads"
#define OPTION_USE_GPU "use_gpu"
#define OPTION_GPU_DEVICE "gpu_device"
#define OPTION_FLASH_ATTN "flash_attn"

#define SAMPLE_RATE 16000
#define DEFAULT_N_SAMPLES (30 * SAMPLE_RATE)
#define DEFAULT_MAX_TEXT_BYTES 4096
#define LANGUAGE_MAX 16

static const uairt_host_api* g_host;

#define fail(...) uairt_backend_fail(g_host, __VA_ARGS__)

typedef struct {
  int64_t n_samples;
  int64_t max_text_bytes;
  char language[LANGUAGE_MAX];
  bool translate;
  int num_threads; /* 0: whisper's default */
  bool use_gpu;
  int gpu_device;
  bool flash_attn;
} engine_t;

typedef struct {
  struct whisper_context* ctx;
  engine_t options;
} model_t;

static bool parse_integer(const char* text, long min, long max, long* out) {
  if (!text || !*text) {
    return false;
  }
  char* end = NULL;
  errno = 0;
  long value = strtol(text, &end, 10);
  if (errno != 0 || *end != '\0' || value < min || value > max) {
    return false;
  }
  *out = value;
  return true;
}

static bool parse_boolean(const char* text, bool* out) {
  if (text && strcmp(text, "true") == 0) {
    *out = true;
    return true;
  }
  if (text && strcmp(text, "false") == 0) {
    *out = false;
    return true;
  }
  return false;
}

static uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    fail("out of memory");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->n_samples = DEFAULT_N_SAMPLES;
  engine->max_text_bytes = DEFAULT_MAX_TEXT_BYTES;
  strcpy(engine->language, "en");
  engine->use_gpu = true;
  for (size_t i = 0; i < num_options; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    long number = 0;
    bool ok = true;
    if (strcmp(key, OPTION_N_SAMPLES) == 0) {
      ok = parse_integer(value, SAMPLE_RATE / 10, 1L << 30, &number);
      engine->n_samples = number;
    } else if (strcmp(key, OPTION_MAX_TEXT_BYTES) == 0) {
      ok = parse_integer(value, 1, 1L << 24, &number);
      engine->max_text_bytes = number;
    } else if (strcmp(key, OPTION_LANGUAGE) == 0) {
      ok = *value && strlen(value) < LANGUAGE_MAX;
      if (ok) {
        strcpy(engine->language, value);
      }
    } else if (strcmp(key, OPTION_THREADS) == 0) {
      ok = parse_integer(value, 1, 1024, &number);
      engine->num_threads = (int)number;
    } else if (strcmp(key, OPTION_GPU_DEVICE) == 0) {
      ok = parse_integer(value, 0, 1024, &number);
      engine->gpu_device = (int)number;
    } else if (strcmp(key, OPTION_TRANSLATE) == 0) {
      ok = parse_boolean(value, &engine->translate);
    } else if (strcmp(key, OPTION_USE_GPU) == 0) {
      ok = parse_boolean(value, &engine->use_gpu);
    } else if (strcmp(key, OPTION_FLASH_ATTN) == 0) {
      ok = parse_boolean(value, &engine->flash_attn);
    } else {
      fail("unknown option '%s'", key);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    if (!ok) {
      fail("invalid value '%s' for option '%s'", value, key);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  *out_engine = engine;
  return UAIRT_OK;
}

static void destroy_engine(void* engine) {
  free(engine);
}

static uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const engine_t* engine = engine_handle;
  if (!source->path) {
    fail("the whispercpp backend loads a ggml .bin model from a path, not from memory");
    return UAIRT_ERR_UNSUPPORTED;
  }
  if (!uairt_backend_file_exists(source->path)) {
    fail("cannot open the whisper.cpp model '%s'", source->path);
    return UAIRT_ERR_IO;
  }
  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    fail("out of memory");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  struct whisper_context_params params = whisper_context_default_params();
  params.use_gpu = engine->use_gpu;
  params.gpu_device = engine->gpu_device;
  params.flash_attn = engine->flash_attn;
  model->ctx = whisper_init_from_file_with_params(source->path, params);
  if (!model->ctx) {
    fail("whisper.cpp cannot load '%s' (is it a whisper ggml .bin model?)", source->path);
    free(model);
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  if (!whisper_is_multilingual(model->ctx) && strcmp(engine->language, "en") != 0 && strcmp(engine->language, "auto") != 0) {
    fail("'%s' is an English-only model; option '%s' is '%s'", source->path, OPTION_LANGUAGE, engine->language);
    whisper_free(model->ctx);
    free(model);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  model->options = *engine;
  *out_model = model;
  return UAIRT_OK;
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  if (model) {
    whisper_free(model->ctx);
    free(model);
  }
}

static size_t num_inputs(const void* model) {
  (void)model;
  return 1;
}

static size_t num_outputs(const void* model) {
  (void)model;
  return 2;
}

static void describe(uairt_tensor* out, const char* name, uairt_dtype dtype, int64_t elements, size_t element_size) {
  memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(*out);
  out->dtype = dtype;
  out->rank = 1;
  out->dims[0] = elements;
  out->domain = UAIRT_MEM_HOST;
  out->dmabuf_fd = -1;
  out->nbytes = (size_t)elements * element_size;
  out->name = name;
}

static uairt_status input_info(const void* handle, size_t index, uairt_tensor* out) {
  (void)index;
  describe(out, "audio", UAIRT_DTYPE_FLOAT32, ((const model_t*)handle)->options.n_samples, sizeof(float));
  return UAIRT_OK;
}

static uairt_status output_info(const void* handle, size_t index, uairt_tensor* out) {
  const model_t* model = handle;
  if (index == 0) {
    describe(out, "text", UAIRT_DTYPE_UINT8, model->options.max_text_bytes, 1);
  } else {
    describe(out, "text_length", UAIRT_DTYPE_INT32, 1, sizeof(int32_t));
  }
  return UAIRT_OK;
}

static uairt_status run(void* handle, const uairt_tensor* inputs, size_t n_in, uairt_tensor* outputs, size_t n_out) {
  model_t* model = handle;
  (void)n_in;
  (void)n_out;
  struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  params.print_progress = false;
  params.print_realtime = false;
  params.print_timestamps = false;
  params.print_special = false;
  params.translate = model->options.translate;
  params.language = model->options.language;
  params.detect_language = strcmp(model->options.language, "auto") == 0;
  if (model->options.num_threads > 0) {
    params.n_threads = model->options.num_threads;
  }
  if (whisper_full(model->ctx, params, (const float*)inputs[0].data, (int)model->options.n_samples) != 0) {
    fail("whisper.cpp failed to transcribe the audio (inputs under 100 ms are rejected)");
    return UAIRT_ERR_RUNTIME;
  }
  uint8_t* text = outputs[0].data;
  size_t used = 0;
  const size_t capacity = (size_t)model->options.max_text_bytes;
  const int segments = whisper_full_n_segments(model->ctx);
  for (int i = 0; i < segments; ++i) {
    const char* piece = whisper_full_get_segment_text(model->ctx, i);
    size_t length = piece ? strlen(piece) : 0;
    if (length > capacity - used) {
      size_t needed = used + length;
      for (int j = i + 1; j < segments; ++j) {
        const char* rest = whisper_full_get_segment_text(model->ctx, j);
        needed += rest ? strlen(rest) : 0;
      }
      fail("the transcript needs %zu bytes but option '%s' is %zu", needed, OPTION_MAX_TEXT_BYTES, capacity);
      return UAIRT_ERR_RUNTIME;
    }
    memcpy(text + used, piece, length);
    used += length;
  }
  memset(text + used, 0, capacity - used);
  *(int32_t*)outputs[1].data = (int32_t)used;
  return UAIRT_OK;
}

static const uairt_backend_api g_api = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "whispercpp",
    .supported_domains = UAIRT_MEM_HOST,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_inputs,
    .num_outputs = num_outputs,
    .input_info = input_info,
    .output_info = output_info,
    .run = run,
};

UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return &g_api;
}
