/* SPDX-License-Identifier: Apache-2.0 */
/*
 * llama.cpp backend (GGUF language models), built as a loadable plugin with llama.cpp linked statically and hidden.
 * A model has no tensors: it is used through sessions (see the session table below).
 */
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <llama.h>
#include <uairt/uairt_backend.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#define OPTION_N_GPU_LAYERS "n_gpu_layers"
#define OPTION_DEVICE "device"
#define DEVICE_NAME_MAX 64

static const uairt_host_api* g_host;

#if defined(__GNUC__)
static void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

static void fail(const char* fmt, ...) {
  char message[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

typedef struct {
  int n_gpu_layers;
  char device[DEVICE_NAME_MAX]; /* empty: llama.cpp uses every device it finds */
} engine_t;

typedef struct {
  struct llama_model* model;
  const struct llama_vocab* vocab;
  int32_t n_vocab;
  int32_t n_ctx_train;
} model_t;

static void log_errors_only(enum ggml_log_level level, const char* text, void* user_data) {
  (void)user_data;
  if (level == GGML_LOG_LEVEL_ERROR) {
    fputs(text, stderr);
  }
}

static void init_llama(void) {
  llama_log_set(log_errors_only, NULL);
  llama_backend_init();
}

#if defined(_WIN32)
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_llama_once(PINIT_ONCE once, PVOID param, PVOID* context) {
  (void)once;
  (void)param;
  (void)context;
  init_llama();
  return TRUE;
}
static void ensure_llama(void) {
  InitOnceExecuteOnce(&g_once, init_llama_once, NULL, NULL);
}
#else
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void ensure_llama(void) {
  pthread_once(&g_once, init_llama);
}
#endif

static uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->n_gpu_layers = 99;
  for (size_t i = 0; i < num_options; ++i) {
    const char* value = options[i].value ? options[i].value : "";
    if (strcmp(options[i].key, OPTION_DEVICE) == 0) {
      if (!*value || strlen(value) >= sizeof(engine->device)) {
        fail("%s must be a device name of 1 to %d characters, got '%s'", OPTION_DEVICE, DEVICE_NAME_MAX - 1, value);
        free(engine);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
      memcpy(engine->device, value, strlen(value) + 1);
      continue;
    }
    if (strcmp(options[i].key, OPTION_N_GPU_LAYERS) != 0) {
      fail("unknown option '%s'", options[i].key);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    char* end = NULL;
    errno = 0;
    long layers = strtol(value, &end, 10);
    if (end == value || *end || errno || layers < 0 || layers > INT_MAX) {
      fail("%s must be a non-negative integer, got '%s'", OPTION_N_GPU_LAYERS, value);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    engine->n_gpu_layers = (int)layers;
  }
  ensure_llama();
  *out_engine = engine;
  return UAIRT_OK;
}

static void destroy_engine(void* engine) {
  free(engine);
}

static uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const engine_t* engine = engine_handle;
  if (!source->path) {
    fail("llamacpp loads models from a path, not from memory");
    return UAIRT_ERR_UNSUPPORTED;
  }
  FILE* file = fopen(source->path, "rb");
  if (!file) {
    fail("cannot open model '%s'", source->path);
    return UAIRT_ERR_IO;
  }
  fclose(file);

  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  struct llama_model_params params = llama_model_default_params();
  params.n_gpu_layers = engine->n_gpu_layers;
  ggml_backend_dev_t devices[2] = {NULL, NULL};
  if (engine->device[0]) {
    devices[0] = ggml_backend_dev_by_name(engine->device);
    if (!devices[0]) {
      fail("no llama.cpp device named '%s' in this build", engine->device);
      free(model);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    params.devices = devices;
  }
  model->model = llama_model_load_from_file(source->path, params);
  if (!model->model) {
    fail("cannot load '%s' as a GGUF model", source->path);
    free(model);
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  model->vocab = llama_model_get_vocab(model->model);
  model->n_vocab = llama_vocab_n_tokens(model->vocab);
  model->n_ctx_train = llama_model_n_ctx_train(model->model);
  *out_model = model;
  return UAIRT_OK;
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  llama_model_free(model->model);
  free(model);
}

static size_t num_io(const void* model) {
  (void)model;
  return 0;
}

static uairt_status io_info(const void* model, size_t index, uairt_tensor* out) {
  (void)model;
  (void)index;
  (void)out;
  fail("llamacpp models have no tensors; use a session");
  return UAIRT_ERR_INVALID_ARGUMENT;
}

static uairt_status run(void* model, const uairt_tensor* in, size_t n_in, uairt_tensor* out, size_t n_out) {
  (void)model;
  (void)in;
  (void)n_in;
  (void)out;
  (void)n_out;
  fail("llamacpp models run through sessions (uairt_session_create), not uairt_model_run");
  return UAIRT_ERR_UNSUPPORTED;
}

typedef struct {
  struct llama_context* ctx;
  const model_t* model;
  uint32_t n_ctx;
  uint32_t n_batch;
  size_t pos;
} session_t;

static bool valid_tokens(const model_t* model, const int32_t* tokens, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= model->n_vocab) {
      fail("token id %d is outside the vocabulary (0..%d)", tokens[i], model->n_vocab - 1);
      return false;
    }
  }
  return true;
}

static uairt_status vocab_size(const void* handle, size_t* out_size) {
  *out_size = (size_t)((const model_t*)handle)->n_vocab;
  return UAIRT_OK;
}

static uairt_status tokenize(
    const void* handle,
    const char* text,
    size_t text_len,
    int add_special,
    int32_t* tokens,
    size_t capacity,
    size_t* out_count) {
  const model_t* model = handle;
  if (text_len > INT32_MAX || capacity > INT32_MAX) {
    fail("text or buffer is too large");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  int32_t n = llama_tokenize(
      model->vocab, text, (int32_t)text_len, tokens, (int32_t)capacity, add_special != 0, true);
  if (n == INT32_MIN) {
    fail("text is too large to tokenize");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (n < 0) {
    *out_count = (size_t)(-(int64_t)n);
    fail("token buffer too small: %zu tokens needed", *out_count);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_count = (size_t)n;
  return UAIRT_OK;
}

static uairt_status detokenize(
    const void* handle,
    const int32_t* tokens,
    size_t count,
    char* out,
    size_t capacity,
    size_t* out_nbytes) {
  const model_t* model = handle;
  if (count > INT32_MAX || capacity > INT32_MAX) {
    fail("token list or buffer is too large");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (!valid_tokens(model, tokens, count)) {
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  int32_t n = llama_detokenize(model->vocab, tokens, (int32_t)count, out, (int32_t)capacity, false, true);
  if (n < 0) {
    *out_nbytes = (size_t)(-(int64_t)n);
    fail("text buffer too small: %zu bytes needed", *out_nbytes);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_nbytes = (size_t)n;
  return UAIRT_OK;
}

static uairt_status create_session(
    void* model_handle,
    const uairt_option* options,
    size_t num_options,
    void** out_session) {
  const model_t* model = model_handle;
  unsigned long n_ctx = 2048;
  for (size_t i = 0; i < num_options; ++i) {
    if (strcmp(options[i].key, "n_ctx") != 0) {
      fail("unknown session option '%s'", options[i].key);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    const char* value = options[i].value ? options[i].value : "";
    char* end = NULL;
    errno = 0;
    n_ctx = strtoul(value, &end, 10);
    if (end == value || *end || errno || n_ctx < 1 || n_ctx > INT32_MAX) {
      fail("n_ctx must be a positive integer, got '%s'", value);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  if (model->n_ctx_train > 0 && n_ctx > (unsigned long)model->n_ctx_train) {
    n_ctx = (unsigned long)model->n_ctx_train;
  }
  session_t* session = calloc(1, sizeof(*session));
  if (!session) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  struct llama_context_params params = llama_context_default_params();
  params.n_ctx = (uint32_t)n_ctx;
  params.n_batch = (uint32_t)(n_ctx < 512 ? n_ctx : 512);
  params.n_ubatch = params.n_batch;
  params.no_perf = true;
  session->ctx = llama_init_from_model(model->model, params);
  if (!session->ctx) {
    fail("cannot create a llama.cpp context with n_ctx=%lu (out of memory?)", n_ctx);
    free(session);
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  session->model = model;
  session->n_ctx = params.n_ctx;
  session->n_batch = params.n_batch;
  *out_session = session;
  return UAIRT_OK;
}

static void destroy_session(void* handle) {
  session_t* session = handle;
  llama_free(session->ctx);
  free(session);
}

static uairt_status append(void* handle, const int32_t* tokens, size_t count) {
  session_t* session = handle;
  if (count == 0) {
    return UAIRT_OK;
  }
  if (!valid_tokens(session->model, tokens, count)) {
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (count > session->n_ctx - session->pos) {
    fail("context full: %zu tokens in a context of %u, cannot add %zu", session->pos, session->n_ctx, count);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  int32_t chunk[512];
  for (size_t done = 0; done < count;) {
    size_t n = count - done < session->n_batch ? count - done : session->n_batch;
    memcpy(chunk, tokens + done, n * sizeof(int32_t));
    int result = llama_decode(session->ctx, llama_batch_get_one(chunk, (int32_t)n));
    if (result != 0) {
      llama_memory_clear(llama_get_memory(session->ctx), true);
      session->pos = 0;
      fail("llama_decode failed (%d); the session was reset", result);
      return UAIRT_ERR_RUNTIME;
    }
    done += n;
    session->pos += n;
  }
  return UAIRT_OK;
}

static uairt_status logits(const void* handle, float* out, size_t capacity, size_t* out_count) {
  const session_t* session = handle;
  size_t n = (size_t)session->model->n_vocab;
  *out_count = n;
  if (session->pos == 0) {
    fail("no tokens have been appended");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (capacity < n) {
    fail("logits buffer too small: %zu values needed", n);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  const float* values = llama_get_logits_ith(session->ctx, -1);
  if (!values) {
    fail("llama.cpp returned no logits");
    return UAIRT_ERR_RUNTIME;
  }
  memcpy(out, values, n * sizeof(float));
  return UAIRT_OK;
}

static uairt_status position(const void* handle, size_t* out_position) {
  *out_position = ((const session_t*)handle)->pos;
  return UAIRT_OK;
}

static uairt_status reset(void* handle) {
  session_t* session = handle;
  llama_memory_clear(llama_get_memory(session->ctx), true);
  session->pos = 0;
  return UAIRT_OK;
}

static const uairt_session_api kSession = {
    .struct_size = sizeof(uairt_session_api),
    .abi_version = UAIRT_SESSION_ABI_VERSION,
    .vocab_size = vocab_size,
    .tokenize = tokenize,
    .detokenize = detokenize,
    .create_session = create_session,
    .destroy_session = destroy_session,
    .append = append,
    .logits = logits,
    .position = position,
    .reset = reset,
};

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "llamacpp",
    .supported_domains = UAIRT_MEM_HOST,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_io,
    .num_outputs = num_io,
    .input_info = io_info,
    .output_info = io_info,
    .run = run,
    .session = &kSession,
};

UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return &kApi;
}
