/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Test-only plugin for the session API, built three times (SESSION_TEST_VARIANT):
 *   1 "fake-session": a complete session table over a byte tokenizer.
 *   2 "fake-old": the backend struct as it was before the `session` field (struct_size ends at free_buffer).
 *   3 "fake-bad": a session table whose `reset` is missing.
 * Tokens are bytes (0..255) plus token 256 as BOS. After appending token t the logits favour t + 1, so greedy
 * generation counts upward.
 */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt_backend.h>

#define VOCAB 257
#define BOS 256
#define DEFAULT_CTX 8

#ifndef SESSION_TEST_VARIANT
#define SESSION_TEST_VARIANT 1
#endif

static const uairt_host_api* g_host;

static void fail(const char* message) {
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

typedef struct {
  size_t n_ctx;
  size_t pos;
  int32_t last;
} session_t;

static uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  (void)options;
  (void)num_options;
  *out_engine = malloc(1);
  return *out_engine ? UAIRT_OK : UAIRT_ERR_OUT_OF_MEMORY;
}

static void destroy_engine(void* engine) {
  free(engine);
}

static uairt_status load_model(void* engine, const uairt_model_source* source, void** out_model) {
  (void)engine;
  (void)source;
  *out_model = malloc(1);
  return *out_model ? UAIRT_OK : UAIRT_ERR_OUT_OF_MEMORY;
}

static void destroy_model(void* model) {
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
  fail("the model has no tensors");
  return UAIRT_ERR_INVALID_ARGUMENT;
}

static uairt_status run(void* model, const uairt_tensor* in, size_t n_in, uairt_tensor* out, size_t n_out) {
  (void)model;
  (void)in;
  (void)n_in;
  (void)out;
  (void)n_out;
  fail("use a session");
  return UAIRT_ERR_UNSUPPORTED;
}

#if SESSION_TEST_VARIANT != 2
static uairt_status vocab_size(const void* model, size_t* out_size) {
  (void)model;
  *out_size = VOCAB;
  return UAIRT_OK;
}

static uairt_status tokenize(const void* model, const char* text, size_t text_len, int add_special, int32_t* tokens,
                             size_t capacity, size_t* out_count) {
  (void)model;
  size_t needed = text_len + (add_special ? 1 : 0);
  *out_count = needed;
  if (capacity < needed) {
    fail("token buffer too small");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  size_t n = 0;
  if (add_special) {
    tokens[n++] = BOS;
  }
  for (size_t i = 0; i < text_len; ++i) {
    tokens[n++] = (unsigned char)text[i];
  }
  return UAIRT_OK;
}

static uairt_status detokenize(const void* model, const int32_t* tokens, size_t count, char* out, size_t capacity,
                               size_t* out_nbytes) {
  (void)model;
  size_t needed = 0;
  for (size_t i = 0; i < count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= VOCAB) {
      fail("token id out of range");
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    needed += tokens[i] < 256 ? 1 : 0;
  }
  *out_nbytes = needed;
  if (capacity < needed) {
    fail("text buffer too small");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  size_t n = 0;
  for (size_t i = 0; i < count; ++i) {
    if (tokens[i] < 256) {
      out[n++] = (char)tokens[i];
    }
  }
  return UAIRT_OK;
}

static uairt_status create_session(void* model, const uairt_option* options, size_t num_options, void** out) {
  (void)model;
  session_t* session = calloc(1, sizeof(*session));
  if (!session) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  session->n_ctx = DEFAULT_CTX;
  for (size_t i = 0; i < num_options; ++i) {
    if (strcmp(options[i].key, "n_ctx") != 0) {
      fail("unknown session option");
      free(session);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    session->n_ctx = (size_t)atoi(options[i].value);
  }
  *out = session;
  return UAIRT_OK;
}

static void destroy_session(void* session) {
  free(session);
}

static uairt_status append(void* handle, const int32_t* tokens, size_t count) {
  session_t* session = handle;
  if (session->pos + count > session->n_ctx) {
    fail("context full");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= VOCAB) {
      fail("token id out of range");
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  for (size_t i = 0; i < count; ++i) {
    session->last = tokens[i];
  }
  session->pos += count;
  return UAIRT_OK;
}

static uairt_status logits(const void* handle, float* out, size_t capacity, size_t* out_count) {
  const session_t* session = handle;
  *out_count = VOCAB;
  if (session->pos == 0) {
    fail("no tokens appended");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (capacity < VOCAB) {
    fail("logits buffer too small");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  memset(out, 0, VOCAB * sizeof(float));
  out[(session->last + 1) % VOCAB] = 1.0f;
  return UAIRT_OK;
}

static uairt_status position(const void* handle, size_t* out_position) {
  *out_position = ((const session_t*)handle)->pos;
  return UAIRT_OK;
}

#if SESSION_TEST_VARIANT != 3
static uairt_status reset(void* handle) {
  session_t* session = handle;
  session->pos = 0;
  session->last = 0;
  return UAIRT_OK;
}
#endif

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
#if SESSION_TEST_VARIANT != 3
    .reset = reset,
#endif
};
#endif

#if SESSION_TEST_VARIANT == 2
#define API_NAME "fake-old"
/* The v1 layout: everything up to and including free_buffer, with no `session`. */
typedef struct {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* name;
  uairt_memory_domain supported_domains;
  uairt_status (*create_engine)(const uairt_option*, size_t, void**);
  void (*destroy_engine)(void*);
  uairt_status (*load_model)(void*, const uairt_model_source*, void**);
  void (*destroy_model)(void*);
  size_t (*num_inputs)(const void*);
  size_t (*num_outputs)(const void*);
  uairt_status (*input_info)(const void*, size_t, uairt_tensor*);
  uairt_status (*output_info)(const void*, size_t, uairt_tensor*);
  uairt_status (*run)(void*, const uairt_tensor*, size_t, uairt_tensor*, size_t);
  uairt_status (*alloc_buffer)(void*, size_t, uairt_memory_domain, void**, void**, int32_t*);
  void (*free_buffer)(void*, void*);
} old_api_t;

_Static_assert(sizeof(old_api_t) == offsetof(uairt_backend_api, session), "old_api_t must be the v1 layout");

static const old_api_t kApi = {
    .struct_size = sizeof(old_api_t),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = API_NAME,
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
};
#else
#if SESSION_TEST_VARIANT == 3
#define API_NAME "fake-bad"
#else
#define API_NAME "fake-session"
#endif
static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = API_NAME,
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
#endif

UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return (const uairt_backend_api*)&kApi;
}
