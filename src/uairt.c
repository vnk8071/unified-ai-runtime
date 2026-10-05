/* SPDX-License-Identifier: Apache-2.0 */
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#if !defined(_WIN32)
#include <pthread.h>
#define LOCK() pthread_mutex_lock(&g_lock)
#define UNLOCK() pthread_mutex_unlock(&g_lock)
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LOCK() AcquireSRWLockExclusive(&g_lock)
#define UNLOCK() ReleaseSRWLockExclusive(&g_lock)
static SRWLOCK g_lock = SRWLOCK_INIT;
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;
#endif

#define MAX_BACKENDS 16
#define ERROR_CAPACITY 256

struct uairt_engine {
  const uairt_backend_api* api;
  void* handle;
};

struct uairt_model {
  const uairt_backend_api* api;
  void* handle;
  size_t num_inputs;
  size_t num_outputs;
};

struct uairt_buffer {
  uairt_engine* engine;
  void* backend_handle;
  void* data;
  int32_t fd;
  size_t nbytes;
  uairt_memory_domain domain;
};

static _Thread_local char g_error[ERROR_CAPACITY];
static const uairt_backend_api* g_backends[MAX_BACKENDS];
static size_t g_backend_count;

void uairt_set_error(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(g_error, ERROR_CAPACITY, fmt, args);
  va_end(args);
}

static void host_set_error(const char* message) {
  snprintf(g_error, ERROR_CAPACITY, "%s", message ? message : "");
}

static const uairt_host_api g_host = {
    .struct_size = sizeof(uairt_host_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .set_error = host_set_error,
};

const char* uairt_version_string(void) {
  return UAIRT_VERSION_STRING;
}

const char* uairt_last_error(void) {
  return g_error;
}

const char* uairt_status_string(uairt_status status) {
  switch (status) {
    case UAIRT_OK:
      return "ok";
    case UAIRT_ERR_INVALID_ARGUMENT:
      return "invalid argument";
    case UAIRT_ERR_NOT_FOUND:
      return "not found";
    case UAIRT_ERR_UNSUPPORTED:
      return "unsupported";
    case UAIRT_ERR_BACKEND_UNAVAILABLE:
      return "backend unavailable";
    case UAIRT_ERR_INCOMPATIBLE_MODEL:
      return "incompatible model";
    case UAIRT_ERR_OUT_OF_MEMORY:
      return "out of memory";
    case UAIRT_ERR_RUNTIME:
      return "runtime error";
    case UAIRT_ERR_VERSION_MISMATCH:
      return "version mismatch";
    case UAIRT_ERR_IO:
      return "io error";
    default:
      return "unknown status";
  }
}

size_t uairt_dtype_size(uairt_dtype dtype) {
  switch (dtype) {
    case UAIRT_DTYPE_FLOAT32:
    case UAIRT_DTYPE_INT32:
      return 4;
    case UAIRT_DTYPE_FLOAT16:
    case UAIRT_DTYPE_BFLOAT16:
    case UAIRT_DTYPE_INT16:
    case UAIRT_DTYPE_UINT16:
      return 2;
    case UAIRT_DTYPE_INT8:
    case UAIRT_DTYPE_UINT8:
    case UAIRT_DTYPE_BOOL:
      return 1;
    case UAIRT_DTYPE_INT64:
      return 8;
    default:
      return 0;
  }
}

void uairt_tensor_init(uairt_tensor* tensor) {
  memset(tensor, 0, sizeof(*tensor));
  tensor->struct_size = sizeof(*tensor);
  tensor->domain = UAIRT_MEM_HOST;
  tensor->dmabuf_fd = -1;
}

int64_t uairt_tensor_num_elements(const uairt_tensor* tensor) {
  if (tensor->rank > UAIRT_MAX_RANK) {
    return -1;
  }
  int64_t count = 1;
  for (uint32_t i = 0; i < tensor->rank; ++i) {
    int64_t dim = tensor->dims[i];
    if (dim < 0 || (dim != 0 && count > INT64_MAX / dim)) {
      return -1;
    }
    count *= dim;
  }
  return count;
}

/* A plugin built before the `session` field has a shorter struct, so the field may only be read when it fits. */
static const uairt_session_api* session_of(const uairt_backend_api* api) {
  return api->struct_size >= offsetof(uairt_backend_api, session) + sizeof(api->session) ? api->session : NULL;
}

static const char* missing_session_entry(const uairt_session_api* sessions) {
  if (sessions->struct_size < sizeof(*sessions)) {
    return "struct_size";
  }
  if (sessions->abi_version != UAIRT_SESSION_ABI_VERSION) {
    return "abi_version";
  }
  if (!sessions->vocab_size) {
    return "vocab_size";
  }
  if (!sessions->tokenize) {
    return "tokenize";
  }
  if (!sessions->detokenize) {
    return "detokenize";
  }
  if (!sessions->create_session) {
    return "create_session";
  }
  if (!sessions->destroy_session) {
    return "destroy_session";
  }
  if (!sessions->append) {
    return "append";
  }
  if (!sessions->logits) {
    return "logits";
  }
  if (!sessions->position) {
    return "position";
  }
  if (!sessions->reset) {
    return "reset";
  }
  return NULL;
}

static void ensure_builtin_backends(void);

static uairt_status register_locked(uairt_backend_get_api_fn get_api) {
  g_error[0] = '\0';
  const uairt_backend_api* api = get_api(&g_host);
  if (!api) {
    if (!g_error[0]) {
      uairt_set_error("backend entry point returned no API");
    }
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  if (api->abi_version != UAIRT_BACKEND_ABI_VERSION) {
    uairt_set_error(
        "backend ABI version %u, host expects %u",
        api->abi_version,
        UAIRT_BACKEND_ABI_VERSION);
    return UAIRT_ERR_VERSION_MISMATCH;
  }
  if (api->struct_size < offsetof(uairt_backend_api, session) || !api->name ||
      !api->name[0] || !api->supported_domains || !api->create_engine ||
      !api->destroy_engine || !api->load_model || !api->destroy_model ||
      !api->num_inputs || !api->num_outputs || !api->input_info ||
      !api->output_info || !api->run || !api->alloc_buffer != !api->free_buffer) {
    uairt_set_error("backend API is incomplete");
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  const uairt_session_api* sessions = session_of(api);
  if (sessions) {
    const char* missing = missing_session_entry(sessions);
    if (missing) {
      uairt_set_error("backend '%s': session API is incomplete (%s)", api->name, missing);
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
  }
  for (size_t i = 0; i < g_backend_count; ++i) {
    if (strcmp(g_backends[i]->name, api->name) == 0) {
      uairt_set_error("backend '%s' is already registered", api->name);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  if (g_backend_count == MAX_BACKENDS) {
    uairt_set_error("too many backends registered");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  g_backends[g_backend_count++] = api;
  return UAIRT_OK;
}

static void register_builtin_backends(void) {
#ifdef UAIRT_WITH_REFERENCE_BACKEND
  (void)register_locked(uairt_reference_backend_get_api);
#endif
}

#if defined(_WIN32)
static BOOL CALLBACK register_builtin_backends_once(PINIT_ONCE once, PVOID param, PVOID* context) {
  (void)once;
  (void)param;
  (void)context;
  register_builtin_backends();
  return TRUE;
}
#endif

static void ensure_builtin_backends(void) {
#if !defined(_WIN32)
  pthread_once(&g_once, register_builtin_backends);
#else
  InitOnceExecuteOnce(&g_once, register_builtin_backends_once, NULL, NULL);
#endif
}

uairt_status uairt_register_backend(uairt_backend_get_api_fn get_api) {
  if (!get_api) {
    uairt_set_error("get_api is null");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  ensure_builtin_backends();
  LOCK();
  uairt_status status = register_locked(get_api);
  UNLOCK();
  return status;
}

uairt_status uairt_load_backend_library(const char* path) {
  if (!path) {
    uairt_set_error("path is null");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  uairt_backend_get_api_fn get_api = NULL;
  uairt_status status = uairt_loader_open(path, &get_api);
  if (status != UAIRT_OK) {
    return status;
  }
  return uairt_register_backend(get_api);
}

size_t uairt_backend_count(void) {
  ensure_builtin_backends();
  LOCK();
  size_t count = g_backend_count;
  UNLOCK();
  return count;
}

const char* uairt_backend_name(size_t index) {
  ensure_builtin_backends();
  LOCK();
  const char* name = index < g_backend_count ? g_backends[index]->name : NULL;
  UNLOCK();
  return name;
}

static const uairt_backend_api* find_backend(const char* name) {
  const uairt_backend_api* found = NULL;
  LOCK();
  for (size_t i = 0; i < g_backend_count; ++i) {
    if (strcmp(g_backends[i]->name, name) == 0) {
      found = g_backends[i];
      break;
    }
  }
  UNLOCK();
  return found;
}

uairt_status uairt_engine_create(
    const char* backend,
    const uairt_option* options,
    size_t num_options,
    uairt_engine** out_engine) {
  if (!backend || !out_engine || (num_options && !options)) {
    uairt_set_error("invalid argument to uairt_engine_create");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_engine = NULL;
  ensure_builtin_backends();
  const uairt_backend_api* api = find_backend(backend);
  if (!api) {
    uairt_set_error("backend '%s' is not registered", backend);
    return UAIRT_ERR_NOT_FOUND;
  }
  uairt_engine* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    uairt_set_error("out of memory");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status = api->create_engine(options, num_options, &engine->handle);
  if (status != UAIRT_OK) {
    free(engine);
    return status;
  }
  engine->api = api;
  *out_engine = engine;
  return UAIRT_OK;
}

void uairt_engine_destroy(uairt_engine* engine) {
  if (!engine) {
    return;
  }
  engine->api->destroy_engine(engine->handle);
  free(engine);
}

uairt_status uairt_model_load(
    uairt_engine* engine,
    const uairt_model_source* source,
    uairt_model** out_model) {
  if (!engine || !source || !out_model ||
      source->struct_size < sizeof(*source)) {
    uairt_set_error("invalid argument to uairt_model_load");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_model = NULL;
  bool has_path = source->path != NULL;
  bool has_data = source->data != NULL && source->size > 0;
  if (has_path == has_data) {
    uairt_set_error("model source needs exactly one of path or data");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  uairt_model* model = calloc(1, sizeof(*model));
  if (!model) {
    uairt_set_error("out of memory");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status =
      engine->api->load_model(engine->handle, source, &model->handle);
  if (status != UAIRT_OK) {
    free(model);
    return status;
  }
  model->api = engine->api;
  model->num_inputs = model->api->num_inputs(model->handle);
  model->num_outputs = model->api->num_outputs(model->handle);
  *out_model = model;
  return UAIRT_OK;
}

void uairt_model_destroy(uairt_model* model) {
  if (!model) {
    return;
  }
  model->api->destroy_model(model->handle);
  free(model);
}

size_t uairt_model_num_inputs(const uairt_model* model) {
  return model ? model->num_inputs : 0;
}

size_t uairt_model_num_outputs(const uairt_model* model) {
  return model ? model->num_outputs : 0;
}

uairt_status
uairt_model_input_info(const uairt_model* model, size_t index, uairt_tensor* out) {
  if (!model || !out || index >= model->num_inputs) {
    uairt_set_error("invalid argument to uairt_model_input_info");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  uairt_tensor_init(out);
  return model->api->input_info(model->handle, index, out);
}

uairt_status uairt_model_output_info(
    const uairt_model* model,
    size_t index,
    uairt_tensor* out) {
  if (!model || !out || index >= model->num_outputs) {
    uairt_set_error("invalid argument to uairt_model_output_info");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  uairt_tensor_init(out);
  return model->api->output_info(model->handle, index, out);
}

static uairt_status check_binding(
    const uairt_backend_api* api,
    const uairt_tensor* expected,
    const uairt_tensor* actual,
    const char* kind,
    size_t index) {
  if (actual->struct_size < sizeof(*actual)) {
    uairt_set_error("%s %zu: struct_size too small", kind, index);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (actual->dtype != expected->dtype || actual->rank != expected->rank) {
    uairt_set_error("%s %zu: dtype or rank mismatch", kind, index);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  for (uint32_t i = 0; i < expected->rank; ++i) {
    if (actual->dims[i] != expected->dims[i]) {
      uairt_set_error("%s %zu: shape mismatch at dim %u", kind, index, i);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  if (actual->domain != UAIRT_MEM_HOST && actual->domain != UAIRT_MEM_DMABUF &&
      actual->domain != UAIRT_MEM_PINNED) {
    uairt_set_error("%s %zu: unknown memory domain", kind, index);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (!(api->supported_domains & actual->domain)) {
    uairt_set_error(
        "%s %zu: backend '%s' does not support memory domain %u",
        kind,
        index,
        api->name,
        actual->domain);
    return UAIRT_ERR_UNSUPPORTED;
  }
  if (((actual->domain == UAIRT_MEM_HOST || actual->domain == UAIRT_MEM_PINNED) && !actual->data) ||
      (actual->domain == UAIRT_MEM_DMABUF && actual->dmabuf_fd < 0)) {
    uairt_set_error("%s %zu: missing buffer", kind, index);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  int64_t elements = uairt_tensor_num_elements(expected);
  size_t needed = (size_t)elements * uairt_dtype_size(expected->dtype);
  if (elements < 0 || actual->nbytes < needed) {
    uairt_set_error("%s %zu: buffer smaller than %zu bytes", kind, index, needed);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  return UAIRT_OK;
}

uairt_status uairt_model_run(
    uairt_model* model,
    const uairt_tensor* inputs,
    size_t num_inputs,
    uairt_tensor* outputs,
    size_t num_outputs) {
  if (!model || (num_inputs && !inputs) || (num_outputs && !outputs)) {
    uairt_set_error("invalid argument to uairt_model_run");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (num_inputs != model->num_inputs || num_outputs != model->num_outputs) {
    uairt_set_error(
        "model takes %zu inputs and %zu outputs, got %zu and %zu",
        model->num_inputs,
        model->num_outputs,
        num_inputs,
        num_outputs);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  uairt_tensor expected;
  for (size_t i = 0; i < num_inputs; ++i) {
    uairt_tensor_init(&expected);
    uairt_status status = model->api->input_info(model->handle, i, &expected);
    if (status == UAIRT_OK) {
      status = check_binding(model->api, &expected, &inputs[i], "input", i);
    }
    if (status != UAIRT_OK) {
      return status;
    }
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    uairt_tensor_init(&expected);
    uairt_status status = model->api->output_info(model->handle, i, &expected);
    if (status == UAIRT_OK) {
      status = check_binding(model->api, &expected, &outputs[i], "output", i);
    }
    if (status != UAIRT_OK) {
      return status;
    }
  }
  return model->api->run(model->handle, inputs, num_inputs, outputs, num_outputs);
}

uairt_status uairt_buffer_alloc(
    uairt_engine* engine,
    size_t nbytes,
    uairt_memory_domain domain,
    uairt_buffer** out_buffer) {
  if (!engine || !out_buffer || nbytes == 0) {
    uairt_set_error("invalid argument to uairt_buffer_alloc");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_buffer = NULL;
  uairt_buffer* buffer = calloc(1, sizeof(*buffer));
  if (!buffer) {
    uairt_set_error("out of memory");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  buffer->engine = engine;
  buffer->nbytes = nbytes;
  buffer->domain = domain;
  buffer->fd = -1;
  uairt_status status = UAIRT_OK;
  if (domain == UAIRT_MEM_HOST) {
    buffer->data = malloc(nbytes);
    if (!buffer->data) {
      uairt_set_error("out of memory");
      status = UAIRT_ERR_OUT_OF_MEMORY;
    }
  } else if (domain == UAIRT_MEM_DMABUF && engine->api->alloc_buffer) {
    status = engine->api->alloc_buffer(
        engine->handle, nbytes, domain, &buffer->backend_handle, &buffer->data, &buffer->fd);
  } else if (domain == UAIRT_MEM_DMABUF) {
    uairt_set_error("backend '%s' cannot allocate DMABUF buffers", engine->api->name);
    status = UAIRT_ERR_UNSUPPORTED;
  } else if (domain == UAIRT_MEM_PINNED) {
    if ((engine->api->supported_domains & UAIRT_MEM_PINNED) && engine->api->alloc_buffer) {
      status = engine->api->alloc_buffer(
          engine->handle, nbytes, domain, &buffer->backend_handle, &buffer->data, &buffer->fd);
    } else {
      uairt_set_error("backend '%s' cannot allocate pinned buffers", engine->api->name);
      status = UAIRT_ERR_UNSUPPORTED;
    }
  } else {
    uairt_set_error("unknown memory domain %u", domain);
    status = UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (status != UAIRT_OK) {
    free(buffer);
    return status;
  }
  *out_buffer = buffer;
  return UAIRT_OK;
}

void uairt_buffer_free(uairt_buffer* buffer) {
  if (!buffer) {
    return;
  }
  if (buffer->domain == UAIRT_MEM_HOST) {
    free(buffer->data);
  } else {
    buffer->engine->api->free_buffer(buffer->engine->handle, buffer->backend_handle);
  }
  free(buffer);
}

void* uairt_buffer_data(const uairt_buffer* buffer) {
  return buffer ? buffer->data : NULL;
}

int32_t uairt_buffer_fd(const uairt_buffer* buffer) {
  return buffer ? buffer->fd : -1;
}

size_t uairt_buffer_size(const uairt_buffer* buffer) {
  return buffer ? buffer->nbytes : 0;
}

void uairt_tensor_use_buffer(uairt_tensor* tensor, const uairt_buffer* buffer) {
  tensor->domain = buffer->domain;
  tensor->data = buffer->data;
  tensor->dmabuf_fd = buffer->fd;
  tensor->nbytes = buffer->nbytes;
}

struct uairt_session {
  const uairt_session_api* sessions;
  void* handle;
};

static const uairt_session_api* require_sessions(const uairt_model* model, const char* call) {
  const uairt_session_api* sessions = session_of(model->api);
  if (!sessions) {
    uairt_set_error("backend '%s' has no session support (%s)", model->api->name, call);
  }
  return sessions;
}

uairt_status uairt_model_vocab_size(const uairt_model* model, size_t* out_size) {
  if (!model || !out_size) {
    uairt_set_error("invalid argument to uairt_model_vocab_size");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  const uairt_session_api* sessions = require_sessions(model, "vocab_size");
  return sessions ? sessions->vocab_size(model->handle, out_size) : UAIRT_ERR_UNSUPPORTED;
}

uairt_status uairt_model_tokenize(
    const uairt_model* model,
    const char* text,
    size_t text_len,
    int add_special,
    int32_t* tokens,
    size_t capacity,
    size_t* out_count) {
  if (!model || (text_len && !text) || (capacity && !tokens) || !out_count) {
    uairt_set_error("invalid argument to uairt_model_tokenize");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  const uairt_session_api* sessions = require_sessions(model, "tokenize");
  return sessions ? sessions->tokenize(model->handle, text ? text : "", text_len, add_special, tokens, capacity,
                                       out_count)
                  : UAIRT_ERR_UNSUPPORTED;
}

uairt_status uairt_model_detokenize(
    const uairt_model* model,
    const int32_t* tokens,
    size_t count,
    char* out,
    size_t capacity,
    size_t* out_nbytes) {
  if (!model || (count && !tokens) || (capacity && !out) || !out_nbytes) {
    uairt_set_error("invalid argument to uairt_model_detokenize");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  const uairt_session_api* sessions = require_sessions(model, "detokenize");
  return sessions ? sessions->detokenize(model->handle, tokens, count, out, capacity, out_nbytes)
                  : UAIRT_ERR_UNSUPPORTED;
}

uairt_status uairt_session_create(
    uairt_model* model,
    const uairt_option* options,
    size_t num_options,
    uairt_session** out_session) {
  if (!model || !out_session || (num_options && !options)) {
    uairt_set_error("invalid argument to uairt_session_create");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_session = NULL;
  const uairt_session_api* sessions = require_sessions(model, "create_session");
  if (!sessions) {
    return UAIRT_ERR_UNSUPPORTED;
  }
  uairt_session* session = calloc(1, sizeof(*session));
  if (!session) {
    uairt_set_error("out of memory");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status = sessions->create_session(model->handle, options, num_options, &session->handle);
  if (status != UAIRT_OK) {
    free(session);
    return status;
  }
  session->sessions = sessions;
  *out_session = session;
  return UAIRT_OK;
}

void uairt_session_destroy(uairt_session* session) {
  if (!session) {
    return;
  }
  session->sessions->destroy_session(session->handle);
  free(session);
}

uairt_status uairt_session_append(uairt_session* session, const int32_t* tokens, size_t count) {
  if (!session || (count && !tokens)) {
    uairt_set_error("invalid argument to uairt_session_append");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  return session->sessions->append(session->handle, tokens, count);
}

uairt_status uairt_session_logits(
    const uairt_session* session,
    float* out,
    size_t capacity,
    size_t* out_count) {
  if (!session || (capacity && !out) || !out_count) {
    uairt_set_error("invalid argument to uairt_session_logits");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  return session->sessions->logits(session->handle, out, capacity, out_count);
}

uairt_status uairt_session_position(const uairt_session* session, size_t* out_position) {
  if (!session || !out_position) {
    uairt_set_error("invalid argument to uairt_session_position");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  return session->sessions->position(session->handle, out_position);
}

uairt_status uairt_session_reset(uairt_session* session) {
  if (!session) {
    uairt_set_error("invalid argument to uairt_session_reset");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  return session->sessions->reset(session->handle);
}
