/* SPDX-License-Identifier: Apache-2.0 */
#ifndef UAIRT_UAIRT_H
#define UAIRT_UAIRT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(UAIRT_BUILDING_SHARED)
#define UAIRT_API __declspec(dllexport)
#elif defined(UAIRT_SHARED)
#define UAIRT_API __declspec(dllimport)
#else
#define UAIRT_API
#endif
#elif defined(__GNUC__)
#define UAIRT_API __attribute__((visibility("default")))
#else
#define UAIRT_API
#endif

#define UAIRT_VERSION_MAJOR 0
#define UAIRT_VERSION_MINOR 1
#define UAIRT_VERSION_PATCH 0

#define UAIRT_STRINGIFY_(x) #x
#define UAIRT_STRINGIFY(x) UAIRT_STRINGIFY_(x)
#define UAIRT_VERSION_STRING                    \
  UAIRT_STRINGIFY(UAIRT_VERSION_MAJOR)           \
  "." UAIRT_STRINGIFY(UAIRT_VERSION_MINOR) "." UAIRT_STRINGIFY(UAIRT_VERSION_PATCH)

#define UAIRT_MAX_RANK 8

typedef int32_t uairt_status;
enum {
  UAIRT_OK = 0,
  UAIRT_ERR_INVALID_ARGUMENT = 1,
  UAIRT_ERR_NOT_FOUND = 2,
  UAIRT_ERR_UNSUPPORTED = 3,
  UAIRT_ERR_BACKEND_UNAVAILABLE = 4,
  UAIRT_ERR_INCOMPATIBLE_MODEL = 5,
  UAIRT_ERR_OUT_OF_MEMORY = 6,
  UAIRT_ERR_RUNTIME = 7,
  UAIRT_ERR_VERSION_MISMATCH = 8,
  UAIRT_ERR_IO = 9,
};

typedef int32_t uairt_dtype;
enum {
  UAIRT_DTYPE_UNKNOWN = 0,
  UAIRT_DTYPE_FLOAT32 = 1,
  UAIRT_DTYPE_FLOAT16 = 2,
  UAIRT_DTYPE_BFLOAT16 = 3,
  UAIRT_DTYPE_INT8 = 4,
  UAIRT_DTYPE_UINT8 = 5,
  UAIRT_DTYPE_INT16 = 6,
  UAIRT_DTYPE_INT32 = 7,
  UAIRT_DTYPE_INT64 = 8,
  UAIRT_DTYPE_BOOL = 9,
  UAIRT_DTYPE_UINT16 = 10,
};

/* Memory domains are bit flags so a backend can advertise a set of them. */
typedef uint32_t uairt_memory_domain;
enum {
  UAIRT_MEM_HOST = 1u << 0,
  UAIRT_MEM_DMABUF = 1u << 1,
  /* Page-locked host memory from uairt_buffer_alloc; usable like host memory and DMA-able by the device. */
  UAIRT_MEM_PINNED = 1u << 2,
};

typedef struct uairt_engine uairt_engine;
typedef struct uairt_model uairt_model;
typedef struct uairt_buffer uairt_buffer;

typedef struct uairt_option {
  const char* key;
  const char* value;
} uairt_option;

/* Exactly one of `path` or `data` must be set. `format` is an optional hint. */
typedef struct uairt_model_source {
  uint32_t struct_size;
  const char* path;
  const void* data;
  size_t size;
  const char* format;
} uairt_model_source;

/* Used both to describe a model I/O (data == NULL) and to bind a buffer. */
typedef struct uairt_tensor {
  uint32_t struct_size;
  uairt_dtype dtype;
  uint32_t rank;
  int64_t dims[UAIRT_MAX_RANK];
  float quant_scale;
  int32_t quant_zero_point;
  uairt_memory_domain domain;
  void* data;
  int32_t dmabuf_fd;
  size_t nbytes;
  /* Set by uairt_model_input_info/output_info; owned by the model. */
  const char* name;
} uairt_tensor;

UAIRT_API const char* uairt_version_string(void);
UAIRT_API const char* uairt_status_string(uairt_status status);
UAIRT_API const char* uairt_last_error(void);

UAIRT_API size_t uairt_dtype_size(uairt_dtype dtype);
UAIRT_API void uairt_tensor_init(uairt_tensor* tensor);
UAIRT_API int64_t uairt_tensor_num_elements(const uairt_tensor* tensor);

UAIRT_API size_t uairt_backend_count(void);
UAIRT_API const char* uairt_backend_name(size_t index);
UAIRT_API uairt_status uairt_load_backend_library(const char* path);

UAIRT_API uairt_status uairt_engine_create(
    const char* backend,
    const uairt_option* options,
    size_t num_options,
    uairt_engine** out_engine);

/* All models created from an engine must be destroyed before the engine. */
UAIRT_API void uairt_engine_destroy(uairt_engine* engine);

UAIRT_API uairt_status uairt_model_load(
    uairt_engine* engine,
    const uairt_model_source* source,
    uairt_model** out_model);
UAIRT_API void uairt_model_destroy(uairt_model* model);

UAIRT_API size_t uairt_model_num_inputs(const uairt_model* model);
UAIRT_API size_t uairt_model_num_outputs(const uairt_model* model);
UAIRT_API uairt_status
uairt_model_input_info(const uairt_model* model, size_t index, uairt_tensor* out);
UAIRT_API uairt_status
uairt_model_output_info(const uairt_model* model, size_t index, uairt_tensor* out);

/*
 * Buffers that a backend can use without copying. HOST buffers are plain memory;
 * DMABUF buffers (FastRPC rpcmem on Qualcomm) need backend support, otherwise
 * allocation returns UAIRT_ERR_UNSUPPORTED. Free every buffer after the models that
 * used it and before the engine that allocated it.
 */
UAIRT_API uairt_status uairt_buffer_alloc(
    uairt_engine* engine,
    size_t nbytes,
    uairt_memory_domain domain,
    uairt_buffer** out_buffer);
UAIRT_API void uairt_buffer_free(uairt_buffer* buffer);
UAIRT_API void* uairt_buffer_data(const uairt_buffer* buffer);
UAIRT_API int32_t uairt_buffer_fd(const uairt_buffer* buffer);
UAIRT_API size_t uairt_buffer_size(const uairt_buffer* buffer);
/* Points `tensor` at `buffer` (domain, data, fd and nbytes). */
UAIRT_API void uairt_tensor_use_buffer(uairt_tensor* tensor, const uairt_buffer* buffer);

/* Outputs are caller-allocated; `nbytes` must cover the model's output size. */
UAIRT_API uairt_status uairt_model_run(
    uairt_model* model,
    const uairt_tensor* inputs,
    size_t num_inputs,
    uairt_tensor* outputs,
    size_t num_outputs);

/*
 * Language models: the vocabulary calls and sessions work on models whose backend supports them (llamacpp) and return
 * UAIRT_ERR_UNSUPPORTED otherwise. Where a call fills a buffer, a `capacity` that is too small returns
 * UAIRT_ERR_INVALID_ARGUMENT and still sets the out count to the size needed, so capacity 0 sizes a buffer.
 * One token can be part of a multi-byte character: detokenize returns bytes, and the caller joins them.
 * Backends may parse special-token text such as `<|eot_id|>` into control tokens when tokenizing (llamacpp does); do
 * not tokenize untrusted text if that matters.
 */
UAIRT_API uairt_status uairt_model_vocab_size(const uairt_model* model, size_t* out_size);
UAIRT_API uairt_status uairt_model_tokenize(
    const uairt_model* model,
    const char* text,
    size_t text_len,
    int add_special,
    int32_t* tokens,
    size_t capacity,
    size_t* out_count);
/* Writes no terminating NUL. */
UAIRT_API uairt_status uairt_model_detokenize(
    const uairt_model* model,
    const int32_t* tokens,
    size_t count,
    char* out,
    size_t capacity,
    size_t* out_nbytes);

/*
 * A session keeps the model's state (the KV cache) between calls. A model can have several sessions, each independent.
 * Do not use one session from two threads at once; different sessions are independent and may be used from different
 * threads. Destroy sessions before their model. Option: `n_ctx`, the tokens of context.
 */
typedef struct uairt_session uairt_session;

UAIRT_API uairt_status uairt_session_create(
    uairt_model* model,
    const uairt_option* options,
    size_t num_options,
    uairt_session** out_session);
UAIRT_API void uairt_session_destroy(uairt_session* session);
/* Feeds tokens at the current position. A call that would pass the context size fails and changes nothing. */
UAIRT_API uairt_status uairt_session_append(uairt_session* session, const int32_t* tokens, size_t count);
/* Logits of the last appended token (one per vocabulary entry). Fails before the first append and after a reset. */
UAIRT_API uairt_status
uairt_session_logits(const uairt_session* session, float* out, size_t capacity, size_t* out_count);
UAIRT_API uairt_status uairt_session_position(const uairt_session* session, size_t* out_position);
UAIRT_API uairt_status uairt_session_reset(uairt_session* session);

#ifdef __cplusplus
}
#endif

#endif
