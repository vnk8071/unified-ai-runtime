/* SPDX-License-Identifier: Apache-2.0 */
#ifndef UAIRT_UAIRT_BACKEND_H
#define UAIRT_UAIRT_BACKEND_H

#include <uairt/uairt.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UAIRT_BACKEND_ABI_VERSION 1
#define UAIRT_BACKEND_ENTRY_SYMBOL "uairt_backend_get_api"

/*
 * The name a backend source gives its entry function. A plugin keeps the default, so the loader finds
 * UAIRT_BACKEND_ENTRY_SYMBOL. A backend linked into an application is compiled with
 * -DUAIRT_BACKEND_ENTRY=uairt_backend_get_api_<name> (see uairt/uairt_static.h), because every plugin would otherwise
 * define the same symbol.
 */
#ifndef UAIRT_BACKEND_ENTRY
#define UAIRT_BACKEND_ENTRY uairt_backend_get_api
#endif

/*
 * Optional, POSIX only. A plugin that exports this symbol (any non-NULL data symbol) is opened with RTLD_GLOBAL, so
 * libraries it loads later (for example a compiled library inside a model) can resolve symbols from the plugin. Leave it
 * out unless needed: global symbols can clash with those of other plugins. Ignored on Windows.
 */
#define UAIRT_BACKEND_GLOBAL_SYMBOLS_SYMBOL "uairt_backend_global_symbols"

typedef struct uairt_host_api {
  uint32_t struct_size;
  uint32_t abi_version;
  void (*set_error)(const char* message);
} uairt_host_api;

#define UAIRT_SESSION_ABI_VERSION 1

/*
 * Optional stateful sessions for language models. Handles are the backend's own: `model` is what load_model
 * returned, `session` what create_session returned. A too-small `capacity` returns UAIRT_ERR_INVALID_ARGUMENT after
 * setting the out count to the size needed.
 */
typedef struct uairt_session_api {
  uint32_t struct_size;
  uint32_t abi_version;
  uairt_status (*vocab_size)(const void* model, size_t* out_size);
  uairt_status (*tokenize)(
      const void* model,
      const char* text,
      size_t text_len,
      int add_special,
      int32_t* tokens,
      size_t capacity,
      size_t* out_count);
  uairt_status (*detokenize)(
      const void* model,
      const int32_t* tokens,
      size_t count,
      char* out,
      size_t capacity,
      size_t* out_nbytes);
  uairt_status (*create_session)(
      void* model,
      const uairt_option* options,
      size_t num_options,
      void** out_session);
  void (*destroy_session)(void* session);
  uairt_status (*append)(void* session, const int32_t* tokens, size_t count);
  uairt_status (*logits)(const void* session, float* out, size_t capacity, size_t* out_count);
  uairt_status (*position)(const void* session, size_t* out_position);
  uairt_status (*reset)(void* session);
  /*
   * Optional, appended after `reset`: read only when struct_size covers it. Returns non-zero when `token` ends a
   * generation (end of text, end of turn). A token outside the vocabulary is not one. NULL: the backend cannot tell.
   */
  int (*is_end_of_generation)(const void* model, int32_t token);
} uairt_session_api;

#define UAIRT_TEXT_ABI_VERSION 1

/*
 * Optional text completion for backends that take text and return text (language model servers). `prompt` is NUL
 * terminated. The backend calls `callback` once per piece of generated text, in order, and stops when it returns
 * non-zero; stopping is not an error. The backend must not retain `prompt`, `options` or the callback.
 */
typedef struct uairt_text_api {
  uint32_t struct_size;
  uint32_t abi_version;
  uairt_status (*complete)(
      void* model,
      const char* prompt,
      const uairt_option* options,
      size_t num_options,
      uairt_text_callback callback,
      void* user_data);
} uairt_text_api;

/*
 * Handles are opaque to the host. Backends must not throw across this boundary
 * and must not retain pointers from `uairt_tensor` or `uairt_model_source` after
 * the call returns. The host validates argument counts, dtypes, shapes, buffer
 * sizes and memory domains before calling `run`.
 */
typedef struct uairt_backend_api {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* name;
  uairt_memory_domain supported_domains;
  uairt_status (*create_engine)(
      const uairt_option* options,
      size_t num_options,
      void** out_engine);
  void (*destroy_engine)(void* engine);
  uairt_status (*load_model)(
      void* engine,
      const uairt_model_source* source,
      void** out_model);
  void (*destroy_model)(void* model);
  size_t (*num_inputs)(const void* model);
  size_t (*num_outputs)(const void* model);
  uairt_status (*input_info)(const void* model, size_t index, uairt_tensor* out);
  uairt_status (*output_info)(const void* model, size_t index, uairt_tensor* out);
  uairt_status (*run)(
      void* model,
      const uairt_tensor* inputs,
      size_t num_inputs,
      uairt_tensor* outputs,
      size_t num_outputs);
  /* Optional: allocate zero-copy buffers (DMABUF). Both must be set, or neither. */
  uairt_status (*alloc_buffer)(
      void* engine,
      size_t nbytes,
      uairt_memory_domain domain,
      void** out_handle,
      void** out_data,
      int32_t* out_fd);
  void (*free_buffer)(void* engine, void* handle);
  /*
   * Optional. Present only when struct_size covers it: a plugin built before this field existed has a shorter struct,
   * and the host never reads past struct_size. NULL means the backend has no sessions.
   */
  const uairt_session_api* session;
  /* Optional, appended after `session` under the same rule: read only when struct_size covers it. NULL: no text API. */
  const uairt_text_api* text;
} uairt_backend_api;

typedef const uairt_backend_api* (*uairt_backend_get_api_fn)(
    const uairt_host_api* host);

/* Static registration; dynamic plugins export UAIRT_BACKEND_ENTRY_SYMBOL. */
UAIRT_API uairt_status uairt_register_backend(uairt_backend_get_api_fn get_api);

#ifdef __cplusplus
}
#endif

#endif
