/* SPDX-License-Identifier: Apache-2.0 */
#ifndef UAIRT_UAIRT_BACKEND_H
#define UAIRT_UAIRT_BACKEND_H

#include <uairt/uairt.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UAIRT_BACKEND_ABI_VERSION 1
#define UAIRT_BACKEND_ENTRY_SYMBOL "uairt_backend_get_api"

typedef struct uairt_host_api {
  uint32_t struct_size;
  uint32_t abi_version;
  void (*set_error)(const char* message);
} uairt_host_api;

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
} uairt_backend_api;

typedef const uairt_backend_api* (*uairt_backend_get_api_fn)(
    const uairt_host_api* host);

/* Static registration; dynamic plugins export UAIRT_BACKEND_ENTRY_SYMBOL. */
UAIRT_API uairt_status uairt_register_backend(uairt_backend_get_api_fn get_api);

#ifdef __cplusplus
}
#endif

#endif
