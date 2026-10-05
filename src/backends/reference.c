/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Test-only backend: one float32 [1, 4] input copied to one float32 [1, 4]
 * output. Compiled twice: built in, and as a loadable plugin that proves the
 * ABI works without linking against the core library.
 */
#include <stdlib.h>
#include <string.h>

#include "../internal.h"

#ifdef UAIRT_REFERENCE_AS_PLUGIN
#define REFERENCE_ENTRY uairt_backend_get_api
#define REFERENCE_NAME "reference-plugin"
#else
#define REFERENCE_ENTRY uairt_reference_backend_get_api
#define REFERENCE_NAME "reference"
#endif

#define REFERENCE_ELEMENTS 4

static const uairt_host_api* g_host;

static void fail(const char* message) {
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

static uairt_status create_engine(
    const uairt_option* options,
    size_t num_options,
    void** out_engine) {
  (void)options;
  (void)num_options;
  *out_engine = malloc(1);
  return *out_engine ? UAIRT_OK : UAIRT_ERR_OUT_OF_MEMORY;
}

static void destroy_engine(void* engine) {
  free(engine);
}

static uairt_status
load_model(void* engine, const uairt_model_source* source, void** out_model) {
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
  return 1;
}

static uairt_status describe(const void* model, size_t index, uairt_tensor* out) {
  (void)model;
  if (index != 0) {
    fail("reference backend has a single input and output");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  out->dtype = UAIRT_DTYPE_FLOAT32;
  out->rank = 2;
  out->dims[0] = 1;
  out->dims[1] = REFERENCE_ELEMENTS;
  return UAIRT_OK;
}

static uairt_status run(
    void* model,
    const uairt_tensor* inputs,
    size_t num_inputs,
    uairt_tensor* outputs,
    size_t num_outputs) {
  (void)model;
  (void)num_inputs;
  (void)num_outputs;
  memcpy(outputs[0].data, inputs[0].data, REFERENCE_ELEMENTS * sizeof(float));
  return UAIRT_OK;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = REFERENCE_NAME,
    .supported_domains = UAIRT_MEM_HOST,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_io,
    .num_outputs = num_io,
    .input_info = describe,
    .output_info = describe,
    .run = run,
};

UAIRT_API const uairt_backend_api* REFERENCE_ENTRY(const uairt_host_api* host) {
  g_host = host;
  return &kApi;
}
