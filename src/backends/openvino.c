/* SPDX-License-Identifier: Apache-2.0 */
/*
 * OpenVINO backend, built as a loadable plugin.
 * v1 limits: OpenVINO IR (.xml with its .bin) or ONNX files given by path, or a directory that holds one .xml
 * file (an ultralytics `_openvino_model` export); static shapes; host memory. Any OpenVINO device works:
 * CPU (default), GPU (Intel integrated and Arc), NPU, AUTO. Inputs are read from, and outputs written to, the
 * caller's memory without a copy where the device allows it.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define strdup _strdup
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#include <openvino/c/openvino.h>
#include <uairt/uairt_backend.h>

#define OPTION_DEVICE "device"
#define OPTION_HINT "performance_hint"
#define OPTION_THREADS "num_threads"
#define OPTION_CACHE_DIR "cache_dir"

typedef struct {
  ov_core_t* core;
  char* device;
  const char* hint; /* "LATENCY" or "THROUGHPUT", or NULL */
  char* threads;
  char* cache_dir;
} engine_t;

typedef struct {
  char* name;
  uairt_tensor desc;
  ov_element_type_e type;
  size_t nbytes;
} io_t;

typedef struct {
  ov_compiled_model_t* compiled;
  ov_infer_request_t* request;
  io_t* inputs;
  size_t num_inputs;
  io_t* outputs;
  size_t num_outputs;
} model_t;

static const uairt_host_api* g_host;

#if defined(__GNUC__)
static void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

static void fail(const char* fmt, ...) {
  char message[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

/* Reports the last OpenVINO error and maps the status code. */
static uairt_status check(ov_status_e status, const char* what) {
  if (status == OK) {
    return UAIRT_OK;
  }
  const char* detail = ov_get_last_err_msg();
  fail("openvino: %s failed: %s", what, detail && detail[0] ? detail : ov_get_error_info(status));
  if (detail) {
    ov_free(detail);
  }
  switch (status) {
    case NOT_IMPLEMENTED:
      return UAIRT_ERR_UNSUPPORTED;
    case INVALID_C_PARAM:
      return UAIRT_ERR_INVALID_ARGUMENT;
    default:
      return UAIRT_ERR_RUNTIME;
  }
}

static bool map_dtype(ov_element_type_e type, uairt_dtype* dtype, size_t* size) {
  switch (type) {
    case F32: *dtype = UAIRT_DTYPE_FLOAT32; *size = 4; return true;
    case F16: *dtype = UAIRT_DTYPE_FLOAT16; *size = 2; return true;
    case BF16: *dtype = UAIRT_DTYPE_BFLOAT16; *size = 2; return true;
    case I8: *dtype = UAIRT_DTYPE_INT8; *size = 1; return true;
    case U8: *dtype = UAIRT_DTYPE_UINT8; *size = 1; return true;
    case I16: *dtype = UAIRT_DTYPE_INT16; *size = 2; return true;
    case U16: *dtype = UAIRT_DTYPE_UINT16; *size = 2; return true;
    case I32: *dtype = UAIRT_DTYPE_INT32; *size = 4; return true;
    case I64: *dtype = UAIRT_DTYPE_INT64; *size = 8; return true;
    case BOOLEAN: *dtype = UAIRT_DTYPE_BOOL; *size = 1; return true;
    default: return false;
  }
}

static void release_engine(engine_t* engine) {
  if (engine->core) {
    ov_core_free(engine->core);
  }
  free(engine->device);
  free(engine->threads);
  free(engine->cache_dir);
  free(engine);
}

static bool device_available(const engine_t* engine, char* available, size_t capacity) {
  ov_available_devices_t devices = {0};
  bool found = false;
  available[0] = '\0';
  if (ov_core_get_available_devices(engine->core, &devices) != OK) {
    return true; /* cannot list devices: let compile report any problem */
  }
  size_t base = strcspn(engine->device, ".(,"); /* "GPU.1" and "AUTO:GPU,CPU" name a device family first */
  size_t used = 0;
  for (size_t i = 0; i < devices.size; ++i) {
    if (strncmp(devices.devices[i], engine->device, base) == 0 || strncmp(engine->device, "AUTO", 4) == 0 ||
        strncmp(engine->device, "HETERO", 6) == 0 || strncmp(engine->device, "MULTI", 5) == 0) {
      found = true;
    }
    if (used < capacity) {
      used += (size_t)snprintf(available + used, capacity - used, "%s%s", used ? ", " : "", devices.devices[i]);
    }
  }
  ov_available_devices_free(&devices);
  return found;
}

static uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->device = strdup("CPU");
  uairt_status status = UAIRT_OK;
  for (size_t i = 0; i < num_options; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    if (strcmp(key, OPTION_DEVICE) == 0) {
      free(engine->device);
      engine->device = strdup(value);
    } else if (strcmp(key, OPTION_HINT) == 0) {
      if (strcmp(value, "latency") == 0) {
        engine->hint = "LATENCY";
      } else if (strcmp(value, "throughput") == 0) {
        engine->hint = "THROUGHPUT";
      } else {
        fail("%s must be latency or throughput", OPTION_HINT);
        status = UAIRT_ERR_INVALID_ARGUMENT;
        goto cleanup;
      }
    } else if (strcmp(key, OPTION_THREADS) == 0) {
      if (atoi(value) < 1) {
        fail("%s must be a positive integer", OPTION_THREADS);
        status = UAIRT_ERR_INVALID_ARGUMENT;
        goto cleanup;
      }
      free(engine->threads);
      engine->threads = strdup(value);
    } else if (strcmp(key, OPTION_CACHE_DIR) == 0) {
      free(engine->cache_dir);
      engine->cache_dir = strdup(value);
    } else {
      fail("unknown option '%s'", key);
      status = UAIRT_ERR_INVALID_ARGUMENT;
      goto cleanup;
    }
  }
  status = check(ov_core_create(&engine->core), "ov_core_create");
  if (status != UAIRT_OK) {
    goto cleanup;
  }
  char available[256];
  if (!device_available(engine, available, sizeof(available))) {
    fail("OpenVINO device '%s' is not available here (available: %s)", engine->device, available);
    status = UAIRT_ERR_BACKEND_UNAVAILABLE;
    goto cleanup;
  }
  *out_engine = engine;
  return UAIRT_OK;

cleanup:
  release_engine(engine);
  return status;
}

static void destroy_engine(void* handle) {
  release_engine(handle);
}

static void free_io(io_t* io, size_t count) {
  if (!io) {
    return;
  }
  for (size_t i = 0; i < count; ++i) {
    free(io[i].name);
  }
  free(io);
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  if (model->request) {
    ov_infer_request_free(model->request);
  }
  if (model->compiled) {
    ov_compiled_model_free(model->compiled);
  }
  free_io(model->inputs, model->num_inputs);
  free_io(model->outputs, model->num_outputs);
  free(model);
}

static uairt_status describe_port(ov_output_const_port_t* port, const char* kind, size_t index, io_t* io) {
  ov_shape_t shape = {0};
  uairt_status status = UAIRT_OK;
  if (ov_const_port_get_shape(port, &shape) != OK) {
    fail("%s %zu has a dynamic shape; export the model with static shapes", kind, index);
    return UAIRT_ERR_UNSUPPORTED;
  }
  size_t element_size = 0;
  ov_element_type_e type = DYNAMIC;
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  if (ov_port_get_element_type(port, &type) != OK || !map_dtype(type, &dtype, &element_size)) {
    fail("%s %zu has an element type the backend does not support", kind, index);
    status = UAIRT_ERR_UNSUPPORTED;
    goto cleanup;
  }
  if (shape.rank > UAIRT_MAX_RANK) {
    fail("%s %zu has rank %lld, above %d", kind, index, (long long)shape.rank, UAIRT_MAX_RANK);
    status = UAIRT_ERR_UNSUPPORTED;
    goto cleanup;
  }
  io->type = type;
  io->desc.struct_size = sizeof(io->desc);
  io->desc.dtype = dtype;
  io->desc.rank = (uint32_t)shape.rank;
  int64_t count = 1;
  for (int64_t d = 0; d < shape.rank; ++d) {
    io->desc.dims[d] = shape.dims[d];
    count *= shape.dims[d];
  }
  io->desc.domain = UAIRT_MEM_HOST;
  io->desc.dmabuf_fd = -1;
  io->nbytes = (size_t)count * element_size;
  io->desc.nbytes = io->nbytes;
  char* name = NULL;
  if (ov_port_get_any_name(port, &name) == OK && name) {
    io->name = strdup(name);
    ov_free(name);
  } else {
    char fallback[32];
    snprintf(fallback, sizeof(fallback), "%s%zu", kind, index);
    io->name = strdup(fallback);
  }
  io->desc.name = io->name;
cleanup:
  ov_shape_free(&shape);
  return status;
}

static uairt_status load_io(
    const ov_compiled_model_t* compiled,
    bool inputs,
    io_t** out,
    size_t* out_count) {
  size_t count = 0;
  uairt_status status = check(inputs ? ov_compiled_model_inputs_size(compiled, &count)
                                     : ov_compiled_model_outputs_size(compiled, &count),
                              "ov_compiled_model_*_size");
  if (status != UAIRT_OK) {
    return status;
  }
  io_t* io = calloc(count ? count : 1, sizeof(*io));
  if (!io) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  *out = io;
  *out_count = count;
  for (size_t i = 0; i < count; ++i) {
    ov_output_const_port_t* port = NULL;
    status = check(inputs ? ov_compiled_model_input_by_index(compiled, i, &port)
                          : ov_compiled_model_output_by_index(compiled, i, &port),
                   "ov_compiled_model_*_by_index");
    if (status != UAIRT_OK) {
      return status;
    }
    status = describe_port(port, inputs ? "input" : "output", i, &io[i]);
    ov_output_const_port_free(port);
    if (status != UAIRT_OK) {
      return status;
    }
  }
  return UAIRT_OK;
}

/* A directory is accepted when it holds exactly the model's .xml; returns a malloc'd path or NULL. */
static char* find_xml_in_directory(const char* directory) {
  char* found = NULL;
#if defined(_WIN32)
  char pattern[1024];
  snprintf(pattern, sizeof(pattern), "%s\\*.xml", directory);
  WIN32_FIND_DATAA entry;
  HANDLE handle = FindFirstFileA(pattern, &entry);
  if (handle == INVALID_HANDLE_VALUE) {
    return NULL;
  }
  size_t length = strlen(directory) + strlen(entry.cFileName) + 2;
  found = malloc(length);
  if (found) {
    snprintf(found, length, "%s\\%s", directory, entry.cFileName);
  }
  FindClose(handle);
#else
  DIR* dir = opendir(directory);
  if (!dir) {
    return NULL;
  }
  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    size_t length = strlen(entry->d_name);
    if (length > 4 && strcmp(entry->d_name + length - 4, ".xml") == 0) {
      size_t total = strlen(directory) + length + 2;
      found = malloc(total);
      if (found) {
        snprintf(found, total, "%s/%s", directory, entry->d_name);
      }
      break;
    }
  }
  closedir(dir);
#endif
  return found;
}

static bool is_directory(const char* path) {
#if defined(_WIN32)
  DWORD attributes = GetFileAttributesA(path);
  return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  struct stat info;
  return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

static uairt_status compile(engine_t* engine, ov_model_t* model, ov_compiled_model_t** compiled) {
  const char* keys[3];
  const char* values[3];
  size_t count = 0;
  if (engine->hint) {
    keys[count] = ov_property_key_hint_performance_mode;
    values[count++] = engine->hint;
  }
  if (engine->threads) {
    keys[count] = ov_property_key_inference_num_threads;
    values[count++] = engine->threads;
  }
  if (engine->cache_dir) {
    keys[count] = ov_property_key_cache_dir;
    values[count++] = engine->cache_dir;
  }
  ov_status_e status;
  switch (count) {
    case 0:
      status = ov_core_compile_model(engine->core, model, engine->device, 0, compiled);
      break;
    case 1:
      status = ov_core_compile_model(engine->core, model, engine->device, 1, compiled, keys[0], values[0]);
      break;
    case 2:
      status = ov_core_compile_model(engine->core, model, engine->device, 2, compiled, keys[0], values[0], keys[1],
                                     values[1]);
      break;
    default:
      status = ov_core_compile_model(engine->core, model, engine->device, 3, compiled, keys[0], values[0], keys[1],
                                     values[1], keys[2], values[2]);
      break;
  }
  return check(status, "ov_core_compile_model");
}

static uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  engine_t* engine = engine_handle;
  if (!source->path) {
    fail("the openvino backend loads models from a path (.xml, .onnx, or a directory with one .xml)");
    return UAIRT_ERR_UNSUPPORTED;
  }
  char* resolved = NULL;
  const char* path = source->path;
  if (is_directory(path)) {
    resolved = find_xml_in_directory(path);
    if (!resolved) {
      fail("no .xml model found in directory '%s'", path);
      return UAIRT_ERR_IO;
    }
    path = resolved;
  }
  FILE* file = fopen(path, "rb");
  if (!file) {
    fail("cannot open model file '%s'", path);
    free(resolved);
    return UAIRT_ERR_IO;
  }
  fclose(file);

  model_t* model = calloc(1, sizeof(*model));
  ov_model_t* graph = NULL;
  uairt_status status = model ? UAIRT_OK : UAIRT_ERR_OUT_OF_MEMORY;
  if (status == UAIRT_OK) {
    status = check(ov_core_read_model(engine->core, path, NULL, &graph), "ov_core_read_model");
    if (status == UAIRT_ERR_RUNTIME) {
      status = UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
  }
  if (status == UAIRT_OK) {
    status = compile(engine, graph, &model->compiled);
  }
  if (status == UAIRT_OK) {
    status = check(ov_compiled_model_create_infer_request(model->compiled, &model->request),
                   "ov_compiled_model_create_infer_request");
  }
  if (status == UAIRT_OK) {
    status = load_io(model->compiled, true, &model->inputs, &model->num_inputs);
  }
  if (status == UAIRT_OK) {
    status = load_io(model->compiled, false, &model->outputs, &model->num_outputs);
  }
  if (graph) {
    ov_model_free(graph);
  }
  free(resolved);
  if (status != UAIRT_OK) {
    if (model) {
      destroy_model(model);
    }
    return status;
  }
  *out_model = model;
  return UAIRT_OK;
}

static size_t num_inputs(const void* handle) {
  return ((const model_t*)handle)->num_inputs;
}

static size_t num_outputs(const void* handle) {
  return ((const model_t*)handle)->num_outputs;
}

static uairt_status input_info(const void* handle, size_t index, uairt_tensor* out) {
  *out = ((const model_t*)handle)->inputs[index].desc;
  return UAIRT_OK;
}

static uairt_status output_info(const void* handle, size_t index, uairt_tensor* out) {
  *out = ((const model_t*)handle)->outputs[index].desc;
  return UAIRT_OK;
}

/* Wraps the caller's memory in a tensor, without copying. */
static uairt_status wrap(const io_t* io, void* data, ov_tensor_t** tensor) {
  ov_shape_t shape = {0};
  uairt_status status = check(ov_shape_create(io->desc.rank, io->desc.dims, &shape), "ov_shape_create");
  if (status != UAIRT_OK) {
    return status;
  }
  status = check(ov_tensor_create_from_host_ptr(io->type, shape, data, tensor), "ov_tensor_create_from_host_ptr");
  ov_shape_free(&shape);
  return status;
}

static uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  model_t* model = handle;
  ov_tensor_t** tensors = calloc(n_in + n_out ? n_in + n_out : 1, sizeof(*tensors));
  if (!tensors) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status = UAIRT_OK;
  for (size_t i = 0; i < n_in && status == UAIRT_OK; ++i) {
    status = wrap(&model->inputs[i], inputs[i].data, &tensors[i]);
    if (status == UAIRT_OK) {
      status = check(ov_infer_request_set_input_tensor_by_index(model->request, i, tensors[i]),
                     "ov_infer_request_set_input_tensor_by_index");
    }
  }
  for (size_t i = 0; i < n_out && status == UAIRT_OK; ++i) {
    status = wrap(&model->outputs[i], outputs[i].data, &tensors[n_in + i]);
    if (status == UAIRT_OK) {
      status = check(ov_infer_request_set_output_tensor_by_index(model->request, i, tensors[n_in + i]),
                     "ov_infer_request_set_output_tensor_by_index");
    }
  }
  if (status == UAIRT_OK) {
    status = check(ov_infer_request_infer(model->request), "ov_infer_request_infer");
  }
  for (size_t i = 0; i < n_in + n_out; ++i) {
    if (tensors[i]) {
      ov_tensor_free(tensors[i]);
    }
  }
  free(tensors);
  return status;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "openvino",
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
  return &kApi;
}
