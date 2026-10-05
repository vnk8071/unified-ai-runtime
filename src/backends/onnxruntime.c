/* SPDX-License-Identifier: Apache-2.0 */
/*
 * ONNX Runtime backend, built as a loadable plugin (libuairt_backend_onnxruntime).
 * v1 limits: host memory, static shapes. Execution providers: CPU, or a plugin execution provider
 * library (ep_library, ep_name, ep_option.<key>) with CPU as the fallback.
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
#include <dlfcn.h>
#endif
#include <onnxruntime_c_api.h>
#include <uairt/uairt_backend.h>

#define OPTION_THREADS "intra_op_threads"
#define OPTION_EP "execution_provider"
#define OPTION_EP_LIBRARY "ep_library"
#define OPTION_EP_NAME "ep_name"
#define OPTION_EP_PREFIX "ep_option."
#define OPTION_LOG_LEVEL "log_level"
#define MAX_EP_OPTIONS 32

typedef struct {
  OrtEnv* env;
  OrtSessionOptions* options;
  char* ep_registration;
} engine_t;

typedef struct {
  char* name;
  uairt_tensor desc;
  ONNXTensorElementDataType element_type;
  size_t nbytes;
} io_t;

typedef struct {
  OrtSession* session;
  OrtMemoryInfo* memory_info;
  io_t* inputs;
  size_t num_inputs;
  io_t* outputs;
  size_t num_outputs;
} model_t;

static const uairt_host_api* g_host;
static const OrtApi* g_ort;

#if defined(__GNUC__)
static void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

#if defined(_WIN32)
/* ONNX Runtime takes wide paths on Windows; the caller frees the result. */
static wchar_t* utf8_to_wide(const char* text) {
  int count = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
  wchar_t* wide = count > 0 ? malloc((size_t)count * sizeof(wchar_t)) : NULL;
  if (wide && MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, count) == 0) {
    free(wide);
    wide = NULL;
  }
  return wide;
}
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

static uairt_status check(OrtStatus* status) {
  if (!status) {
    return UAIRT_OK;
  }
  OrtErrorCode code = g_ort->GetErrorCode(status);
  fail("onnxruntime: %s", g_ort->GetErrorMessage(status));
  g_ort->ReleaseStatus(status);
  switch (code) {
    case ORT_INVALID_ARGUMENT:
      return UAIRT_ERR_INVALID_ARGUMENT;
    case ORT_NO_SUCHFILE:
    case ORT_NO_MODEL:
      return UAIRT_ERR_IO;
    case ORT_INVALID_PROTOBUF:
    case ORT_INVALID_GRAPH:
    case ORT_MODEL_LOAD_CANCELED:
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    case ORT_NOT_IMPLEMENTED:
      return UAIRT_ERR_UNSUPPORTED;
    default:
      return UAIRT_ERR_RUNTIME;
  }
}

#define TRY(expr)                \
  do {                           \
    status = check(expr);        \
    if (status != UAIRT_OK) {     \
      goto cleanup;              \
    }                            \
  } while (0)

/* Plugins do not link the core, so element sizes are computed here. */
static bool map_element_type(
    ONNXTensorElementDataType type,
    uairt_dtype* dtype,
    size_t* size) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      *dtype = UAIRT_DTYPE_FLOAT32, *size = 4;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      *dtype = UAIRT_DTYPE_FLOAT16, *size = 2;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      *dtype = UAIRT_DTYPE_BFLOAT16, *size = 2;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      *dtype = UAIRT_DTYPE_INT8, *size = 1;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      *dtype = UAIRT_DTYPE_UINT8, *size = 1;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      *dtype = UAIRT_DTYPE_INT16, *size = 2;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      *dtype = UAIRT_DTYPE_INT32, *size = 4;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      *dtype = UAIRT_DTYPE_INT64, *size = 8;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      *dtype = UAIRT_DTYPE_BOOL, *size = 1;
      return true;
    default:
      return false;
  }
}

static void release_status(OrtStatus* status) {
  if (status) {
    g_ort->ReleaseStatus(status);
  }
}

static void release_engine(engine_t* engine) {
  if (engine->ep_registration) {
    release_status(g_ort->UnregisterExecutionProviderLibrary(engine->env, engine->ep_registration));
    free(engine->ep_registration);
  }
  if (engine->options) {
    g_ort->ReleaseSessionOptions(engine->options);
  }
  if (engine->env) {
    g_ort->ReleaseEnv(engine->env);
  }
  free(engine);
}

/* Registers a plugin execution provider library and appends its device (CPU stays as the fallback). */
static uairt_status append_plugin_ep(
    engine_t* engine,
    const char* library,
    const char* name,
    const char* const* keys,
    const char* const* values,
    size_t num_ep_options) {
  uairt_status status = UAIRT_OK;
#if defined(_WIN32)
  wchar_t* wide_library = utf8_to_wide(library);
  if (!wide_library) {
    fail("cannot convert the plugin library path to UTF-16");
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  OrtStatus* register_status =
      g_ort->RegisterExecutionProviderLibrary(engine->env, name, wide_library);
  free(wide_library);
  TRY(register_status);
#else
  TRY(g_ort->RegisterExecutionProviderLibrary(engine->env, name, library));
#endif
  engine->ep_registration = strdup(name);
  const OrtEpDevice* const* devices = NULL;
  size_t num_devices = 0;
  TRY(g_ort->GetEpDevices(engine->env, &devices, &num_devices));
  const OrtEpDevice* chosen = NULL;
  char available[256] = "";
  size_t used = 0;
  for (size_t i = 0; i < num_devices; ++i) {
    const char* device_name = g_ort->EpDevice_EpName(devices[i]);
    if (!chosen && strcmp(device_name, name) == 0) {
      chosen = devices[i];
    }
    if (used < sizeof(available)) {
      used += (size_t)snprintf(available + used, sizeof(available) - used, "%s%s", used ? ", " : "", device_name);
    }
  }
  if (!chosen) {
    fail("no execution provider device named '%s' (available: %s); the library may not match this "
         "ONNX Runtime version, or ONNX Runtime found no hardware the provider supports",
         name, available);
    return UAIRT_ERR_UNSUPPORTED;
  }
  TRY(g_ort->SessionOptionsAppendExecutionProvider_V2(engine->options, engine->env, &chosen, 1, keys,
                                                      values, num_ep_options));
cleanup:
  return status;
}

/*
 * Appends a built-in GPU execution provider ("cuda" or "tensorrt") with the ep_option.<key> settings. The provider
 * libraries ship with an ONNX Runtime GPU build; a CPU-only build fails here. Nodes the provider cannot take run on
 * the CPU provider, which ONNX Runtime adds last.
 */
static uairt_status append_gpu_ep(
    engine_t* engine,
    const char* provider,
    const char* const* keys,
    const char* const* values,
    size_t num_ep_options) {
  uairt_status status = UAIRT_OK;
  OrtCUDAProviderOptionsV2* cuda = NULL;
  OrtTensorRTProviderOptionsV2* tensorrt = NULL;
  if (strcmp(provider, "cuda") == 0) {
    TRY(g_ort->CreateCUDAProviderOptions(&cuda));
    TRY(g_ort->UpdateCUDAProviderOptions(cuda, keys, values, num_ep_options));
    status = check(g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(engine->options, cuda));
  } else {
    TRY(g_ort->CreateTensorRTProviderOptions(&tensorrt));
    TRY(g_ort->UpdateTensorRTProviderOptions(tensorrt, keys, values, num_ep_options));
    status = check(g_ort->SessionOptionsAppendExecutionProvider_TensorRT_V2(engine->options, tensorrt));
  }
cleanup:
  if (cuda) {
    g_ort->ReleaseCUDAProviderOptions(cuda);
  }
  if (tensorrt) {
    g_ort->ReleaseTensorRTProviderOptions(tensorrt);
  }
  if (status == UAIRT_ERR_RUNTIME) {
    status = UAIRT_ERR_BACKEND_UNAVAILABLE;  // a CPU-only build, or the provider's libraries cannot be loaded
  }
  return status;
}

static bool parse_log_level(const char* value, int* level) {
  static const char* const names[] = {"verbose", "info", "warning", "error", "fatal"};
  for (int i = 0; i < 5; ++i) {
    if (strcmp(value, names[i]) == 0) {
      *level = i;
      return true;
    }
  }
  return false;
}

static uairt_status create_engine(
    const uairt_option* options,
    size_t num_options,
    void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status = UAIRT_OK;
  const char* ep_library = NULL;
  const char* ep_name = NULL;
  const char* ep_keys[MAX_EP_OPTIONS];
  const char* ep_values[MAX_EP_OPTIONS];
  size_t num_ep_options = 0;
  bool cpu_requested = false;
  const char* gpu_ep = NULL;
  TRY(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "uairt", &engine->env));
  TRY(g_ort->CreateSessionOptions(&engine->options));
  for (size_t i = 0; i < num_options; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    if (strcmp(key, OPTION_THREADS) == 0) {
      TRY(g_ort->SetIntraOpNumThreads(engine->options, atoi(value)));
    } else if (strcmp(key, OPTION_LOG_LEVEL) == 0) {
      int level = 0;
      if (!parse_log_level(value, &level)) {
        fail("%s must be verbose, info, warning, error or fatal", OPTION_LOG_LEVEL);
        status = UAIRT_ERR_INVALID_ARGUMENT;
        goto cleanup;
      }
      TRY(g_ort->SetSessionLogSeverityLevel(engine->options, level));
    } else if (strcmp(key, OPTION_EP) == 0) {
      if (strcmp(value, "cuda") == 0 || strcmp(value, "tensorrt") == 0) {
        gpu_ep = value;
      } else if (strcmp(value, "cpu") == 0) {
        cpu_requested = true;
      } else {
        fail("execution provider '%s' is not supported (cpu, cuda, tensorrt); use ep_library and ep_name for a "
             "plugin", value);
        status = UAIRT_ERR_UNSUPPORTED;
        goto cleanup;
      }
    } else if (strcmp(key, OPTION_EP_LIBRARY) == 0) {
      ep_library = value;
    } else if (strcmp(key, OPTION_EP_NAME) == 0) {
      ep_name = value;
    } else if (strncmp(key, OPTION_EP_PREFIX, strlen(OPTION_EP_PREFIX)) == 0) {
      if (num_ep_options == MAX_EP_OPTIONS) {
        fail("at most %d '%s*' options are supported", MAX_EP_OPTIONS, OPTION_EP_PREFIX);
        status = UAIRT_ERR_INVALID_ARGUMENT;
        goto cleanup;
      }
      ep_keys[num_ep_options] = key + strlen(OPTION_EP_PREFIX);
      ep_values[num_ep_options++] = value;
    } else {
      fail("unknown option '%s'", key);
      status = UAIRT_ERR_INVALID_ARGUMENT;
      goto cleanup;
    }
  }
  if (gpu_ep) {
    if (ep_library || ep_name) {
      fail("execution_provider=%s cannot be combined with '%s' or '%s'", gpu_ep, OPTION_EP_LIBRARY, OPTION_EP_NAME);
      status = UAIRT_ERR_INVALID_ARGUMENT;
      goto cleanup;
    }
    status = append_gpu_ep(engine, gpu_ep, ep_keys, ep_values, num_ep_options);
    if (status != UAIRT_OK) {
      goto cleanup;
    }
  } else if (ep_library) {
    if (!ep_name || cpu_requested) {
      fail("'%s' needs '%s' and cannot be combined with execution_provider=cpu", OPTION_EP_LIBRARY, OPTION_EP_NAME);
      status = UAIRT_ERR_INVALID_ARGUMENT;
      goto cleanup;
    }
    status = append_plugin_ep(engine, ep_library, ep_name, ep_keys, ep_values, num_ep_options);
    if (status != UAIRT_OK) {
      goto cleanup;
    }
  } else if (ep_name || num_ep_options) {
    fail("'%s' and '%s*' need '%s' or execution_provider=cuda|tensorrt", OPTION_EP_NAME, OPTION_EP_PREFIX,
         OPTION_EP_LIBRARY);
    status = UAIRT_ERR_INVALID_ARGUMENT;
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
  if (model->session) {
    g_ort->ReleaseSession(model->session);
  }
  if (model->memory_info) {
    g_ort->ReleaseMemoryInfo(model->memory_info);
  }
  free_io(model->inputs, model->num_inputs);
  free_io(model->outputs, model->num_outputs);
  free(model);
}

static uairt_status describe_io(
    OrtSession* session,
    OrtAllocator* allocator,
    bool is_output,
    size_t index,
    io_t* io) {
  uairt_status status = UAIRT_OK;
  char* name = NULL;
  OrtTypeInfo* type_info = NULL;
  const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
  const char* kind = is_output ? "output" : "input";

  TRY(is_output ? g_ort->SessionGetOutputName(session, index, allocator, &name)
                : g_ort->SessionGetInputName(session, index, allocator, &name));
  io->name = strdup(name);
  if (!io->name) {
    status = UAIRT_ERR_OUT_OF_MEMORY;
    goto cleanup;
  }
  TRY(is_output ? g_ort->SessionGetOutputTypeInfo(session, index, &type_info)
                : g_ort->SessionGetInputTypeInfo(session, index, &type_info));
  TRY(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));
  if (!tensor_info) {
    fail("%s '%s' is not a tensor", kind, io->name);
    status = UAIRT_ERR_UNSUPPORTED;
    goto cleanup;
  }

  size_t rank = 0;
  TRY(g_ort->GetTensorElementType(tensor_info, &io->element_type));
  TRY(g_ort->GetDimensionsCount(tensor_info, &rank));
  size_t element_size = 0;
  if (!map_element_type(io->element_type, &io->desc.dtype, &element_size) ||
      rank > UAIRT_MAX_RANK) {
    fail("%s '%s' has an unsupported element type or rank", kind, io->name);
    status = UAIRT_ERR_UNSUPPORTED;
    goto cleanup;
  }
  io->desc.rank = (uint32_t)rank;
  io->desc.name = io->name;
  TRY(g_ort->GetDimensions(tensor_info, io->desc.dims, rank));
  io->nbytes = element_size;
  for (size_t d = 0; d < rank; ++d) {
    if (io->desc.dims[d] < 0) {
      fail("%s '%s' has a dynamic dimension %zu; static shapes only for now",
           kind, io->name, d);
      status = UAIRT_ERR_UNSUPPORTED;
      goto cleanup;
    }
    io->nbytes *= (size_t)io->desc.dims[d];
  }

cleanup:
  if (type_info) {
    g_ort->ReleaseTypeInfo(type_info);
  }
  if (name) {
    OrtStatus* free_status = g_ort->AllocatorFree(allocator, name);
    if (free_status) {
      g_ort->ReleaseStatus(free_status);
    }
  }
  return status;
}

static uairt_status load_io(
    OrtSession* session,
    OrtAllocator* allocator,
    bool is_output,
    io_t** out_io,
    size_t* out_count) {
  size_t count = 0;
  uairt_status status = check(
      is_output ? g_ort->SessionGetOutputCount(session, &count)
                : g_ort->SessionGetInputCount(session, &count));
  if (status != UAIRT_OK) {
    return status;
  }
  io_t* io = calloc(count ? count : 1, sizeof(*io));
  if (!io) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  *out_io = io;
  *out_count = count;
  for (size_t i = 0; i < count; ++i) {
    io[i].desc.struct_size = sizeof(uairt_tensor);
    io[i].desc.domain = UAIRT_MEM_HOST;
    io[i].desc.dmabuf_fd = -1;
    status = describe_io(session, allocator, is_output, i, &io[i]);
    if (status != UAIRT_OK) {
      return status;
    }
  }
  return UAIRT_OK;
}

static uairt_status load_model(
    void* engine_handle,
    const uairt_model_source* source,
    void** out_model) {
  engine_t* engine = engine_handle;
  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status = UAIRT_OK;
  OrtAllocator* allocator = NULL;

  if (source->path) {
#ifdef _WIN32
    wchar_t* wide_path = utf8_to_wide(source->path);
    if (!wide_path) {
      fail("cannot convert the model path to UTF-16");
      status = UAIRT_ERR_OUT_OF_MEMORY;
      goto cleanup;
    }
    OrtStatus* create_status =
        g_ort->CreateSession(engine->env, wide_path, engine->options, &model->session);
    free(wide_path);
    TRY(create_status);
#else
    TRY(g_ort->CreateSession(engine->env, source->path, engine->options,
                             &model->session));
#endif
  } else {
    TRY(g_ort->CreateSessionFromArray(engine->env, source->data, source->size,
                                      engine->options, &model->session));
  }
  TRY(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                 &model->memory_info));
  TRY(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  status = load_io(model->session, allocator, false, &model->inputs,
                   &model->num_inputs);
  if (status != UAIRT_OK) {
    goto cleanup;
  }
  status = load_io(model->session, allocator, true, &model->outputs,
                   &model->num_outputs);
  if (status != UAIRT_OK) {
    goto cleanup;
  }
  *out_model = model;
  return UAIRT_OK;

cleanup:
  destroy_model(model);
  return status;
}

static size_t num_inputs(const void* handle) {
  return ((const model_t*)handle)->num_inputs;
}

static size_t num_outputs(const void* handle) {
  return ((const model_t*)handle)->num_outputs;
}

static uairt_status
input_info(const void* handle, size_t index, uairt_tensor* out) {
  *out = ((const model_t*)handle)->inputs[index].desc;
  return UAIRT_OK;
}

static uairt_status
output_info(const void* handle, size_t index, uairt_tensor* out) {
  *out = ((const model_t*)handle)->outputs[index].desc;
  return UAIRT_OK;
}

static uairt_status bind(
    model_t* model,
    const io_t* io,
    void* data,
    OrtValue** out_value) {
  return check(g_ort->CreateTensorWithDataAsOrtValue(
      model->memory_info, data, io->nbytes, io->desc.dims, io->desc.rank,
      io->element_type, out_value));
}

static uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  model_t* model = handle;
  OrtValue** in_values = calloc(n_in ? n_in : 1, sizeof(*in_values));
  OrtValue** out_values = calloc(n_out ? n_out : 1, sizeof(*out_values));
  const char** in_names = calloc(n_in ? n_in : 1, sizeof(*in_names));
  const char** out_names = calloc(n_out ? n_out : 1, sizeof(*out_names));
  uairt_status status = UAIRT_OK;
  if (!in_values || !out_values || !in_names || !out_names) {
    status = UAIRT_ERR_OUT_OF_MEMORY;
    goto cleanup;
  }
  for (size_t i = 0; i < n_in; ++i) {
    in_names[i] = model->inputs[i].name;
    status = bind(model, &model->inputs[i], inputs[i].data, &in_values[i]);
    if (status != UAIRT_OK) {
      goto cleanup;
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    out_names[i] = model->outputs[i].name;
    status = bind(model, &model->outputs[i], outputs[i].data, &out_values[i]);
    if (status != UAIRT_OK) {
      goto cleanup;
    }
  }
  status = check(g_ort->Run(model->session, NULL, in_names, (const OrtValue* const*)in_values,
                            n_in, out_names, n_out, out_values));

cleanup:
  for (size_t i = 0; in_values && i < n_in; ++i) {
    g_ort->ReleaseValue(in_values[i]);
  }
  for (size_t i = 0; out_values && i < n_out; ++i) {
    g_ort->ReleaseValue(out_values[i]);
  }
  free(in_values);
  free(out_values);
  free(in_names);
  free(out_names);
  return status;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "onnxruntime",
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

/*
 * ONNX Runtime is loaded when the plugin loads, not linked, so one plugin works with whichever release is installed (a pip
 * install of onnxruntime, a release directory, a system copy) as long as it is at least as new as the headers this was built
 * with. Search order: the UAIRT_ONNXRUNTIME_LIBRARY environment variable, a directory baked in at build time
 * (UAIRT_ORT_LIBRARY_DIR, for source builds), then the loader's default search.
 */
#if defined(_WIN32)
static const char* const kLibraryNames[] = {"onnxruntime.dll"};
#elif defined(__APPLE__)
static const char* const kLibraryNames[] = {"libonnxruntime.dylib"};
#else
static const char* const kLibraryNames[] = {"libonnxruntime.so", "libonnxruntime.so.1"};
#endif

typedef const OrtApiBase*(ORT_API_CALL* get_api_base_fn)(void);

static get_api_base_fn open_ort(const char* path) {
  get_api_base_fn function = NULL;
#if defined(_WIN32)
  HMODULE library = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  FARPROC symbol = library ? GetProcAddress(library, "OrtGetApiBase") : NULL;
#else
  void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  void* symbol = library ? dlsym(library, "OrtGetApiBase") : NULL;
#endif
  if (symbol) {
    memcpy(&function, &symbol, sizeof(function));
  }
  return function;
}

/* On Windows the Win32 environment is read, not the C runtime's copy: a host built with a different C runtime (Python, or a
 * debug build) sets variables the plugin's own copy never sees. */
static bool read_variable(const char* name, char* out, size_t capacity) {
#if defined(_WIN32)
  DWORD length = GetEnvironmentVariableA(name, out, (DWORD)capacity);
  return length > 0 && length < capacity;
#else
  const char* value = getenv(name);
  if (!value || !value[0]) {
    return false;
  }
  snprintf(out, capacity, "%s", value);
  return true;
#endif
}

/* Returns the first ONNX Runtime found, and the paths tried when none loads. The library stays loaded. */
static const OrtApiBase* load_ort(char* tried, size_t capacity) {
  char candidates[8][1024];
  size_t count = 0;
  if (read_variable("UAIRT_ONNXRUNTIME_LIBRARY", candidates[count], sizeof(candidates[0]))) {
    ++count;
  }
  for (size_t i = 0; i < sizeof(kLibraryNames) / sizeof(kLibraryNames[0]); ++i) {
#if defined(UAIRT_ORT_LIBRARY_DIR)
    snprintf(candidates[count], sizeof(candidates[0]), "%s/%s", UAIRT_ORT_LIBRARY_DIR, kLibraryNames[i]);
#if defined(_WIN32)
    for (char* c = candidates[count]; *c; ++c) {
      *c = *c == '/' ? '\\' : *c;  /* the altered search path needs backslashes */
    }
#endif
    ++count;
#endif
    snprintf(candidates[count++], sizeof(candidates[0]), "%s", kLibraryNames[i]);
  }
  size_t used = 0;
  tried[0] = '\0';
  for (size_t i = 0; i < count; ++i) {
    get_api_base_fn get_api_base = open_ort(candidates[i]);
    if (get_api_base) {
      return get_api_base();
    }
    if (used < capacity) {
      used += (size_t)snprintf(tried + used, capacity - used, "%s%s", used ? ", " : "", candidates[i]);
    }
  }
  return NULL;
}

UAIRT_API const uairt_backend_api* uairt_backend_get_api(
    const uairt_host_api* host) {
  g_host = host;
  char tried[2048];
  const OrtApiBase* base = load_ort(tried, sizeof(tried));
  if (!base) {
    fail("cannot load the ONNX Runtime library (tried: %s); install onnxruntime (pip install onnxruntime) or set "
         "UAIRT_ONNXRUNTIME_LIBRARY to its path",
         tried);
    return NULL;
  }
  g_ort = base->GetApi(ORT_API_VERSION);
  if (!g_ort) {
    fail("the installed ONNX Runtime (%s) is older than the headers this "
         "backend was built with (API %d)",
         base->GetVersionString(), ORT_API_VERSION);
    return NULL;
  }
  return &kApi;
}
