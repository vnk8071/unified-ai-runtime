/* SPDX-License-Identifier: Apache-2.0 */
/*
 * TFLite backend, built as a loadable plugin. Runs .tflite models with TFLite's own CPU kernels,
 * or through the QNN TFLite delegate (HTP) when the plugin is built with the delegate header.
 * v1 limits: static shapes, host memory, one delegate (QNN HTP). The delegate library is loaded
 * at run time from a path given as an engine option.
 */
#define _POSIX_C_SOURCE 200809L
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define strdup _strdup
#else
#include <dlfcn.h>
#endif
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/lite/c/c_api.h"
#include <uairt/uairt_backend.h>

#ifdef UAIRT_TFLITE_QNN_DELEGATE
#include "QnnTFLiteDelegate.h"
#endif

#define OPTION_NUM_THREADS "num_threads"
#define OPTION_DELEGATE "delegate"
#define OPTION_DELEGATE_LIBRARY "delegate_library"
#define OPTION_BACKEND_LIBRARY "backend_library"
#define OPTION_SKEL_DIR "skel_dir"
#define OPTION_PERFORMANCE_MODE "htp_performance_mode"
#define OPTION_PRECISION "htp_precision"
#define OPTION_ADSP_LIBRARY_PATH "adsp_library_path"
#define MESSAGE_CAPACITY 512

typedef struct {
  int num_threads;
  bool use_qnn_delegate;
  char* delegate_library;
  char* backend_library;
  char* skel_dir;
  char* performance_mode;
  char* precision;
#ifdef UAIRT_TFLITE_QNN_DELEGATE
  void* delegate_handle;
  TfLiteQnnDelegateOptions (*options_default)(void);
  TfLiteDelegate* (*delegate_create)(const TfLiteQnnDelegateOptions*);
  void (*delegate_delete)(TfLiteDelegate*);
#endif
} engine_t;

typedef struct {
  char* name;
  uairt_tensor desc;
} io_t;

typedef struct {
  engine_t* engine;
  void* model_data;
  TfLiteModel* model;
  TfLiteInterpreterOptions* options;
  TfLiteDelegate* delegate;
  TfLiteInterpreter* interpreter;
  io_t* inputs;
  size_t num_inputs;
  io_t* outputs;
  size_t num_outputs;
} model_t;

static const uairt_host_api* g_host;
static char g_tflite_message[MESSAGE_CAPACITY];

#if defined(__GNUC__)
static void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
#endif

#if defined(_WIN32)
static void* uairt_dlopen(const char* path) {
  return (void*)LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
}
static void* uairt_dlsym(void* handle, const char* name) {
  FARPROC symbol = GetProcAddress((HMODULE)handle, name);
  void* out;
  memcpy(&out, &symbol, sizeof(out));
  return out;
}
static void uairt_dlclose(void* handle) {
  FreeLibrary((HMODULE)handle);
}
static const char* uairt_dlerror(void) {
  static char text[64];
  snprintf(text, sizeof(text), "error %lu", GetLastError());
  return text;
}
#else
#define uairt_dlopen(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define uairt_dlsym dlsym
#define uairt_dlclose dlclose
#define uairt_dlerror dlerror
#endif

static void fail(const char* fmt, ...) {
  char message[MESSAGE_CAPACITY];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

static void fail_with_tflite_message(const char* what) {
  fail("%s%s%s", what, g_tflite_message[0] ? ": " : "", g_tflite_message);
}

static void error_reporter(void* user_data, const char* format, va_list args) {
  (void)user_data;
  vsnprintf(g_tflite_message, sizeof(g_tflite_message), format, args);
}

static char* copy_string(const char* text) {
  return text ? strdup(text) : NULL;
}

static bool map_type(TfLiteType type, uairt_dtype* dtype) {
  switch (type) {
    case kTfLiteFloat32:
      *dtype = UAIRT_DTYPE_FLOAT32;
      return true;
    case kTfLiteFloat16:
      *dtype = UAIRT_DTYPE_FLOAT16;
      return true;
    case kTfLiteInt8:
      *dtype = UAIRT_DTYPE_INT8;
      return true;
    case kTfLiteUInt8:
      *dtype = UAIRT_DTYPE_UINT8;
      return true;
    case kTfLiteInt16:
      *dtype = UAIRT_DTYPE_INT16;
      return true;
    case kTfLiteInt32:
      *dtype = UAIRT_DTYPE_INT32;
      return true;
    case kTfLiteInt64:
      *dtype = UAIRT_DTYPE_INT64;
      return true;
    case kTfLiteBool:
      *dtype = UAIRT_DTYPE_BOOL;
      return true;
    default:
      return false;
  }
}

static void destroy_engine(void* handle) {
  engine_t* engine = handle;
#ifdef UAIRT_TFLITE_QNN_DELEGATE
  if (engine->delegate_handle) {
    uairt_dlclose(engine->delegate_handle);
  }
#endif
  free(engine->delegate_library);
  free(engine->backend_library);
  free(engine->skel_dir);
  free(engine->performance_mode);
  free(engine->precision);
  free(engine);
}

#ifdef UAIRT_TFLITE_QNN_DELEGATE
static uairt_status open_delegate(engine_t* engine) {
  if (!engine->delegate_library || !engine->backend_library) {
    fail("delegate=qnn needs the options '%s' and '%s'", OPTION_DELEGATE_LIBRARY,
         OPTION_BACKEND_LIBRARY);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  engine->delegate_handle = uairt_dlopen(engine->delegate_library);
  if (!engine->delegate_handle) {
    fail("cannot load the QNN delegate library: %s", uairt_dlerror());
    return UAIRT_ERR_BACKEND_UNAVAILABLE;
  }
  void* symbol = uairt_dlsym(engine->delegate_handle, "TfLiteQnnDelegateOptionsDefault");
  memcpy(&engine->options_default, &symbol, sizeof(symbol));
  symbol = uairt_dlsym(engine->delegate_handle, "TfLiteQnnDelegateCreate");
  memcpy(&engine->delegate_create, &symbol, sizeof(symbol));
  symbol = uairt_dlsym(engine->delegate_handle, "TfLiteQnnDelegateDelete");
  memcpy(&engine->delegate_delete, &symbol, sizeof(symbol));
  if (!engine->options_default || !engine->delegate_create || !engine->delegate_delete) {
    fail("the library does not export the QNN delegate API");
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  return UAIRT_OK;
}

static bool parse_performance_mode(const char* value, TfLiteQnnDelegateHtpPerformanceMode* mode) {
  static const struct {
    const char* name;
    TfLiteQnnDelegateHtpPerformanceMode mode;
  } kModes[] = {
      {"default", kHtpDefault},
      {"burst", kHtpBurst},
      {"high_performance", kHtpHighPerformance},
      {"sustained", kHtpSustainedHighPerformance},
      {"balanced", kHtpBalanced},
      {"power_saver", kHtpPowerSaver},
  };
  for (size_t i = 0; i < sizeof(kModes) / sizeof(kModes[0]); ++i) {
    if (strcmp(value, kModes[i].name) == 0) {
      *mode = kModes[i].mode;
      return true;
    }
  }
  return false;
}
#endif

static uairt_status create_engine(
    const uairt_option* options,
    size_t num_options,
    void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->num_threads = 1;
  uairt_status status = UAIRT_OK;
  for (size_t i = 0; i < num_options && status == UAIRT_OK; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    if (strcmp(key, OPTION_NUM_THREADS) == 0) {
      engine->num_threads = atoi(value);
    } else if (strcmp(key, OPTION_DELEGATE) == 0) {
      if (strcmp(value, "qnn") == 0) {
        engine->use_qnn_delegate = true;
      } else if (strcmp(value, "none") != 0) {
        fail("%s must be none or qnn, got '%s'", OPTION_DELEGATE, value);
        status = UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (strcmp(key, OPTION_DELEGATE_LIBRARY) == 0) {
      engine->delegate_library = copy_string(value);
    } else if (strcmp(key, OPTION_BACKEND_LIBRARY) == 0) {
      engine->backend_library = copy_string(value);
    } else if (strcmp(key, OPTION_SKEL_DIR) == 0) {
      engine->skel_dir = copy_string(value);
    } else if (strcmp(key, OPTION_PERFORMANCE_MODE) == 0) {
      engine->performance_mode = copy_string(value);
    } else if (strcmp(key, OPTION_PRECISION) == 0) {
      engine->precision = copy_string(value);
    } else if (strcmp(key, OPTION_ADSP_LIBRARY_PATH) == 0) {
#if defined(_WIN32)
      _putenv_s("ADSP_LIBRARY_PATH", value);
#else
      setenv("ADSP_LIBRARY_PATH", value, 1);
#endif
    } else {
      fail("unknown option '%s'", key);
      status = UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  if (status == UAIRT_OK && engine->use_qnn_delegate) {
#ifdef UAIRT_TFLITE_QNN_DELEGATE
    status = open_delegate(engine);
    TfLiteQnnDelegateHtpPerformanceMode mode;
    if (status == UAIRT_OK && engine->performance_mode &&
        !parse_performance_mode(engine->performance_mode, &mode)) {
      fail("%s must be default, burst, high_performance, sustained, balanced or power_saver",
           OPTION_PERFORMANCE_MODE);
      status = UAIRT_ERR_INVALID_ARGUMENT;
    }
    if (status == UAIRT_OK && engine->precision && strcmp(engine->precision, "quantized") != 0 &&
        strcmp(engine->precision, "fp16") != 0) {
      fail("%s must be quantized or fp16", OPTION_PRECISION);
      status = UAIRT_ERR_INVALID_ARGUMENT;
    }
#else
    fail("this build has no QNN delegate support (configure with QNN_SDK_ROOT)");
    status = UAIRT_ERR_UNSUPPORTED;
#endif
  }
  if (status != UAIRT_OK) {
    destroy_engine(engine);
    return status;
  }
  *out_engine = engine;
  return UAIRT_OK;
}

static void free_io(io_t* io, size_t count) {
  for (size_t i = 0; io && i < count; ++i) {
    free(io[i].name);
  }
  free(io);
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  if (model->interpreter) {
    TfLiteInterpreterDelete(model->interpreter);
  }
#ifdef UAIRT_TFLITE_QNN_DELEGATE
  if (model->delegate) {
    model->engine->delegate_delete(model->delegate);
  }
#endif
  if (model->options) {
    TfLiteInterpreterOptionsDelete(model->options);
  }
  if (model->model) {
    TfLiteModelDelete(model->model);
  }
  free(model->model_data);
  free_io(model->inputs, model->num_inputs);
  free_io(model->outputs, model->num_outputs);
  free(model);
}

static uairt_status describe(const TfLiteTensor* tensor, const char* kind, size_t index, io_t* io) {
  const char* name = TfLiteTensorName(tensor);
  io->name = strdup(name ? name : "");
  if (!io->name) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  int rank = TfLiteTensorNumDims(tensor);
  if (!map_type(TfLiteTensorType(tensor), &dtype) || rank < 0 || rank > UAIRT_MAX_RANK) {
    fail("%s %zu ('%s') has an unsupported type or rank", kind, index, io->name);
    return UAIRT_ERR_UNSUPPORTED;
  }
  io->desc.struct_size = sizeof(uairt_tensor);
  io->desc.domain = UAIRT_MEM_HOST;
  io->desc.dmabuf_fd = -1;
  io->desc.dtype = dtype;
  io->desc.rank = (uint32_t)rank;
  io->desc.name = io->name;
  for (int d = 0; d < rank; ++d) {
    int dim = TfLiteTensorDim(tensor, d);
    if (dim <= 0) {
      fail("%s %zu ('%s') has a dynamic dimension; static shapes only for now", kind, index,
           io->name);
      return UAIRT_ERR_UNSUPPORTED;
    }
    io->desc.dims[d] = dim;
  }
  TfLiteQuantizationParams quantization = TfLiteTensorQuantizationParams(tensor);
  io->desc.quant_scale = quantization.scale;
  io->desc.quant_zero_point = quantization.zero_point;
  return UAIRT_OK;
}

static uairt_status load_io(model_t* model) {
  size_t n_in = (size_t)TfLiteInterpreterGetInputTensorCount(model->interpreter);
  size_t n_out = (size_t)TfLiteInterpreterGetOutputTensorCount(model->interpreter);
  model->inputs = calloc(n_in ? n_in : 1, sizeof(io_t));
  model->outputs = calloc(n_out ? n_out : 1, sizeof(io_t));
  if (!model->inputs || !model->outputs) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  model->num_inputs = n_in;
  model->num_outputs = n_out;
  for (size_t i = 0; i < n_in; ++i) {
    uairt_status status = describe(
        TfLiteInterpreterGetInputTensor(model->interpreter, (int32_t)i), "input", i, &model->inputs[i]);
    if (status != UAIRT_OK) {
      return status;
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    uairt_status status = describe(
        TfLiteInterpreterGetOutputTensor(model->interpreter, (int32_t)i), "output", i,
        &model->outputs[i]);
    if (status != UAIRT_OK) {
      return status;
    }
  }
  return UAIRT_OK;
}

static uairt_status attach_qnn_delegate(model_t* model) {
#ifdef UAIRT_TFLITE_QNN_DELEGATE
  engine_t* engine = model->engine;
  TfLiteQnnDelegateOptions options = engine->options_default();
  options.backend_type = kHtpBackend;
  options.library_path = engine->backend_library;
  options.skel_library_dir = engine->skel_dir ? engine->skel_dir : "";
  options.htp_options.precision =
      engine->precision && strcmp(engine->precision, "fp16") == 0 ? kHtpFp16 : kHtpQuantized;
  if (engine->performance_mode) {
    parse_performance_mode(engine->performance_mode, &options.htp_options.performance_mode);
  }
  model->delegate = engine->delegate_create(&options);
  if (!model->delegate) {
    fail("TfLiteQnnDelegateCreate failed");
    return UAIRT_ERR_RUNTIME;
  }
  TfLiteInterpreterOptionsAddDelegate(model->options, model->delegate);
  return UAIRT_OK;
#else
  (void)model;
  return UAIRT_ERR_UNSUPPORTED;
#endif
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
  model->engine = engine;
  g_tflite_message[0] = '\0';
  uairt_status status = UAIRT_OK;

  if (source->path) {
    FILE* file = fopen(source->path, "rb");
    if (!file) {
      fail("cannot open model file '%s'", source->path);
      status = UAIRT_ERR_IO;
    } else {
      fclose(file);
      model->model = TfLiteModelCreateFromFile(source->path);
    }
  } else {
    model->model_data = malloc(source->size);
    if (!model->model_data) {
      status = UAIRT_ERR_OUT_OF_MEMORY;
    } else {
      memcpy(model->model_data, source->data, source->size);
      model->model = TfLiteModelCreate(model->model_data, source->size);
    }
  }
  if (status == UAIRT_OK && !model->model) {
    fail("not a valid TFLite model");
    status = UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  if (status == UAIRT_OK) {
    model->options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(model->options, engine->num_threads);
    TfLiteInterpreterOptionsSetErrorReporter(model->options, error_reporter, NULL);
    if (engine->use_qnn_delegate) {
      status = attach_qnn_delegate(model);
    }
  }
  if (status == UAIRT_OK) {
    model->interpreter = TfLiteInterpreterCreate(model->model, model->options);
    if (!model->interpreter ||
        TfLiteInterpreterAllocateTensors(model->interpreter) != kTfLiteOk) {
      fail_with_tflite_message("cannot create the interpreter");
      status = UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
  }
  if (status == UAIRT_OK) {
    status = load_io(model);
  }
  if (status != UAIRT_OK) {
    destroy_model(model);
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

static uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  model_t* model = handle;
  g_tflite_message[0] = '\0';
  for (size_t i = 0; i < n_in; ++i) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(model->interpreter, (int32_t)i);
    if (TfLiteTensorCopyFromBuffer(tensor, inputs[i].data, TfLiteTensorByteSize(tensor)) != kTfLiteOk) {
      fail_with_tflite_message("cannot set an input");
      return UAIRT_ERR_RUNTIME;
    }
  }
  if (TfLiteInterpreterInvoke(model->interpreter) != kTfLiteOk) {
    fail_with_tflite_message("invoke failed");
    return UAIRT_ERR_RUNTIME;
  }
  for (size_t i = 0; i < n_out; ++i) {
    const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(model->interpreter, (int32_t)i);
    if (TfLiteTensorCopyToBuffer(tensor, outputs[i].data, TfLiteTensorByteSize(tensor)) != kTfLiteOk) {
      fail_with_tflite_message("cannot read an output");
      return UAIRT_ERR_RUNTIME;
    }
  }
  return UAIRT_OK;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "tflite",
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
