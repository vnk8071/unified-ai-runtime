/* SPDX-License-Identifier: Apache-2.0 */
/*
 * QNN (Qualcomm AI Runtime) backend, built as a loadable plugin.
 * v1 limits: DLC models only, one graph per model, host memory, static shapes.
 * The QNN backend and system libraries are loaded at runtime from paths given as
 * engine options; QNN headers come from the user's QAIRT SDK (QNN_SDK_ROOT).
 */
#define _POSIX_C_SOURCE 200809L
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define strdup _strdup
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <HTP/QnnHtpDevice.h>
#include <QnnInterface.h>
#include <System/QnnSystemDlc.h>
#include <System/QnnSystemInterface.h>
#include <uairt/uairt_backend.h>

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

#define OPTION_BACKEND_LIBRARY "backend_library"
#define OPTION_SYSTEM_LIBRARY "system_library"
#define OPTION_GRAPH_NAME "graph_name"
#define OPTION_ADSP_LIBRARY_PATH "adsp_library_path"
#define OPTION_HTP_PERFORMANCE_MODE "htp_performance_mode"
#define OPTION_DEVICE "device"
#define OPTION_SDK_ROOT "sdk_root"
#define OPTION_TARGET "target"
#define OPTION_HEXAGON_ARCH "hexagon_arch"
#define OPTION_CACHE_DIR "cache_dir"
#define OPTION_CACHE_WRITE "cache_write"
#define DEFAULT_HEXAGON_ARCH "v73"
#if defined(_WIN32)
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif
#if defined(_WIN32)
#define RPCMEM_LIBRARY "libcdsprpc.dll"
#else
#define RPCMEM_LIBRARY "libcdsprpc.so"
#endif
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

typedef void* (*rpcmem_alloc_fn)(int heap_id, uint32_t flags, int size);
typedef void (*rpcmem_free_fn)(void* data);
typedef int (*rpcmem_to_fd_fn)(void* data);

#define QNN(engine) ((engine)->provider->QNN_INTERFACE_VER_NAME)
#define QNN_SYSTEM(engine) ((engine)->system_provider->QNN_SYSTEM_INTERFACE_VER_NAME)

typedef enum {
  PERF_DEFAULT,
  PERF_BURST,
  PERF_HIGH_PERFORMANCE,
  PERF_BALANCED,
  PERF_POWER_SAVER,
} perf_mode_t;

typedef struct {
  void* backend_library;
  void* system_library;
  const QnnInterface_t* provider;
  const QnnSystemInterface_t* system_provider;
  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device;
  char* graph_name;
  char* cache_dir;
  bool cache_write;
  bool is_htp;
  char* backend_path;
  perf_mode_t perf_mode;
  bool has_power_config;
  uint32_t power_config_id;
  QnnHtpDevice_PerfInfrastructure_t perf_infra;
  void* rpcmem_library;
  rpcmem_alloc_fn rpcmem_alloc;
  rpcmem_free_fn rpcmem_free;
  rpcmem_to_fd_fn rpcmem_to_fd;
} engine_t;

typedef struct {
  char* name;
  Qnn_Tensor_t tensor;
  uint32_t dims[UAIRT_MAX_RANK];
  uairt_tensor desc;
  size_t nbytes;
  Qnn_MemHandle_t mem_handle;
  int32_t registered_fd;
} io_t;

typedef struct {
  engine_t* engine;
  Qnn_ContextHandle_t context;
  QnnSystemDlc_Handle_t dlc;
  Qnn_GraphHandle_t graph;
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
  char message[320];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

static uairt_status check(const engine_t* engine, Qnn_ErrorHandle_t error, const char* what) {
  if (error == QNN_SUCCESS) {
    return UAIRT_OK;
  }
  const char* message = NULL;
  if (engine && QNN(engine).errorGetMessage) {
    QNN(engine).errorGetMessage(error, &message);
  }
  fail("%s failed: %s (QNN error %llu)", what, message ? message : "no message",
       (unsigned long long)error);
  return error == QNN_COMMON_ERROR_NOT_SUPPORTED ? UAIRT_ERR_UNSUPPORTED : UAIRT_ERR_RUNTIME;
}

static bool map_data_type(Qnn_DataType_t type, uairt_dtype* dtype, size_t* size) {
  switch (type) {
    case QNN_DATATYPE_FLOAT_32:
      *dtype = UAIRT_DTYPE_FLOAT32, *size = 4;
      return true;
    case QNN_DATATYPE_FLOAT_16:
      *dtype = UAIRT_DTYPE_FLOAT16, *size = 2;
      return true;
    case QNN_DATATYPE_BFLOAT_16:
      *dtype = UAIRT_DTYPE_BFLOAT16, *size = 2;
      return true;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      *dtype = UAIRT_DTYPE_INT8, *size = 1;
      return true;
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      *dtype = UAIRT_DTYPE_UINT8, *size = 1;
      return true;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
      *dtype = UAIRT_DTYPE_INT16, *size = 2;
      return true;
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      *dtype = UAIRT_DTYPE_UINT16, *size = 2;
      return true;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
      *dtype = UAIRT_DTYPE_INT32, *size = 4;
      return true;
    case QNN_DATATYPE_INT_64:
      *dtype = UAIRT_DTYPE_INT64, *size = 8;
      return true;
    case QNN_DATATYPE_BOOL_8:
      *dtype = UAIRT_DTYPE_BOOL, *size = 1;
      return true;
    default:
      return false;
  }
}

typedef struct {
  const char* name;
  Qnn_DataType_t data_type;
  Qnn_QuantizeParams_t quant;
  uint32_t rank;
  const uint32_t* dims;
  const uint8_t* dynamic;
} tensor_view_t;

static bool view_tensor(const Qnn_Tensor_t* tensor, tensor_view_t* view) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1:
      view->name = tensor->v1.name;
      view->data_type = tensor->v1.dataType;
      view->quant = tensor->v1.quantizeParams;
      view->rank = tensor->v1.rank;
      view->dims = tensor->v1.dimensions;
      view->dynamic = NULL;
      return true;
    case QNN_TENSOR_VERSION_2:
      view->name = tensor->v2.name;
      view->data_type = tensor->v2.dataType;
      view->quant = tensor->v2.quantizeParams;
      view->rank = tensor->v2.rank;
      view->dims = tensor->v2.dimensions;
      view->dynamic = tensor->v2.isDynamicDimensions;
      return true;
    default:
      return false;
  }
}

static void bind_buffer(Qnn_Tensor_t* tensor, void* data, size_t nbytes) {
  Qnn_ClientBuffer_t buffer = {.data = data, .dataSize = (uint32_t)nbytes};
  if (tensor->version == QNN_TENSOR_VERSION_1) {
    tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v1.clientBuf = buffer;
  } else {
    tensor->v2.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v2.clientBuf = buffer;
  }
}

static uairt_status describe_io(const Qnn_Tensor_t* source, const char* kind, io_t* io) {
  tensor_view_t view;
  if (!view_tensor(source, &view)) {
    fail("%s tensor uses an unsupported QNN tensor version", kind);
    return UAIRT_ERR_UNSUPPORTED;
  }
  io->name = strdup(view.name ? view.name : "");
  if (!io->name) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  size_t element_size = 0;
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  if (!map_data_type(view.data_type, &dtype, &element_size) || view.rank > UAIRT_MAX_RANK) {
    fail("%s '%s' has an unsupported data type (0x%x) or rank %u", kind, io->name,
         (unsigned)view.data_type, view.rank);
    return UAIRT_ERR_UNSUPPORTED;
  }
  memset(&io->desc, 0, sizeof(io->desc));
  io->desc.struct_size = sizeof(uairt_tensor);
  io->desc.domain = UAIRT_MEM_HOST;
  io->desc.dmabuf_fd = -1;
  io->desc.dtype = dtype;
  io->desc.rank = view.rank;
  io->desc.name = io->name;
  io->nbytes = element_size;
  for (uint32_t i = 0; i < view.rank; ++i) {
    if (view.dynamic && view.dynamic[i]) {
      fail("%s '%s' has a dynamic dimension %u; static shapes only for now", kind,
           io->name, i);
      return UAIRT_ERR_UNSUPPORTED;
    }
    io->dims[i] = view.dims[i];
    io->desc.dims[i] = view.dims[i];
    io->nbytes *= view.dims[i];
  }
  if (view.quant.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    io->desc.quant_scale = view.quant.scaleOffsetEncoding.scale;
    io->desc.quant_zero_point = -view.quant.scaleOffsetEncoding.offset;
  }

  io->tensor = *source;
  if (io->tensor.version == QNN_TENSOR_VERSION_1) {
    io->tensor.v1.dimensions = io->dims;
    io->tensor.v1.name = io->name;
  } else {
    io->tensor.v2.dimensions = io->dims;
    io->tensor.v2.name = io->name;
    io->tensor.v2.isDynamicDimensions = NULL;
  }
  return UAIRT_OK;
}

static uairt_status load_io(
    const Qnn_Tensor_t* tensors,
    uint32_t count,
    const char* kind,
    io_t** out_io) {
  io_t* io = calloc(count ? count : 1, sizeof(*io));
  if (!io) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  *out_io = io;
  for (uint32_t i = 0; i < count; ++i) {
    uairt_status status = describe_io(&tensors[i], kind, &io[i]);
    if (status != UAIRT_OK) {
      return status;
    }
  }
  return UAIRT_OK;
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

static void destroy_engine(void* handle) {
  engine_t* engine = handle;
  if (engine->has_power_config) {
    engine->perf_infra.destroyPowerConfigId(engine->power_config_id);
  }
  if (engine->device) {
    QNN(engine).deviceFree(engine->device);
  }
  if (engine->backend) {
    QNN(engine).backendFree(engine->backend);
  }
  free(engine->graph_name);
  free(engine->cache_dir);
  free(engine->backend_path);
  if (engine->rpcmem_library) {
    uairt_dlclose(engine->rpcmem_library);
  }
  if (engine->system_library) {
    uairt_dlclose(engine->system_library);
  }
  if (engine->backend_library) {
    uairt_dlclose(engine->backend_library);
  }
  free(engine);
}

static char* join_path(const char* a, const char* b, const char* c) {
  size_t length = strlen(a) + strlen(b) + (c ? strlen(c) : 0) + 3;
  char* out = malloc(length);
  if (out) {
    snprintf(out, length, "%s%s%s%s%s", a, PATH_SEPARATOR, b, c ? PATH_SEPARATOR : "", c ? c : "");
  }
  return out;
}

static bool contains_ignore_case(const char* text, const char* needle) {
  size_t length = strlen(needle);
  for (; *text; ++text) {
    size_t i = 0;
    while (i < length && text[i] &&
           tolower((unsigned char)text[i]) == tolower((unsigned char)needle[i])) {
      ++i;
    }
    if (i == length) {
      return true;
    }
  }
  return false;
}

typedef struct {
  char* backend;
  char* system;
  char* skel_dir;
} sdk_paths_t;

static void free_sdk_paths(sdk_paths_t* paths) {
  free(paths->backend);
  free(paths->system);
  free(paths->skel_dir);
}

/* Maps device=cpu|gpu|npu to the SDK's backend and system libraries and the Hexagon skel directory. */
static uairt_status resolve_sdk_paths(
    const char* device,
    const char* sdk_root,
    const char* target,
    const char* arch,
    sdk_paths_t* paths) {
  const char* name = strcmp(device, "cpu") == 0   ? "QnnCpu"
                     : strcmp(device, "gpu") == 0 ? "QnnGpu"
                     : strcmp(device, "npu") == 0 ? "QnnHtp"
                                                  : NULL;
  if (!name) {
    fail("%s must be auto, cpu, gpu or npu", OPTION_DEVICE);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (!sdk_root || !*sdk_root) {
    fail("'%s' needs the '%s' option or the QNN_SDK_ROOT environment variable", OPTION_DEVICE,
         OPTION_SDK_ROOT);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (!target) {
#if defined(_WIN32) && (defined(_M_ARM64) || defined(__aarch64__))
    target = "aarch64-windows-msvc";
#else
    fail("'%s' needs the '%s' option (the lib/<target> directory name) on this platform",
         OPTION_DEVICE, OPTION_TARGET);
    return UAIRT_ERR_INVALID_ARGUMENT;
#endif
  }
#if defined(_WIN32)
  const char* prefix = "";
  const char* suffix = ".dll";
#else
  const char* prefix = "lib";
  const char* suffix = ".so";
#endif
  char* lib_dir = join_path(sdk_root, "lib", target);
  if (!lib_dir) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  char file[64];
  snprintf(file, sizeof(file), "%s%s%s", prefix, name, suffix);
  paths->backend = join_path(lib_dir, file, NULL);
  snprintf(file, sizeof(file), "%sQnnSystem%s", prefix, suffix);
  paths->system = join_path(lib_dir, file, NULL);
  free(lib_dir);
  if (strcmp(device, "npu") == 0) {
    char hexagon[32];
    snprintf(hexagon, sizeof(hexagon), "hexagon-%s", arch);
    char* hexagon_dir = join_path(sdk_root, "lib", hexagon);
    paths->skel_dir = hexagon_dir ? join_path(hexagon_dir, "unsigned", NULL) : NULL;
    free(hexagon_dir);
  }
  return paths->backend && paths->system ? UAIRT_OK : UAIRT_ERR_OUT_OF_MEMORY;
}

static bool parse_bool(const char* value, bool* out) {
  if (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 || strcmp(value, "on") == 0) {
    *out = true;
    return true;
  }
  if (strcmp(value, "0") == 0 || strcmp(value, "false") == 0 || strcmp(value, "off") == 0) {
    *out = false;
    return true;
  }
  return false;
}

static bool version_matches(Qnn_Version_t version, uint32_t major, uint32_t minor) {
  return version.major == major && version.minor == minor;
}

static uairt_status open_providers(engine_t* engine, const char* backend_path, const char* system_path) {
  engine->backend_library = uairt_dlopen(backend_path);
  if (!engine->backend_library) {
    fail("cannot load QNN backend library: %s", uairt_dlerror());
    return UAIRT_ERR_BACKEND_UNAVAILABLE;
  }
  engine->system_library = uairt_dlopen(system_path);
  if (!engine->system_library) {
    fail("cannot load QNN system library: %s", uairt_dlerror());
    return UAIRT_ERR_BACKEND_UNAVAILABLE;
  }

  typedef Qnn_ErrorHandle_t (*get_providers_fn)(const QnnInterface_t***, uint32_t*);
  typedef Qnn_ErrorHandle_t (*get_system_providers_fn)(const QnnSystemInterface_t***, uint32_t*);
  get_providers_fn get_providers;
  get_system_providers_fn get_system_providers;
  void* symbol = uairt_dlsym(engine->backend_library, "QnnInterface_getProviders");
  memcpy(&get_providers, &symbol, sizeof(symbol));
  symbol = uairt_dlsym(engine->system_library, "QnnSystemInterface_getProviders");
  memcpy(&get_system_providers, &symbol, sizeof(symbol));
  if (!get_providers || !get_system_providers) {
    fail("QNN library does not export its provider entry points");
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }

  const QnnInterface_t** providers = NULL;
  uint32_t count = 0;
  if (get_providers(&providers, &count) != QNN_SUCCESS || !providers) {
    fail("QnnInterface_getProviders failed");
    return UAIRT_ERR_RUNTIME;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (version_matches(providers[i]->apiVersion.coreApiVersion, QNN_API_VERSION_MAJOR,
                        QNN_API_VERSION_MINOR)) {
      engine->provider = providers[i];
      break;
    }
  }
  if (!engine->provider) {
    fail("backend has no QNN API %u.%u provider (built against these headers)",
         QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
    return UAIRT_ERR_VERSION_MISMATCH;
  }

  const QnnSystemInterface_t** system_providers = NULL;
  count = 0;
  if (get_system_providers(&system_providers, &count) != QNN_SUCCESS || !system_providers) {
    fail("QnnSystemInterface_getProviders failed");
    return UAIRT_ERR_RUNTIME;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (version_matches(system_providers[i]->systemApiVersion, QNN_SYSTEM_API_VERSION_MAJOR,
                        QNN_SYSTEM_API_VERSION_MINOR)) {
      engine->system_provider = system_providers[i];
      break;
    }
  }
  if (!engine->system_provider) {
    fail("system library has no QNN system API %u.%u provider", QNN_SYSTEM_API_VERSION_MAJOR,
         QNN_SYSTEM_API_VERSION_MINOR);
    return UAIRT_ERR_VERSION_MISMATCH;
  }
  return UAIRT_OK;
}

static bool parse_perf_mode(const char* value, perf_mode_t* mode) {
  static const struct {
    const char* name;
    perf_mode_t mode;
  } kModes[] = {
      {"default", PERF_DEFAULT},
      {"burst", PERF_BURST},
      {"high_performance", PERF_HIGH_PERFORMANCE},
      {"balanced", PERF_BALANCED},
      {"power_saver", PERF_POWER_SAVER},
  };
  for (size_t i = 0; i < sizeof(kModes) / sizeof(kModes[0]); ++i) {
    if (strcmp(value, kModes[i].name) == 0) {
      *mode = kModes[i].mode;
      return true;
    }
  }
  return false;
}

/*
 * Maps a mode to HTP DCVS v3 votes. These are UAIRT's own settings, not Qualcomm's
 * definitions of the same names. The votes last as long as the engine does.
 */
static uairt_status apply_performance_mode(engine_t* engine) {
  if (engine->perf_mode == PERF_DEFAULT) {
    return UAIRT_OK;
  }
  if (!engine->device || !QNN(engine).deviceGetInfrastructure) {
    fail("'%s' needs the HTP backend", OPTION_HTP_PERFORMANCE_MODE);
    return UAIRT_ERR_UNSUPPORTED;
  }
  QnnDevice_Infrastructure_t infrastructure = NULL;
  uairt_status status = check(engine, QNN(engine).deviceGetInfrastructure(&infrastructure),
                             "QnnDevice_getInfrastructure");
  if (status != UAIRT_OK) {
    return status;
  }
  const QnnHtpDevice_Infrastructure_t* htp = (const QnnHtpDevice_Infrastructure_t*)infrastructure;
  if (!htp || htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF ||
      !htp->perfInfra.createPowerConfigId || !htp->perfInfra.setPowerConfig ||
      !htp->perfInfra.destroyPowerConfigId) {
    fail("'%s' needs the HTP backend", OPTION_HTP_PERFORMANCE_MODE);
    return UAIRT_ERR_UNSUPPORTED;
  }
  engine->perf_infra = htp->perfInfra;
  status = check(engine, engine->perf_infra.createPowerConfigId(0, 0, &engine->power_config_id),
                 "createPowerConfigId");
  if (status != UAIRT_OK) {
    return status;
  }
  engine->has_power_config = true;

  QnnHtpPerfInfrastructure_VoltageCorner_t corner_min, corner_target, corner_max;
  QnnHtpPerfInfrastructure_PowerMode_t power_mode;
  uint32_t dcvs_enable, sleep_disable, sleep_latency;
  bool polling = false;
  switch (engine->perf_mode) {
    case PERF_BURST:
      corner_min = corner_target = corner_max = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
      power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
      dcvs_enable = 0, sleep_disable = 1, sleep_latency = 40, polling = true;
      break;
    case PERF_HIGH_PERFORMANCE:
      corner_min = corner_target = corner_max = DCVS_VOLTAGE_VCORNER_TURBO;
      power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
      dcvs_enable = 0, sleep_disable = 0, sleep_latency = 100, polling = true;
      break;
    case PERF_BALANCED:
      corner_min = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      corner_target = DCVS_VOLTAGE_VCORNER_NOM;
      corner_max = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN;
      dcvs_enable = 1, sleep_disable = 0, sleep_latency = 1000;
      break;
    default:
      corner_min = DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER;
      corner_target = DCVS_VOLTAGE_VCORNER_SVS;
      corner_max = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
      dcvs_enable = 1, sleep_disable = 0, sleep_latency = 1000;
      break;
  }

  QnnHtpPerfInfrastructure_PowerConfig_t dcvs = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  dcvs.dcvsV3Config.contextId = engine->power_config_id;
  dcvs.dcvsV3Config.setDcvsEnable = 1;
  dcvs.dcvsV3Config.dcvsEnable = dcvs_enable;
  dcvs.dcvsV3Config.powerMode = power_mode;
  dcvs.dcvsV3Config.setSleepLatency = 1;
  dcvs.dcvsV3Config.sleepLatency = sleep_latency;
  dcvs.dcvsV3Config.setSleepDisable = 1;
  dcvs.dcvsV3Config.sleepDisable = sleep_disable;
  dcvs.dcvsV3Config.setBusParams = 1;
  dcvs.dcvsV3Config.busVoltageCornerMin = corner_min;
  dcvs.dcvsV3Config.busVoltageCornerTarget = corner_target;
  dcvs.dcvsV3Config.busVoltageCornerMax = corner_max;
  dcvs.dcvsV3Config.setCoreParams = 1;
  dcvs.dcvsV3Config.coreVoltageCornerMin = corner_min;
  dcvs.dcvsV3Config.coreVoltageCornerTarget = corner_target;
  dcvs.dcvsV3Config.coreVoltageCornerMax = corner_max;

  QnnHtpPerfInfrastructure_PowerConfig_t polling_time = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  polling_time.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  polling_time.rpcPollingTimeConfig = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME;
  QnnHtpPerfInfrastructure_PowerConfig_t control_latency = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT;
  control_latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  control_latency.rpcControlLatencyConfig = 100;

  const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = {&dcvs, NULL, NULL, NULL};
  if (polling) {
    configs[1] = &polling_time;
    configs[2] = &control_latency;
  }
  return check(engine, engine->perf_infra.setPowerConfig(engine->power_config_id, configs),
               "setPowerConfig");
}

static uairt_status create_engine(
    const uairt_option* options,
    size_t num_options,
    void** out_engine) {
  const char* backend_path = NULL;
  const char* system_path = NULL;
  const char* graph_name = NULL;
  const char* adsp_path = NULL;
  const char* device = "auto";
  const char* sdk_root = getenv("QNN_SDK_ROOT");
  const char* target = NULL;
  const char* hexagon_arch = DEFAULT_HEXAGON_ARCH;
  const char* cache_dir = NULL;
#if defined(_WIN32)
  bool cache_write = false;
#else
  bool cache_write = true;
#endif
  perf_mode_t perf_mode = PERF_DEFAULT;
  for (size_t i = 0; i < num_options; ++i) {
    if (strcmp(options[i].key, OPTION_BACKEND_LIBRARY) == 0) {
      backend_path = options[i].value;
    } else if (strcmp(options[i].key, OPTION_SYSTEM_LIBRARY) == 0) {
      system_path = options[i].value;
    } else if (strcmp(options[i].key, OPTION_GRAPH_NAME) == 0) {
      graph_name = options[i].value;
    } else if (strcmp(options[i].key, OPTION_ADSP_LIBRARY_PATH) == 0) {
      adsp_path = options[i].value;
    } else if (strcmp(options[i].key, OPTION_DEVICE) == 0) {
      device = options[i].value;
    } else if (strcmp(options[i].key, OPTION_SDK_ROOT) == 0) {
      sdk_root = options[i].value;
    } else if (strcmp(options[i].key, OPTION_TARGET) == 0) {
      target = options[i].value;
    } else if (strcmp(options[i].key, OPTION_HEXAGON_ARCH) == 0) {
      hexagon_arch = options[i].value;
    } else if (strcmp(options[i].key, OPTION_CACHE_DIR) == 0) {
      cache_dir = options[i].value;
    } else if (strcmp(options[i].key, OPTION_CACHE_WRITE) == 0) {
      if (!options[i].value || !parse_bool(options[i].value, &cache_write)) {
        fail("%s must be true or false", OPTION_CACHE_WRITE);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (strcmp(options[i].key, OPTION_HTP_PERFORMANCE_MODE) == 0) {
      if (!options[i].value || !parse_perf_mode(options[i].value, &perf_mode)) {
        fail("%s must be default, burst, high_performance, balanced or power_saver",
             OPTION_HTP_PERFORMANCE_MODE);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else {
      fail("unknown option '%s'", options[i].key);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  sdk_paths_t resolved = {0};
  if (strcmp(device, "auto") != 0) {
    uairt_status resolve_status =
        resolve_sdk_paths(device, sdk_root, target, hexagon_arch, &resolved);
    if (resolve_status != UAIRT_OK) {
      free_sdk_paths(&resolved);
      return resolve_status;
    }
    if (!backend_path) {
      backend_path = resolved.backend;
    }
    if (!system_path) {
      system_path = resolved.system;
    }
    if (!adsp_path && !getenv("ADSP_LIBRARY_PATH")) {
      adsp_path = resolved.skel_dir;
    }
  }
  if (!backend_path || !system_path) {
    fail("options '%s' and '%s' are required (paths to the QNN backend and system libraries), "
         "or use '%s'",
         OPTION_BACKEND_LIBRARY, OPTION_SYSTEM_LIBRARY, OPTION_DEVICE);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    free_sdk_paths(&resolved);
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->is_htp = contains_ignore_case(backend_path, "QnnHtp");
  engine->cache_write = cache_write;
  if (cache_dir && *cache_dir) {
    engine->cache_dir = strdup(cache_dir);
  }
  engine->backend_path = strdup(backend_path);
  if (adsp_path) {
#if defined(_WIN32)
    _putenv_s("ADSP_LIBRARY_PATH", adsp_path);
#else
    setenv("ADSP_LIBRARY_PATH", adsp_path, 1);
#endif
  }
  engine->perf_mode = perf_mode;
  if (graph_name) {
    engine->graph_name = strdup(graph_name);
  }
  uairt_status status = open_providers(engine, backend_path, system_path);
  free_sdk_paths(&resolved);
  if (status == UAIRT_OK) {
    status = check(engine, QNN(engine).backendCreate(NULL, NULL, &engine->backend),
                   "QnnBackend_create");
  }
  if (status == UAIRT_OK && QNN(engine).deviceCreate) {
    Qnn_ErrorHandle_t error = QNN(engine).deviceCreate(NULL, NULL, &engine->device);
    if (error == QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      engine->device = NULL;
    } else {
      status = check(engine, error, "QnnDevice_create");
    }
  }
  if (status == UAIRT_OK) {
    status = apply_performance_mode(engine);
  }
  if (status != UAIRT_OK) {
    destroy_engine(engine);
    return status;
  }
  *out_engine = engine;
  return UAIRT_OK;
}

static void deregister_io(model_t* model, io_t* io, size_t count) {
  for (size_t i = 0; io && i < count; ++i) {
    if (io[i].mem_handle) {
      QNN(model->engine).memDeRegister(&io[i].mem_handle, 1);
      io[i].mem_handle = NULL;
    }
  }
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  deregister_io(model, model->inputs, model->num_inputs);
  deregister_io(model, model->outputs, model->num_outputs);
  if (model->context) {
    QNN(model->engine).contextFree(model->context, NULL);
  }
  if (model->dlc) {
    QNN_SYSTEM(model->engine).systemDlcFree(model->dlc);
  }
  free_io(model->inputs, model->num_inputs);
  free_io(model->outputs, model->num_outputs);
  free(model);
}

static bool graph_info_fields(
    const QnnSystemContext_GraphInfo_t* info,
    const char** name,
    uint32_t* num_inputs,
    Qnn_Tensor_t** inputs,
    uint32_t* num_outputs,
    Qnn_Tensor_t** outputs) {
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1:
      *name = info->graphInfoV1.graphName;
      *num_inputs = info->graphInfoV1.numGraphInputs;
      *inputs = info->graphInfoV1.graphInputs;
      *num_outputs = info->graphInfoV1.numGraphOutputs;
      *outputs = info->graphInfoV1.graphOutputs;
      return true;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
      *name = info->graphInfoV2.graphName;
      *num_inputs = info->graphInfoV2.numGraphInputs;
      *inputs = info->graphInfoV2.graphInputs;
      *num_outputs = info->graphInfoV2.numGraphOutputs;
      *outputs = info->graphInfoV2.graphOutputs;
      return true;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
      *name = info->graphInfoV3.graphName;
      *num_inputs = info->graphInfoV3.numGraphInputs;
      *inputs = info->graphInfoV3.graphInputs;
      *num_outputs = info->graphInfoV3.numGraphOutputs;
      *outputs = info->graphInfoV3.graphOutputs;
      return true;
    default:
      return false;
  }
}

static bool has_suffix(const char* text, const char* suffix) {
  size_t length = strlen(text);
  size_t suffix_length = strlen(suffix);
  return length >= suffix_length && strcmp(text + length - suffix_length, suffix) == 0;
}

static const char* model_format(const uairt_model_source* source) {
  if (source->format) {
    return source->format;
  }
  if (source->path && has_suffix(source->path, ".dlc")) {
    return "dlc";
  }
  if (source->path && has_suffix(source->path, ".bin")) {
    return "context-binary";
  }
  return NULL;
}

static uairt_status select_graph(
    const engine_t* engine,
    const QnnSystemContext_GraphInfo_t* graphs,
    uint32_t count,
    const QnnSystemContext_GraphInfo_t** out) {
  if (count == 0) {
    fail("the model contains no graphs");
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  if (!engine->graph_name && count == 1) {
    *out = &graphs[0];
    return UAIRT_OK;
  }
  char names[256] = "";
  size_t used = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const char* name = NULL;
    uint32_t num_in = 0, num_out = 0;
    Qnn_Tensor_t *in = NULL, *out_tensors = NULL;
    if (!graph_info_fields(&graphs[i], &name, &num_in, &in, &num_out, &out_tensors) || !name) {
      continue;
    }
    if (engine->graph_name && strcmp(name, engine->graph_name) == 0) {
      *out = &graphs[i];
      return UAIRT_OK;
    }
    if (used < sizeof(names)) {
      used += (size_t)snprintf(names + used, sizeof(names) - used, "%s%s", used ? ", " : "", name);
    }
  }
  if (engine->graph_name) {
    fail("graph '%s' not found; available: %s", engine->graph_name, names);
  } else {
    fail("the model has %u graphs; select one with option '%s' (available: %s)", count,
         OPTION_GRAPH_NAME, names);
  }
  return UAIRT_ERR_INVALID_ARGUMENT;
}

static uairt_status bind_graph(
    engine_t* engine,
    model_t* model,
    const QnnSystemContext_GraphInfo_t* info,
    bool finalize) {
  const char* graph_name = NULL;
  uint32_t num_inputs = 0, num_outputs = 0;
  Qnn_Tensor_t *inputs = NULL, *outputs = NULL;
  if (!graph_info_fields(info, &graph_name, &num_inputs, &inputs, &num_outputs, &outputs)) {
    fail("unsupported QNN graph info version");
    return UAIRT_ERR_UNSUPPORTED;
  }
  uairt_status status = check(
      engine, QNN(engine).graphRetrieve(model->context, graph_name, &model->graph),
      "QnnGraph_retrieve");
  if (status == UAIRT_OK && finalize) {
    status = check(engine, QNN(engine).graphFinalize(model->graph, NULL, NULL),
                   "QnnGraph_finalize");
  }
  if (status != UAIRT_OK) {
    return status;
  }
  model->num_inputs = num_inputs;
  model->num_outputs = num_outputs;
  status = load_io(inputs, num_inputs, "input", &model->inputs);
  if (status == UAIRT_OK) {
    status = load_io(outputs, num_outputs, "output", &model->outputs);
  }
  return status;
}

static uairt_status load_dlc(engine_t* engine, model_t* model, const uairt_model_source* source) {
  if (source->path) {
    FILE* file = fopen(source->path, "rb");
    if (!file) {
      fail("cannot open model file '%s'", source->path);
      return UAIRT_ERR_IO;
    }
    fclose(file);
  }
  QnnSystemContext_GraphInfo_t* graphs = NULL;
  uint32_t num_graphs = 0;
  uairt_status status = check(
      engine,
      QNN(engine).contextCreate(engine->backend, engine->device, NULL, &model->context),
      "QnnContext_create");
  if (status != UAIRT_OK) {
    return status;
  }
  Qnn_ErrorHandle_t error =
      source->path ? QNN_SYSTEM(engine).systemDlcCreateFromFile(NULL, source->path, &model->dlc)
                   : QNN_SYSTEM(engine).systemDlcCreateFromBinary(
                         NULL, (const uint8_t*)source->data, (uint32_t)source->size, &model->dlc);
  if (check(engine, error, "QnnSystemDlc_create") != UAIRT_OK) {
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  status = check(
      engine,
      QNN_SYSTEM(engine).systemDlcComposeGraphs(
          model->dlc, NULL, 0, engine->backend, model->context, *engine->provider,
          QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &graphs, &num_graphs),
      "QnnSystemDlc_composeGraphs");
  if (status != UAIRT_OK) {
    return status;
  }
  const QnnSystemContext_GraphInfo_t* graph = NULL;
  status = select_graph(engine, graphs, num_graphs, &graph);
  return status == UAIRT_OK ? bind_graph(engine, model, graph, true) : status;
}

#if defined(_WIN32)
static uairt_status map_file(const char* path, const void** out, size_t* size) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    fail("cannot open model file '%s'", path);
    return UAIRT_ERR_IO;
  }
  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  void* data = length > 0 ? malloc((size_t)length) : NULL;
  if (!data || fread(data, 1, (size_t)length, file) != (size_t)length) {
    free(data);
    fclose(file);
    fail("cannot read model file '%s'", path);
    return UAIRT_ERR_IO;
  }
  fclose(file);
  *out = data;
  *size = (size_t)length;
  return UAIRT_OK;
}

static void unmap_file(const void* data, size_t size) {
  (void)size;
  free((void*)data);
}
#else
static uairt_status map_file(const char* path, const void** out, size_t* size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fail("cannot open model file '%s'", path);
    return UAIRT_ERR_IO;
  }
  struct stat info;
  if (fstat(fd, &info) != 0 || info.st_size <= 0) {
    close(fd);
    fail("cannot read model file '%s'", path);
    return UAIRT_ERR_IO;
  }
  void* mapping = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED) {
    fail("cannot map model file '%s'", path);
    return UAIRT_ERR_IO;
  }
  *out = mapping;
  *size = (size_t)info.st_size;
  return UAIRT_OK;
}

static void unmap_file(const void* data, size_t size) {
  munmap((void*)data, size);
}
#endif

static uairt_status load_context_binary(
    engine_t* engine,
    model_t* model,
    const uairt_model_source* source) {
  const void* buffer = source->data;
  size_t size = source->size;
  bool mapped = false;
  if (source->path) {
    uairt_status status = map_file(source->path, &buffer, &size);
    if (status != UAIRT_OK) {
      return status;
    }
    mapped = true;
  }

  QnnSystemContext_Handle_t system_context = NULL;
  const QnnSystemContext_BinaryInfo_t* info = NULL;
  Qnn_ContextBinarySize_t info_size = 0;
  uairt_status status = check(engine, QNN_SYSTEM(engine).systemContextCreate(&system_context),
                             "QnnSystemContext_create");
  if (status == UAIRT_OK &&
      check(engine,
            QNN_SYSTEM(engine).systemContextGetBinaryInfo(system_context, (void*)buffer, size,
                                                          &info, &info_size),
            "QnnSystemContext_getBinaryInfo") != UAIRT_OK) {
    status = UAIRT_ERR_INCOMPATIBLE_MODEL;
  }

  const QnnSystemContext_GraphInfo_t* graphs = NULL;
  uint32_t num_graphs = 0;
  if (status == UAIRT_OK) {
    switch (info->version) {
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
        graphs = info->contextBinaryInfoV1.graphs;
        num_graphs = info->contextBinaryInfoV1.numGraphs;
        break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
        graphs = info->contextBinaryInfoV2.graphs;
        num_graphs = info->contextBinaryInfoV2.numGraphs;
        break;
      case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
        graphs = info->contextBinaryInfoV3.graphs;
        num_graphs = info->contextBinaryInfoV3.numGraphs;
        break;
      default:
        fail("unsupported QNN context binary info version");
        status = UAIRT_ERR_UNSUPPORTED;
    }
  }
  const QnnSystemContext_GraphInfo_t* graph = NULL;
  if (status == UAIRT_OK) {
    status = select_graph(engine, graphs, num_graphs, &graph);
  }
  if (status == UAIRT_OK) {
    status = check(engine,
                   QNN(engine).contextCreateFromBinary(engine->backend, engine->device, NULL,
                                                       buffer, size, &model->context, NULL),
                   "QnnContext_createFromBinary");
    if (status != UAIRT_OK) {
      status = UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
  }
  if (status == UAIRT_OK) {
    status = bind_graph(engine, model, graph, false);
  }

  if (system_context) {
    QNN_SYSTEM(engine).systemContextFree(system_context);
  }
  if (mapped) {
    unmap_file(buffer, size);
  }
  return status;
}

/* A compiled-context cache for DLC models: composing a DLC on the device can take tens of seconds. */
static bool file_signature(const char* path, uint64_t* size, uint64_t* mtime) {
#if defined(_WIN32)
  struct _stat64 info;
  if (_stat64(path, &info) != 0) {
    return false;
  }
#else
  struct stat info;
  if (stat(path, &info) != 0) {
    return false;
  }
#endif
  *size = (uint64_t)info.st_size;
  *mtime = (uint64_t)info.st_mtime;
  return true;
}

static uint64_t fnv1a(uint64_t hash, const void* data, size_t length) {
  const unsigned char* bytes = data;
  for (size_t i = 0; i < length; ++i) {
    hash = (hash ^ bytes[i]) * 1099511628211ULL;
  }
  return hash;
}

/* The key covers the model path, size and mtime plus the backend library, so a changed model or SDK misses. */
static char* cache_file_path(const engine_t* engine, const char* model_path) {
  uint64_t size, mtime;
  if (!engine->cache_dir || !engine->is_htp || !model_path ||
      !file_signature(model_path, &size, &mtime)) {
    return NULL;
  }
  uint64_t hash = 14695981039346656037ULL;
  hash = fnv1a(hash, model_path, strlen(model_path));
  hash = fnv1a(hash, &size, sizeof(size));
  hash = fnv1a(hash, &mtime, sizeof(mtime));
  hash = fnv1a(hash, engine->backend_path, strlen(engine->backend_path));
  char name[40];
  snprintf(name, sizeof(name), "%016llx.qnn_ctx.bin", (unsigned long long)hash);
  return join_path(engine->cache_dir, name, NULL);
}

static void write_cache(engine_t* engine, model_t* model, const char* cache_path) {
  Qnn_ContextBinarySize_t size = 0, written = 0;
  if (!QNN(engine).contextGetBinarySize || !QNN(engine).contextGetBinary ||
      QNN(engine).contextGetBinarySize(model->context, &size) != QNN_SUCCESS || size == 0) {
    return;
  }
  void* buffer = malloc((size_t)size);
  if (!buffer) {
    return;
  }
  size_t tmp_length = strlen(cache_path) + 5;
  char* tmp_path = malloc(tmp_length);
  if (tmp_path &&
      QNN(engine).contextGetBinary(model->context, buffer, size, &written) == QNN_SUCCESS &&
      written > 0) {
    snprintf(tmp_path, tmp_length, "%s.tmp", cache_path);
    FILE* file = fopen(tmp_path, "wb");
    bool ok = file && fwrite(buffer, 1, (size_t)written, file) == (size_t)written;
    if (file) {
      ok = (fclose(file) == 0) && ok;
    }
    if (ok) {
      remove(cache_path);
      ok = rename(tmp_path, cache_path) == 0;
    }
    if (!ok) {
      remove(tmp_path);
    }
  }
  free(tmp_path);
  free(buffer);
}

static uairt_status load_cached_dlc(engine_t* engine, const char* cache_path, model_t* model) {
  FILE* file = fopen(cache_path, "rb");
  if (!file) {
    return UAIRT_ERR_NOT_FOUND;
  }
  fclose(file);
  uairt_model_source cached;
  memset(&cached, 0, sizeof(cached));
  cached.struct_size = sizeof(cached);
  cached.path = cache_path;
  cached.format = "context-binary";
  return load_context_binary(engine, model, &cached);
}

static uairt_status load_model(
    void* engine_handle,
    const uairt_model_source* source,
    void** out_model) {
  engine_t* engine = engine_handle;
  const char* format = model_format(source);
  bool is_dlc = format && strcmp(format, "dlc") == 0;
  bool is_binary = format && strcmp(format, "context-binary") == 0;
  if (!is_dlc && !is_binary) {
    fail("unknown model format; use a .dlc or .bin path, or format \"dlc\" / \"context-binary\"");
    return UAIRT_ERR_UNSUPPORTED;
  }
  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  model->engine = engine;
  char* cache_path = is_dlc ? cache_file_path(engine, source->path) : NULL;
  uairt_status status = UAIRT_ERR_NOT_FOUND;
  if (cache_path) {
    status = load_cached_dlc(engine, cache_path, model);
    if (status != UAIRT_OK) {
      /* A missing or stale cache entry is not an error: recompile from the DLC. */
      destroy_model(model);
      model = calloc(1, sizeof(*model));
      if (!model) {
        free(cache_path);
        return UAIRT_ERR_OUT_OF_MEMORY;
      }
      model->engine = engine;
    }
  }
  if (status != UAIRT_OK) {
    status = is_dlc ? load_dlc(engine, model, source) : load_context_binary(engine, model, source);
    if (status == UAIRT_OK && cache_path && engine->cache_write) {
      write_cache(engine, model, cache_path);
    }
  }
  free(cache_path);
  if (status != UAIRT_OK) {
    destroy_model(model);
    return status;
  }
  *out_model = model;
  return UAIRT_OK;
}

static size_t num_inputs_fn(const void* handle) {
  return ((const model_t*)handle)->num_inputs;
}

static size_t num_outputs_fn(const void* handle) {
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

static void bind_handle(Qnn_Tensor_t* tensor, Qnn_MemHandle_t handle) {
  if (tensor->version == QNN_TENSOR_VERSION_1) {
    tensor->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor->v1.memHandle = handle;
  } else {
    tensor->v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
    tensor->v2.memHandle = handle;
  }
}

/* One registration per I/O, reused while the same buffer is bound. */
static uairt_status ensure_registered(model_t* model, io_t* io, int32_t fd) {
  if (io->mem_handle && io->registered_fd == fd) {
    return UAIRT_OK;
  }
  if (io->mem_handle) {
    QNN(model->engine).memDeRegister(&io->mem_handle, 1);
    io->mem_handle = NULL;
  }
  tensor_view_t view;
  if (!view_tensor(&io->tensor, &view)) {
    return UAIRT_ERR_UNSUPPORTED;
  }
  Qnn_MemDescriptor_t descriptor;
  memset(&descriptor, 0, sizeof(descriptor));
  descriptor.memShape.numDim = io->desc.rank;
  descriptor.memShape.dimSize = io->dims;
  descriptor.dataType = view.data_type;
  descriptor.memType = QNN_MEM_TYPE_ION;
  descriptor.ionInfo.fd = fd;
  uairt_status status =
      check(model->engine,
            QNN(model->engine).memRegister(model->context, &descriptor, 1, &io->mem_handle),
            "QnnMem_register");
  if (status == UAIRT_OK) {
    io->registered_fd = fd;
  }
  return status;
}

static uairt_status load_rpcmem(engine_t* engine) {
  if (engine->rpcmem_alloc) {
    return UAIRT_OK;
  }
  engine->rpcmem_library = uairt_dlopen(RPCMEM_LIBRARY);
  if (!engine->rpcmem_library) {
    fail("cannot load %s (needed for DMABUF buffers): %s", RPCMEM_LIBRARY, uairt_dlerror());
    return UAIRT_ERR_UNSUPPORTED;
  }
  void* symbol = uairt_dlsym(engine->rpcmem_library, "rpcmem_alloc");
  memcpy(&engine->rpcmem_alloc, &symbol, sizeof(symbol));
  symbol = uairt_dlsym(engine->rpcmem_library, "rpcmem_free");
  memcpy(&engine->rpcmem_free, &symbol, sizeof(symbol));
  symbol = uairt_dlsym(engine->rpcmem_library, "rpcmem_to_fd");
  memcpy(&engine->rpcmem_to_fd, &symbol, sizeof(symbol));
  if (!engine->rpcmem_alloc || !engine->rpcmem_free || !engine->rpcmem_to_fd) {
    engine->rpcmem_alloc = NULL;
    fail("%s does not export the rpcmem functions", RPCMEM_LIBRARY);
    return UAIRT_ERR_UNSUPPORTED;
  }
  return UAIRT_OK;
}

static uairt_status alloc_buffer(
    void* engine_handle,
    size_t nbytes,
    uairt_memory_domain domain,
    void** out_handle,
    void** out_data,
    int32_t* out_fd) {
  engine_t* engine = engine_handle;
  if (domain != UAIRT_MEM_DMABUF || nbytes > 0x7fffffffu) {
    fail("rpcmem buffers must be DMABUF and smaller than 2 GiB");
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  uairt_status status = load_rpcmem(engine);
  if (status != UAIRT_OK) {
    return status;
  }
  void* data = engine->rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)nbytes);
  if (!data) {
    fail("rpcmem_alloc failed for %zu bytes", nbytes);
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  int fd = engine->rpcmem_to_fd(data);
  if (fd < 0) {
    engine->rpcmem_free(data);
    fail("rpcmem_to_fd failed");
    return UAIRT_ERR_RUNTIME;
  }
  *out_handle = data;
  *out_data = data;
  *out_fd = fd;
  return UAIRT_OK;
}

static void free_buffer(void* engine_handle, void* handle) {
  engine_t* engine = engine_handle;
  engine->rpcmem_free(handle);
}

static uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  model_t* model = handle;
  Qnn_Tensor_t* in = calloc(n_in ? n_in : 1, sizeof(*in));
  Qnn_Tensor_t* out = calloc(n_out ? n_out : 1, sizeof(*out));
  uairt_status status = UAIRT_OK;
  if (!in || !out) {
    status = UAIRT_ERR_OUT_OF_MEMORY;
    goto cleanup;
  }
  for (size_t i = 0; i < n_in; ++i) {
    in[i] = model->inputs[i].tensor;
    if (inputs[i].domain == UAIRT_MEM_DMABUF) {
      status = ensure_registered(model, &model->inputs[i], inputs[i].dmabuf_fd);
      if (status != UAIRT_OK) {
        goto cleanup;
      }
      bind_handle(&in[i], model->inputs[i].mem_handle);
    } else {
      bind_buffer(&in[i], inputs[i].data, model->inputs[i].nbytes);
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    out[i] = model->outputs[i].tensor;
    if (outputs[i].domain == UAIRT_MEM_DMABUF) {
      status = ensure_registered(model, &model->outputs[i], outputs[i].dmabuf_fd);
      if (status != UAIRT_OK) {
        goto cleanup;
      }
      bind_handle(&out[i], model->outputs[i].mem_handle);
    } else {
      bind_buffer(&out[i], outputs[i].data, model->outputs[i].nbytes);
    }
  }
  status = check(model->engine,
                 QNN(model->engine).graphExecute(model->graph, in, (uint32_t)n_in, out,
                                                 (uint32_t)n_out, NULL, NULL),
                 "QnnGraph_execute");

cleanup:
  free(in);
  free(out);
  return status;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "qnn",
    .supported_domains = UAIRT_MEM_HOST | UAIRT_MEM_DMABUF,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_inputs_fn,
    .num_outputs = num_outputs_fn,
    .input_info = input_info,
    .output_info = output_info,
    .run = run,
    .alloc_buffer = alloc_buffer,
    .free_buffer = free_buffer,
};

UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return &kApi;
}
