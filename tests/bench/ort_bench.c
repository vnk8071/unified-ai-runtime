/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Times an ONNX model through ONNX Runtime with the QNN execution provider, so it can be
 * compared with UAIRT's QNN backend. The ONNX Runtime core library is loaded with dlopen.
 * Needs the ONNX Runtime C header (1.24 or later). Two ways to get the QNN EP:
 *   - a core library built with the QNN EP compiled in: omit --plugin, the EP is added by name;
 *   - the onnxruntime-qnn plugin library: pass --plugin (see the limitation below).
 *
 *   ort_bench --core libonnxruntime.so [--plugin libonnxruntime_providers_qnn.so]
 *             [--reg-name name] [--by-name] [--log 0-4] [--opt key=value]...
 *             <model.onnx> <input.bin> <out-prefix> <runs>
 *
 * --by-name appends the provider with the legacy by-name call instead of choosing an EP device.
 *
 * Provider options use the QNN EP names, for example backend_path=...libQnnHtp.so and
 * htp_performance_mode=burst. Outputs go to <out-prefix><index>.bin (in model order).
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <onnxruntime_c_api.h>

#define MAX_OPTIONS 32

static const OrtApi* g_ort;

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

static int compare_doubles(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static void check(OrtStatus* status, const char* what) {
  if (status) {
    fprintf(stderr, "%s failed: %s\n", what, g_ort->GetErrorMessage(status));
    exit(1);
  }
}

static void* read_file(const char* path, size_t* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *size = (size_t)ftell(f);
  fseek(f, 0, SEEK_SET);
  void* data = malloc(*size);
  if (data && fread(data, 1, *size, f) != *size) {
    free(data);
    data = NULL;
  }
  fclose(f);
  return data;
}

int main(int argc, char** argv) {
  const char *core = NULL, *plugin = NULL, *reg_name = "qnn_plugin";
  int log_level = ORT_LOGGING_LEVEL_WARNING;
  int by_name = 0;
  const char* keys[MAX_OPTIONS];
  const char* values[MAX_OPTIONS];
  size_t num_options = 0;
  int arg = 1;
  for (; arg + 1 < argc && argv[arg][0] == '-'; arg += 2) {
    if (!strcmp(argv[arg], "--by-name")) { by_name = 1; arg -= 1; }
    else if (!strcmp(argv[arg], "--reg-name")) reg_name = argv[arg + 1];
    else if (!strcmp(argv[arg], "--log")) log_level = atoi(argv[arg + 1]);
    else if (!strcmp(argv[arg], "--core")) core = argv[arg + 1];
    else if (!strcmp(argv[arg], "--plugin")) plugin = argv[arg + 1];
    else if (!strcmp(argv[arg], "--opt") && num_options < MAX_OPTIONS) {
      char* pair = argv[arg + 1];
      char* equals = strchr(pair, '=');
      if (!equals) { fprintf(stderr, "--opt expects key=value\n"); return 2; }
      *equals = '\0';
      keys[num_options] = pair;
      values[num_options++] = equals + 1;
    } else { fprintf(stderr, "unknown option %s\n", argv[arg]); return 2; }
  }
  if (!core || argc - arg != 4) {
    fprintf(stderr, "usage: %s --core <libonnxruntime.so> [--plugin <provider.so>] [--opt k=v]... "
                    "<model.onnx> <input.bin> <out-prefix> <runs>\n", argv[0]);
    return 2;
  }
  const char *model_path = argv[arg], *input_path = argv[arg + 1], *prefix = argv[arg + 2];
  int runs = atoi(argv[arg + 3]);

  void* library = dlopen(core, RTLD_NOW | RTLD_LOCAL);
  if (!library) { fprintf(stderr, "cannot load %s: %s\n", core, dlerror()); return 1; }
  const OrtApiBase* (*get_base)(void);
  void* symbol = dlsym(library, "OrtGetApiBase");
  memcpy(&get_base, &symbol, sizeof(symbol));
  g_ort = get_base()->GetApi(ORT_API_VERSION);
  if (!g_ort) { fprintf(stderr, "ONNX Runtime %s is older than the header API %d\n", get_base()->GetVersionString(), ORT_API_VERSION); return 1; }

  double start = now_ms();
  if (!plugin) by_name = 1;
  OrtEnv* env;
  check(g_ort->CreateEnv((OrtLoggingLevel)log_level, "ort_bench", &env), "CreateEnv");
  const OrtEpDevice* const* devices = NULL;
  size_t num_devices = 0;
  if (plugin) {
    check(g_ort->RegisterExecutionProviderLibrary(env, reg_name, plugin), "RegisterExecutionProviderLibrary");
    check(g_ort->GetEpDevices(env, &devices, &num_devices), "GetEpDevices");
  }
  const OrtEpDevice* qnn = NULL;
  for (size_t i = 0; i < num_devices; ++i) {
    printf("ep device %zu: %s\n", i, g_ort->EpDevice_EpName(devices[i]));
    if (!qnn && strstr(g_ort->EpDevice_EpName(devices[i]), "QNN")) qnn = devices[i];
  }
  if (!qnn && !by_name) { fprintf(stderr, "no QNN execution provider device found (try --by-name)\n"); return 1; }

  OrtSessionOptions* options;
  check(g_ort->CreateSessionOptions(&options), "CreateSessionOptions");
  if (by_name) {
    check(g_ort->SessionOptionsAppendExecutionProvider(options, plugin ? "QNNExecutionProvider" : "QNN", keys, values, num_options),
          "SessionOptionsAppendExecutionProvider");
  } else {
    check(g_ort->SessionOptionsAppendExecutionProvider_V2(options, env, &qnn, 1, keys, values, num_options),
          "SessionOptionsAppendExecutionProvider_V2");
  }
  OrtSession* session;
  check(g_ort->CreateSession(env, model_path, options, &session), "CreateSession");
  double init_ms = now_ms() - start;

  OrtAllocator* allocator;
  check(g_ort->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions");
  size_t n_in, n_out;
  check(g_ort->SessionGetInputCount(session, &n_in), "SessionGetInputCount");
  check(g_ort->SessionGetOutputCount(session, &n_out), "SessionGetOutputCount");
  if (n_in != 1) { fprintf(stderr, "this benchmark takes models with one input, got %zu\n", n_in); return 1; }

  char *in_name, **out_names = calloc(n_out, sizeof(char*));
  check(g_ort->SessionGetInputName(session, 0, allocator, &in_name), "SessionGetInputName");
  for (size_t i = 0; i < n_out; ++i) check(g_ort->SessionGetOutputName(session, i, allocator, &out_names[i]), "SessionGetOutputName");

  OrtTypeInfo* type_info;
  check(g_ort->SessionGetInputTypeInfo(session, 0, &type_info), "SessionGetInputTypeInfo");
  const OrtTensorTypeAndShapeInfo* tensor_info;
  check(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info), "CastTypeInfoToTensorInfo");
  ONNXTensorElementDataType element_type;
  size_t rank;
  check(g_ort->GetTensorElementType(tensor_info, &element_type), "GetTensorElementType");
  check(g_ort->GetDimensionsCount(tensor_info, &rank), "GetDimensionsCount");
  int64_t dims[8];
  check(g_ort->GetDimensions(tensor_info, dims, rank), "GetDimensions");

  size_t size = 0;
  void* input = read_file(input_path, &size);
  if (!input) { fprintf(stderr, "cannot read %s\n", input_path); return 1; }
  OrtMemoryInfo* memory_info;
  check(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), "CreateCpuMemoryInfo");
  OrtValue* input_value;
  check(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input, size, dims, rank, element_type, &input_value),
        "CreateTensorWithDataAsOrtValue");

  OrtValue** outputs = calloc(n_out, sizeof(OrtValue*));
  const char* in_names[1] = {in_name};
  check(g_ort->Run(session, NULL, in_names, (const OrtValue* const*)&input_value, 1, (const char* const*)out_names, n_out, outputs), "Run");
  for (size_t i = 0; i < n_out; ++i) g_ort->ReleaseValue(outputs[i]), outputs[i] = NULL;

  double* times = malloc((size_t)(runs ? runs : 1) * sizeof(*times));
  for (int r = 0; r < runs; ++r) {
    for (size_t i = 0; i < n_out; ++i) outputs[i] = NULL;
    double t0 = now_ms();
    check(g_ort->Run(session, NULL, in_names, (const OrtValue* const*)&input_value, 1, (const char* const*)out_names, n_out, outputs), "Run");
    times[r] = now_ms() - t0;
    if (r + 1 < runs) for (size_t i = 0; i < n_out; ++i) g_ort->ReleaseValue(outputs[i]);
  }
  for (size_t i = 0; i < n_out; ++i) {
    OrtTensorTypeAndShapeInfo* info;
    check(g_ort->GetTensorTypeAndShape(outputs[i], &info), "GetTensorTypeAndShape");
    size_t count;
    check(g_ort->GetTensorShapeElementCount(info, &count), "GetTensorShapeElementCount");
    ONNXTensorElementDataType out_type;
    check(g_ort->GetTensorElementType(info, &out_type), "GetTensorElementType");
    size_t element_size = out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ? 1 : out_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 1;
    void* data;
    check(g_ort->GetTensorMutableData(outputs[i], &data), "GetTensorMutableData");
    printf("output %zu name=%s type=%d elements=%zu\n", i, out_names[i], (int)out_type, count);
    char path[1024];
    snprintf(path, sizeof(path), "%s%zu.bin", prefix, i);
    FILE* f = fopen(path, "wb");
    fwrite(data, element_size, count, f);
    fclose(f);
    g_ort->ReleaseTensorTypeAndShapeInfo(info);
  }
  if (runs > 0) {
    qsort(times, (size_t)runs, sizeof(*times), compare_doubles);
    double sum = 0;
    for (int r = 0; r < runs; ++r) sum += times[r];
    fprintf(stderr, "init_ms=%.1f latency_ms runs=%d min=%.2f median=%.2f mean=%.2f\n", init_ms, runs,
            times[0], times[runs / 2], sum / runs);
  }
  return 0;
}
