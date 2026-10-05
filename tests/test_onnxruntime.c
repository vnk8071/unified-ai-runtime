/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt.h>

static int g_failures;

#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__,\
              __LINE__, #cond, uairt_last_error());               \
      ++g_failures;                                              \
    }                                                            \
  } while (0)

static void* read_file(const char* path, size_t* size) {
  FILE* file = fopen(path, "rb");
  if (!file) {
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  *size = (size_t)ftell(file);
  fseek(file, 0, SEEK_SET);
  void* data = malloc(*size);
  if (data && fread(data, 1, *size, file) != *size) {
    free(data);
    data = NULL;
  }
  fclose(file);
  return data;
}

static void run_add_mul(uairt_model* model) {
  CHECK(uairt_model_num_inputs(model) == 2);
  CHECK(uairt_model_num_outputs(model) == 2);

  uairt_tensor in[2], out[2];
  for (size_t i = 0; i < 2; ++i) {
    CHECK(uairt_model_input_info(model, i, &in[i]) == UAIRT_OK);
    CHECK(uairt_model_output_info(model, i, &out[i]) == UAIRT_OK);
  }
  CHECK(in[0].dtype == UAIRT_DTYPE_FLOAT32 && in[0].rank == 2);
  CHECK(in[0].dims[0] == 1 && in[0].dims[1] == 4);

  float x[4] = {1, 2, 3, 4};
  float y[4] = {10, 20, 30, 40};
  float sum[4] = {0}, product[4] = {0};
  in[0].data = x;
  in[0].nbytes = sizeof(x);
  in[1].data = y;
  in[1].nbytes = sizeof(y);
  out[0].data = sum;
  out[0].nbytes = sizeof(sum);
  out[1].data = product;
  out[1].nbytes = sizeof(product);

  CHECK(uairt_model_run(model, in, 2, out, 2) == UAIRT_OK);
  for (int i = 0; i < 4; ++i) {
    CHECK(sum[i] == x[i] + y[i]);
    CHECK(product[i] == x[i] * y[i]);
  }

  /* Running twice must give the same result (no state leaks between runs). */
  memset(sum, 0, sizeof(sum));
  CHECK(uairt_model_run(model, in, 2, out, 2) == UAIRT_OK);
  CHECK(sum[3] == 44.f);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <plugin> <add_mul.onnx> <dynamic.onnx>\n", argv[0]);
    return 2;
  }
  const char* plugin = argv[1];
  const char* add_mul_path = argv[2];
  const char* dynamic_path = argv[3];

  CHECK(uairt_load_backend_library(plugin) == UAIRT_OK);

  uairt_option bad_option = {"no_such_option", "1"};
  uairt_engine* engine = NULL;
  CHECK(uairt_engine_create("onnxruntime", &bad_option, 1, &engine) ==
        UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option unknown_provider = {"execution_provider", "no_such_provider"};
  CHECK(uairt_engine_create("onnxruntime", &unknown_provider, 1, &engine) ==
        UAIRT_ERR_UNSUPPORTED);

  uairt_option threads = {"intra_op_threads", "1"};
  CHECK(uairt_engine_create("onnxruntime", &threads, 1, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }

  uairt_model_source from_path = {.struct_size = sizeof(from_path), .path = add_mul_path};
  uairt_model* model = NULL;
  CHECK(uairt_model_load(engine, &from_path, &model) == UAIRT_OK);
  if (model) {
    run_add_mul(model);
    uairt_model_destroy(model);
  }

  size_t size = 0;
  void* bytes = read_file(add_mul_path, &size);
  CHECK(bytes != NULL);
  if (bytes) {
    uairt_model_source from_memory = {
        .struct_size = sizeof(from_memory), .data = bytes, .size = size};
    model = NULL;
    CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_OK);
    if (model) {
      run_add_mul(model);
      uairt_model_destroy(model);
    }
    free(bytes);
  }

  uairt_model_source dynamic = {.struct_size = sizeof(dynamic), .path = dynamic_path};
  CHECK(uairt_model_load(engine, &dynamic, &model) == UAIRT_ERR_UNSUPPORTED);
  CHECK(strstr(uairt_last_error(), "dynamic") != NULL);

  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.onnx"};
  CHECK(uairt_model_load(engine, &missing, &model) != UAIRT_OK);
  CHECK(uairt_last_error()[0] != '\0');

  uairt_engine_destroy(engine);

  /* Plugin execution provider options are validated before anything is loaded. */
  uairt_option no_name = {"ep_library", "/some/lib.dylib"};
  CHECK(uairt_engine_create("onnxruntime", &no_name, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option name_only = {"ep_name", "SomeExecutionProvider"};
  CHECK(uairt_engine_create("onnxruntime", &name_only, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option ep_only = {"ep_option.x", "1"};
  CHECK(uairt_engine_create("onnxruntime", &ep_only, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option missing_library[] = {{"ep_library", "/no/such/ep.dylib"}, {"ep_name", "SomeExecutionProvider"}};
  CHECK(uairt_engine_create("onnxruntime", missing_library, 2, &engine) != UAIRT_OK);
  CHECK(uairt_last_error()[0] != '\0');

  /* Optional: a real plugin execution provider, from UAIRT_TEST_ORT_EP_LIBRARY and UAIRT_TEST_ORT_EP_NAME. */
  const char* ep_library = getenv("UAIRT_TEST_ORT_EP_LIBRARY");
  const char* ep_name = getenv("UAIRT_TEST_ORT_EP_NAME");
  if (ep_library && ep_name) {
    uairt_option plugin[] = {{"ep_library", ep_library}, {"ep_name", ep_name}};
    CHECK(uairt_engine_create("onnxruntime", plugin, 2, &engine) == UAIRT_OK);
    if (engine) {
      uairt_model* plugin_model = NULL;
      CHECK(uairt_model_load(engine, &from_path, &plugin_model) == UAIRT_OK);
      if (plugin_model) {
        run_add_mul(plugin_model);
        uairt_model_destroy(plugin_model);
      }
      uairt_engine_destroy(engine);
    }
  }

  /* Optional: a built-in GPU provider ("cuda" or "tensorrt") from UAIRT_TEST_ORT_PROVIDER; needs a GPU build. */
  const char* provider = getenv("UAIRT_TEST_ORT_PROVIDER");
  if (provider && provider[0]) {
    uairt_option gpu[] = {{"execution_provider", provider}};
    CHECK(uairt_engine_create("onnxruntime", gpu, 1, &engine) == UAIRT_OK);
    if (engine) {
      uairt_model* gpu_model = NULL;
      CHECK(uairt_model_load(engine, &from_path, &gpu_model) == UAIRT_OK);
      if (gpu_model) {
        run_add_mul(gpu_model);
        uairt_model_destroy(gpu_model);
      }
      uairt_engine_destroy(engine);
    }
    uairt_option bad_combo[] = {{"execution_provider", provider}, {"ep_name", "x"}};
    CHECK(uairt_engine_create("onnxruntime", bad_combo, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  }

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all onnxruntime checks passed");
  return 0;
}
