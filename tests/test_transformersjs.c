/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt.h>

static int g_failures;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__, __LINE__,      \
              #cond, uairt_last_error());                                         \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

static uairt_status load(uairt_engine* engine, const char* path, uairt_model** model) {
  uairt_model_source source = {.struct_size = sizeof(source), .path = path};
  return uairt_model_load(engine, &source, model);
}

/* Runs the add_mul model directory: sum = x + y and product = x * y on four floats. */
static void run_add_mul(uairt_model* model) {
  CHECK(uairt_model_num_inputs(model) == 2 && uairt_model_num_outputs(model) == 2);
  uairt_tensor info;
  uairt_tensor_init(&info);
  CHECK(uairt_model_input_info(model, 0, &info) == UAIRT_OK && info.name && strcmp(info.name, "x") == 0);
  CHECK(uairt_model_output_info(model, 1, &info) == UAIRT_OK && info.name && strcmp(info.name, "product") == 0);
  float x[4] = {1, 2, 3, 4};
  float y[4] = {10, 20, 30, 40};
  float sum[4] = {0};
  float product[4] = {0};
  uairt_tensor in[2], out[2];
  for (int i = 0; i < 2; ++i) {
    uairt_tensor_init(&in[i]);
    uairt_tensor_init(&out[i]);
    CHECK(uairt_model_input_info(model, (size_t)i, &in[i]) == UAIRT_OK);
    CHECK(uairt_model_output_info(model, (size_t)i, &out[i]) == UAIRT_OK);
    in[i].data = i == 0 ? x : y;
    in[i].nbytes = sizeof(x);
    out[i].data = i == 0 ? sum : product;
    out[i].nbytes = sizeof(sum);
  }
  CHECK(uairt_model_run(model, in, 2, out, 2) == UAIRT_OK);
  for (int i = 0; i < 4; ++i) {
    CHECK(sum[i] == x[i] + y[i]);
    CHECK(product[i] == x[i] * y[i]);
  }
}

/* argv[2] is the add_mul model directory (config.json and onnx/model.onnx), argv[3] tests/data/fake_transformersjs_worker.mjs. With UAIRT_TRANSFORMERSJS_DIR set to a directory that has
 * @huggingface/transformers installed, the test also runs the model; without it only the error paths run. */
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <plugin> <add_mul model directory> <stand-in worker>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("transformersjs", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_device = {"device", "cuda"};
  CHECK(uairt_engine_create("transformersjs", &bad_device, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option newline = {"package_dir", "a\nb"};
  CHECK(uairt_engine_create("transformersjs", &newline, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);

  uairt_model* model = NULL;
  CHECK(uairt_engine_create("transformersjs", NULL, 0, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  CHECK(load(engine, "/no/such/model", &model) == UAIRT_ERR_NOT_FOUND);
  CHECK(strstr(uairt_last_error(), "/no/such/model") != NULL);
  char garbage[16] = {1, 2, 3};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_UNSUPPORTED);
  uairt_engine_destroy(engine);

  uairt_option no_node[] = {{"node", "/no/such/node"}};
  CHECK(uairt_engine_create("transformersjs", no_node, 1, &engine) == UAIRT_OK);
  CHECK(load(engine, argv[2], &model) == UAIRT_ERR_BACKEND_UNAVAILABLE);
  uairt_engine_destroy(engine);

  uairt_option no_worker[] = {{"worker", "/no/such/worker.mjs"}};
  CHECK(uairt_engine_create("transformersjs", no_worker, 1, &engine) == UAIRT_OK);
  CHECK(load(engine, argv[2], &model) == UAIRT_ERR_BACKEND_UNAVAILABLE);
  CHECK(strstr(uairt_last_error(), "worker") != NULL);
  uairt_engine_destroy(engine);

  uairt_option no_package[] = {{"package_dir", "/no/such/dir"}};
  CHECK(uairt_engine_create("transformersjs", no_package, 1, &engine) == UAIRT_OK);
  CHECK(load(engine, argv[2], &model) == UAIRT_ERR_BACKEND_UNAVAILABLE);
  CHECK(strstr(uairt_last_error(), "@huggingface/transformers") != NULL);
  uairt_engine_destroy(engine);

  /* a stand-in worker (argv[3]) that logs no WebGPU placement: cpu loads and echoes, webgpu is refused */
  {
    uairt_option fake_cpu[] = {{"worker", argv[3]}, {"device", "cpu"}};
    CHECK(uairt_engine_create("transformersjs", fake_cpu, 2, &engine) == UAIRT_OK);
    CHECK(load(engine, argv[2], &model) == UAIRT_OK);
    if (model) {
      float in_data[4] = {1, 2, 3, 4};
      float out_data[4] = {0};
      uairt_tensor in, out;
      uairt_tensor_init(&in);
      uairt_tensor_init(&out);
      CHECK(uairt_model_input_info(model, 0, &in) == UAIRT_OK && uairt_model_output_info(model, 0, &out) == UAIRT_OK);
      in.data = in_data;
      in.nbytes = sizeof(in_data);
      out.data = out_data;
      out.nbytes = sizeof(out_data);
      CHECK(uairt_model_run(model, &in, 1, &out, 1) == UAIRT_OK);
      CHECK(memcmp(in_data, out_data, sizeof(in_data)) == 0);
      uairt_model_destroy(model);
      model = NULL;
    }
    uairt_engine_destroy(engine);
    uairt_option fake_webgpu[] = {{"worker", argv[3]}, {"device", "webgpu"}};
    CHECK(uairt_engine_create("transformersjs", fake_webgpu, 2, &engine) == UAIRT_OK);
    CHECK(load(engine, argv[2], &model) == UAIRT_ERR_BACKEND_UNAVAILABLE);
    CHECK(strstr(uairt_last_error(), "WebGpuExecutionProvider") != NULL);
    uairt_engine_destroy(engine);
  }

  const char* package_dir = getenv("UAIRT_TRANSFORMERSJS_DIR");
  if (package_dir && package_dir[0]) {
    uairt_option cpu[] = {{"package_dir", package_dir}, {"device", "cpu"}};
    CHECK(uairt_engine_create("transformersjs", cpu, 2, &engine) == UAIRT_OK);
    if (engine) {
      CHECK(load(engine, argv[2], &model) == UAIRT_OK);
      if (model) {
        run_add_mul(model);
        uairt_model_destroy(model);
      }
      uairt_engine_destroy(engine);
    }
    /* webgpu either runs on WebGPU or is refused; it never runs on the CPU quietly */
    uairt_option webgpu[] = {{"package_dir", package_dir}, {"device", "webgpu"}};
    CHECK(uairt_engine_create("transformersjs", webgpu, 2, &engine) == UAIRT_OK);
    if (engine) {
      uairt_status status = load(engine, argv[2], &model);
      CHECK(status == UAIRT_OK || status == UAIRT_ERR_BACKEND_UNAVAILABLE);
      if (status == UAIRT_OK) {
        run_add_mul(model);
        uairt_model_destroy(model);
      }
      uairt_engine_destroy(engine);
    }
  } else {
    puts("UAIRT_TRANSFORMERSJS_DIR not set: the model run was skipped");
  }

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all transformersjs checks passed");
  return 0;
}
