/* SPDX-License-Identifier: Apache-2.0 */
#include <math.h>
#include <stdio.h>
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

static int near(float a, float b) {
  return fabsf(a - b) < 1e-6f;
}

/* Loads add_mul.mnn with the given engine options, feeds x and y of `n` elements, and checks sum and product. */
static void run_add_mul(const uairt_option* options, size_t num_options, const char* path, int n) {
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  CHECK(uairt_engine_create("mnn", options, num_options, &engine) == UAIRT_OK);
  if (!engine) {
    return;
  }
  uairt_model_source source = {.struct_size = sizeof(source), .path = path};
  CHECK(uairt_model_load(engine, &source, &model) == UAIRT_OK);
  if (!model) {
    uairt_engine_destroy(engine);
    return;
  }
  /* MNN numbers inputs and outputs by name: x, y and product, sum. */
  CHECK(uairt_model_num_inputs(model) == 2 && uairt_model_num_outputs(model) == 2);
  uairt_tensor in[2], out[2];
  CHECK(uairt_model_input_info(model, 0, &in[0]) == UAIRT_OK && strcmp(in[0].name, "x") == 0);
  CHECK(uairt_model_input_info(model, 1, &in[1]) == UAIRT_OK && strcmp(in[1].name, "y") == 0);
  CHECK(uairt_model_output_info(model, 0, &out[0]) == UAIRT_OK && strcmp(out[0].name, "product") == 0);
  CHECK(uairt_model_output_info(model, 1, &out[1]) == UAIRT_OK && strcmp(out[1].name, "sum") == 0);
  CHECK(in[0].dtype == UAIRT_DTYPE_FLOAT32 && in[0].rank == 2 && in[0].dims[1] == n && in[0].nbytes == (size_t)n * 4);

  float x[16], y[16], product[16], sum[16];
  for (int i = 0; i < n; ++i) {
    x[i] = (float)(i + 1);
    y[i] = (float)(10 * (i + 1));
  }
  in[0].data = x;
  in[1].data = y;
  out[0].data = product;
  out[1].data = sum;
  CHECK(uairt_model_run(model, in, 2, out, 2) == UAIRT_OK);
  for (int i = 0; i < n; ++i) {
    CHECK(near(sum[i], x[i] + y[i]) && near(product[i], x[i] * y[i]));
  }
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

/* argv[2] is a file that is not a model, argv[3] is tests/data/add_mul.mnn. */
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <plugin> <non-model file> <add_mul.mnn>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("mnn", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_device = {"device", "tpu"};
  CHECK(uairt_engine_create("mnn", &bad_device, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_threads = {"num_threads", "0"};
  CHECK(uairt_engine_create("mnn", &bad_threads, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_precision = {"precision", "fp64"};
  CHECK(uairt_engine_create("mnn", &bad_precision, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_shape = {"input_shapes", "1xbad"};
  CHECK(uairt_engine_create("mnn", &bad_shape, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(engine == NULL);

  CHECK(uairt_engine_create("mnn", NULL, 0, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.mnn"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.mnn") != NULL);
  uairt_model_source not_a_model = {.struct_size = sizeof(not_a_model), .path = argv[2]};
  CHECK(uairt_model_load(engine, &not_a_model, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);
  char garbage[64] = {1, 2, 3};
  uairt_model_source from_garbage = {.struct_size = sizeof(from_garbage), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_garbage, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);
  CHECK(model == NULL);
  uairt_engine_destroy(engine);

  /* A device this MNN was not built with is an error, not a quiet run on the CPU. */
  uairt_option cuda = {"device", "cuda"};
  CHECK(uairt_engine_create("mnn", &cuda, 1, &engine) == UAIRT_OK);
  if (engine) {
    uairt_model_source real = {.struct_size = sizeof(real), .path = argv[3]};
    CHECK(uairt_model_load(engine, &real, &model) == UAIRT_ERR_BACKEND_UNAVAILABLE);
    uairt_engine_destroy(engine);
  }

  run_add_mul(NULL, 0, argv[3], 4);
  uairt_option threads[] = {{"num_threads", "2"}, {"precision", "high"}};
  run_add_mul(threads, 2, argv[3], 4);
  uairt_option wider = {"input_shapes", "1x8;1x8"};  /* resizes both inputs: the outputs follow */
  run_add_mul(&wider, 1, argv[3], 8);
  uairt_option wrong_count = {"input_shapes", "1x8"};
  CHECK(uairt_engine_create("mnn", &wrong_count, 1, &engine) == UAIRT_OK);
  if (engine) {
    uairt_model_source real = {.struct_size = sizeof(real), .path = argv[3]};
    CHECK(uairt_model_load(engine, &real, &model) == UAIRT_ERR_INVALID_ARGUMENT);
    uairt_engine_destroy(engine);
  }

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all mnn checks passed");
  fflush(stdout);
  return 0;
}
