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

/* argv[2] is a file that is not a model, argv[3] is tests/data/add_mul.nb: inputs x and y [1, 4], outputs x + y and x * y. */
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <plugin> <non-model file> <add_mul.nb>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option shapes = {"input_shapes", "1x4;1x4"};
  uairt_option bad_option[] = {{"no_such_option", "1"}, shapes};
  CHECK(uairt_engine_create("paddlelite", bad_option, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_engine_create("paddlelite", NULL, 0, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "input_shapes") != NULL);
  uairt_option bad_shape = {"input_shapes", "1xbad"};
  CHECK(uairt_engine_create("paddlelite", &bad_shape, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_threads[] = {{"num_threads", "0"}, shapes};
  CHECK(uairt_engine_create("paddlelite", bad_threads, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_power[] = {{"power_mode", "turbo"}, shapes};
  CHECK(uairt_engine_create("paddlelite", bad_power, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_precision[] = {{"precision", "fp64"}, shapes};
  CHECK(uairt_engine_create("paddlelite", bad_precision, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(engine == NULL);

  uairt_model* model = NULL;
  CHECK(uairt_engine_create("paddlelite", &shapes, 1, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.nb"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.nb") != NULL);
  uairt_model_source not_a_model = {.struct_size = sizeof(not_a_model), .path = argv[2]};
  CHECK(uairt_model_load(engine, &not_a_model, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);
  CHECK(model == NULL);
  uairt_engine_destroy(engine);

  /* the wrong number of shapes for the model's two inputs */
  uairt_option one_shape = {"input_shapes", "1x4"};
  CHECK(uairt_engine_create("paddlelite", &one_shape, 1, &engine) == UAIRT_OK);
  uairt_model_source real = {.struct_size = sizeof(real), .path = argv[3]};
  CHECK(uairt_model_load(engine, &real, &model) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_engine_destroy(engine);

  /* a real run, with two threads, from the path and from memory */
  for (int from_memory = 0; from_memory < 2; ++from_memory) {
    uairt_option options[] = {shapes, {"num_threads", "2"}};
    CHECK(uairt_engine_create("paddlelite", options, 2, &engine) == UAIRT_OK);
    if (!engine) {
      return 1;
    }
    static char buffer[1 << 16];
    uairt_model_source source = real;
    if (from_memory) {
      FILE* f = fopen(argv[3], "rb");
      size_t n = f ? fread(buffer, 1, sizeof(buffer), f) : 0;
      if (f) {
        fclose(f);
      }
      source = (uairt_model_source){.struct_size = sizeof(source), .data = buffer, .size = n};
      CHECK(n > 0);
    }
    CHECK(uairt_model_load(engine, &source, &model) == UAIRT_OK);
    if (!model) {
      uairt_engine_destroy(engine);
      return 1;
    }
    CHECK(uairt_model_num_inputs(model) == 2 && uairt_model_num_outputs(model) == 2);
    uairt_tensor in[2], out[2];
    CHECK(uairt_model_input_info(model, 0, &in[0]) == UAIRT_OK && strcmp(in[0].name, "x") == 0);
    CHECK(uairt_model_input_info(model, 1, &in[1]) == UAIRT_OK && strcmp(in[1].name, "y") == 0);
    CHECK(uairt_model_output_info(model, 0, &out[0]) == UAIRT_OK && uairt_model_output_info(model, 1, &out[1]) == UAIRT_OK);
    CHECK(in[0].dtype == UAIRT_DTYPE_FLOAT32 && in[0].rank == 2 && in[0].dims[1] == 4 && in[0].nbytes == 16);
    CHECK(out[0].rank == 2 && out[0].dims[1] == 4 && out[0].nbytes == 16);
    float x[4] = {1, 2, 3, 4}, y[4] = {10, 20, 30, 40}, sum[4], product[4];
    in[0].data = x;
    in[1].data = y;
    out[0].data = sum;      /* the model's outputs keep its order: x + y, then x * y */
    out[1].data = product;
    CHECK(uairt_model_run(model, in, 2, out, 2) == UAIRT_OK);
    for (int i = 0; i < 4; ++i) {
      CHECK(near(sum[i], x[i] + y[i]) && near(product[i], x[i] * y[i]));
    }
    uairt_model_destroy(model);
    uairt_engine_destroy(engine);
  }

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all paddlelite checks passed");
  fflush(stdout);
  return 0;
}
