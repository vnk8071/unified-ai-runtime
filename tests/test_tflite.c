/* SPDX-License-Identifier: Apache-2.0 */
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

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <plugin>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("tflite", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_delegate = {"delegate", "gpu"};
  CHECK(uairt_engine_create("tflite", &bad_delegate, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);

  /* delegate=qnn without its library or backend path, or without delegate support built in */
  uairt_option qnn = {"delegate", "qnn"};
  uairt_status status = uairt_engine_create("tflite", &qnn, 1, &engine);
  CHECK(status == UAIRT_ERR_INVALID_ARGUMENT || status == UAIRT_ERR_UNSUPPORTED);
  CHECK(engine == NULL);

  uairt_option threads = {"num_threads", "2"};
  CHECK(uairt_engine_create("tflite", &threads, 1, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.tflite"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.tflite") != NULL);

  uairt_model_source not_a_model = {.struct_size = sizeof(not_a_model), .path = "/etc/hosts"};
  CHECK(uairt_model_load(engine, &not_a_model, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  char garbage[64] = {1, 2, 3};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  uairt_engine_destroy(engine);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all tflite error-path checks passed");
  return 0;
}
