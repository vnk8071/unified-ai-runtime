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
  CHECK(uairt_engine_create("coreml", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_units = {"compute_units", "tpu"};
  CHECK(uairt_engine_create("coreml", &bad_units, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);

  uairt_option units = {"compute_units", "cpu_only"};
  CHECK(uairt_engine_create("coreml", &units, 1, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  char bytes[8] = {0};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = bytes, .size = 8};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_UNSUPPORTED);

  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.mlmodelc"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.mlmodelc") != NULL);

  uairt_model_source not_a_model = {.struct_size = sizeof(not_a_model), .path = "/etc/hosts"};
  CHECK(uairt_model_load(engine, &not_a_model, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);
  CHECK(uairt_last_error()[0] != '\0');

  uairt_engine_destroy(engine);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all coreml error-path checks passed");
  return 0;
}
