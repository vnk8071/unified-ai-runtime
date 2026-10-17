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

/* argv[2] is an existing file that is not a .pte program. */
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <plugin> <file that is not a .pte>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("executorch", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "no_such_option") != NULL);
  uairt_option empty_method = {"method", ""};
  CHECK(uairt_engine_create("executorch", &empty_method, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(engine == NULL);

  CHECK(uairt_engine_create("executorch", NULL, 0, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.pte"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.pte") != NULL);

  uairt_model_source not_a_program = {.struct_size = sizeof(not_a_program), .path = argv[2]};
  CHECK(uairt_model_load(engine, &not_a_program, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);
  CHECK(model == NULL);

  char garbage[64] = {1, 2, 3};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_UNSUPPORTED);

  uairt_engine_destroy(engine);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all executorch error-path checks passed");
  return 0;
}
