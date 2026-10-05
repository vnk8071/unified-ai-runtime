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

/* argv[2] is a file that is not a model, argv[3] a directory without an .xml file. */
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <plugin> <non-model file> <directory without .xml>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("openvino", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_hint = {"performance_hint", "fastest"};
  CHECK(uairt_engine_create("openvino", &bad_hint, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_threads = {"num_threads", "0"};
  CHECK(uairt_engine_create("openvino", &bad_threads, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_device = {"device", "NO_SUCH_DEVICE"};
  CHECK(uairt_engine_create("openvino", &bad_device, 1, &engine) == UAIRT_ERR_BACKEND_UNAVAILABLE);
  CHECK(strstr(uairt_last_error(), "NO_SUCH_DEVICE") != NULL);
  CHECK(engine == NULL);

  uairt_option threads = {"num_threads", "2"};
  CHECK(uairt_engine_create("openvino", &threads, 1, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.xml"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.xml") != NULL);

  uairt_model_source empty_dir = {.struct_size = sizeof(empty_dir), .path = argv[3]};
  CHECK(uairt_model_load(engine, &empty_dir, &model) == UAIRT_ERR_IO);

  uairt_model_source not_a_model = {.struct_size = sizeof(not_a_model), .path = argv[2]};
  CHECK(uairt_model_load(engine, &not_a_model, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  char garbage[64] = {1, 2, 3};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_UNSUPPORTED);

  uairt_engine_destroy(engine);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all openvino error-path checks passed");
  fflush(stdout);
  return 0;
}
