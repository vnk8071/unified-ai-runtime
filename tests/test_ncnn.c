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

/* argv[2] is a directory without a .param file. */
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <plugin> <directory without .param>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option shapes = {"input_shapes", "3x4x4"};
  uairt_option bad_option[] = {{"no_such_option", "1"}, shapes};
  CHECK(uairt_engine_create("ncnn", bad_option, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_engine_create("ncnn", NULL, 0, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "input_shapes") != NULL);
  uairt_option bad_shape = {"input_shapes", "3xbadx4"};
  CHECK(uairt_engine_create("ncnn", &bad_shape, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option batch_shape = {"input_shapes", "1x1x3x4x4"};
  CHECK(uairt_engine_create("ncnn", &batch_shape, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_device[] = {{"device", "cuda"}, shapes};
  CHECK(uairt_engine_create("ncnn", bad_device, 2, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option no_gpu[] = {{"device", "vulkan"}, {"vulkan_device", "99"}, shapes};
  CHECK(uairt_engine_create("ncnn", no_gpu, 3, &engine) == UAIRT_ERR_BACKEND_UNAVAILABLE);
  CHECK(engine == NULL);

  CHECK(uairt_engine_create("ncnn", &shapes, 1, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.param"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.param") != NULL);

  uairt_model_source empty_dir = {.struct_size = sizeof(empty_dir), .path = argv[2]};
  CHECK(uairt_model_load(engine, &empty_dir, &model) == UAIRT_ERR_IO);

  char garbage[64] = {1, 2, 3};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_UNSUPPORTED);

  uairt_engine_destroy(engine);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all ncnn error-path checks passed");
  return 0;
}
