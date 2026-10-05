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

/* argv[2] is any file that is not a TensorRT engine; the test file itself works. */
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <plugin> <non-engine file>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("tensorrt", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_device = {"device", "-1"};
  CHECK(uairt_engine_create("tensorrt", &bad_device, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(engine == NULL);

  CHECK(uairt_engine_create("tensorrt", NULL, 0, &engine) == UAIRT_OK);
  if (!engine) {
    return 1;
  }
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.engine"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/no/such/model.engine") != NULL);

  uairt_model_source not_an_engine = {.struct_size = sizeof(not_an_engine), .path = argv[2]};
  CHECK(uairt_model_load(engine, &not_an_engine, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  char garbage[64] = {1, 2, 3};
  uairt_model_source from_memory = {.struct_size = sizeof(from_memory), .data = garbage, .size = sizeof(garbage)};
  CHECK(uairt_model_load(engine, &from_memory, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  uairt_model_source wrong_format = {
      .struct_size = sizeof(wrong_format), .path = argv[2], .format = "onnx"};
  CHECK(uairt_model_load(engine, &wrong_format, &model) == UAIRT_ERR_UNSUPPORTED);

  uairt_buffer* pinned = NULL;
  CHECK(uairt_buffer_alloc(engine, 4096, UAIRT_MEM_PINNED, &pinned) == UAIRT_OK);
  if (pinned) {
    CHECK(uairt_buffer_data(pinned) != NULL && uairt_buffer_size(pinned) == 4096);
    memset(uairt_buffer_data(pinned), 0xAB, 4096);
    uairt_tensor tensor;
    uairt_tensor_init(&tensor);
    uairt_tensor_use_buffer(&tensor, pinned);
    CHECK(tensor.domain == UAIRT_MEM_PINNED && tensor.data == uairt_buffer_data(pinned));
    uairt_buffer_free(pinned);
  }
  uairt_buffer* dmabuf = NULL;
  CHECK(uairt_buffer_alloc(engine, 4096, UAIRT_MEM_DMABUF, &dmabuf) == UAIRT_ERR_UNSUPPORTED);

  uairt_engine_destroy(engine);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all tensorrt error-path checks passed");
  return 0;
}
