/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <string.h>

#include <uairt/uairt.h>

static int g_failures;

#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,     \
              __LINE__, #cond);                                  \
      ++g_failures;                                              \
    }                                                            \
  } while (0)

static bool has_backend(const char* name) {
  for (size_t i = 0; i < uairt_backend_count(); ++i) {
    if (strcmp(uairt_backend_name(i), name) == 0) {
      return true;
    }
  }
  return false;
}

static void test_buffers(void) {
  uairt_engine* engine = NULL;
  CHECK(uairt_engine_create("reference", NULL, 0, &engine) == UAIRT_OK);
  if (!engine) {
    return;
  }
  uairt_buffer* buffer = NULL;
  CHECK(uairt_buffer_alloc(engine, 16, UAIRT_MEM_HOST, &buffer) == UAIRT_OK);
  if (buffer) {
    CHECK(uairt_buffer_data(buffer) != NULL);
    CHECK(uairt_buffer_fd(buffer) == -1);
    CHECK(uairt_buffer_size(buffer) == 16);
    uairt_tensor tensor;
    uairt_tensor_init(&tensor);
    uairt_tensor_use_buffer(&tensor, buffer);
    CHECK(tensor.data == uairt_buffer_data(buffer) && tensor.nbytes == 16);
    CHECK(tensor.domain == UAIRT_MEM_HOST);
    uairt_buffer_free(buffer);
  }
  buffer = NULL;
  CHECK(uairt_buffer_alloc(engine, 16, UAIRT_MEM_DMABUF, &buffer) == UAIRT_ERR_UNSUPPORTED);
  CHECK(buffer == NULL);
  CHECK(uairt_buffer_alloc(engine, 0, UAIRT_MEM_HOST, &buffer) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_buffer_alloc(NULL, 16, UAIRT_MEM_HOST, &buffer) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_buffer_alloc(engine, 16, 1u << 7, &buffer) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_buffer_free(NULL);
  uairt_engine_destroy(engine);
}

static void run_roundtrip(const char* backend) {
  uairt_engine* engine = NULL;
  CHECK(uairt_engine_create(backend, NULL, 0, &engine) == UAIRT_OK);
  if (!engine) {
    return;
  }
  uairt_model_source source = {.struct_size = sizeof(source), .path = "unused"};
  uairt_model* model = NULL;
  CHECK(uairt_model_load(engine, &source, &model) == UAIRT_OK);
  if (!model) {
    uairt_engine_destroy(engine);
    return;
  }
  CHECK(uairt_model_num_inputs(model) == 1);
  CHECK(uairt_model_num_outputs(model) == 1);

  uairt_tensor in, out;
  CHECK(uairt_model_input_info(model, 0, &in) == UAIRT_OK);
  CHECK(uairt_model_output_info(model, 0, &out) == UAIRT_OK);
  CHECK(in.dtype == UAIRT_DTYPE_FLOAT32 && in.rank == 2);

  float input[4] = {1.f, 2.f, 3.f, 4.f};
  float output[4] = {0};
  in.data = input;
  in.nbytes = sizeof(input);
  out.data = output;
  out.nbytes = sizeof(output);
  CHECK(uairt_model_run(model, &in, 1, &out, 1) == UAIRT_OK);
  CHECK(memcmp(input, output, sizeof(input)) == 0);

  out.nbytes = 8;
  CHECK(uairt_model_run(model, &in, 1, &out, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  out.nbytes = sizeof(output);

  uairt_tensor dma = in;
  dma.domain = UAIRT_MEM_DMABUF;
  dma.dmabuf_fd = 3;
  CHECK(uairt_model_run(model, &dma, 1, &out, 1) == UAIRT_ERR_UNSUPPORTED);

  uairt_tensor wrong = in;
  wrong.dims[1] = 5;
  CHECK(uairt_model_run(model, &wrong, 1, &out, 1) == UAIRT_ERR_INVALID_ARGUMENT);

  CHECK(uairt_model_run(model, &in, 0, &out, 1) == UAIRT_ERR_INVALID_ARGUMENT);

  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

int main(int argc, char** argv) {
  CHECK(strcmp(uairt_version_string(), UAIRT_VERSION_STRING) == 0);
  CHECK(strcmp(uairt_status_string(UAIRT_ERR_NOT_FOUND), "not found") == 0);
  CHECK(has_backend("reference"));

  uairt_engine* engine = NULL;
  CHECK(uairt_engine_create("nope", NULL, 0, &engine) == UAIRT_ERR_NOT_FOUND);
  CHECK(engine == NULL);
  CHECK(strstr(uairt_last_error(), "nope") != NULL);
  CHECK(uairt_engine_create(NULL, NULL, 0, &engine) == UAIRT_ERR_INVALID_ARGUMENT);

  uairt_tensor t;
  uairt_tensor_init(&t);
  t.rank = 2;
  t.dims[0] = 3;
  t.dims[1] = 5;
  CHECK(uairt_tensor_num_elements(&t) == 15);
  t.dims[1] = -1;
  CHECK(uairt_tensor_num_elements(&t) == -1);

  test_buffers();
  run_roundtrip("reference");

  if (argc > 1) {
    CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);
    CHECK(has_backend("reference-plugin"));
    CHECK(uairt_load_backend_library(argv[1]) == UAIRT_ERR_INVALID_ARGUMENT);
    run_roundtrip("reference-plugin");
  }
  CHECK(uairt_load_backend_library("/nonexistent/lib.so") == UAIRT_ERR_IO);

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all checks passed");
  return 0;
}
