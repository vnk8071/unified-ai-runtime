/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Smallest complete program: pick a backend by name, load a model, run it.
 * It uses the built-in "reference" backend, which needs no SDK and copies one
 * float32 [1, 4] input to the output, so it runs anywhere UAIRT builds.
 *
 * With a real backend the same code runs after loading the plugin first:
 *   uairt_load_backend_library("libuairt_backend_onnxruntime.so");
 * and passing that backend's name, options and model path.
 */
#include <stdio.h>

#include <uairt/uairt.h>

#define TRY(call)                                                              \
  do {                                                                         \
    uairt_status status_ = (call);                                              \
    if (status_ != UAIRT_OK) {                                                  \
      fprintf(stderr, "%s failed: %s (%s)\n", #call, uairt_status_string(status_), \
              uairt_last_error());                                              \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  printf("UAIRT %s, backends:", uairt_version_string());
  for (size_t i = 0; i < uairt_backend_count(); ++i) {
    printf(" %s", uairt_backend_name(i));
  }
  printf("\n");

  uairt_engine* engine = NULL;
  TRY(uairt_engine_create("reference", NULL, 0, &engine));

  uairt_model_source source = {.struct_size = sizeof(source), .path = "unused"};
  uairt_model* model = NULL;
  TRY(uairt_model_load(engine, &source, &model));

  uairt_tensor input, output;
  TRY(uairt_model_input_info(model, 0, &input));
  TRY(uairt_model_output_info(model, 0, &output));

  float in_data[4] = {1.f, 2.f, 3.f, 4.f};
  float out_data[4] = {0};
  input.data = in_data;
  input.nbytes = sizeof(in_data);
  output.data = out_data;
  output.nbytes = sizeof(out_data);
  TRY(uairt_model_run(model, &input, 1, &output, 1));

  printf("output: %g %g %g %g\n", out_data[0], out_data[1], out_data[2], out_data[3]);

  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return out_data[3] == 4.f ? 0 : 1;
}
