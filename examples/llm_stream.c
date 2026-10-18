/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Streaming text generation with uairt_llm (sampling and the generation loop):
 *   llm_stream <plugin> <model.gguf> "<prompt>" <max_tokens> [temperature [top_k [top_p [seed [n_gpu_layers]]]]]
 * Prints the prompt, then the generated text as it arrives, then the finish reason on stderr.
 */
#include <stdio.h>
#include <stdlib.h>

#include <uairt/uairt_llm.h>

static int on_text(const char* text, size_t len, void* user_data) {
  (void)user_data;
  fwrite(text, 1, len, stdout);
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 10) {
    fprintf(stderr, "usage: %s <plugin> <model.gguf> \"<prompt>\" <max_tokens> [temperature [top_k [top_p [seed [n_gpu_layers]]]]]\n",
            argv[0]);
    return 2;
  }
  uairt_status status = uairt_load_backend_library(argv[1]);
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  uairt_option engine_options[1] = {{"n_gpu_layers", argc > 9 ? argv[9] : "99"}};
  uairt_model_source source = {.struct_size = sizeof(source), .path = argv[2]};
  if (status != UAIRT_OK || (status = uairt_engine_create("llamacpp", engine_options, 1, &engine)) != UAIRT_OK ||
      (status = uairt_model_load(engine, &source, &model)) != UAIRT_OK) {
    fprintf(stderr, "%s (%s)\n", uairt_status_string(status), uairt_last_error());
    return 1;
  }

  uairt_generate_options options = uairt_generate_options_default();
  options.max_tokens = (size_t)atoi(argv[4]);
  options.sampling.temperature = argc > 5 ? (float)atof(argv[5]) : 0.0f;
  options.sampling.top_k = argc > 6 ? atoi(argv[6]) : 0;
  options.sampling.top_p = argc > 7 ? (float)atof(argv[7]) : 1.0f;
  options.sampling.seed = argc > 8 ? strtoull(argv[8], NULL, 10) : 0;

  fputs(argv[3], stdout);
  uairt_generate_result result = {.struct_size = sizeof(result)};
  status = uairt_llm_generate(model, NULL, argv[3], &options, on_text, NULL, &result);
  putchar('\n');
  if (status != UAIRT_OK) {
    fprintf(stderr, "generate: %s (%s)\n", uairt_status_string(status), uairt_last_error());
  } else {
    fprintf(stderr, "%zu tokens, finish_reason=%d\n", result.num_tokens, result.finish_reason);
  }
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return status == UAIRT_OK ? 0 : 1;
}
