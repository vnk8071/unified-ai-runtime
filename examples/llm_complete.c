/* SPDX-License-Identifier: Apache-2.0 */
/* usage: llm_complete <plugin> <llamacpp|ollama|lmstudio|sglang|http://host:port> <model.gguf or model name> "<prompt>" [key=value ...] */
/* llamacpp loads the plugin libuairt_backend_llamacpp; the others need libuairt_backend_llmserver. */
/* The key=value pairs are options of uairt_model_complete: max_tokens, temperature, top_p, seed, system. */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <uairt/uairt.h>

static int print_piece(const char* text, size_t len, void* user_data) {
  (void)user_data;
  fwrite(text, 1, len, stdout);
  fflush(stdout);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <plugin> <llamacpp|preset|base url> <model> <prompt> [key=value ...]\n", argv[0]);
    return 2;
  }
  bool local = strcmp(argv[2], "llamacpp") == 0;
  uairt_option engine_option = {strncmp(argv[2], "http", 4) == 0 ? "base_url" : "preset", argv[2]};
  uairt_option complete_options[16];
  size_t num_complete_options = 0;
  for (int i = 5; i < argc && num_complete_options < 16; ++i) {
    char* eq = strchr(argv[i], '=');
    if (!eq) {
      fprintf(stderr, "option '%s' is not key=value\n", argv[i]);
      return 2;
    }
    *eq = '\0';
    complete_options[num_complete_options++] = (uairt_option){argv[i], eq + 1};
  }
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  uairt_model_source source = {.struct_size = sizeof(source), .path = argv[3]};
  uairt_status status = uairt_load_backend_library(argv[1]);
  if (status == UAIRT_OK) {
    status = local ? uairt_engine_create("llamacpp", NULL, 0, &engine)
                   : uairt_engine_create("llmserver", &engine_option, 1, &engine);
  }
  if (status == UAIRT_OK) {
    status = uairt_model_load(engine, &source, &model);
  }
  if (status == UAIRT_OK) {
    status = uairt_model_complete(model, argv[4], complete_options, num_complete_options, print_piece, NULL);
    putchar('\n');
  }
  if (status != UAIRT_OK) {
    fprintf(stderr, "error: %s\n", uairt_last_error());
  }
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return status == UAIRT_OK ? 0 : 1;
}
