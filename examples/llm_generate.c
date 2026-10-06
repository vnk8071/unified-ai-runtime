/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Greedy text generation through the session API:
 *   llm_generate <plugin> <model.gguf> "<prompt>" <n_tokens> [n_gpu_layers]
 * Prints the prompt followed by the generated text. Tokenizing and sampling are the caller's job; here sampling is
 * "pick the largest logit".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt.h>

static int die(const char* what, uairt_status status) {
  fprintf(stderr, "%s: %s (%s)\n", what, uairt_status_string(status), uairt_last_error());
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    fprintf(stderr, "usage: %s <plugin> <model.gguf> \"<prompt>\" <n_tokens> [n_gpu_layers]\n", argv[0]);
    return 2;
  }
  const char* prompt = argv[3];
  int steps = atoi(argv[4]);
  uairt_status status = uairt_load_backend_library(argv[1]);
  if (status != UAIRT_OK) {
    return die("load plugin", status);
  }
  uairt_option layers = {"n_gpu_layers", argc == 6 ? argv[5] : "99"};
  uairt_engine* engine = NULL;
  if ((status = uairt_engine_create("llamacpp", &layers, 1, &engine)) != UAIRT_OK) {
    return die("create engine", status);
  }
  uairt_model_source source = {.struct_size = sizeof(source), .path = argv[2]};
  uairt_model* model = NULL;
  if ((status = uairt_model_load(engine, &source, &model)) != UAIRT_OK) {
    return die("load model", status);
  }

  size_t vocab = 0, count = 0;
  uairt_model_vocab_size(model, &vocab);
  uairt_model_tokenize(model, prompt, strlen(prompt), 1, NULL, 0, &count);  /* capacity 0: reports the size */
  int32_t* tokens = malloc(count * sizeof(int32_t));
  if ((status = uairt_model_tokenize(model, prompt, strlen(prompt), 1, tokens, count, &count)) != UAIRT_OK) {
    return die("tokenize", status);
  }

  uairt_session* session = NULL;
  if ((status = uairt_session_create(model, NULL, 0, &session)) != UAIRT_OK) {
    return die("create session", status);
  }
  if ((status = uairt_session_append(session, tokens, count)) != UAIRT_OK) {
    return die("append", status);
  }
  free(tokens);

  float* logits = malloc(vocab * sizeof(float));
  int32_t* generated = malloc((steps > 0 ? (size_t)steps : 1) * sizeof(int32_t));
  char* text = NULL;
  size_t text_capacity = 0, printed = 0;
  fputs(prompt, stdout);
  for (int i = 0; i < steps; ++i) {
    size_t n = 0;
    if ((status = uairt_session_logits(session, logits, vocab, &n)) != UAIRT_OK) {
      return die("logits", status);
    }
    size_t best = 0;
    for (size_t k = 1; k < n; ++k) {
      if (logits[k] > logits[best]) {
        best = k;
      }
    }
    generated[i] = (int32_t)best;

    /* Detokenize everything generated so far and print only the new bytes: one token can be half a character. */
    size_t nbytes = 0;
    uairt_model_detokenize(model, generated, (size_t)i + 1, NULL, 0, &nbytes);
    if (nbytes + 1 > text_capacity) {
      text_capacity = nbytes + 1;
      text = realloc(text, text_capacity);
    }
    if ((status = uairt_model_detokenize(model, generated, (size_t)i + 1, text, nbytes, &nbytes)) != UAIRT_OK) {
      return die("detokenize", status);
    }
    if (nbytes > printed) {
      fwrite(text + printed, 1, nbytes - printed, stdout);
      printed = nbytes;
    }
    if (i + 1 < steps && (status = uairt_session_append(session, &generated[i], 1)) != UAIRT_OK) {
      return die("append", status);
    }
  }
  putchar('\n');
  /* Some runtimes (OpenCL on Windows) end the process without flushing C stdio, which loses redirected output. */
  fflush(stdout);

  free(text);
  free(generated);
  free(logits);
  uairt_session_destroy(session);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
  return 0;
}
