/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include <uairt/uairt.h>

static int g_failures;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__, __LINE__, #cond,    \
              uairt_last_error());                                                    \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

static void write_file(const char* path, const void* data, size_t size) {
  FILE* file = fopen(path, "wb");
  if (file) {
    fwrite(data, 1, size, file);
    fclose(file);
  }
}

/* Loads `path` on a fresh engine and returns the status; the model, if any, is destroyed. */
static uairt_status try_load(const char* path) {
  uairt_engine* engine = NULL;
  uairt_option cpu = {"n_gpu_layers", "0"};
  if (uairt_engine_create("llamacpp", &cpu, 1, &engine) != UAIRT_OK) {
    return UAIRT_ERR_RUNTIME;
  }
  uairt_model_source source = {.struct_size = sizeof(source), .path = path};
  uairt_model* model = NULL;
  uairt_status status = uairt_model_load(engine, &source, &model);
  if (model) {
    uairt_model_destroy(model);
  }
  uairt_engine_destroy(engine);
  return status;
}

static size_t argmax(const float* values, size_t count) {
  size_t best = 0;
  for (size_t i = 1; i < count; ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

/* Tokenizes `text` into a malloc'd array, sizing the buffer with a capacity-0 call. */
static int32_t* tokenize_all(const uairt_model* model, const char* text, int add_special, size_t* count) {
  size_t needed = 0;
  size_t length = strlen(text);
  uairt_status status = uairt_model_tokenize(model, text, length, add_special, NULL, 0, &needed);
  CHECK(status == UAIRT_OK || status == UAIRT_ERR_INVALID_ARGUMENT);
  int32_t* tokens = malloc((needed ? needed : 1) * sizeof(int32_t));
  CHECK(uairt_model_tokenize(model, text, length, add_special, tokens, needed, count) == UAIRT_OK);
  CHECK(*count == needed);
  return tokens;
}

static char* detokenize_all(const uairt_model* model, const int32_t* tokens, size_t count) {
  size_t nbytes = 0;
  uairt_status status = uairt_model_detokenize(model, tokens, count, NULL, 0, &nbytes);
  CHECK(status == UAIRT_OK || status == UAIRT_ERR_INVALID_ARGUMENT);
  char* text = malloc(nbytes + 1);
  CHECK(uairt_model_detokenize(model, tokens, count, text, nbytes, &nbytes) == UAIRT_OK);
  text[nbytes] = 0;
  return text;
}

static void test_model(const char* model_path) {
  uairt_engine* engine = NULL;
  uairt_option cpu = {"n_gpu_layers", "0"};
  CHECK(uairt_engine_create("llamacpp", &cpu, 1, &engine) == UAIRT_OK);
  uairt_model_source source = {.struct_size = sizeof(source), .path = model_path};
  uairt_model* model = NULL;
  CHECK(uairt_model_load(engine, &source, &model) == UAIRT_OK);
  if (!model) {
    uairt_engine_destroy(engine);
    return;
  }
  CHECK(uairt_model_num_inputs(model) == 0 && uairt_model_num_outputs(model) == 0);
  CHECK(uairt_model_run(model, NULL, 0, NULL, 0) == UAIRT_ERR_UNSUPPORTED);

  size_t vocab = 0;
  CHECK(uairt_model_vocab_size(model, &vocab) == UAIRT_OK && vocab > 1000);

  /* Round trips, including multi-byte text. */
  const char* texts[] = {"Hello, world", "h\xc3\xa9llo \xe6\x97\xa5\xe6\x9c\xac"};
  for (size_t i = 0; i < 2; ++i) {
    size_t count = 0;
    int32_t* tokens = tokenize_all(model, texts[i], 0, &count);
    CHECK(count > 0);
    char* back = detokenize_all(model, tokens, count);
    CHECK(strcmp(back, texts[i]) == 0);
    free(back);
    free(tokens);
  }

  /* Empty input. */
  size_t count = 99;
  int32_t one[1];
  CHECK(uairt_model_tokenize(model, "", 0, 0, one, 1, &count) == UAIRT_OK && count == 0);
  size_t nbytes = 99;
  CHECK(uairt_model_detokenize(model, NULL, 0, NULL, 0, &nbytes) == UAIRT_OK && nbytes == 0);

  /* A buffer that is too small reports the size needed. */
  size_t needed = 0;
  int32_t tiny[1];
  CHECK(uairt_model_tokenize(model, "Hello, world", 12, 0, tiny, 1, &needed) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(needed > 1);

  /* Token ids outside the vocabulary are an error, not an abort. */
  char buffer[64];
  int32_t out_of_range[1] = {(int32_t)vocab};
  CHECK(uairt_model_detokenize(model, out_of_range, 1, buffer, sizeof(buffer), &nbytes) == UAIRT_ERR_INVALID_ARGUMENT);
  int32_t negative[1] = {-5};
  CHECK(uairt_model_detokenize(model, negative, 1, buffer, sizeof(buffer), &nbytes) == UAIRT_ERR_INVALID_ARGUMENT);

  /* Session options. */
  uairt_session* session = NULL;
  uairt_option zero = {"n_ctx", "0"};
  uairt_option word = {"n_ctx", "lots"};
  uairt_option unknown = {"nope", "1"};
  CHECK(uairt_session_create(model, &zero, 1, &session) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_create(model, &word, 1, &session) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_create(model, &unknown, 1, &session) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(session == NULL);

  uairt_option ctx = {"n_ctx", "1024"};
  CHECK(uairt_session_create(model, &ctx, 1, &session) == UAIRT_OK);
  if (!session) {
    uairt_model_destroy(model);
    uairt_engine_destroy(engine);
    return;
  }
  float* logits = malloc(vocab * sizeof(float));
  size_t n = 0, position = 99;
  CHECK(uairt_session_logits(session, logits, vocab, &n) == UAIRT_ERR_INVALID_ARGUMENT);

  size_t prompt_count = 0;
  int32_t* prompt = tokenize_all(model, "The capital of France is", 1, &prompt_count);
  CHECK(uairt_session_append(session, prompt, prompt_count) == UAIRT_OK);
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == prompt_count);
  CHECK(uairt_session_logits(session, logits, vocab, &n) == UAIRT_OK && n == vocab);
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(logits[i])) {
      CHECK(!"logits must be finite");
      break;
    }
  }
  n = 0;
  CHECK(uairt_session_logits(session, logits, 10, &n) == UAIRT_ERR_INVALID_ARGUMENT && n == vocab);

  /* Deterministic: the same prompt after a reset gives identical logits (the cache is really cleared). */
  float* first = malloc(vocab * sizeof(float));
  memcpy(first, logits, vocab * sizeof(float));
  CHECK(uairt_session_reset(session) == UAIRT_OK);
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 0);
  CHECK(uairt_session_logits(session, logits, vocab, &n) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_append(session, prompt, prompt_count) == UAIRT_OK);
  CHECK(uairt_session_logits(session, logits, vocab, &n) == UAIRT_OK);
  CHECK(memcmp(first, logits, vocab * sizeof(float)) == 0);

  /* Token ids out of range do not reach llama.cpp. */
  int32_t bad[1] = {(int32_t)vocab};
  CHECK(uairt_session_append(session, bad, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  bad[0] = -1;
  CHECK(uairt_session_append(session, bad, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_append(session, NULL, 0) == UAIRT_OK);

  /* Two sessions on one model do not share a cache. */
  uairt_session* other = NULL;
  CHECK(uairt_session_create(model, &ctx, 1, &other) == UAIRT_OK);
  if (other) {
    size_t other_count = 0;
    int32_t* other_prompt = tokenize_all(model, "Once upon a time", 1, &other_count);
    CHECK(uairt_session_append(other, other_prompt, other_count) == UAIRT_OK);
    CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == prompt_count);
    float* other_logits = malloc(vocab * sizeof(float));
    CHECK(uairt_session_logits(other, other_logits, vocab, &n) == UAIRT_OK);
    CHECK(memcmp(first, other_logits, vocab * sizeof(float)) != 0);
    CHECK(uairt_session_logits(session, logits, vocab, &n) == UAIRT_OK);
    CHECK(memcmp(first, logits, vocab * sizeof(float)) == 0);
    free(other_logits);
    free(other_prompt);
    uairt_session_destroy(other);
  }
  uairt_session_destroy(session);

  /* A prompt longer than the batch size (512) goes in through chunking. */
  uairt_option big = {"n_ctx", "1024"};
  CHECK(uairt_session_create(model, &big, 1, &session) == UAIRT_OK);
  if (session) {
    char* long_text = malloc(8192);
    long_text[0] = 0;
    for (int i = 0; i < 70; ++i) {
      strcat(long_text, "The quick brown fox jumps over the lazy dog. ");
    }
    size_t long_count = 0;
    int32_t* long_tokens = tokenize_all(model, long_text, 1, &long_count);
    CHECK(long_count > 512 && long_count < 1024);
    CHECK(uairt_session_append(session, long_tokens, long_count) == UAIRT_OK);
    CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == long_count);
    CHECK(uairt_session_logits(session, logits, vocab, &n) == UAIRT_OK);
    CHECK(isfinite(logits[argmax(logits, n)]));

    /* The context is full after enough tokens: the failing call changes nothing. */
    int32_t* filler = malloc(1024 * sizeof(int32_t));
    for (size_t i = 0; i < 1024; ++i) {
      filler[i] = long_tokens[0];
    }
    CHECK(uairt_session_append(session, filler, 1024) == UAIRT_ERR_INVALID_ARGUMENT);
    CHECK(strstr(uairt_last_error(), "context full") != NULL);
    CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == long_count);
    free(filler);
    free(long_tokens);
    free(long_text);
    uairt_session_destroy(session);
  }

  free(first);
  free(prompt);
  free(logits);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

/* argv[1] plugin, argv[2] a writable directory. */
int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    fprintf(stderr, "usage: %s <plugin> <scratch directory> [model.gguf]\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);

  uairt_engine* engine = NULL;
  uairt_option unknown = {"no_such_option", "1"};
  CHECK(uairt_engine_create("llamacpp", &unknown, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option not_a_number = {"n_gpu_layers", "many"};
  CHECK(uairt_engine_create("llamacpp", &not_a_number, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option negative = {"n_gpu_layers", "-1"};
  CHECK(uairt_engine_create("llamacpp", &negative, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option cpu = {"n_gpu_layers", "0"};
  CHECK(uairt_engine_create("llamacpp", &cpu, 1, &engine) == UAIRT_OK);

  char path[1024];
  uairt_model* model = NULL;
  uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/nonexistent/model.gguf"};
  CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "/nonexistent/model.gguf") != NULL);
  uint8_t bytes[4] = {1, 2, 3, 4};
  uairt_model_source memory = {.struct_size = sizeof(memory), .data = bytes, .size = sizeof(bytes)};
  CHECK(uairt_model_load(engine, &memory, &model) == UAIRT_ERR_UNSUPPORTED);
  CHECK(model == NULL);
  uairt_engine_destroy(engine);

  snprintf(path, sizeof(path), "%s/not_a_model.gguf", argv[2]);
  write_file(path, "this is plain text, not a GGUF file", 35);
  CHECK(try_load(path) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  /* Valid magic and version, absurd counts and then garbage: a clean error, not a crash. */
  unsigned char corrupt[64] = {'G', 'G', 'U', 'F', 3, 0, 0, 0, 0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0,
                               0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0};
  memset(corrupt + 24, 0xa5, sizeof(corrupt) - 24);
  snprintf(path, sizeof(path), "%s/corrupt.gguf", argv[2]);
  write_file(path, corrupt, sizeof(corrupt));
  CHECK(try_load(path) == UAIRT_ERR_INCOMPATIBLE_MODEL);

  uairt_option empty_device = {"device", ""};
  CHECK(uairt_engine_create("llamacpp", &empty_device, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option no_device = {"device", "no_such_device"};
  CHECK(uairt_engine_create("llamacpp", &no_device, 1, &engine) == UAIRT_OK);
  snprintf(path, sizeof(path), "%s/not_a_model.gguf", argv[2]);
  uairt_model_source plain = {.struct_size = sizeof(plain), .path = path};
  CHECK(uairt_model_load(engine, &plain, &model) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "no_such_device") != NULL);
  uairt_engine_destroy(engine);

  if (argc == 4) {
    test_model(argv[3]);
  }

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all checks passed");
  return 0;
}
