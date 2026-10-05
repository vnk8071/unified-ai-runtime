/* SPDX-License-Identifier: Apache-2.0 */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

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

static bool has_backend(const char* name) {
  for (size_t i = 0; i < uairt_backend_count(); ++i) {
    if (strcmp(uairt_backend_name(i), name) == 0) {
      return true;
    }
  }
  return false;
}

static bool open_model(const char* backend, uairt_engine** engine, uairt_model** model) {
  *engine = NULL;
  *model = NULL;
  uairt_model_source source = {.struct_size = sizeof(source), .path = "unused"};
  CHECK(uairt_engine_create(backend, NULL, 0, engine) == UAIRT_OK);
  if (!*engine) {
    return false;
  }
  CHECK(uairt_model_load(*engine, &source, model) == UAIRT_OK);
  if (!*model) {
    uairt_engine_destroy(*engine);
    return false;
  }
  return true;
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

static void test_without_sessions(const char* backend) {
  uairt_engine* engine;
  uairt_model* model;
  if (!open_model(backend, &engine, &model)) {
    return;
  }
  size_t size = 0, count = 0, nbytes = 0;
  int32_t tokens[4] = {0};
  char text[8];
  uairt_session* session = NULL;
  CHECK(uairt_model_vocab_size(model, &size) == UAIRT_ERR_UNSUPPORTED);
  CHECK(strstr(uairt_last_error(), backend) != NULL);
  CHECK(uairt_model_tokenize(model, "hi", 2, 0, tokens, 4, &count) == UAIRT_ERR_UNSUPPORTED);
  CHECK(uairt_model_detokenize(model, tokens, 1, text, 8, &nbytes) == UAIRT_ERR_UNSUPPORTED);
  CHECK(uairt_session_create(model, NULL, 0, &session) == UAIRT_ERR_UNSUPPORTED);
  CHECK(session == NULL);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

static void test_sessions(void) {
  uairt_engine* engine;
  uairt_model* model;
  if (!open_model("fake-session", &engine, &model)) {
    return;
  }
  size_t vocab = 0;
  CHECK(uairt_model_vocab_size(model, &vocab) == UAIRT_OK && vocab == 257);
  CHECK(uairt_model_vocab_size(model, NULL) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_model_vocab_size(NULL, &vocab) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_model_run(model, NULL, 0, NULL, 0) == UAIRT_ERR_UNSUPPORTED);

  int32_t tokens[8];
  size_t count = 0;
  CHECK(uairt_model_tokenize(model, "hi", 2, 1, tokens, 8, &count) == UAIRT_OK);
  CHECK(count == 3 && tokens[0] == 256 && tokens[1] == 'h' && tokens[2] == 'i');
  count = 0;
  CHECK(uairt_model_tokenize(model, "hi", 2, 1, tokens, 2, &count) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(count == 3);
  count = 0;
  CHECK(uairt_model_tokenize(model, "hi", 2, 1, NULL, 0, &count) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(count == 3);
  count = 99;
  CHECK(uairt_model_tokenize(model, "", 0, 0, NULL, 0, &count) == UAIRT_OK);
  CHECK(count == 0);
  CHECK(uairt_model_tokenize(model, NULL, 2, 0, tokens, 8, &count) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_model_tokenize(model, "hi", 2, 0, NULL, 4, &count) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_model_tokenize(model, "hi", 2, 0, tokens, 8, NULL) == UAIRT_ERR_INVALID_ARGUMENT);

  int32_t ids[3] = {256, 'o', 'k'};
  char text[8] = {0};
  size_t nbytes = 0;
  CHECK(uairt_model_detokenize(model, ids, 3, text, 8, &nbytes) == UAIRT_OK);
  CHECK(nbytes == 2 && memcmp(text, "ok", 2) == 0);
  nbytes = 0;
  CHECK(uairt_model_detokenize(model, ids, 3, text, 1, &nbytes) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(nbytes == 2);
  nbytes = 9;
  CHECK(uairt_model_detokenize(model, NULL, 0, NULL, 0, &nbytes) == UAIRT_OK);
  CHECK(nbytes == 0);
  int32_t out_of_range[1] = {5000};
  CHECK(uairt_model_detokenize(model, out_of_range, 1, text, 8, &nbytes) == UAIRT_ERR_INVALID_ARGUMENT);

  uairt_session* session = NULL;
  uairt_option unknown = {"nope", "1"};
  CHECK(uairt_session_create(model, &unknown, 1, &session) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(session == NULL);
  CHECK(uairt_session_create(NULL, NULL, 0, &session) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_create(model, NULL, 1, &session) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option n_ctx = {"n_ctx", "4"};
  CHECK(uairt_session_create(model, &n_ctx, 1, &session) == UAIRT_OK);
  if (!session) {
    uairt_model_destroy(model);
    uairt_engine_destroy(engine);
    return;
  }

  float logits[257];
  size_t n = 0, position = 99;
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 0);
  CHECK(uairt_session_logits(session, logits, 257, &n) == UAIRT_ERR_INVALID_ARGUMENT);

  int32_t prompt[3] = {256, 'a', 'b'};
  CHECK(uairt_session_append(session, prompt, 3) == UAIRT_OK);
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 3);
  CHECK(uairt_session_logits(session, logits, 257, &n) == UAIRT_OK);
  CHECK(n == 257 && argmax(logits, n) == 'b' + 1);
  n = 0;
  CHECK(uairt_session_logits(session, logits, 10, &n) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(n == 257);

  int32_t more[2] = {1, 2};
  CHECK(uairt_session_append(session, more, 2) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "context full") != NULL);
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 3);
  CHECK(uairt_session_append(session, NULL, 0) == UAIRT_OK);
  int32_t bad_id = 1000, negative = -1;
  CHECK(uairt_session_append(session, &bad_id, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_append(session, &negative, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_append(session, more, 1) == UAIRT_OK);
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 4);
  CHECK(uairt_session_append(session, NULL, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_position(session, NULL) == UAIRT_ERR_INVALID_ARGUMENT);

  CHECK(uairt_session_reset(session) == UAIRT_OK);
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 0);
  CHECK(uairt_session_logits(session, logits, 257, &n) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_session_reset(NULL) == UAIRT_ERR_INVALID_ARGUMENT);

  uairt_session_destroy(session);
  uairt_session_destroy(NULL);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

/* argv[1] complete session table, argv[2] old-sized struct, argv[3] session table without reset. */
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <fake-session plugin> <fake-old plugin> <fake-bad plugin>\n", argv[0]);
    return 2;
  }
  CHECK(uairt_load_backend_library(argv[3]) == UAIRT_ERR_INCOMPATIBLE_MODEL);
  CHECK(strstr(uairt_last_error(), "reset") != NULL);
  CHECK(!has_backend("fake-bad"));
  CHECK(uairt_load_backend_library(argv[2]) == UAIRT_OK);
  CHECK(has_backend("fake-old"));
  CHECK(uairt_load_backend_library(argv[1]) == UAIRT_OK);
  CHECK(has_backend("fake-session"));

  test_without_sessions("reference");
  test_without_sessions("fake-old");
  test_sessions();

  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("all checks passed");
  return 0;
}
