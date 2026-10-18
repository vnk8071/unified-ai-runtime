/* SPDX-License-Identifier: Apache-2.0 */
/* usage: test_llmserver <plugin> <reference plugin or -> <base url> <base url of a server that needs the key "secret"> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt.h>

static int g_failures;

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__, __LINE__, #cond, \
              uairt_last_error());                                                 \
      ++g_failures;                                                                \
    }                                                                              \
  } while (0)

typedef struct {
  char text[4096];
  size_t pieces;
  size_t stop_after; /* 0: never */
} collector_t;

static int collect(const char* text, size_t len, void* user_data) {
  collector_t* c = user_data;
  if (c->pieces == 0) {
    c->text[0] = '\0';
  }
  size_t used = strlen(c->text);
  if (used + len < sizeof(c->text)) {
    memcpy(c->text + used, text, len);
    c->text[used + len] = '\0';
  }
  ++c->pieces;
  return c->stop_after && c->pieces >= c->stop_after;
}

static uairt_engine* make_engine(const char* url, const uairt_option* extra, size_t n_extra) {
  uairt_option options[4] = {{"base_url", url}};
  for (size_t i = 0; i < n_extra && i < 3; ++i) {
    options[i + 1] = extra[i];
  }
  uairt_engine* engine = NULL;
  CHECK(uairt_engine_create("llmserver", options, n_extra + 1, &engine) == UAIRT_OK);
  return engine;
}

static uairt_status load(uairt_engine* engine, const char* name, uairt_model** model) {
  uairt_model_source source = {.struct_size = sizeof(source), .path = name};
  *model = NULL;
  return uairt_model_load(engine, &source, model);
}

static uairt_status engine_status(const uairt_option* options, size_t n) {
  uairt_engine* engine = NULL;
  uairt_status status = uairt_engine_create("llmserver", options, n, &engine);
  if (engine) {
    uairt_engine_destroy(engine);
  }
  return status;
}

static void test_engine_options(void) {
  CHECK(engine_status(NULL, 0) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_preset[] = {{"preset", "vllm"}};
  CHECK(engine_status(bad_preset, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_url[] = {{"base_url", "ftp://host"}};
  CHECK(engine_status(bad_url, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option empty_url[] = {{"base_url", "http://"}};
  CHECK(engine_status(empty_url, 1) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_timeout[] = {{"preset", "ollama"}, {"timeout_s", "0"}};
  CHECK(engine_status(bad_timeout, 2) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option bad_key[] = {{"preset", "ollama"}, {"api_key", "a\nb"}};
  CHECK(engine_status(bad_key, 2) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option unknown[] = {{"preset", "ollama"}, {"flavour", "x"}};
  CHECK(engine_status(unknown, 2) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option good[] = {{"preset", "lmstudio"}};
  CHECK(engine_status(good, 1) == UAIRT_OK);
}

static void test_load(const char* url, const char* keyed_url) {
  uairt_model* model = NULL;

  uairt_engine* engine = make_engine(url, NULL, 0);
  CHECK(load(engine, "nope", &model) == UAIRT_ERR_NOT_FOUND);
  CHECK(strstr(uairt_last_error(), "nope") != NULL);
  uairt_model_source memory = {.struct_size = sizeof(memory), .data = "x", .size = 1};
  CHECK(uairt_model_load(engine, &memory, &model) == UAIRT_ERR_UNSUPPORTED);
  CHECK(load(engine, "", &model) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(load(engine, "tagged", &model) == UAIRT_OK); /* the server lists "tagged:latest" */
  uairt_model_destroy(model);
  CHECK(load(engine, "tagged:lat", &model) == UAIRT_ERR_NOT_FOUND);
  CHECK(load(engine, "fake-model", &model) == UAIRT_OK);
  if (model) {
    CHECK(uairt_model_num_inputs(model) == 0);
    CHECK(uairt_model_run(model, NULL, 0, NULL, 0) == UAIRT_ERR_UNSUPPORTED);
    size_t count = 0;
    CHECK(uairt_model_tokenize(model, "a", 1, 0, NULL, 0, &count) == UAIRT_ERR_UNSUPPORTED);
    uairt_model_destroy(model);
  }
  uairt_engine_destroy(engine);

  /* SGLang serves one model, which it also answers to as "default". */
  uairt_option options[] = {{"base_url", url}, {"preset", "sglang"}};
  uairt_engine* sg = NULL;
  CHECK(uairt_engine_create("llmserver", options, 2, &sg) == UAIRT_OK);
  CHECK(load(sg, "default", &model) == UAIRT_OK);
  if (model) {
    uairt_model_destroy(model);
  }
  uairt_engine_destroy(sg);

  uairt_engine* nobody = make_engine("http://127.0.0.1:1", NULL, 0);
  CHECK(load(nobody, "fake-model", &model) == UAIRT_ERR_IO);
  CHECK(strstr(uairt_last_error(), "cannot reach") != NULL);
  uairt_engine_destroy(nobody);

  uairt_engine* keyless = make_engine(keyed_url, NULL, 0);
  CHECK(load(keyless, "fake-model", &model) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "bad key") != NULL);
  uairt_engine_destroy(keyless);
}

static void test_complete(const char* url, const char* keyed_url) {
  uairt_engine* engine = make_engine(url, NULL, 0);
  uairt_model* model = NULL;
  CHECK(load(engine, "fake-model", &model) == UAIRT_OK);
  if (!model) {
    uairt_engine_destroy(engine);
    return;
  }

  collector_t out = {0};
  CHECK(uairt_model_complete(model, "hi", NULL, 0, collect, &out) == UAIRT_OK);
  CHECK(strcmp(out.text, "model=fake-model;sys=;max=None;temp=None;top_p=None;seed=None;key=;prompt=hi") == 0);
  CHECK(out.pieces > 1);
  CHECK(strstr(out.text, "AFTER-DONE") == NULL);

  /* Quotes, a newline, a control character and non-ASCII text survive both directions. */
  const char* awkward = "say \"hi\"\n\tback\\slash \x01 h\xC3\xA9llo \xE2\x9C\x93 \xF0\x9F\x98\x80";
  uairt_option options[] = {{"system", "be brief"}, {"max_tokens", "16"}, {"temperature", "0.5"},
                            {"top_p", "1"}, {"seed", "7"}};
  memset(&out, 0, sizeof(out));
  CHECK(uairt_model_complete(model, awkward, options, 5, collect, &out) == UAIRT_OK);
  char expected[1024];
  snprintf(expected, sizeof(expected),
           "model=fake-model;sys=be brief;max=16;temp=0.5;top_p=1;seed=7;key=;prompt=%s", awkward);
  CHECK(strcmp(out.text, expected) == 0);

  memset(&out, 0, sizeof(out));
  out.stop_after = 2;
  CHECK(uairt_model_complete(model, "__count__", NULL, 0, collect, &out) == UAIRT_OK);
  CHECK(out.pieces == 2);

  memset(&out, 0, sizeof(out));
  CHECK(uairt_model_complete(model, "__error500__", NULL, 0, collect, &out) == UAIRT_ERR_RUNTIME);
  CHECK(strstr(uairt_last_error(), "boom") != NULL);
  CHECK(out.pieces == 0);

  memset(&out, 0, sizeof(out));
  CHECK(uairt_model_complete(model, "__midstream__", NULL, 0, collect, &out) == UAIRT_ERR_RUNTIME);
  CHECK(strstr(uairt_last_error(), "out of memory") != NULL);
  CHECK(strcmp(out.text, "partial") == 0);

  uairt_option zero_tokens[] = {{"max_tokens", "0"}};
  uairt_option hot[] = {{"temperature", "3"}};
  uairt_option bad_number[] = {{"temperature", "1e"}};
  uairt_option hex[] = {{"temperature", "0x1"}};
  uairt_option no_top_p[] = {{"top_p", "0"}};
  uairt_option negative_seed[] = {{"seed", "-1"}};
  uairt_option unknown[] = {{"mirostat", "1"}};
  const uairt_option* invalid[] = {zero_tokens, hot, bad_number, hex, no_top_p, negative_seed, unknown};
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
    CHECK(uairt_model_complete(model, "hi", invalid[i], 1, collect, &out) == UAIRT_ERR_INVALID_ARGUMENT);
  }
  CHECK(uairt_model_complete(model, NULL, NULL, 0, collect, &out) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_model_complete(model, "hi", NULL, 0, NULL, &out) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_model_destroy(model);

  CHECK(load(engine, "gone", &model) == UAIRT_OK);
  if (model) {
    CHECK(uairt_model_complete(model, "hi", NULL, 0, collect, &out) == UAIRT_ERR_NOT_FOUND);
    uairt_model_destroy(model);
  }
  uairt_engine_destroy(engine);

  uairt_option key[] = {{"api_key", "secret"}};
  uairt_engine* keyed = make_engine(keyed_url, key, 1);
  CHECK(load(keyed, "fake-model", &model) == UAIRT_OK);
  if (model) {
    memset(&out, 0, sizeof(out));
    CHECK(uairt_model_complete(model, "hi", NULL, 0, collect, &out) == UAIRT_OK);
    CHECK(strstr(out.text, ";key=secret;") != NULL);
    uairt_model_destroy(model);
  }
  uairt_engine_destroy(keyed);
}

static void test_unsupported_backend(const char* reference) {
  if (strcmp(reference, "-") == 0) {
    return;
  }
  CHECK(uairt_load_backend_library(reference) == UAIRT_OK);
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  CHECK(uairt_engine_create("reference", NULL, 0, &engine) == UAIRT_OK);
  if (engine && load(engine, "unused", &model) == UAIRT_OK) {
    collector_t out = {0};
    CHECK(uairt_model_complete(model, "hi", NULL, 0, collect, &out) == UAIRT_ERR_UNSUPPORTED);
    CHECK(out.pieces == 0);
    uairt_model_destroy(model);
  }
  uairt_engine_destroy(engine);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s <plugin> <reference plugin or -> <url> <keyed url>\n", argv[0]);
    return 2;
  }
  if (uairt_load_backend_library(argv[1]) != UAIRT_OK) {
    fprintf(stderr, "cannot load %s: %s\n", argv[1], uairt_last_error());
    return 2;
  }
  test_engine_options();
  test_load(argv[3], argv[4]);
  test_complete(argv[3], argv[4]);
  test_unsupported_backend(argv[2]);
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("llmserver: all checks passed");
  return 0;
}
