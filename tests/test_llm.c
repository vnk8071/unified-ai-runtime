/* SPDX-License-Identifier: Apache-2.0 */
/* uairt_llm on the fake session plugin (tests/session_test_plugin.c): after token t the logits favour t + 1. */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt_llm.h>

static int g_failures;

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                                \
    }                                                                              \
  } while (0)

static uairt_sampling make_sampling(float temperature, int32_t top_k, float top_p, uint64_t seed) {
  uairt_sampling s = uairt_sampling_default();
  s.temperature = temperature;
  s.top_k = top_k;
  s.top_p = top_p;
  s.seed = seed;
  return s;
}

static void test_sampling_validation(void) {
  float logits[3] = {0.0f, 1.0f, 2.0f};
  int32_t token = -1;
  uairt_rng rng;
  uairt_llm_rng_init(&rng, 1);
  uairt_sampling good = make_sampling(1.0f, 0, 1.0f, 1);
  CHECK(uairt_llm_sample(logits, 3, &good, &rng, &token) == UAIRT_OK);

  uairt_sampling bad[] = {make_sampling(-1.0f, 0, 1.0f, 0), make_sampling(NAN, 0, 1.0f, 0),
                          make_sampling(INFINITY, 0, 1.0f, 0), make_sampling(1.0f, -1, 1.0f, 0),
                          make_sampling(1.0f, 0, 0.0f, 0), make_sampling(1.0f, 0, 1.5f, 0)};
  for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
    CHECK(uairt_llm_sample(logits, 3, &bad[i], &rng, &token) == UAIRT_ERR_INVALID_ARGUMENT);
  }
  uairt_sampling short_struct = good;
  short_struct.struct_size = sizeof(size_t);
  CHECK(uairt_llm_sample(logits, 3, &short_struct, &rng, &token) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_llm_sample(logits, 0, &good, &rng, &token) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_llm_sample(NULL, 3, &good, &rng, &token) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_llm_sample(logits, 3, NULL, &rng, &token) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_llm_sample(logits, 3, &good, &rng, NULL) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_llm_sample(logits, 3, &good, NULL, &token) == UAIRT_ERR_INVALID_ARGUMENT);  /* sampling needs a state */
  float with_nan[3] = {0.0f, NAN, 1.0f};
  CHECK(uairt_llm_sample(with_nan, 3, &good, &rng, &token) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_sampling greedy = uairt_sampling_default();
  CHECK(uairt_llm_sample(with_nan, 3, &greedy, NULL, &token) == UAIRT_ERR_INVALID_ARGUMENT);
}

static void test_greedy(void) {
  float logits[5] = {0.5f, 3.0f, -1.0f, 3.0f, 2.0f};
  uairt_sampling greedy = uairt_sampling_default();
  int32_t token = -1;
  CHECK(uairt_llm_sample(logits, 5, &greedy, NULL, &token) == UAIRT_OK);
  CHECK(token == 1);  /* the first of equal maxima */
  uairt_rng rng;
  uairt_llm_rng_init(&rng, 9);
  uairt_sampling top1 = make_sampling(5.0f, 1, 1.0f, 9);  /* top_k 1 is greedy whatever the temperature */
  for (int i = 0; i < 20; ++i) {
    CHECK(uairt_llm_sample(logits, 5, &top1, &rng, &token) == UAIRT_OK && token == 1);
  }
}

static void test_distribution(void) {
  uairt_rng rng;
  uairt_llm_rng_init(&rng, 123);
  uairt_sampling s = make_sampling(1.0f, 0, 1.0f, 123);
  float even[2] = {0.0f, 0.0f};
  int ones = 0;
  for (int i = 0; i < 10000; ++i) {
    int32_t token = 0;
    CHECK(uairt_llm_sample(even, 2, &s, &rng, &token) == UAIRT_OK);
    ones += token == 1;
  }
  CHECK(ones > 4600 && ones < 5400);

  float skew[2] = {logf(3.0f), 0.0f};  /* probabilities 0.75 and 0.25 */
  int zeros = 0;
  for (int i = 0; i < 10000; ++i) {
    int32_t token = 0;
    CHECK(uairt_llm_sample(skew, 2, &s, &rng, &token) == UAIRT_OK);
    zeros += token == 0;
  }
  CHECK(zeros > 7300 && zeros < 7700);

  /* temperature flattens: at a high temperature the unlikely token is drawn more often than at 1 */
  uairt_sampling hot = make_sampling(100.0f, 0, 1.0f, 5);
  int hot_zeros = 0;
  for (int i = 0; i < 10000; ++i) {
    int32_t token = 0;
    CHECK(uairt_llm_sample(skew, 2, &hot, &rng, &token) == UAIRT_OK);
    hot_zeros += token == 0;
  }
  CHECK(hot_zeros > 4600 && hot_zeros < 5400);
}

static void test_cuts(void) {
  uairt_rng rng;
  uairt_llm_rng_init(&rng, 7);
  float probs[4] = {0.5f, 0.3f, 0.15f, 0.05f};
  float logits[4];
  for (int i = 0; i < 4; ++i) {
    logits[i] = logf(probs[i]);
  }
  int seen[4] = {0};
  uairt_sampling p70 = make_sampling(1.0f, 0, 0.7f, 7);  /* 0.5 + 0.3 reaches 0.7: tokens 0 and 1 only */
  for (int i = 0; i < 2000; ++i) {
    int32_t token = -1;
    CHECK(uairt_llm_sample(logits, 4, &p70, &rng, &token) == UAIRT_OK && token >= 0 && token < 4);
    seen[token]++;
  }
  CHECK(seen[0] > 0 && seen[1] > 0 && seen[2] == 0 && seen[3] == 0);
  CHECK(seen[0] > seen[1]);

  uairt_sampling p45 = make_sampling(1.0f, 0, 0.45f, 7);  /* the first token alone is enough */
  for (int i = 0; i < 200; ++i) {
    int32_t token = -1;
    CHECK(uairt_llm_sample(logits, 4, &p45, &rng, &token) == UAIRT_OK && token == 0);
  }

  uairt_sampling k2 = make_sampling(1.0f, 2, 1.0f, 7);
  memset(seen, 0, sizeof(seen));
  for (int i = 0; i < 2000; ++i) {
    int32_t token = -1;
    CHECK(uairt_llm_sample(logits, 4, &k2, &rng, &token) == UAIRT_OK && token >= 0 && token < 4);
    seen[token]++;
  }
  CHECK(seen[0] > 0 && seen[1] > 0 && seen[2] == 0 && seen[3] == 0);

  uairt_sampling k_big = make_sampling(1.0f, 100, 1.0f, 7);  /* larger than the vocabulary: no cut */
  memset(seen, 0, sizeof(seen));
  for (int i = 0; i < 4000; ++i) {
    int32_t token = -1;
    CHECK(uairt_llm_sample(logits, 4, &k_big, &rng, &token) == UAIRT_OK && token >= 0 && token < 4);
    seen[token]++;
  }
  CHECK(seen[0] > 0 && seen[1] > 0 && seen[2] > 0 && seen[3] > 0);
}

/* top_k picks from exactly the k largest of a big vocabulary, checked against a brute-force ranking */
static void test_top_k_selection(void) {
  enum { N = 1000, K = 7 };
  static float logits[N];
  uint32_t x = 12345;
  for (int i = 0; i < N; ++i) {
    x = x * 1664525u + 1013904223u;
    logits[i] = (float)(x >> 8) / 1000.0f + (float)i * 1e-6f;  /* distinct values */
  }
  bool in_top[N] = {false};
  for (int rank = 0; rank < K; ++rank) {
    int best = -1;
    for (int i = 0; i < N; ++i) {
      if (!in_top[i] && (best < 0 || logits[i] > logits[best])) {
        best = i;
      }
    }
    in_top[best] = true;
  }
  uairt_rng rng;
  uairt_llm_rng_init(&rng, 99);
  uairt_sampling s = make_sampling(1000.0f, K, 1.0f, 99);  /* near-uniform over the kept tokens */
  bool drawn[N] = {false};
  for (int i = 0; i < 3000; ++i) {
    int32_t token = -1;
    CHECK(uairt_llm_sample(logits, N, &s, &rng, &token) == UAIRT_OK && token >= 0 && token < N);
    CHECK(in_top[token]);
    drawn[token] = true;
  }
  int distinct = 0;
  for (int i = 0; i < N; ++i) {
    distinct += drawn[i];
  }
  CHECK(distinct == K);
}

static void test_seed(void) {
  float logits[10];
  for (int i = 0; i < 10; ++i) {
    logits[i] = (float)i * 0.1f;
  }
  int32_t a[40], b[40], c[40];
  uairt_sampling s = make_sampling(2.0f, 0, 0.9f, 77);
  uairt_sampling other = make_sampling(2.0f, 0, 0.9f, 78);
  uairt_rng rng;
  uairt_llm_rng_init(&rng, s.seed);
  for (int i = 0; i < 40; ++i) {
    CHECK(uairt_llm_sample(logits, 10, &s, &rng, &a[i]) == UAIRT_OK);
  }
  uairt_llm_rng_init(&rng, s.seed);
  for (int i = 0; i < 40; ++i) {
    CHECK(uairt_llm_sample(logits, 10, &s, &rng, &b[i]) == UAIRT_OK);
  }
  uairt_llm_rng_init(&rng, other.seed);
  for (int i = 0; i < 40; ++i) {
    CHECK(uairt_llm_sample(logits, 10, &other, &rng, &c[i]) == UAIRT_OK);
  }
  CHECK(memcmp(a, b, sizeof(a)) == 0);
  CHECK(memcmp(a, c, sizeof(a)) != 0);
}

typedef struct {
  char text[512];
  size_t length;
  int calls;
  int stop_after;
} collector;

static int collect(const char* text, size_t len, void* user_data) {
  collector* c = user_data;
  if (c->length + len < sizeof(c->text)) {
    memcpy(c->text + c->length, text, len);
    c->length += len;
  }
  c->calls++;
  return c->stop_after > 0 && c->calls >= c->stop_after;
}

static void test_generate(const char* plugin) {
  CHECK(uairt_load_backend_library(plugin) == UAIRT_OK);
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  uairt_model_source source = {.struct_size = sizeof(source), .path = "unused"};
  CHECK(uairt_engine_create("fake-session", NULL, 0, &engine) == UAIRT_OK);
  CHECK(uairt_model_load(engine, &source, &model) == UAIRT_OK);

  uairt_generate_options options = uairt_generate_options_default();
  uairt_generate_result result = {.struct_size = sizeof(result)};
  collector c;

  /* greedy: the fake model counts upward, so after "A" comes "BCD" */
  memset(&c, 0, sizeof(c));
  options.max_tokens = 3;
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &c, &result) == UAIRT_OK);
  CHECK(c.length == 3 && memcmp(c.text, "BCD", 3) == 0);
  CHECK(result.num_tokens == 3 && result.finish_reason == UAIRT_FINISH_LENGTH);

  /* no callback and no result are allowed; NULL options mean the defaults (64 tokens), which overrun this 8-token context */
  CHECK(uairt_llm_generate(model, NULL, "A", &options, NULL, NULL, NULL) == UAIRT_OK);
  CHECK(uairt_llm_generate(model, NULL, "A", NULL, NULL, NULL, NULL) != UAIRT_OK);

  /* a stop token ends the run and is not emitted */
  memset(&c, 0, sizeof(c));
  int32_t stop[1] = {'D'};
  options.max_tokens = 10;
  options.stop_tokens = stop;
  options.num_stop_tokens = 1;
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &c, &result) == UAIRT_OK);
  CHECK(c.length == 2 && memcmp(c.text, "BC", 2) == 0);
  CHECK(result.num_tokens == 2 && result.finish_reason == UAIRT_FINISH_STOP);
  options.stop_tokens = NULL;
  options.num_stop_tokens = 0;

  /* the callback can stop the run; the piece it was handed was delivered */
  memset(&c, 0, sizeof(c));
  c.stop_after = 2;
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &c, &result) == UAIRT_OK);
  CHECK(c.length == 2 && memcmp(c.text, "BC", 2) == 0);
  CHECK(result.num_tokens == 2 && result.finish_reason == UAIRT_FINISH_CALLBACK);

  /* max_tokens 0 generates nothing */
  memset(&c, 0, sizeof(c));
  options.max_tokens = 0;
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &c, &result) == UAIRT_OK);
  CHECK(c.calls == 0 && result.num_tokens == 0 && result.finish_reason == UAIRT_FINISH_LENGTH);

  /* the context is 8 tokens: running past it returns the backend's error, after the text produced so far */
  memset(&c, 0, sizeof(c));
  options.max_tokens = 20;
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &c, &result) != UAIRT_OK);
  CHECK(c.length >= 6 && memcmp(c.text, "BCDEFG", 6) == 0);
  CHECK(strstr(uairt_last_error(), "context") != NULL);

  /* a session passed in is left open and keeps its state, so a second call continues from it */
  uairt_session* session = NULL;
  CHECK(uairt_session_create(model, NULL, 0, &session) == UAIRT_OK);
  options.max_tokens = 2;
  memset(&c, 0, sizeof(c));
  CHECK(uairt_llm_generate(model, session, "A", &options, collect, &c, &result) == UAIRT_OK);
  size_t position = 0;
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 3);  /* BOS, 'A', 'B' */
  memset(&c, 0, sizeof(c));
  options.add_special = 0;
  CHECK(uairt_llm_generate(model, session, "X", &options, collect, &c, &result) == UAIRT_OK);
  CHECK(c.length == 2 && memcmp(c.text, "YZ", 2) == 0);  /* continues after the new prompt, not after the old text */
  CHECK(uairt_session_position(session, &position) == UAIRT_OK && position == 5);
  options.add_special = 1;

  /* invalid arguments */
  CHECK(uairt_llm_generate(NULL, NULL, "A", &options, collect, &c, &result) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(uairt_llm_generate(model, NULL, NULL, &options, collect, &c, &result) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_generate_options bad = options;
  bad.sampling.top_p = 2.0f;
  CHECK(uairt_llm_generate(model, NULL, "A", &bad, collect, &c, &result) == UAIRT_ERR_INVALID_ARGUMENT);
  bad = options;
  bad.struct_size = 4;
  CHECK(uairt_llm_generate(model, NULL, "A", &bad, collect, &c, &result) == UAIRT_ERR_INVALID_ARGUMENT);
  bad = options;
  bad.num_stop_tokens = 1;
  CHECK(uairt_llm_generate(model, NULL, "A", &bad, collect, &c, &result) == UAIRT_ERR_INVALID_ARGUMENT);
  options.add_special = 0;
  CHECK(uairt_llm_generate(model, NULL, "", &options, collect, &c, &result) == UAIRT_ERR_INVALID_ARGUMENT);  /* empty */
  options.add_special = 1;

  /* sampled generation is repeatable from the seed; the fake logits are nearly flat, so the text is noise */
  collector first, second, third;
  memset(&first, 0, sizeof(first));
  memset(&second, 0, sizeof(second));
  memset(&third, 0, sizeof(third));
  options.max_tokens = 6;
  options.sampling = make_sampling(1.0f, 0, 1.0f, 4242);
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &first, &result) == UAIRT_OK);
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &second, &result) == UAIRT_OK);
  options.sampling.seed = 4243;
  CHECK(uairt_llm_generate(model, NULL, "A", &options, collect, &third, &result) == UAIRT_OK);
  CHECK(first.length == second.length && memcmp(first.text, second.text, first.length) == 0);
  CHECK(first.length != third.length || memcmp(first.text, third.text, first.length) != 0);

  uairt_session_destroy(session);
  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

/* The model's own end of generation ('~' in the fake plugin) stops the run like a stop token, unless the caller opts out; a
 * backend whose table predates the entry never reports one. */
static void test_end_of_generation(const char* backend) {
  uairt_engine* engine = NULL;
  uairt_model* model = NULL;
  uairt_model_source source = {.struct_size = sizeof(source), .path = "unused"};
  CHECK(uairt_engine_create(backend, NULL, 0, &engine) == UAIRT_OK);
  CHECK(uairt_model_load(engine, &source, &model) == UAIRT_OK);
  int knows = strcmp(backend, "fake-session") == 0;

  uairt_generate_options options = uairt_generate_options_default();
  uairt_generate_result result = {.struct_size = sizeof(result)};
  collector c;
  memset(&c, 0, sizeof(c));
  options.max_tokens = 6;
  CHECK(uairt_llm_generate(model, NULL, "x", &options, collect, &c, &result) == UAIRT_OK);  /* y z { | } ~ */
  if (knows) {
    CHECK(c.length == 5 && memcmp(c.text, "yz{|}", 5) == 0);
    CHECK(result.num_tokens == 5 && result.finish_reason == UAIRT_FINISH_STOP);
  } else {
    CHECK(c.length == 6 && memcmp(c.text, "yz{|}~", 6) == 0);
    CHECK(result.num_tokens == 6 && result.finish_reason == UAIRT_FINISH_LENGTH);
  }

  memset(&c, 0, sizeof(c));
  options.ignore_end_of_generation = 1;
  CHECK(uairt_llm_generate(model, NULL, "x", &options, collect, &c, &result) == UAIRT_OK);
  CHECK(c.length == 6 && memcmp(c.text, "yz{|}~", 6) == 0);
  CHECK(result.num_tokens == 6 && result.finish_reason == UAIRT_FINISH_LENGTH);

  /* an options struct from before the field was added does not read it */
  uairt_generate_options older = uairt_generate_options_default();
  older.max_tokens = 6;
  older.ignore_end_of_generation = 1;  /* set, but outside what struct_size says the caller knows */
  older.struct_size = offsetof(uairt_generate_options, ignore_end_of_generation);
  memset(&c, 0, sizeof(c));
  CHECK(uairt_llm_generate(model, NULL, "x", &older, collect, &c, &result) == UAIRT_OK);
  CHECK(result.num_tokens == (knows ? 5u : 6u));

  uairt_model_destroy(model);
  uairt_engine_destroy(engine);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <fake-session plugin> <fake-short plugin>\n", argv[0]);
    return 2;
  }
  test_sampling_validation();
  test_greedy();
  test_distribution();
  test_cuts();
  test_top_k_selection();
  test_seed();
  test_generate(argv[1]);
  CHECK(uairt_load_backend_library(argv[2]) == UAIRT_OK);
  test_end_of_generation("fake-session");
  test_end_of_generation("fake-short");
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("ok");
  return 0;
}
