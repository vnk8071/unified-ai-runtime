/* SPDX-License-Identifier: Apache-2.0 */
#ifndef UAIRT_UAIRT_LLM_H
#define UAIRT_UAIRT_LLM_H

/*
 * Text generation over any backend that has sessions (today llamacpp): sampling, the generation loop, stop conditions
 * and streaming text. It is a separate library (libuairt_llm) on top of the public C API, not part of the core ABI, and
 * it uses nothing a plugin does not already offer: uairt_model_tokenize, uairt_model_detokenize and the session calls.
 */

#include <uairt/uairt.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sampling. Greedy when temperature is 0; otherwise the logits are divided by temperature, cut to the `top_k` largest
 * (0 means no limit), softmaxed, cut to the smallest set whose probability reaches `top_p` (1 means no cut), and one
 * token is drawn. `seed` makes a run repeatable: the same seed, options and logits give the same tokens.
 */
typedef struct uairt_sampling {
  size_t struct_size;
  float temperature; /* finite, >= 0 */
  int32_t top_k;     /* >= 0 */
  float top_p;       /* in (0, 1] */
  uint64_t seed;
} uairt_sampling;

/* temperature 0, top_k 0, top_p 1, seed 0, with struct_size set. */
UAIRT_API uairt_sampling uairt_sampling_default(void);

/* The random state advances on each draw; start it at the seed with uairt_llm_rng_init. Greedy sampling ignores it. */
typedef struct uairt_rng {
  uint64_t state;
} uairt_rng;

UAIRT_API void uairt_llm_rng_init(uairt_rng* rng, uint64_t seed);

/* One token id from `n` logits (`n` > 0, all finite). */
UAIRT_API uairt_status uairt_llm_sample(
    const float* logits,
    size_t n,
    const uairt_sampling* sampling,
    uairt_rng* rng,
    int32_t* out_token);

typedef struct uairt_generate_options {
  size_t struct_size;
  size_t max_tokens;
  uairt_sampling sampling;
  const int32_t* stop_tokens; /* generation ends, without emitting the token, when one of these is drawn */
  size_t num_stop_tokens;
  int add_special; /* passed to uairt_model_tokenize for the prompt (adds BOS where the model has one) */
  /*
   * Appended fields: read only when struct_size covers them. By default generation also ends, like llama.cpp's own tools,
   * when the model draws a token that ends a generation (uairt_model_is_end_of_generation); set this to keep going.
   */
  int ignore_end_of_generation;
} uairt_generate_options;

UAIRT_API uairt_generate_options uairt_generate_options_default(void);

enum {
  UAIRT_FINISH_LENGTH = 0,   /* max_tokens reached */
  UAIRT_FINISH_STOP = 1,     /* a stop token, or the model's own end of generation, was drawn */
  UAIRT_FINISH_CALLBACK = 2, /* the callback returned non-zero */
};

typedef struct uairt_generate_result {
  size_t struct_size;
  size_t num_tokens; /* tokens generated and kept, not counting a stop token */
  int finish_reason;
} uairt_generate_result;

/*
 * Tokenizes `prompt`, appends it to `session`, then samples up to `max_tokens` tokens and passes the text to `callback`
 * as it is produced (UTF-8; the bytes of a character split across tokens are held back until it is complete, and a
 * truncated character at the end is replaced by U+FFFD). `session` may already hold earlier text, so a conversation
 * continues where it left off; pass NULL to use a fresh session that is destroyed before returning. `options` may be
 * NULL for the defaults and `callback` may be NULL to generate without streaming; `out_result` may be NULL.
 *
 * Like uairt_model_tokenize this parses special-token text such as `<|eot_id|>` in the prompt where the backend does:
 * do not pass untrusted text if that matters. Errors from the backend (a prompt longer than the context, for example)
 * are returned as they are; text already delivered stays delivered.
 */
UAIRT_API uairt_status uairt_llm_generate(
    uairt_model* model,
    uairt_session* session,
    const char* prompt,
    const uairt_generate_options* options,
    uairt_text_callback callback,
    void* user_data,
    uairt_generate_result* out_result);

#ifdef __cplusplus
}
#endif

#endif
