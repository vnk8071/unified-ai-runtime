// SPDX-License-Identifier: Apache-2.0
// Sampling and the generation loop over the session API. Like the core: no exceptions, RTTI, STL or operator new.
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <uairt/uairt_llm.h>

namespace {

struct Candidates {
  const float* logits;
  bool less(int32_t a, int32_t b) const {  // a ranks below b; the lower index wins a tie so results are reproducible
    return logits[a] < logits[b] || (logits[a] == logits[b] && a > b);
  }
};

void sift_down(const Candidates& c, int32_t* heap, size_t size, size_t i) {
  for (;;) {
    size_t smallest = i, left = 2 * i + 1, right = left + 1;
    if (left < size && c.less(heap[left], heap[smallest])) {
      smallest = left;
    }
    if (right < size && c.less(heap[right], heap[smallest])) {
      smallest = right;
    }
    if (smallest == i) {
      return;
    }
    int32_t t = heap[i];
    heap[i] = heap[smallest];
    heap[smallest] = t;
    i = smallest;
  }
}

// The `k` largest logits as indices, largest first. A min-heap of size k keeps the best seen so far.
void top_k_indices(const float* logits, size_t n, size_t k, int32_t* out) {
  Candidates c{logits};
  size_t size = 0;
  for (size_t i = 0; i < n; ++i) {
    int32_t index = (int32_t)i;
    if (size < k) {
      out[size++] = index;
      if (size == k) {
        for (size_t j = k / 2; j-- > 0;) {
          sift_down(c, out, size, j);
        }
      }
    } else if (c.less(out[0], index)) {
      out[0] = index;
      sift_down(c, out, size, 0);
    }
  }
  if (size < k) {  // unreachable for k <= n, kept so a short heap is still ordered
    for (size_t j = size / 2; j-- > 0;) {
      sift_down(c, out, size, j);
    }
  }
  for (size_t end = size; end > 1; --end) {  // heapsort: repeatedly move the smallest to the back
    int32_t t = out[0];
    out[0] = out[end - 1];
    out[end - 1] = t;
    sift_down(c, out, end - 1, 0);
  }
}

uint64_t next_u64(uairt_rng* rng) {  // splitmix64
  uint64_t z = (rng->state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

double next_unit(uairt_rng* rng) {  // [0, 1)
  return (double)(next_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

bool sampling_valid(const uairt_sampling* s) {
  return s && s->struct_size >= sizeof(uairt_sampling) && isfinite(s->temperature) && s->temperature >= 0.0f &&
         s->top_k >= 0 && s->top_p > 0.0f && s->top_p <= 1.0f;
}

// Length of the prefix of `data` that ends on a character boundary: a lead byte whose continuation bytes have not all
// arrived is held back. Bytes that are invalid for other reasons pass through.
size_t utf8_complete_prefix(const unsigned char* data, size_t size) {
  for (size_t back = 1; back <= 3 && back <= size; ++back) {
    unsigned char byte = data[size - back];
    if ((byte & 0xC0) == 0x80) {
      continue;  // continuation byte: keep looking for its lead byte
    }
    size_t need = byte >= 0xF0 && byte < 0xF8 ? 4 : byte >= 0xE0 ? 3 : byte >= 0xC0 ? 2 : 1;
    if (byte >= 0xF8) {
      need = 1;
    }
    return back < need ? size - back : size;
  }
  return size;
}

}  // namespace

extern "C" {

uairt_sampling uairt_sampling_default(void) {
  uairt_sampling s = {};
  s.struct_size = sizeof(s);
  s.top_p = 1.0f;
  return s;
}

uairt_generate_options uairt_generate_options_default(void) {
  uairt_generate_options o = {};
  o.struct_size = sizeof(o);
  o.max_tokens = 64;
  o.sampling = uairt_sampling_default();
  o.add_special = 1;
  return o;
}

void uairt_llm_rng_init(uairt_rng* rng, uint64_t seed) {
  if (rng) {
    rng->state = seed;
  }
}

uairt_status uairt_llm_sample(const float* logits, size_t n, const uairt_sampling* sampling, uairt_rng* rng,
                              int32_t* out_token) {
  if (!logits || n == 0 || n > (size_t)INT32_MAX || !sampling_valid(sampling) || !out_token) {
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(logits[i])) {
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  if (sampling->temperature == 0.0f) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
      if (logits[i] > logits[best]) {
        best = i;
      }
    }
    *out_token = (int32_t)best;
    return UAIRT_OK;
  }
  if (!rng) {
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  const double inv_t = 1.0 / (double)sampling->temperature;
  const bool cut_k = sampling->top_k > 0 && (size_t)sampling->top_k < n;
  const bool cut_p = sampling->top_p < 1.0f;

  if (!cut_k && !cut_p) {  // the whole vocabulary, in index order: no sorting needed
    float peak = logits[0];
    for (size_t i = 1; i < n; ++i) {
      peak = logits[i] > peak ? logits[i] : peak;
    }
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
      total += exp(((double)logits[i] - (double)peak) * inv_t);
    }
    double target = next_unit(rng) * total, running = 0.0;
    size_t pick = n - 1;
    for (size_t i = 0; i < n; ++i) {
      running += exp(((double)logits[i] - (double)peak) * inv_t);
      if (running > target) {
        pick = i;
        break;
      }
    }
    *out_token = (int32_t)pick;
    return UAIRT_OK;
  }

  size_t k = cut_k ? (size_t)sampling->top_k : n;
  int32_t* order = static_cast<int32_t*>(malloc(k * sizeof(int32_t)));
  double* probs = static_cast<double*>(malloc(k * sizeof(double)));
  if (!order || !probs) {
    free(order);
    free(probs);
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  top_k_indices(logits, n, k, order);
  double total = 0.0;
  for (size_t i = 0; i < k; ++i) {
    probs[i] = exp(((double)logits[order[i]] - (double)logits[order[0]]) * inv_t);
    total += probs[i];
  }
  size_t keep = k;
  if (cut_p) {  // the smallest prefix whose normalised probability reaches top_p
    double running = 0.0;
    for (size_t i = 0; i < k; ++i) {
      running += probs[i] / total;
      if (running >= (double)sampling->top_p) {
        keep = i + 1;
        break;
      }
    }
    total = 0.0;
    for (size_t i = 0; i < keep; ++i) {
      total += probs[i];
    }
  }
  double target = next_unit(rng) * total, running = 0.0;
  size_t pick = keep - 1;
  for (size_t i = 0; i < keep; ++i) {
    running += probs[i];
    if (running > target) {
      pick = i;
      break;
    }
  }
  *out_token = order[pick];
  free(order);
  free(probs);
  return UAIRT_OK;
}

uairt_status uairt_llm_generate(uairt_model* model, uairt_session* session, const char* prompt,
                                const uairt_generate_options* options, uairt_text_callback callback, void* user_data,
                                uairt_generate_result* out_result) {
  uairt_generate_options defaults = uairt_generate_options_default();
  const uairt_generate_options* opts = options ? options : &defaults;
  if (!model || !prompt || opts->struct_size < offsetof(uairt_generate_options, ignore_end_of_generation) ||
      !sampling_valid(&opts->sampling) ||
      (opts->num_stop_tokens && !opts->stop_tokens) || (out_result && out_result->struct_size < sizeof(*out_result))) {
    return UAIRT_ERR_INVALID_ARGUMENT;
  }

  const bool stop_at_end =
      opts->struct_size < offsetof(uairt_generate_options, ignore_end_of_generation) + sizeof(int) ||
      !opts->ignore_end_of_generation;
  uairt_status status = UAIRT_OK;
  uairt_session* own = NULL;
  int32_t* prompt_tokens = NULL;
  float* logits = NULL;
  int32_t* generated = NULL;
  unsigned char* text = NULL;
  size_t text_capacity = 0, emitted = 0, num_generated = 0;
  int finish = UAIRT_FINISH_LENGTH;

  if (!session) {
    if ((status = uairt_session_create(model, NULL, 0, &own)) != UAIRT_OK) {
      return status;
    }
    session = own;
  }

  size_t vocab = 0, prompt_count = 0;
  if ((status = uairt_model_vocab_size(model, &vocab)) != UAIRT_OK) {
    goto done;
  }
  size_t prompt_len;
  prompt_len = strlen(prompt);
  uairt_model_tokenize(model, prompt, prompt_len, opts->add_special, NULL, 0, &prompt_count);  // capacity 0: the size
  if (prompt_count == 0) {
    status = UAIRT_ERR_INVALID_ARGUMENT;
    goto done;
  }
  prompt_tokens = static_cast<int32_t*>(malloc(prompt_count * sizeof(int32_t)));
  logits = static_cast<float*>(malloc(vocab * sizeof(float)));
  generated = static_cast<int32_t*>(malloc((opts->max_tokens ? opts->max_tokens : 1) * sizeof(int32_t)));
  if (!prompt_tokens || !logits || !generated) {
    status = UAIRT_ERR_OUT_OF_MEMORY;
    goto done;
  }
  if ((status = uairt_model_tokenize(model, prompt, prompt_len, opts->add_special, prompt_tokens, prompt_count,
                                     &prompt_count)) != UAIRT_OK ||
      (status = uairt_session_append(session, prompt_tokens, prompt_count)) != UAIRT_OK) {
    goto done;
  }

  uairt_rng rng;
  uairt_llm_rng_init(&rng, opts->sampling.seed);
  while (num_generated < opts->max_tokens) {
    size_t n = 0;
    int32_t token = 0;
    if ((status = uairt_session_logits(session, logits, vocab, &n)) != UAIRT_OK ||
        (status = uairt_llm_sample(logits, n, &opts->sampling, &rng, &token)) != UAIRT_OK) {
      goto done;
    }
    bool stop = false;
    for (size_t i = 0; i < opts->num_stop_tokens; ++i) {
      stop = stop || opts->stop_tokens[i] == token;
    }
    int ends = 0;  // a backend that cannot tell (UNSUPPORTED) leaves this to the caller's stop tokens
    if (!stop && stop_at_end && uairt_model_is_end_of_generation(model, token, &ends) == UAIRT_OK) {
      stop = ends != 0;
    }
    if (stop) {
      finish = UAIRT_FINISH_STOP;
      break;
    }
    generated[num_generated++] = token;

    // Detokenize everything generated so far and emit only the new bytes: one token can be half a character, and a
    // token's text can depend on the tokens before it.
    size_t nbytes = 0;
    uairt_model_detokenize(model, generated, num_generated, NULL, 0, &nbytes);
    if (nbytes > text_capacity) {
      unsigned char* grown = static_cast<unsigned char*>(realloc(text, nbytes));
      if (!grown) {
        status = UAIRT_ERR_OUT_OF_MEMORY;
        goto done;
      }
      text = grown;
      text_capacity = nbytes;
    }
    if (nbytes && (status = uairt_model_detokenize(model, generated, num_generated, reinterpret_cast<char*>(text),
                                                   nbytes, &nbytes)) != UAIRT_OK) {
      goto done;
    }
    size_t ready = utf8_complete_prefix(text, nbytes);
    if (ready > emitted) {
      if (callback && callback(reinterpret_cast<const char*>(text) + emitted, ready - emitted, user_data) != 0) {
        emitted = ready;
        finish = UAIRT_FINISH_CALLBACK;
        break;
      }
      emitted = ready;
    }
    if (num_generated < opts->max_tokens &&
        (status = uairt_session_append(session, &token, 1)) != UAIRT_OK) {
      goto done;
    }
  }
  if (finish != UAIRT_FINISH_CALLBACK) {  // a truncated character at the end becomes U+FFFD
    size_t nbytes = 0;
    if (num_generated) {
      uairt_model_detokenize(model, generated, num_generated, NULL, 0, &nbytes);
      if (nbytes > emitted && nbytes <= text_capacity && callback) {
        callback("\xEF\xBF\xBD", 3, user_data);
      }
    }
  }
  if (out_result) {
    out_result->num_tokens = num_generated;
    out_result->finish_reason = finish;
  }

done:
  free(text);
  free(generated);
  free(logits);
  free(prompt_tokens);
  if (own) {
    uairt_session_destroy(own);
  }
  return status;
}

}  // extern "C"
