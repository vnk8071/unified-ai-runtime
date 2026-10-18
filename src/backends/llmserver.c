/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Language model server backend: attaches to a running server that speaks the OpenAI chat completions API (Ollama,
 * LM Studio, SGLang) over HTTP with libcurl. It never starts or installs a server. A model is a model name; it has no
 * tensors and no sessions, only the text API (uairt_model_complete).
 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curl/curl.h>
#include <uairt/uairt_backend.h>

#include "common.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#define OPTION_PRESET "preset"
#define OPTION_BASE_URL "base_url"
#define OPTION_API_KEY "api_key"
#define OPTION_TIMEOUT "timeout_s"
#define OPTION_MAX_TOKENS "max_tokens"
#define OPTION_TEMPERATURE "temperature"
#define OPTION_TOP_P "top_p"
#define OPTION_SEED "seed"
#define OPTION_SYSTEM "system"

#define URL_MAX 256
#define KEY_MAX 256
#define NAME_MAX_LEN 256
#define DEFAULT_TIMEOUT_S 120
#define CONNECT_TIMEOUT_S 5
#define MODELS_BODY_MAX (4u << 20)
#define LINE_MAX_BYTES (8u << 20)
#define ERROR_BODY_MAX (64u << 10)
#define JSON_MAX_DEPTH 64

static const uairt_host_api* g_host;

#define fail(...) uairt_backend_fail(g_host, __VA_ARGS__)

typedef struct {
  const char* name;
  const char* base_url;
} preset_t;

static const preset_t kPresets[] = {
    {"ollama", "http://127.0.0.1:11434"},
    {"lmstudio", "http://127.0.0.1:1234"},
    {"sglang", "http://127.0.0.1:30000"},
};

typedef struct {
  char preset[16];
  char base_url[URL_MAX];
  char api_key[KEY_MAX];
  long timeout_s;
} engine_t;

typedef struct {
  const engine_t* engine;
  char name[NAME_MAX_LEN];
} model_t;

/* ---- growable buffer ---- */

typedef struct {
  char* data;
  size_t len;
  size_t cap;
} buf_t;

static bool buf_append(buf_t* b, const char* data, size_t n) {
  if (n >= SIZE_MAX - b->len - 1) {
    return false;
  }
  if (b->len + n + 1 > b->cap) {
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + n + 1) {
      if (cap > SIZE_MAX / 2) {
        return false;
      }
      cap *= 2;
    }
    char* grown = realloc(b->data, cap);
    if (!grown) {
      return false;
    }
    b->data = grown;
    b->cap = cap;
  }
  memcpy(b->data + b->len, data, n);
  b->len += n;
  b->data[b->len] = '\0';
  return true;
}

static bool buf_puts(buf_t* b, const char* text) {
  return buf_append(b, text, strlen(text));
}

static void buf_free(buf_t* b) {
  free(b->data);
  memset(b, 0, sizeof(*b));
}

/* ---- JSON: just enough to read the fields of the replies ---- */

typedef struct {
  const char* p;
  const char* end;
} cursor_t;

static void skip_ws(cursor_t* c) {
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) {
    ++c->p;
  }
}

static int hex4(const char* p) {
  int value = 0;
  for (int i = 0; i < 4; ++i) {
    char ch = p[i];
    value <<= 4;
    if (ch >= '0' && ch <= '9') {
      value |= ch - '0';
    } else if (ch >= 'a' && ch <= 'f') {
      value |= ch - 'a' + 10;
    } else if (ch >= 'A' && ch <= 'F') {
      value |= ch - 'A' + 10;
    } else {
      return -1;
    }
  }
  return value;
}

static bool put_utf8(buf_t* out, unsigned long cp) {
  char bytes[4];
  size_t n;
  if (cp < 0x80) {
    bytes[0] = (char)cp;
    n = 1;
  } else if (cp < 0x800) {
    bytes[0] = (char)(0xC0 | (cp >> 6));
    bytes[1] = (char)(0x80 | (cp & 0x3F));
    n = 2;
  } else if (cp < 0x10000) {
    bytes[0] = (char)(0xE0 | (cp >> 12));
    bytes[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    bytes[2] = (char)(0x80 | (cp & 0x3F));
    n = 3;
  } else {
    bytes[0] = (char)(0xF0 | (cp >> 18));
    bytes[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    bytes[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    bytes[3] = (char)(0x80 | (cp & 0x3F));
    n = 4;
  }
  return buf_append(out, bytes, n);
}

/* `c` is at the opening quote. Appends the unescaped text to `out` (NULL: skip). Leaves `c` after the closing quote. */
static bool parse_string(cursor_t* c, buf_t* out) {
  if (c->p >= c->end || *c->p != '"') {
    return false;
  }
  ++c->p;
  while (c->p < c->end) {
    unsigned char ch = (unsigned char)*c->p++;
    if (ch == '"') {
      return true;
    }
    if (ch < 0x20) {
      return false;
    }
    if (ch != '\\') {
      if (out && !buf_append(out, (const char*)&ch, 1)) {
        return false;
      }
      continue;
    }
    if (c->p >= c->end) {
      return false;
    }
    char esc = *c->p++;
    char plain = 0;
    switch (esc) {
      case '"': plain = '"'; break;
      case '\\': plain = '\\'; break;
      case '/': plain = '/'; break;
      case 'b': plain = '\b'; break;
      case 'f': plain = '\f'; break;
      case 'n': plain = '\n'; break;
      case 'r': plain = '\r'; break;
      case 't': plain = '\t'; break;
      case 'u': {
        if (c->end - c->p < 4) {
          return false;
        }
        int hi = hex4(c->p);
        if (hi < 0) {
          return false;
        }
        c->p += 4;
        unsigned long cp = (unsigned long)hi;
        if (hi >= 0xD800 && hi <= 0xDBFF) {
          if (c->end - c->p < 6 || c->p[0] != '\\' || c->p[1] != 'u') {
            return false;
          }
          int lo = hex4(c->p + 2);
          if (lo < 0xDC00 || lo > 0xDFFF) {
            return false;
          }
          c->p += 6;
          cp = 0x10000 + (((unsigned long)hi - 0xD800) << 10) + ((unsigned long)lo - 0xDC00);
        } else if (hi >= 0xDC00 && hi <= 0xDFFF) {
          return false;
        }
        if (out && !put_utf8(out, cp)) {
          return false;
        }
        continue;
      }
      default:
        return false;
    }
    if (out && !buf_append(out, &plain, 1)) {
      return false;
    }
  }
  return false;
}

static bool skip_value(cursor_t* c, int depth) {
  if (depth > JSON_MAX_DEPTH) {
    return false;
  }
  skip_ws(c);
  if (c->p >= c->end) {
    return false;
  }
  if (*c->p == '"') {
    return parse_string(c, NULL);
  }
  if (*c->p == '{' || *c->p == '[') {
    char close = *c->p == '{' ? '}' : ']';
    bool is_object = close == '}';
    ++c->p;
    skip_ws(c);
    if (c->p < c->end && *c->p == close) {
      ++c->p;
      return true;
    }
    for (;;) {
      skip_ws(c);
      if (is_object) {
        if (!parse_string(c, NULL)) {
          return false;
        }
        skip_ws(c);
        if (c->p >= c->end || *c->p != ':') {
          return false;
        }
        ++c->p;
      }
      if (!skip_value(c, depth + 1)) {
        return false;
      }
      skip_ws(c);
      if (c->p >= c->end) {
        return false;
      }
      if (*c->p == ',') {
        ++c->p;
        continue;
      }
      if (*c->p == close) {
        ++c->p;
        return true;
      }
      return false;
    }
  }
  const char* start = c->p;
  while (c->p < c->end && !strchr(",]} \t\r\n", *c->p)) {
    ++c->p;
  }
  return c->p > start;
}

/* `c` is at an object. On success `*out` is at the value of `key`. */
static bool obj_find(cursor_t c, const char* key, cursor_t* out) {
  skip_ws(&c);
  if (c.p >= c.end || *c.p != '{') {
    return false;
  }
  ++c.p;
  skip_ws(&c);
  if (c.p < c.end && *c.p == '}') {
    return false;
  }
  for (;;) {
    skip_ws(&c);
    buf_t name = {0};
    bool ok = parse_string(&c, &name);
    bool match = ok && name.data && strcmp(name.data, key) == 0;
    buf_free(&name);
    if (!ok) {
      return false;
    }
    skip_ws(&c);
    if (c.p >= c.end || *c.p != ':') {
      return false;
    }
    ++c.p;
    skip_ws(&c);
    if (match) {
      *out = c;
      return true;
    }
    if (!skip_value(&c, 1)) {
      return false;
    }
    skip_ws(&c);
    if (c.p < c.end && *c.p == ',') {
      ++c.p;
      continue;
    }
    return false;
  }
}

/* `c` is at an array. On success `*out` is at element `index`. */
static bool arr_at(cursor_t c, size_t index, cursor_t* out) {
  skip_ws(&c);
  if (c.p >= c.end || *c.p != '[') {
    return false;
  }
  ++c.p;
  skip_ws(&c);
  if (c.p < c.end && *c.p == ']') {
    return false;
  }
  for (size_t i = 0;; ++i) {
    skip_ws(&c);
    if (i == index) {
      *out = c;
      return true;
    }
    if (!skip_value(&c, 1)) {
      return false;
    }
    skip_ws(&c);
    if (c.p < c.end && *c.p == ',') {
      ++c.p;
      continue;
    }
    return false;
  }
}

static bool value_is(cursor_t v, char first) {
  skip_ws(&v);
  return v.p < v.end && *v.p == first;
}

/* The text of an error reply: "error" as a string or as an object with "message", or a top-level "message". */
static void error_text(const char* json, size_t len, buf_t* out) {
  cursor_t root = {json, json + len};
  cursor_t v;
  if (obj_find(root, "error", &v)) {
    if (value_is(v, '"')) {
      skip_ws(&v);
      parse_string(&v, out);
      return;
    }
    cursor_t m;
    if (value_is(v, '{') && obj_find(v, "message", &m) && value_is(m, '"')) {
      skip_ws(&m);
      parse_string(&m, out);
      return;
    }
  }
  if (obj_find(root, "message", &v) && value_is(v, '"')) {
    skip_ws(&v);
    parse_string(&v, out);
  }
}

/* ---- HTTP ---- */

#if defined(_WIN32)
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_curl_once(PINIT_ONCE once, PVOID param, PVOID* context) {
  (void)once;
  (void)param;
  (void)context;
  curl_global_init(CURL_GLOBAL_DEFAULT);
  return TRUE;
}
static void ensure_curl(void) {
  InitOnceExecuteOnce(&g_once, init_curl_once, NULL, NULL);
}
#else
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void init_curl(void) {
  curl_global_init(CURL_GLOBAL_DEFAULT);
}
static void ensure_curl(void) {
  pthread_once(&g_once, init_curl);
}
#endif

typedef struct {
  CURL* curl;
  long http_code;
  buf_t body;      /* whole reply (GET), or the error body of a failed POST */
  size_t body_max;
  bool stream;     /* POST: parse server-sent events */
  buf_t line;
  uairt_text_callback callback;
  void* user_data;
  bool stopped;    /* the caller's callback asked to stop */
  bool done;       /* the stream ended with [DONE]; the rest of the reply is ignored */
  bool oom;
  bool overflow;
  buf_t stream_error;
} request_t;

/* Handles one "data:" payload of the event stream. Returns false to stop reading. */
static bool handle_event(request_t* r, const char* json, size_t len) {
  if (len == 6 && memcmp(json, "[DONE]", 6) == 0) {
    r->done = true;
    return false;
  }
  cursor_t root = {json, json + len};
  cursor_t choices, first, delta, content;
  if (obj_find(root, "choices", &choices) && arr_at(choices, 0, &first) && obj_find(first, "delta", &delta) &&
      obj_find(delta, "content", &content) && value_is(content, '"')) {
    buf_t text = {0};
    skip_ws(&content);
    bool ok = parse_string(&content, &text);
    if (ok && text.len) {
      if (r->callback(text.data, text.len, r->user_data) != 0) {
        r->stopped = true;
        buf_free(&text);
        return false;
      }
    } else if (!ok && !text.data) {
      r->oom = true;
    }
    buf_free(&text);
    return true;
  }
  if (!r->stream_error.len) {
    error_text(json, len, &r->stream_error);
  }
  return true;
}

static void consume_lines(request_t* r, bool* more) {
  size_t start = 0;
  while (*more) {
    char* nl = memchr(r->line.data + start, '\n', r->line.len - start);
    if (!nl) {
      break;
    }
    size_t end = (size_t)(nl - r->line.data);
    size_t line_len = end - start;
    const char* line = r->line.data + start;
    if (line_len && line[line_len - 1] == '\r') {
      --line_len;
    }
    if (line_len >= 5 && memcmp(line, "data:", 5) == 0) {
      line += 5;
      line_len -= 5;
      if (line_len && *line == ' ') {
        ++line;
        --line_len;
      }
      *more = handle_event(r, line, line_len);
    }
    start = end + 1;
  }
  memmove(r->line.data, r->line.data + start, r->line.len - start);
  r->line.len -= start;
  r->line.data[r->line.len] = '\0';
}

static size_t on_write(char* ptr, size_t size, size_t nmemb, void* userdata) {
  request_t* r = userdata;
  size_t n = size * nmemb;
  curl_easy_getinfo(r->curl, CURLINFO_RESPONSE_CODE, &r->http_code);
  if (!r->stream || r->http_code >= 400) {
    if (r->body.len + n > r->body_max) {
      r->overflow = true;
      return 0;
    }
    if (!buf_append(&r->body, ptr, n)) {
      r->oom = true;
      return 0;
    }
    return n;
  }
  if (r->done) {
    return n;
  }
  if (r->line.len + n > LINE_MAX_BYTES) {
    r->overflow = true;
    return 0;
  }
  if (!buf_append(&r->line, ptr, n)) {
    r->oom = true;
    return 0;
  }
  bool more = true;
  consume_lines(r, &more);
  (void)more;
  return r->stopped || r->oom ? 0 : n;
}

static uairt_status http_error_status(long code) {
  if (code == 404) {
    return UAIRT_ERR_NOT_FOUND;
  }
  return code >= 500 ? UAIRT_ERR_RUNTIME : UAIRT_ERR_INVALID_ARGUMENT;
}

/*
 * GET (body NULL) or POST `body` as JSON to `path`. The reply is read into `r`. A transport failure, an HTTP error
 * status or an unreadable reply returns a status with the error already set.
 */
static uairt_status perform(const engine_t* engine, const char* path, const char* body, request_t* r) {
  char url[URL_MAX + 64];
  snprintf(url, sizeof(url), "%s%s", engine->base_url, path);
  char errbuf[CURL_ERROR_SIZE] = "";
  CURL* curl = curl_easy_init();
  if (!curl) {
    fail("cannot initialize libcurl");
    return UAIRT_ERR_RUNTIME;
  }
  r->curl = curl;
  struct curl_slist* headers = NULL;
  if (body) {
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: text/event-stream");
  }
  if (engine->api_key[0]) {
    char auth[KEY_MAX + 32];
    snprintf(auth, sizeof(auth), "Authorization: Bearer %s", engine->api_key);
    headers = curl_slist_append(headers, auth);
  }
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errbuf);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, (long)CONNECT_TIMEOUT_S);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
#if LIBCURL_VERSION_NUM >= 0x075500
  curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https");
#else
  curl_easy_setopt(curl, CURLOPT_PROTOCOLS, (long)(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
  if (body) {
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, engine->timeout_s);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
  } else {
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, engine->timeout_s);
  }
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, on_write);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, r);

  CURLcode rc = curl_easy_perform(curl);
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r->http_code);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  r->curl = NULL;

  if (r->oom) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  if (r->stopped && rc == CURLE_WRITE_ERROR) {
    return UAIRT_OK;
  }
  if (r->http_code >= 400) {
    buf_t text = {0};
    if (r->body.len) {
      error_text(r->body.data, r->body.len, &text);
    }
    fail("%s answered HTTP %ld%s%s", engine->base_url, r->http_code, text.len ? ": " : "", text.len ? text.data : "");
    buf_free(&text);
    return http_error_status(r->http_code);
  }
  if (r->overflow) {
    fail("reply from %s is larger than the backend accepts", engine->base_url);
    return UAIRT_ERR_RUNTIME;
  }
  if (rc != CURLE_OK) {
    fail("cannot reach %s: %s", engine->base_url, errbuf[0] ? errbuf : curl_easy_strerror(rc));
    return UAIRT_ERR_IO;
  }
  return UAIRT_OK;
}

static void request_free(request_t* r) {
  buf_free(&r->body);
  buf_free(&r->line);
  buf_free(&r->stream_error);
}

/* ---- options ---- */

/* The JSON number grammar, so that a user value can be copied into the request without being reformatted. */
static bool is_json_number(const char* s) {
  const char* p = s;
  if (*p == '-') {
    ++p;
  }
  if (*p == '0') {
    ++p;
  } else if (*p >= '1' && *p <= '9') {
    while (*p >= '0' && *p <= '9') {
      ++p;
    }
  } else {
    return false;
  }
  if (*p == '.') {
    ++p;
    if (*p < '0' || *p > '9') {
      return false;
    }
    while (*p >= '0' && *p <= '9') {
      ++p;
    }
  }
  if (*p == 'e' || *p == 'E') {
    ++p;
    if (*p == '+' || *p == '-') {
      ++p;
    }
    if (*p < '0' || *p > '9') {
      return false;
    }
    while (*p >= '0' && *p <= '9') {
      ++p;
    }
  }
  return *p == '\0';
}

static bool parse_long(const char* text, long min, long max, long* out) {
  char* end = NULL;
  errno = 0;
  long value = strtol(text, &end, 10);
  if (end == text || *end || errno || value < min || value > max) {
    return false;
  }
  *out = value;
  return true;
}

static bool parse_real(const char* text, double min, double max, bool min_open, const char** out) {
  if (!is_json_number(text)) {
    return false;
  }
  double value = strtod(text, NULL);
  if (!isfinite(value) || value > max || value < min || (min_open && value == min)) {
    return false;
  }
  *out = text;
  return true;
}

static uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->timeout_s = DEFAULT_TIMEOUT_S;
  const char* preset_url = NULL;
  for (size_t i = 0; i < num_options; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    if (strcmp(key, OPTION_PRESET) == 0) {
      for (size_t p = 0; p < sizeof(kPresets) / sizeof(kPresets[0]); ++p) {
        if (strcmp(value, kPresets[p].name) == 0) {
          snprintf(engine->preset, sizeof(engine->preset), "%s", kPresets[p].name);
          preset_url = kPresets[p].base_url;
        }
      }
      if (!preset_url) {
        fail("%s must be ollama, lmstudio or sglang, got '%s'", OPTION_PRESET, value);
        free(engine);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (strcmp(key, OPTION_BASE_URL) == 0) {
      size_t n = strlen(value);
      while (n && value[n - 1] == '/') {
        --n;
      }
      bool http = strncmp(value, "http://", 7) == 0 || strncmp(value, "https://", 8) == 0;
      if (!http || n >= sizeof(engine->base_url) || n <= (value[4] == 's' ? 8u : 7u)) {
        fail("%s must be an http:// or https:// URL of at most %d characters, got '%s'", OPTION_BASE_URL, URL_MAX - 1,
             value);
        free(engine);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
      memcpy(engine->base_url, value, n);
      engine->base_url[n] = '\0';
    } else if (strcmp(key, OPTION_API_KEY) == 0) {
      if (strlen(value) >= sizeof(engine->api_key) || strpbrk(value, "\r\n")) {
        fail("%s must be a single line of at most %d characters", OPTION_API_KEY, KEY_MAX - 1);
        free(engine);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
      snprintf(engine->api_key, sizeof(engine->api_key), "%s", value);
    } else if (strcmp(key, OPTION_TIMEOUT) == 0) {
      if (!parse_long(value, 1, 86400, &engine->timeout_s)) {
        fail("%s must be an integer from 1 to 86400, got '%s'", OPTION_TIMEOUT, value);
        free(engine);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else {
      fail("unknown option '%s'", key);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  if (!engine->base_url[0]) {
    if (!preset_url) {
      fail("set the engine option %s (ollama, lmstudio or sglang) or %s", OPTION_PRESET, OPTION_BASE_URL);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    snprintf(engine->base_url, sizeof(engine->base_url), "%s", preset_url);
  }
  ensure_curl();
  *out_engine = engine;
  return UAIRT_OK;
}

static void destroy_engine(void* engine) {
  free(engine);
}

/* ---- model ---- */

/* Ollama lists "llama3.2:latest" and also accepts "llama3.2". */
static bool is_latest_of(const char* listed, const char* name) {
  size_t n = strlen(name);
  return strncmp(listed, name, n) == 0 && strcmp(listed + n, ":latest") == 0;
}

static uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const engine_t* engine = engine_handle;
  if (!source->path) {
    fail("llmserver takes a model name in `path`, not a buffer");
    return UAIRT_ERR_UNSUPPORTED;
  }
  if (!source->path[0] || strlen(source->path) >= NAME_MAX_LEN) {
    fail("the model name must be 1 to %d characters", NAME_MAX_LEN - 1);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }

  request_t r = {.body_max = MODELS_BODY_MAX};
  uairt_status status = perform(engine, "/v1/models", NULL, &r);
  if (status != UAIRT_OK) {
    request_free(&r);
    return status;
  }
  /* SGLang serves one model and accepts the name "default" for it. */
  bool listed = strcmp(engine->preset, "sglang") == 0 && strcmp(source->path, "default") == 0;
  cursor_t root = {r.body.data ? r.body.data : "", r.body.data ? r.body.data + r.body.len : ""};
  cursor_t data, item, id;
  bool parsed = obj_find(root, "data", &data) && value_is(data, '[');
  for (size_t i = 0; parsed && !listed && arr_at(data, i, &item); ++i) {
    buf_t name = {0};
    if (obj_find(item, "id", &id) && value_is(id, '"')) {
      skip_ws(&id);
      parse_string(&id, &name);
    }
    listed = name.data && (strcmp(name.data, source->path) == 0 || is_latest_of(name.data, source->path));
    buf_free(&name);
  }
  request_free(&r);
  if (!parsed && !listed) {
    fail("%s/v1/models did not return a model list; is it an OpenAI-compatible server?", engine->base_url);
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  if (!listed) {
    fail("the server at %s does not list a model named '%s'", engine->base_url, source->path);
    return UAIRT_ERR_NOT_FOUND;
  }

  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  model->engine = engine;
  snprintf(model->name, sizeof(model->name), "%s", source->path);
  *out_model = model;
  return UAIRT_OK;
}

static void destroy_model(void* model) {
  free(model);
}

static size_t num_io(const void* model) {
  (void)model;
  return 0;
}

static uairt_status io_info(const void* model, size_t index, uairt_tensor* out) {
  (void)model;
  (void)index;
  (void)out;
  fail("llmserver models have no tensors; use uairt_model_complete");
  return UAIRT_ERR_INVALID_ARGUMENT;
}

static uairt_status run(void* model, const uairt_tensor* in, size_t n_in, uairt_tensor* out, size_t n_out) {
  (void)model;
  (void)in;
  (void)n_in;
  (void)out;
  (void)n_out;
  fail("llmserver models run through uairt_model_complete, not uairt_model_run");
  return UAIRT_ERR_UNSUPPORTED;
}

/* Appends `text` as a JSON string literal. */
static bool json_string(buf_t* out, const char* text) {
  if (!buf_puts(out, "\"")) {
    return false;
  }
  for (const char* p = text; *p; ++p) {
    unsigned char ch = (unsigned char)*p;
    char escaped[8];
    switch (ch) {
      case '"': strcpy(escaped, "\\\""); break;
      case '\\': strcpy(escaped, "\\\\"); break;
      case '\n': strcpy(escaped, "\\n"); break;
      case '\r': strcpy(escaped, "\\r"); break;
      case '\t': strcpy(escaped, "\\t"); break;
      default:
        if (ch < 0x20) {
          snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
        } else {
          escaped[0] = (char)ch;
          escaped[1] = '\0';
        }
    }
    if (!buf_puts(out, escaped)) {
      return false;
    }
  }
  return buf_puts(out, "\"");
}

static uairt_status complete(
    void* handle,
    const char* prompt,
    const uairt_option* options,
    size_t num_options,
    uairt_text_callback callback,
    void* user_data) {
  const model_t* model = handle;
  const char* system = NULL;
  const char* temperature = NULL;
  const char* top_p = NULL;
  long max_tokens = 0;
  long seed = 0;
  bool has_seed = false;
  for (size_t i = 0; i < num_options; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    bool ok = true;
    if (strcmp(key, OPTION_SYSTEM) == 0) {
      system = value;
    } else if (strcmp(key, OPTION_MAX_TOKENS) == 0) {
      ok = parse_long(value, 1, INT_MAX, &max_tokens);
    } else if (strcmp(key, OPTION_SEED) == 0) {
      ok = parse_long(value, 0, INT_MAX, &seed);
      has_seed = ok;
    } else if (strcmp(key, OPTION_TEMPERATURE) == 0) {
      ok = parse_real(value, 0.0, 2.0, false, &temperature);
    } else if (strcmp(key, OPTION_TOP_P) == 0) {
      ok = parse_real(value, 0.0, 1.0, true, &top_p);
    } else {
      fail("unknown option '%s'", key);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    if (!ok) {
      fail("invalid value '%s' for option '%s'", value, key);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }

  buf_t body = {0};
  bool ok = buf_puts(&body, "{\"model\":") && json_string(&body, model->name) && buf_puts(&body, ",\"stream\":true");
  char number[48];
  if (ok && max_tokens) {
    snprintf(number, sizeof(number), ",\"max_tokens\":%ld", max_tokens);
    ok = buf_puts(&body, number);
  }
  if (ok && has_seed) {
    snprintf(number, sizeof(number), ",\"seed\":%ld", seed);
    ok = buf_puts(&body, number);
  }
  if (ok && temperature) {
    ok = buf_puts(&body, ",\"temperature\":") && buf_puts(&body, temperature);
  }
  if (ok && top_p) {
    ok = buf_puts(&body, ",\"top_p\":") && buf_puts(&body, top_p);
  }
  ok = ok && buf_puts(&body, ",\"messages\":[");
  if (ok && system && *system) {
    ok = buf_puts(&body, "{\"role\":\"system\",\"content\":") && json_string(&body, system) && buf_puts(&body, "},");
  }
  ok = ok && buf_puts(&body, "{\"role\":\"user\",\"content\":") && json_string(&body, prompt) &&
       buf_puts(&body, "}]}");
  if (!ok) {
    buf_free(&body);
    return UAIRT_ERR_OUT_OF_MEMORY;
  }

  request_t r = {.stream = true, .body_max = ERROR_BODY_MAX, .callback = callback, .user_data = user_data};
  uairt_status status = perform(model->engine, "/v1/chat/completions", body.data, &r);
  if (status == UAIRT_OK && r.stream_error.len) {
    fail("the server reported an error: %s", r.stream_error.data);
    status = UAIRT_ERR_RUNTIME;
  }
  request_free(&r);
  buf_free(&body);
  return status;
}

static const uairt_text_api kText = {
    .struct_size = sizeof(uairt_text_api),
    .abi_version = UAIRT_TEXT_ABI_VERSION,
    .complete = complete,
};

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "llmserver",
    .supported_domains = UAIRT_MEM_HOST,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_io,
    .num_outputs = num_io,
    .input_info = io_info,
    .output_info = io_info,
    .run = run,
    .text = &kText,
};

UAIRT_API const uairt_backend_api* UAIRT_BACKEND_ENTRY(const uairt_host_api* host) {
  g_host = host;
  return &kApi;
}
