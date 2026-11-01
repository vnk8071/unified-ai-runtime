/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Transformers.js backend: a Node process (transformersjs_worker.mjs) runs a Transformers.js model and this plugin talks to
 * it over a socket pair. The model is a Transformers.js model directory (config.json and onnx/model.onnx or a dtype variant of it).
 * Shapes are static: the plugin reports `input_shapes` (or the model's own static shapes) and learns the output shapes from
 * one run on zeros at load time. POSIX only. The user installs Node and @huggingface/transformers.
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1 /* SO_NOSIGPIPE */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common.h"

#ifndef UAIRT_TRANSFORMERSJS_WORKER
#define UAIRT_TRANSFORMERSJS_WORKER "transformersjs_worker.mjs"
#endif

#define MAX_TENSORS 16
#define MAX_NAME 128
#define MAX_HEADER (1u << 20)
#define MAX_LOG (4u << 20)

extern char** environ;

static const uairt_host_api* g_host;
#define fail(...) uairt_backend_fail(g_host, __VA_ARGS__)

typedef struct {
  char name[MAX_NAME];
  uairt_dtype dtype;
  uint32_t rank;
  int64_t dims[UAIRT_MAX_RANK];
  size_t nbytes;
} spec_t;

typedef struct {
  char* node;
  char* worker;
  char* package_dir;
  char* device;
  char* dtype;
  char* input_shapes;
} engine_t;

typedef struct {
  pid_t pid;
  int fd;
  int log_fd;
  spec_t inputs[MAX_TENSORS];
  spec_t outputs[MAX_TENSORS];
  size_t num_inputs;
  size_t num_outputs;
  size_t input_nbytes;
  size_t output_nbytes;
} model_t;

static char* copy_string(const char* text) {
  size_t length = strlen(text);
  char* copy = malloc(length + 1);
  if (copy) {
    memcpy(copy, text, length + 1);
  }
  return copy;
}

static void free_engine(engine_t* engine) {
  if (engine) {
    free(engine->node);
    free(engine->worker);
    free(engine->package_dir);
    free(engine->device);
    free(engine->dtype);
    free(engine->input_shapes);
    free(engine);
  }
}

static int set_string(char** slot, const char* value) {
  char* copy = copy_string(value);
  if (!copy) {
    return 0;
  }
  free(*slot);
  *slot = copy;
  return 1;
}

static uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  const char* env_dir = getenv("UAIRT_TRANSFORMERSJS_DIR");
  const char* env_worker = getenv("UAIRT_TRANSFORMERSJS_WORKER");
  if (!set_string(&engine->node, "node") || !set_string(&engine->worker, env_worker && env_worker[0] ? env_worker : UAIRT_TRANSFORMERSJS_WORKER) ||
      !set_string(&engine->package_dir, env_dir && env_dir[0] ? env_dir : ".") || !set_string(&engine->device, "cpu") ||
      !set_string(&engine->dtype, "fp32") || !set_string(&engine->input_shapes, "")) {
    free_engine(engine);
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  for (size_t i = 0; i < num_options; ++i) {
    const char* key = options[i].key;
    const char* value = options[i].value ? options[i].value : "";
    char** slot = NULL;
    if (strcmp(key, "node") == 0) {
      slot = &engine->node;
    } else if (strcmp(key, "worker") == 0) {
      slot = &engine->worker;
    } else if (strcmp(key, "package_dir") == 0) {
      slot = &engine->package_dir;
    } else if (strcmp(key, "device") == 0) {
      if (strcmp(value, "cpu") != 0 && strcmp(value, "webgpu") != 0) {
        fail("device must be 'cpu' or 'webgpu', got '%s'", value);
        free_engine(engine);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
      slot = &engine->device;
    } else if (strcmp(key, "dtype") == 0) {
      slot = &engine->dtype;
    } else if (strcmp(key, "input_shapes") == 0) {
      slot = &engine->input_shapes;
    } else {
      fail("unknown option '%s'", key);
      free_engine(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    if (strchr(value, '\n') || !set_string(slot, value)) {
      fail("option '%s' contains a newline or the plugin is out of memory", key);
      free_engine(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  *out_engine = engine;
  return UAIRT_OK;
}

static void destroy_engine(void* engine) {
  free_engine(engine);
}

static int write_all(int fd, const void* data, size_t size) {
  const char* bytes = data;
  while (size > 0) {
#ifdef MSG_NOSIGNAL
    ssize_t n = send(fd, bytes, size, MSG_NOSIGNAL);
#else
    ssize_t n = send(fd, bytes, size, 0);
#endif
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return 0;
    }
    bytes += n;
    size -= (size_t)n;
  }
  return 1;
}

static int read_all(int fd, void* data, size_t size) {
  char* bytes = data;
  while (size > 0) {
    ssize_t n = read(fd, bytes, size);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return 0;
    }
    bytes += n;
    size -= (size_t)n;
  }
  return 1;
}

static void put_u32(unsigned char* out, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out[i] = (unsigned char)(value >> (8 * i));
  }
}

static uint32_t get_u32(const unsigned char* in) {
  return (uint32_t)in[0] | (uint32_t)in[1] << 8 | (uint32_t)in[2] << 16 | (uint32_t)in[3] << 24;
}

static int send_frame(int fd, const char* header, const void* payload, size_t payload_size) {
  unsigned char prefix[8];
  size_t header_size = strlen(header);
  if (header_size > MAX_HEADER || payload_size > UINT32_MAX) {
    return 0;
  }
  put_u32(prefix, (uint32_t)header_size);
  put_u32(prefix + 4, (uint32_t)payload_size);
  return write_all(fd, prefix, 8) && write_all(fd, header, header_size) && (payload_size == 0 || write_all(fd, payload, payload_size));
}

/* Reads a frame: `*header` is a NUL terminated malloc'd string, `*payload` malloc'd (NULL when empty). */
static int recv_frame(int fd, char** header, unsigned char** payload, size_t* payload_size) {
  unsigned char prefix[8];
  *header = NULL;
  *payload = NULL;
  if (!read_all(fd, prefix, 8)) {
    return 0;
  }
  uint32_t header_size = get_u32(prefix);
  uint32_t size = get_u32(prefix + 4);
  if (header_size > MAX_HEADER) {
    return 0;
  }
  *header = malloc((size_t)header_size + 1);
  if (!*header || !read_all(fd, *header, header_size)) {
    free(*header);
    *header = NULL;
    return 0;
  }
  (*header)[header_size] = '\0';
  *payload_size = size;
  if (size > 0) {
    *payload = malloc(size);
    if (!*payload || !read_all(fd, *payload, size)) {
      free(*header);
      free(*payload);
      *header = NULL;
      *payload = NULL;
      return 0;
    }
  }
  return 1;
}

/* Copies the value of `key=` from a header into `out`; 0 when the key is missing or the value does not fit. */
static int header_value(const char* header, const char* key, char* out, size_t capacity) {
  size_t key_length = strlen(key);
  for (const char* line = strchr(header, '\n'); line; line = strchr(line + 1, '\n')) {
    const char* start = line + 1;
    if (strncmp(start, key, key_length) == 0 && start[key_length] == '=') {
      const char* value = start + key_length + 1;
      const char* end = strchr(value, '\n');
      size_t length = end ? (size_t)(end - value) : strlen(value);
      if (length >= capacity) {
        return 0;
      }
      memcpy(out, value, length);
      out[length] = '\0';
      return 1;
    }
  }
  return 0;
}

static uairt_dtype parse_dtype(const char* text, size_t* size) {
  static const struct {
    const char* name;
    uairt_dtype dtype;
    size_t size;
  } kTypes[] = {{"float32", UAIRT_DTYPE_FLOAT32, 4}, {"int64", UAIRT_DTYPE_INT64, 8}, {"int32", UAIRT_DTYPE_INT32, 4},
                {"int8", UAIRT_DTYPE_INT8, 1},       {"uint8", UAIRT_DTYPE_UINT8, 1},  {"bool", UAIRT_DTYPE_BOOL, 1}};
  for (size_t i = 0; i < sizeof(kTypes) / sizeof(kTypes[0]); ++i) {
    if (strcmp(text, kTypes[i].name) == 0) {
      *size = kTypes[i].size;
      return kTypes[i].dtype;
    }
  }
  return UAIRT_DTYPE_UNKNOWN;
}

/* "name|dtype|1x16x384" */
static int parse_spec(const char* text, spec_t* spec) {
  const char* bar1 = strchr(text, '|');
  const char* bar2 = bar1 ? strchr(bar1 + 1, '|') : NULL;
  if (!bar2 || (size_t)(bar1 - text) >= MAX_NAME) {
    return 0;
  }
  memcpy(spec->name, text, (size_t)(bar1 - text));
  spec->name[bar1 - text] = '\0';
  char dtype[16];
  if ((size_t)(bar2 - bar1 - 1) >= sizeof(dtype)) {
    return 0;
  }
  memcpy(dtype, bar1 + 1, (size_t)(bar2 - bar1 - 1));
  dtype[bar2 - bar1 - 1] = '\0';
  size_t element_size = 0;
  spec->dtype = parse_dtype(dtype, &element_size);
  if (spec->dtype == UAIRT_DTYPE_UNKNOWN) {
    return 0;
  }
  spec->rank = 0;
  size_t count = 1;
  const char* cursor = bar2 + 1;
  while (*cursor) {
    char* end = NULL;
    long long dim = strtoll(cursor, &end, 10);
    if (end == cursor || dim <= 0 || spec->rank == UAIRT_MAX_RANK) {
      return 0;
    }
    spec->dims[spec->rank++] = dim;
    count *= (size_t)dim;
    if (*end == 'x') {
      cursor = end + 1;
    } else if (*end == '\0') {
      cursor = end;
    } else {
      return 0;
    }
  }
  spec->nbytes = count * element_size;
  return spec->rank > 0;
}

/* The worker's stderr goes to an unlinked temp file, opened O_APPEND, so the plugin can read what ONNX Runtime logged. */
static int open_log(void) {
  const char* dir = getenv("TMPDIR");
  char path[512];
  snprintf(path, sizeof(path), "%s/uairt-transformersjs-XXXXXX", dir && dir[0] ? dir : "/tmp");
  int fd = mkstemp(path);
  if (fd >= 0) {
    unlink(path);
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_APPEND);
  }
  return fd;
}

/* Copies the log into a malloc'd string and empties the file. */
static char* take_log(int fd) {
  off_t size = lseek(fd, 0, SEEK_END);
  if (size < 0 || (size_t)size > MAX_LOG) {
    size = size < 0 ? 0 : MAX_LOG;
  }
  char* text = calloc(1, (size_t)size + 1);
  if (text && size > 0 && pread(fd, text, (size_t)size, 0) < 0) {
    text[0] = '\0';
  }
  if (ftruncate(fd, 0) != 0) {
    /* the file only grows; the next read still finds the markers */
  }
  return text;
}

static void stop_worker(model_t* model) {
  if (model->fd >= 0) {
    send_frame(model->fd, "quit", NULL, 0);
    close(model->fd);
  }
  if (model->pid > 0) {
    kill(model->pid, SIGTERM);
    int status;
    while (waitpid(model->pid, &status, 0) < 0 && errno == EINTR) {
    }
  }
  if (model->log_fd >= 0) {
    close(model->log_fd);
  }
}

static uairt_status spawn_worker(const engine_t* engine, model_t* model) {
  int pair[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
    fail("socketpair failed: %s", strerror(errno));
    return UAIRT_ERR_IO;
  }
#ifdef SO_NOSIGPIPE
  int one = 1;
  setsockopt(pair[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
  fcntl(pair[0], F_SETFD, FD_CLOEXEC);
  fcntl(pair[1], F_SETFD, FD_CLOEXEC);
  model->log_fd = open_log();
  if (model->log_fd < 0) {
    fail("cannot create a temporary file for the worker's log: %s", strerror(errno));
    close(pair[0]);
    close(pair[1]);
    return UAIRT_ERR_IO;
  }
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pair[1], 0);
  posix_spawn_file_actions_adddup2(&actions, pair[1], 1);
  posix_spawn_file_actions_adddup2(&actions, model->log_fd, 2);
  char* argv[] = {engine->node, engine->worker, engine->package_dir, NULL};
  int rc = posix_spawnp(&model->pid, engine->node, &actions, NULL, argv, environ);
  posix_spawn_file_actions_destroy(&actions);
  close(pair[1]);
  if (rc != 0) {
    close(pair[0]);
    model->pid = 0;
    fail("cannot start '%s': %s (install Node.js or pass the node option)", engine->node, strerror(rc));
    return UAIRT_ERR_BACKEND_UNAVAILABLE;
  }
  model->fd = pair[0];
  return UAIRT_OK;
}

static uairt_status worker_error(const char* header, const char* log, const char* what) {
  char message[600] = "";
  char code[32] = "";
  header_value(header, "message", message, sizeof(message));
  header_value(header, "code", code, sizeof(code));
  fail("%s: %s", what, message[0] ? message : "the worker reported an error");
  (void)log;
  if (strcmp(code, "unavailable") == 0) {
    return UAIRT_ERR_BACKEND_UNAVAILABLE;
  }
  if (strcmp(code, "invalid_argument") == 0) {
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  if (strcmp(code, "unsupported") == 0) {
    return UAIRT_ERR_UNSUPPORTED;
  }
  return UAIRT_ERR_INCOMPATIBLE_MODEL;
}

static uairt_status lost_worker(model_t* model, const char* what) {
  char* log = take_log(model->log_fd);
  size_t length = log ? strlen(log) : 0;
  fail("%s: the Node worker stopped%s%s", what, length ? "; its last output: " : "", length ? log + (length > 300 ? length - 300 : 0) : "");
  free(log);
  return UAIRT_ERR_RUNTIME;
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  if (model) {
    stop_worker(model);
    free(model);
  }
}

static uairt_status load_model(void* handle, const uairt_model_source* source, void** out_model) {
  const engine_t* engine = handle;
  if (!source->path || source->data) {
    fail("the transformersjs backend loads a Transformers.js model directory from a path, not from memory");
    return UAIRT_ERR_UNSUPPORTED;
  }
  struct stat info;
  if (strchr(source->path, '\n') || stat(source->path, &info) != 0 || !S_ISDIR(info.st_mode)) {
    fail("'%s' is not a Transformers.js model directory (a folder with config.json and onnx/model.onnx)", source->path);
    return UAIRT_ERR_NOT_FOUND;
  }
  if (!uairt_backend_file_exists(engine->worker)) {
    fail("worker script '%s' not found; set the worker option or UAIRT_TRANSFORMERSJS_WORKER", engine->worker);
    return UAIRT_ERR_BACKEND_UNAVAILABLE;
  }
  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  model->fd = -1;
  model->log_fd = -1;
  uairt_status status = spawn_worker(engine, model);
  if (status != UAIRT_OK) {
    destroy_model(model);
    return status;
  }
  char request[4096];
  int request_size = snprintf(request, sizeof(request), "load\npackage_dir=%s\npath=%s\ndevice=%s\ndtype=%s\nshapes=%s", engine->package_dir, source->path,
           engine->device, engine->dtype, engine->input_shapes);
  if (request_size < 0 || (size_t)request_size >= sizeof(request)) {
    fail("the model path and options are longer than %zu bytes", sizeof(request) - 1);
    destroy_model(model);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  char* header = NULL;
  unsigned char* payload = NULL;
  size_t payload_size = 0;
  if (!send_frame(model->fd, request, NULL, 0) || !recv_frame(model->fd, &header, &payload, &payload_size)) {
    status = lost_worker(model, "load");
    destroy_model(model);
    return status;
  }
  char* log = take_log(model->log_fd);
  if (strncmp(header, "ok", 2) != 0) {
    status = worker_error(header, log, "load");
    free(header);
    free(payload);
    free(log);
    destroy_model(model);
    return status;
  }
  char value[MAX_NAME + 64];
  char key[24];
  size_t counts[2] = {0, 0};
  const char* kinds[2] = {"input", "output"};
  status = UAIRT_OK;
  for (int kind = 0; kind < 2 && status == UAIRT_OK; ++kind) {
    snprintf(key, sizeof(key), "n_%ss", kinds[kind]);
    if (!header_value(header, key, value, sizeof(value)) || (counts[kind] = strtoul(value, NULL, 10)) == 0 || counts[kind] > MAX_TENSORS) {
      fail("the worker reported an unusable number of %ss", kinds[kind]);
      status = UAIRT_ERR_INCOMPATIBLE_MODEL;
      break;
    }
    for (size_t i = 0; i < counts[kind]; ++i) {
      spec_t* spec = kind == 0 ? &model->inputs[i] : &model->outputs[i];
      snprintf(key, sizeof(key), "%s%zu", kinds[kind], i);
      if (!header_value(header, key, value, sizeof(value)) || !parse_spec(value, spec)) {
        fail("%s %zu: the worker described it as '%s', which is not a supported tensor (float32, int64, int32, int8, uint8 or bool)",
             kinds[kind], i, value);
        status = UAIRT_ERR_UNSUPPORTED;
        break;
      }
      (kind == 0 ? &model->input_nbytes : &model->output_nbytes)[0] += spec->nbytes;
    }
  }
  model->num_inputs = counts[0];
  model->num_outputs = counts[1];
  free(header);
  free(payload);
  if (status == UAIRT_OK && strcmp(engine->device, "webgpu") == 0 && (!log || !strstr(log, "placed on [WebGpuExecutionProvider]"))) {
    fail("device=webgpu: ONNX Runtime in Node did not place the model on WebGpuExecutionProvider, so it would run on the CPU; "
         "this onnxruntime-node build or machine has no usable WebGPU");
    status = UAIRT_ERR_BACKEND_UNAVAILABLE;
  }
  free(log);
  if (status != UAIRT_OK) {
    destroy_model(model);
    return status;
  }
  *out_model = model;
  return UAIRT_OK;
}

static size_t num_inputs(const void* model) {
  return ((const model_t*)model)->num_inputs;
}

static size_t num_outputs(const void* model) {
  return ((const model_t*)model)->num_outputs;
}

static uairt_status describe(const spec_t* specs, size_t count, size_t index, uairt_tensor* out) {
  if (index >= count) {
    fail("tensor index %zu out of range (%zu)", index, count);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  const spec_t* spec = &specs[index];
  out->dtype = spec->dtype;
  out->rank = spec->rank;
  for (uint32_t i = 0; i < spec->rank; ++i) {
    out->dims[i] = spec->dims[i];
  }
  out->name = spec->name;
  return UAIRT_OK;
}

static uairt_status input_info(const void* model, size_t index, uairt_tensor* out) {
  const model_t* m = model;
  return describe(m->inputs, m->num_inputs, index, out);
}

static uairt_status output_info(const void* model, size_t index, uairt_tensor* out) {
  const model_t* m = model;
  return describe(m->outputs, m->num_outputs, index, out);
}

static uairt_status run(void* handle, const uairt_tensor* inputs, size_t count_in, uairt_tensor* outputs, size_t count_out) {
  model_t* model = handle;
  unsigned char* packed = malloc(model->input_nbytes ? model->input_nbytes : 1);
  if (!packed) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  size_t offset = 0;
  for (size_t i = 0; i < count_in; ++i) {
    memcpy(packed + offset, inputs[i].data, model->inputs[i].nbytes);
    offset += model->inputs[i].nbytes;
  }
  char* header = NULL;
  unsigned char* payload = NULL;
  size_t payload_size = 0;
  uairt_status status = UAIRT_OK;
  if (!send_frame(model->fd, "run", packed, offset) || !recv_frame(model->fd, &header, &payload, &payload_size)) {
    status = lost_worker(model, "run");
  } else if (strncmp(header, "ok", 2) != 0) {
    status = worker_error(header, NULL, "run");
  } else if (payload_size != model->output_nbytes) {
    fail("run: the worker returned %zu bytes, the outputs take %zu", payload_size, model->output_nbytes);
    status = UAIRT_ERR_RUNTIME;
  } else {
    offset = 0;
    for (size_t i = 0; i < count_out; ++i) {
      memcpy(outputs[i].data, payload + offset, model->outputs[i].nbytes);
      offset += model->outputs[i].nbytes;
    }
  }
  free(packed);
  free(header);
  free(payload);
  free(take_log(model->log_fd));
  return status;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "transformersjs",
    .supported_domains = UAIRT_MEM_HOST,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_inputs,
    .num_outputs = num_outputs,
    .input_info = input_info,
    .output_info = output_info,
    .run = run,
};

UAIRT_API const uairt_backend_api* UAIRT_BACKEND_ENTRY(const uairt_host_api* host) {
  g_host = host;
  return &kApi;
}
