/* SPDX-License-Identifier: Apache-2.0 */
/* The CoreML backend linked into the program and registered by hand, as an iOS app does (no plugin is loaded). */
#include <stdio.h>
#include <string.h>

#include <uairt/uairt.h>
#include <uairt/uairt_static.h>

static int g_failures;

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__, __LINE__,      \
              #cond, uairt_last_error());                                         \
      ++g_failures;                                                              \
    }                                                                            \
  } while (0)

static int has_backend(const char* name) {
  for (size_t i = 0; i < uairt_backend_count(); ++i) {
    if (strcmp(uairt_backend_name(i), name) == 0) {
      return 1;
    }
  }
  return 0;
}

int main(void) {
  CHECK(!has_backend("coreml"));
  CHECK(uairt_register_backend(uairt_backend_get_api_coreml) == UAIRT_OK);
  CHECK(has_backend("coreml"));
  CHECK(uairt_register_backend(uairt_backend_get_api_coreml) == UAIRT_ERR_INVALID_ARGUMENT);
  CHECK(strstr(uairt_last_error(), "already registered") != NULL);

  uairt_engine* engine = NULL;
  uairt_option bad_option = {"no_such_option", "1"};
  CHECK(uairt_engine_create("coreml", &bad_option, 1, &engine) == UAIRT_ERR_INVALID_ARGUMENT);
  uairt_option units = {"compute_units", "cpu_only"};
  CHECK(uairt_engine_create("coreml", &units, 1, &engine) == UAIRT_OK);
  if (engine) {
    uairt_model* model = NULL;
    uairt_model_source missing = {.struct_size = sizeof(missing), .path = "/no/such/model.mlmodelc"};
    CHECK(uairt_model_load(engine, &missing, &model) == UAIRT_ERR_IO);
    uairt_model_source not_a_model = {.struct_size = sizeof(not_a_model), .path = "/etc/hosts"};
    CHECK(uairt_model_load(engine, &not_a_model, &model) == UAIRT_ERR_INCOMPATIBLE_MODEL);
    uairt_engine_destroy(engine);
  }
  if (g_failures) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  puts("static coreml backend: all checks passed");
  return 0;
}
