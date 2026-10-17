/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Test-only plugin: the reference backend plus a probe symbol, built with and without the RTLD_GLOBAL opt-in marker
 * (WITH_MARKER).
 */
#define UAIRT_REFERENCE_AS_PLUGIN
#include "../src/backends/reference.c"

UAIRT_API int uairt_test_global_probe(void) {
  return 42;
}

#if WITH_MARKER
UAIRT_API const int uairt_backend_global_symbols = 1;
#endif
