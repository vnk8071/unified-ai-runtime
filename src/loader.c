/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include <string.h>

#include "internal.h"

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

/* Libraries stay loaded for the process lifetime; backends hold static state. */
uairt_status uairt_loader_open(const char* path, uairt_backend_get_api_fn* out) {
  HMODULE module = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!module) {
    uairt_set_error("LoadLibrary failed for %s (error %lu)", path, GetLastError());
    return UAIRT_ERR_IO;
  }
  FARPROC symbol = GetProcAddress(module, UAIRT_BACKEND_ENTRY_SYMBOL);
  if (!symbol) {
    uairt_set_error("%s does not export %s", path, UAIRT_BACKEND_ENTRY_SYMBOL);
    FreeLibrary(module);
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  memcpy(out, &symbol, sizeof(symbol));
  return UAIRT_OK;
}

#else

#include <dlfcn.h>

/* Libraries stay loaded for the process lifetime; backends hold static state. */
uairt_status uairt_loader_open(const char* path, uairt_backend_get_api_fn* out) {
  void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    uairt_set_error("dlopen failed: %s", dlerror());
    return UAIRT_ERR_IO;
  }
  void* symbol = dlsym(handle, UAIRT_BACKEND_ENTRY_SYMBOL);
  if (!symbol) {
    uairt_set_error("%s does not export %s", path, UAIRT_BACKEND_ENTRY_SYMBOL);
    dlclose(handle);
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
  memcpy(out, &symbol, sizeof(symbol));
  return UAIRT_OK;
}

#endif
