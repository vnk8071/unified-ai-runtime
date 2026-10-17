/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Helpers shared by the backend plugins, header-only so that no plugin links the core (docs/writing-a-backend.md, rule 2).
 * C11, C++ and Objective-C. Each plugin includes this file and keeps its own `g_host`.
 */
#ifndef UAIRT_BACKENDS_COMMON_H
#define UAIRT_BACKENDS_COMMON_H

#include <stdarg.h>
#include <stdio.h>

#include <uairt/uairt_backend.h>

#if defined(__GNUC__)
#define UAIRT_BACKEND_PRINTF(fmt_index, first_arg) __attribute__((format(printf, fmt_index, first_arg)))
#else
#define UAIRT_BACKEND_PRINTF(fmt_index, first_arg)
#endif

#define UAIRT_BACKEND_MESSAGE_CAPACITY 1024

/* Formats a message and hands it to the host, which returns it from uairt_last_error(). Longer messages are truncated. */
static inline void uairt_backend_fail(const uairt_host_api* host, const char* fmt, ...) UAIRT_BACKEND_PRINTF(2, 3);

static inline void uairt_backend_fail(const uairt_host_api* host, const char* fmt, ...) {
  char message[UAIRT_BACKEND_MESSAGE_CAPACITY];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (host && host->set_error) {
    host->set_error(message);
  }
}

/* True if `path` can be opened for reading. */
static inline int uairt_backend_file_exists(const char* path) {
  FILE* file = fopen(path, "rb");
  if (file) {
    fclose(file);
  }
  return file != NULL;
}

#endif
