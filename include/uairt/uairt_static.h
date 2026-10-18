/* SPDX-License-Identifier: Apache-2.0 */
#ifndef UAIRT_UAIRT_STATIC_H
#define UAIRT_UAIRT_STATIC_H

#include <uairt/uairt_backend.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Backends linked into the application instead of loaded as plugins, for platforms that cannot load plugins at run
 * time (an iOS app, a sandboxed Android app). Build with -DUAIRT_BUILD_STATIC_BACKENDS=ON, link
 * libuairt_backend_<name>_static.a (and its system libraries) with libuairt, and register once at start-up:
 *
 *     uairt_register_backend(uairt_backend_get_api_coreml);
 *
 * Only the backends listed here have a static target; registering one that was not linked is a link error.
 */
const uairt_backend_api* uairt_backend_get_api_coreml(const uairt_host_api* host);

#ifdef __cplusplus
}
#endif

#endif
