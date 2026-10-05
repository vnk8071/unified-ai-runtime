/* SPDX-License-Identifier: Apache-2.0 */
#ifndef UAIRT_INTERNAL_H
#define UAIRT_INTERNAL_H

#include <uairt/uairt_backend.h>

void uairt_set_error(const char* fmt, ...);

uairt_status uairt_loader_open(const char* path, uairt_backend_get_api_fn* out);

UAIRT_API const uairt_backend_api* uairt_reference_backend_get_api(
    const uairt_host_api* host);

#endif
