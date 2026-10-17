/* SPDX-License-Identifier: Apache-2.0 */
#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <uairt/uairt.h>

/* argv[1] plugin, argv[2] "1" if it opts in to global symbols, "0" if it does not. */
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <plugin> <0|1>\n", argv[0]);
    return 2;
  }
  if (uairt_load_backend_library(argv[1]) != UAIRT_OK) {
    fprintf(stderr, "load failed: %s\n", uairt_last_error());
    return 1;
  }
  int visible = dlsym(RTLD_DEFAULT, "uairt_test_global_probe") != NULL;
  int expected = atoi(argv[2]);
  if (visible != expected) {
    fprintf(stderr, "probe symbol is %s in the global namespace, expected %s\n", visible ? "visible" : "hidden",
            expected ? "visible" : "hidden");
    return 1;
  }
  puts(expected ? "opted-in plugin symbols are global" : "plugin symbols stay local");
  return 0;
}
