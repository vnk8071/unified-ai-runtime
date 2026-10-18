// SPDX-License-Identifier: Apache-2.0
// The helpers in src/llm.cpp that have no public entry point. Including the source gives the test its internal linkage.
#include <stdio.h>

#include "llm.cpp"

static int g_failures;

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                                \
    }                                                                              \
  } while (0)

static size_t prefix(const char* bytes, size_t size) {
  return utf8_complete_prefix(reinterpret_cast<const unsigned char*>(bytes), size);
}

int main() {
  CHECK(prefix("", 0) == 0);
  CHECK(prefix("abc", 3) == 3);
  CHECK(prefix("\xC3\xA9", 2) == 2);            // é, complete
  CHECK(prefix("a\xC3", 2) == 1);               // lead byte of a 2-byte character, continuation missing
  CHECK(prefix("\xE2\x82", 2) == 0);            // 3-byte character, one byte missing
  CHECK(prefix("\xE2\x82\xAC", 3) == 3);        // € complete
  CHECK(prefix("x\xF0\x9F\x98", 4) == 1);       // 4-byte character, one byte missing
  CHECK(prefix("\xF0\x9F\x98\x80", 4) == 4);    // 😀 complete
  CHECK(prefix("\xF0\x9F", 2) == 0);
  CHECK(prefix("ab\x80", 3) == 3);              // a stray continuation byte is not held back forever
  CHECK(prefix("\xFF", 1) == 1);                // not a valid lead byte: pass it through
  if (g_failures) {
    return 1;
  }
  puts("ok");
  return 0;
}
