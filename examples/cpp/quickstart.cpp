// SPDX-License-Identifier: Apache-2.0
// quickstart.c in C++: the built-in "reference" backend copies one float32 [1, 4] input to the output, so it runs anywhere.
#include <cstdio>
#include <exception>

#include "uairt.hpp"

int main() {
  try {
    std::printf("UAIRT %s\n", uairt_version_string());
    uairt::Engine engine("reference");
    uairt::Model model(engine, "unused");

    float in_data[4] = {1.f, 2.f, 3.f, 4.f};
    float out_data[4] = {};
    auto input = model.input_info(0);
    auto output = model.output_info(0);
    input.data = in_data;
    input.nbytes = sizeof(in_data);
    output.data = out_data;
    output.nbytes = sizeof(out_data);
    std::vector<uairt_tensor> outputs{output};
    model.run({input}, outputs);

    std::printf("output: %g %g %g %g\n", out_data[0], out_data[1], out_data[2], out_data[3]);
    return out_data[3] == 4.f ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
