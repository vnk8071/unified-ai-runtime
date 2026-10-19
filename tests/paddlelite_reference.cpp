// SPDX-License-Identifier: Apache-2.0
// The reference for tests/compare_with_paddlelite.py: runs a .nb model through Paddle Lite's own C++ API, the way its demos do.
//   paddlelite_reference <model.nb> <threads> <normal|low> <out-prefix> <shape=input.bin>...
// Shapes are like 1x3x224x224; one shape=file pair per input, in the model's input order. Outputs go to <out-prefix><i>.bin.
#include <paddle_api.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace paddle::lite_api;

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s <model.nb> <threads> <normal|low> <out-prefix> <shape=input.bin>...\n", argv[0]);
    return 2;
  }
  MobileConfig config;
  config.set_model_from_file(argv[1]);
  config.set_threads(std::atoi(argv[2]));
  config.set_precision_mode(std::string(argv[3]) == "low" ? LITE_PRECISION_LOW : LITE_PRECISION_NORMAL);
  auto predictor = CreatePaddlePredictor<MobileConfig>(config);
  for (int i = 5; i < argc; ++i) {
    const std::string pair = argv[i];
    const size_t equals = pair.find('=');
    shape_t shape;
    size_t elements = 1;
    for (size_t start = 0; start <= equals;) {
      size_t end = pair.find('x', start);
      if (end == std::string::npos || end > equals) {
        end = equals;
      }
      shape.push_back(std::atoll(pair.substr(start, end - start).c_str()));
      elements *= static_cast<size_t>(shape.back());
      start = end + 1;
    }
    auto input = predictor->GetInput(i - 5);
    input->Resize(shape);
    std::ifstream file(pair.substr(equals + 1), std::ios::binary);
    file.read(reinterpret_cast<char*>(input->mutable_data<float>()), static_cast<std::streamsize>(elements * sizeof(float)));
  }
  predictor->Run();
  const auto names = predictor->GetOutputNames();
  for (size_t i = 0; i < names.size(); ++i) {
    auto output = predictor->GetOutput(static_cast<int>(i));
    size_t elements = 1;
    for (int64_t d : output->shape()) {
      elements *= static_cast<size_t>(d);
    }
    std::ofstream out(std::string(argv[4]) + std::to_string(i) + ".bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(output->data<float>()), static_cast<std::streamsize>(elements * sizeof(float)));
  }
  return 0;
}
