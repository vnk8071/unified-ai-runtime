// SPDX-License-Identifier: Apache-2.0
// Streams text from a GGUF model through the llamacpp backend with uairt_llm sampling:
//   llm_stream_cpp <libuairt_backend_llamacpp> <model.gguf> "<prompt>" <max_tokens> [temperature [top_k [top_p [seed]]]]
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "uairt.hpp"

int main(int argc, char** argv) {
  if (argc < 5 || argc > 9) {
    std::fprintf(stderr, "usage: %s <plugin> <model.gguf> \"<prompt>\" <max_tokens> [temperature [top_k [top_p [seed]]]]\n", argv[0]);
    return 2;
  }
  try {
    uairt::load_backend_library(argv[1]);
    uairt::Engine engine("llamacpp", {{"n_gpu_layers", "99"}});
    uairt::Model model(engine, argv[2]);

    uairt_generate_options options = uairt_generate_options_default();
    options.max_tokens = static_cast<size_t>(std::atoi(argv[4]));
    if (argc > 5) options.sampling.temperature = static_cast<float>(std::atof(argv[5]));
    if (argc > 6) options.sampling.top_k = std::atoi(argv[6]);
    if (argc > 7) options.sampling.top_p = static_cast<float>(std::atof(argv[7]));
    if (argc > 8) options.sampling.seed = std::strtoull(argv[8], nullptr, 10);

    std::fputs(argv[3], stdout);
    auto result = uairt::generate(model, argv[3], options, [](const std::string& piece) {
      std::fwrite(piece.data(), 1, piece.size(), stdout);
      std::fflush(stdout);
      return false;
    });
    std::printf("\n");
    std::fprintf(stderr, "%zu tokens, finish_reason=%d\n", result.num_tokens, result.finish_reason);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
