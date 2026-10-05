// SPDX-License-Identifier: Apache-2.0
// Runs a model through the QNN backend plugin with pseudo-random inputs and prints the latency.
//   run_qnn <plugin> <sdk-root> <model> [npu|cpu|gpu] [runs] [cache-dir]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <uairt/uairt.hpp>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: run_qnn <plugin> <sdk-root> <model> [npu|cpu|gpu] [runs] [cache-dir]\n";
    return 2;
  }
  using Clock = std::chrono::steady_clock;
  const std::string device = argc > 4 ? argv[4] : "npu";
  const int runs = std::max(1, argc > 5 ? std::atoi(argv[5]) : 20);
  std::map<std::string, std::string> options{{"device", device}, {"sdk_root", argv[2]}};
  if (argc > 6) options["cache_dir"] = argv[6];

  try {
    uairt::load_backend_library(argv[1]);
    const auto started = Clock::now();
    uairt::Engine engine("qnn", options);
    uairt::Model model = engine.load_model(argv[3]);
    std::printf("init_ms=%.1f\n", std::chrono::duration<double, std::milli>(Clock::now() - started).count());

    std::uint32_t seed = 1;
    std::vector<std::vector<std::uint8_t>> inputs, outputs;
    for (const auto& info : model.inputs()) {
      std::vector<std::uint8_t> data(info.nbytes());
      for (auto& byte : data) {
        seed = seed * 1664525u + 1013904223u;
        byte = static_cast<std::uint8_t>(seed >> 24);
      }
      inputs.push_back(std::move(data));
    }
    for (const auto& info : model.outputs()) outputs.emplace_back(info.nbytes());

    std::vector<uairt::ConstBytes> in_blocks;
    std::vector<uairt::MutableBytes> out_blocks;
    for (auto& v : inputs) in_blocks.push_back(uairt::bytes(v));
    for (auto& v : outputs) out_blocks.push_back(uairt::bytes(v));

    std::vector<double> latencies;
    for (int i = 0; i < runs; ++i) {
      const auto start = Clock::now();
      model.run(in_blocks, out_blocks);
      latencies.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    std::sort(latencies.begin(), latencies.end());
    std::printf("latency_ms runs=%d min=%.2f median=%.2f\n", runs, latencies.front(), latencies[latencies.size() / 2]);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      unsigned long long sum = 0;
      for (std::uint8_t byte : outputs[i]) sum += byte;
      std::printf("output %zu: %zu bytes, byte sum %llu\n", i, outputs[i].size(), sum);
    }
  } catch (const uairt::Error& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
