// SPDX-License-Identifier: Apache-2.0
#include <uairt/uairt.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

static int g_failures;
#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                \
    }                                                                              \
  } while (0)

template <class F>
static bool throws_status(uairt::Status status, F&& fn) {
  try {
    fn();
  } catch (const uairt::Error& e) {
    return e.status() == status;
  }
  return false;
}

int main(int argc, char** argv) {
  CHECK(uairt::version() == UAIRT_VERSION_STRING);
  bool has_reference = false;
  for (const auto& name : uairt::backends()) has_reference = has_reference || name == "reference";
  CHECK(has_reference);

  {
    uairt::Engine engine("reference");
    uairt::Model model = engine.load_model("unused");
    CHECK(model.inputs().size() == 1 && model.outputs().size() == 1);
    CHECK(model.inputs()[0].dtype == uairt::DType::Float32 && model.inputs()[0].nbytes() == 16);
    std::vector<float> in{1, 2, 3, 4}, out(4);
    model.run({uairt::bytes(in)}, {uairt::bytes(out)});
    CHECK(out == in);

    std::vector<float> short_in(3);
    CHECK(throws_status(uairt::Status::InvalidArgument, [&] { model.run({uairt::bytes(short_in)}, {uairt::bytes(out)}); }));
    CHECK(throws_status(uairt::Status::InvalidArgument, [&] { model.run({}, {uairt::bytes(out)}); }));

    // A moved-from model is empty; the moved-to one still runs and still keeps the engine alive.
    uairt::Model moved = std::move(model);
    moved.run({uairt::bytes(in)}, {uairt::bytes(out)});

    uairt::Buffer src = engine.alloc_buffer(16), dst = engine.alloc_buffer(16);
    const float values[4] = {5, 6, 7, 8};
    std::memcpy(src.data(), values, 16);
    moved.run_buffers({&src}, {&dst});
    CHECK(std::memcmp(dst.data(), values, 16) == 0);
    CHECK(src.fd() == -1 && src.size() == 16);
    CHECK(throws_status(uairt::Status::Unsupported, [&] { engine.alloc_buffer(16, uairt::Domain::DmaBuf); }));
    CHECK(throws_status(uairt::Status::Unsupported, [&] { engine.alloc_buffer(16, uairt::Domain::Pinned); }));
  }

  CHECK(throws_status(uairt::Status::NotFound, [] { uairt::Engine engine("no_such_backend"); }));
  CHECK(throws_status(uairt::Status::Io, [] { uairt::load_backend_library("/no/such/plugin.so"); }));

  {  // the engine outlives the scope that created it while a model still exists
    auto make_model = [] {
      uairt::Engine engine("reference");
      return engine.load_model("unused");
    };
    uairt::Model model = make_model();
    std::vector<float> in(4, 1.f), out(4);
    model.run({uairt::bytes(in)}, {uairt::bytes(out)});
    CHECK(out == in);
  }

  if (argc > 2) {  // optional: ONNX Runtime plugin and a model with two inputs and two outputs
    uairt::load_backend_library(argv[1]);
    uairt::Engine engine("onnxruntime", {{"intra_op_threads", "1"}});
    uairt::Model model = engine.load_model(argv[2]);
    CHECK(model.inputs().size() == 2 && model.outputs().size() == 2);
    CHECK(model.inputs()[0].name == "x" && model.outputs()[1].name == "product");
    std::vector<float> x{1, 2, 3, 4}, y{10, 20, 30, 40}, sum(4), product(4);
    model.run({uairt::bytes(x), uairt::bytes(y)}, {uairt::bytes(sum), uairt::bytes(product)});
    for (int i = 0; i < 4; ++i) CHECK(sum[i] == x[i] + y[i] && product[i] == x[i] * y[i]);

    std::ifstream file(argv[2], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    uairt::Model from_memory = engine.load_model(bytes.data(), bytes.size());
    CHECK(from_memory.outputs().size() == 2);
    CHECK(throws_status(uairt::Status::Io, [&] { engine.load_model("/no/such/model.onnx"); }));
    CHECK(throws_status(uairt::Status::InvalidArgument, [&] { uairt::Engine bad("onnxruntime", {{"no_such_option", "1"}}); }));
  }

  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::puts("all C++ binding checks passed");
  return 0;
}
