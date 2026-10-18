// SPDX-License-Identifier: Apache-2.0
// A small header-only C++17 wrapper over the C API: RAII handles and exceptions instead of status codes. The library itself stays C;
// this file is an example of how an application can wrap it, not part of the ABI.
#pragma once

#include <uairt/uairt.h>
#include <uairt/uairt_llm.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uairt {

class Error : public std::runtime_error {
 public:
  Error(const std::string& what, uairt_status status)
      : std::runtime_error(what + ": " + uairt_status_string(status) + " (" + uairt_last_error() + ")"), status_(status) {}
  uairt_status status() const { return status_; }

 private:
  uairt_status status_;
};

inline void check(uairt_status status, const char* what) {
  if (status != UAIRT_OK) {
    throw Error(what, status);
  }
}

inline void load_backend_library(const std::string& path) { check(uairt_load_backend_library(path.c_str()), "load backend library"); }

class Engine {
 public:
  using Options = std::vector<std::pair<std::string, std::string>>;

  explicit Engine(const std::string& backend, const Options& options = {}) {
    std::vector<uairt_option> raw;
    for (const auto& kv : options) {
      raw.push_back({kv.first.c_str(), kv.second.c_str()});
    }
    check(uairt_engine_create(backend.c_str(), raw.data(), raw.size(), &handle_), "create engine");
  }
  Engine(Engine&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Engine& operator=(Engine&& other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  ~Engine() { uairt_engine_destroy(handle_); }

  uairt_engine* get() const { return handle_; }

 private:
  uairt_engine* handle_ = nullptr;
};

class Model {
 public:
  // The engine must outlive the model.
  Model(const Engine& engine, const std::string& path) {
    uairt_model_source source = {};
    source.struct_size = sizeof(source);
    source.path = path.c_str();
    check(uairt_model_load(engine.get(), &source, &handle_), "load model");
  }
  Model(Model&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
  Model& operator=(Model&& other) noexcept {
    std::swap(handle_, other.handle_);
    return *this;
  }
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model() { uairt_model_destroy(handle_); }

  uairt_model* get() const { return handle_; }
  size_t num_inputs() const { return uairt_model_num_inputs(handle_); }
  size_t num_outputs() const { return uairt_model_num_outputs(handle_); }

  // dtype, shape and quantization of input/output `index`; data is unset until you point it at a buffer
  uairt_tensor input_info(size_t index) const {
    uairt_tensor t;
    uairt_tensor_init(&t);
    check(uairt_model_input_info(handle_, index, &t), "input info");
    return t;
  }
  uairt_tensor output_info(size_t index) const {
    uairt_tensor t;
    uairt_tensor_init(&t);
    check(uairt_model_output_info(handle_, index, &t), "output info");
    return t;
  }

  void run(const std::vector<uairt_tensor>& inputs, std::vector<uairt_tensor>& outputs) {
    check(uairt_model_run(handle_, inputs.data(), inputs.size(), outputs.data(), outputs.size()), "run");
  }

 private:
  uairt_model* handle_ = nullptr;
};

struct Generation {
  size_t num_tokens = 0;
  int finish_reason = UAIRT_FINISH_LENGTH;
};

// Streams generated text to `on_text`; returning true from it stops the generation.
inline Generation generate(const Model& model, const std::string& prompt, const uairt_generate_options& options,
                           const std::function<bool(const std::string&)>& on_text) {
  auto trampoline = [](const char* text, size_t len, void* user) -> int {
    return (*static_cast<const std::function<bool(const std::string&)>*>(user))(std::string(text, len)) ? 1 : 0;
  };
  uairt_generate_result result = {};
  result.struct_size = sizeof(result);
  check(uairt_llm_generate(model.get(), nullptr, prompt.c_str(), &options, trampoline,
                           const_cast<std::function<bool(const std::string&)>*>(&on_text), &result),
        "generate");
  return {result.num_tokens, result.finish_reason};
}

}  // namespace uairt
