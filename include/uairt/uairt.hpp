// SPDX-License-Identifier: Apache-2.0
// C++17 header-only wrapper for the Unified AI Runtime C API.
//
//   #include <uairt/uairt.hpp>
//   uairt::load_backend_library("libuairt_backend_onnxruntime.so");
//   uairt::Engine engine("onnxruntime", {{"intra_op_threads", "1"}});
//   uairt::Model model = engine.load_model("model.onnx");
//   std::vector<float> x(model.inputs()[0].num_elements()), y(model.outputs()[0].num_elements());
//   model.run({uairt::bytes(x)}, {uairt::bytes(y)});
//
// Failures throw uairt::Error. Engine, Model and Buffer are move-only RAII handles; a Model or Buffer keeps
// its Engine alive, so the C API's destruction order cannot be violated. Free a Buffer only after the
// models that ran with it. Threading follows the C API: do not run one model from several threads at once.
#ifndef UAIRT_UAIRT_HPP
#define UAIRT_UAIRT_HPP

#include <uairt/uairt.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uairt {

enum class Status : int32_t {
  Ok = UAIRT_OK,
  InvalidArgument = UAIRT_ERR_INVALID_ARGUMENT,
  NotFound = UAIRT_ERR_NOT_FOUND,
  Unsupported = UAIRT_ERR_UNSUPPORTED,
  BackendUnavailable = UAIRT_ERR_BACKEND_UNAVAILABLE,
  IncompatibleModel = UAIRT_ERR_INCOMPATIBLE_MODEL,
  OutOfMemory = UAIRT_ERR_OUT_OF_MEMORY,
  Runtime = UAIRT_ERR_RUNTIME,
  VersionMismatch = UAIRT_ERR_VERSION_MISMATCH,
  Io = UAIRT_ERR_IO,
};

class Error : public std::runtime_error {
 public:
  Error(Status status, const std::string& message)
      : std::runtime_error(describe(status, message)), status_(status), message_(message) {}
  Status status() const noexcept { return status_; }
  const std::string& message() const noexcept { return message_; }

 private:
  static std::string describe(Status status, const std::string& message) {
    std::string name = uairt_status_string(static_cast<uairt_status>(status));
    return message.empty() ? name : name + ": " + message;
  }
  Status status_;
  std::string message_;
};

namespace detail {
inline void check(uairt_status status) {
  if (status != UAIRT_OK) {
    const char* message = uairt_last_error();
    throw Error(static_cast<Status>(status), message ? message : "");
  }
}
}  // namespace detail

enum class DType : int32_t {
  Unknown = UAIRT_DTYPE_UNKNOWN,
  Float32 = UAIRT_DTYPE_FLOAT32,
  Float16 = UAIRT_DTYPE_FLOAT16,
  BFloat16 = UAIRT_DTYPE_BFLOAT16,
  Int8 = UAIRT_DTYPE_INT8,
  UInt8 = UAIRT_DTYPE_UINT8,
  Int16 = UAIRT_DTYPE_INT16,
  UInt16 = UAIRT_DTYPE_UINT16,
  Int32 = UAIRT_DTYPE_INT32,
  Int64 = UAIRT_DTYPE_INT64,
  Bool = UAIRT_DTYPE_BOOL,
};

enum class Domain : uint32_t { Host = UAIRT_MEM_HOST, DmaBuf = UAIRT_MEM_DMABUF, Pinned = UAIRT_MEM_PINNED };

inline std::size_t dtype_size(DType dtype) { return uairt_dtype_size(static_cast<uairt_dtype>(dtype)); }

struct TensorInfo {
  std::string name;
  DType dtype = DType::Unknown;
  std::vector<int64_t> shape;
  float quant_scale = 0.f;        // zero when the tensor is not quantized
  int32_t quant_zero_point = 0;   // real value = (stored - quant_zero_point) * quant_scale

  int64_t num_elements() const {
    int64_t count = 1;
    for (int64_t dim : shape) count *= dim;
    return count;
  }
  std::size_t nbytes() const { return static_cast<std::size_t>(num_elements()) * dtype_size(dtype); }
};

// A writable block of memory (an output), and a read-only one (an input). A writable block converts to a
// read-only one, so bytes(vector) works for both inputs and outputs.
struct MutableBytes {
  void* data;
  std::size_t size;
};
struct ConstBytes {
  const void* data;
  std::size_t size;
  ConstBytes(const void* d, std::size_t n) : data(d), size(n) {}
  ConstBytes(const MutableBytes& block) : data(block.data), size(block.size) {}  // NOLINT: implicit on purpose
};

template <class T>
ConstBytes bytes(const std::vector<T>& values) { return ConstBytes(values.data(), values.size() * sizeof(T)); }
template <class T>
MutableBytes bytes(std::vector<T>& values) { return MutableBytes{values.data(), values.size() * sizeof(T)}; }

inline std::string version() { return uairt_version_string(); }

inline std::vector<std::string> backends() {
  std::vector<std::string> names;
  for (std::size_t i = 0; i < uairt_backend_count(); ++i) names.emplace_back(uairt_backend_name(i));
  return names;
}

// Loads a backend plugin and registers it. Plugins stay loaded for the process lifetime.
inline void load_backend_library(const std::string& path) {
  detail::check(uairt_load_backend_library(path.c_str()));
}

class Model;
class Buffer;

class Engine {
 public:
  Engine(const std::string& backend, const std::map<std::string, std::string>& options = {}) {
    std::vector<uairt_option> raw;
    raw.reserve(options.size());
    for (const auto& entry : options) raw.push_back({entry.first.c_str(), entry.second.c_str()});
    uairt_engine* handle = nullptr;
    detail::check(uairt_engine_create(backend.c_str(), raw.data(), raw.size(), &handle));
    handle_ = std::shared_ptr<uairt_engine>(handle, [](uairt_engine* e) { uairt_engine_destroy(e); });
  }

  inline Model load_model(const std::string& path) const;
  inline Model load_model(const void* data, std::size_t size, const std::string& format = "") const;
  inline Buffer alloc_buffer(std::size_t nbytes, Domain domain = Domain::Host) const;

  uairt_engine* native() const noexcept { return handle_.get(); }
  const std::shared_ptr<uairt_engine>& shared() const noexcept { return handle_; }

 private:
  std::shared_ptr<uairt_engine> handle_;
};

// A zero-copy buffer allocated by a backend.
class Buffer {
 public:
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept : engine_(std::move(other.engine_)), handle_(other.handle_) { other.handle_ = nullptr; }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      reset();
      engine_ = std::move(other.engine_);
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }
  ~Buffer() { reset(); }

  void* data() const noexcept { return uairt_buffer_data(handle_); }
  std::size_t size() const noexcept { return uairt_buffer_size(handle_); }
  int fd() const noexcept { return uairt_buffer_fd(handle_); }
  uairt_buffer* native() const noexcept { return handle_; }

 private:
  friend class Engine;
  Buffer(std::shared_ptr<uairt_engine> engine, uairt_buffer* handle) : engine_(std::move(engine)), handle_(handle) {}
  void reset() noexcept {
    if (handle_) uairt_buffer_free(handle_);
    handle_ = nullptr;
  }
  std::shared_ptr<uairt_engine> engine_;
  uairt_buffer* handle_;
};

class Model {
 public:
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&& other) noexcept
      : engine_(std::move(other.engine_)), handle_(other.handle_), inputs_(std::move(other.inputs_)),
        outputs_(std::move(other.outputs_)) {
    other.handle_ = nullptr;
  }
  Model& operator=(Model&& other) noexcept {
    if (this != &other) {
      reset();
      engine_ = std::move(other.engine_);
      handle_ = other.handle_;
      inputs_ = std::move(other.inputs_);
      outputs_ = std::move(other.outputs_);
      other.handle_ = nullptr;
    }
    return *this;
  }
  ~Model() { reset(); }

  const std::vector<TensorInfo>& inputs() const noexcept { return inputs_; }
  const std::vector<TensorInfo>& outputs() const noexcept { return outputs_; }

  // Runs the model. One block per input and output, sized exactly as TensorInfo::nbytes reports.
  void run(const std::vector<ConstBytes>& inputs, const std::vector<MutableBytes>& outputs) {
    if (inputs.size() != inputs_.size() || outputs.size() != outputs_.size())
      throw Error(Status::InvalidArgument, "model takes " + std::to_string(inputs_.size()) + " inputs and " +
                                               std::to_string(outputs_.size()) + " outputs, got " +
                                               std::to_string(inputs.size()) + " and " + std::to_string(outputs.size()));
    std::vector<uairt_tensor> in(inputs.size()), out(outputs.size());
    bind(in, inputs_, inputs, false, "input");
    bind(out, outputs_, outputs, true, "output");
    detail::check(uairt_model_run(handle_, in.data(), in.size(), out.data(), out.size()));
  }

  // Runs with zero-copy buffers bound to the model's inputs and outputs.
  void run_buffers(const std::vector<const Buffer*>& inputs, const std::vector<const Buffer*>& outputs) {
    if (inputs.size() != inputs_.size() || outputs.size() != outputs_.size())
      throw Error(Status::InvalidArgument, "one buffer is needed per model input and output");
    std::vector<uairt_tensor> in(inputs.size()), out(outputs.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
      detail::check(uairt_model_input_info(handle_, i, &in[i]));
      uairt_tensor_use_buffer(&in[i], inputs[i]->native());
    }
    for (std::size_t i = 0; i < out.size(); ++i) {
      detail::check(uairt_model_output_info(handle_, i, &out[i]));
      uairt_tensor_use_buffer(&out[i], outputs[i]->native());
    }
    detail::check(uairt_model_run(handle_, in.data(), in.size(), out.data(), out.size()));
  }

  uairt_model* native() const noexcept { return handle_; }

 private:
  friend class Engine;
  Model(std::shared_ptr<uairt_engine> engine, uairt_model* handle) : engine_(std::move(engine)), handle_(handle) {
    for (std::size_t i = 0; i < uairt_model_num_inputs(handle_); ++i) inputs_.push_back(describe(i, false));
    for (std::size_t i = 0; i < uairt_model_num_outputs(handle_); ++i) outputs_.push_back(describe(i, true));
  }

  TensorInfo describe(std::size_t index, bool output) {
    uairt_tensor tensor;
    detail::check(output ? uairt_model_output_info(handle_, index, &tensor) : uairt_model_input_info(handle_, index, &tensor));
    TensorInfo info;
    info.name = tensor.name ? tensor.name : "";
    info.dtype = static_cast<DType>(tensor.dtype);
    info.shape.assign(tensor.dims, tensor.dims + tensor.rank);
    info.quant_scale = tensor.quant_scale;
    info.quant_zero_point = tensor.quant_zero_point;
    return info;
  }

  template <class Block>
  void bind(std::vector<uairt_tensor>& tensors, const std::vector<TensorInfo>& infos, const std::vector<Block>& blocks,
            bool output, const char* kind) {
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      if (blocks[i].size != infos[i].nbytes())
        throw Error(Status::InvalidArgument, std::string(kind) + " " + std::to_string(i) + " ('" + infos[i].name +
                                                 "') needs " + std::to_string(infos[i].nbytes()) + " bytes, got " +
                                                 std::to_string(blocks[i].size));
      detail::check(output ? uairt_model_output_info(handle_, i, &tensors[i]) : uairt_model_input_info(handle_, i, &tensors[i]));
      tensors[i].data = const_cast<void*>(blocks[i].data);
      tensors[i].nbytes = blocks[i].size;
    }
  }

  void reset() noexcept {
    if (handle_) uairt_model_destroy(handle_);
    handle_ = nullptr;
  }

  std::shared_ptr<uairt_engine> engine_;
  uairt_model* handle_;
  std::vector<TensorInfo> inputs_, outputs_;
};

inline Model Engine::load_model(const std::string& path) const {
  uairt_model_source source{};
  source.struct_size = sizeof(source);
  source.path = path.c_str();
  uairt_model* handle = nullptr;
  detail::check(uairt_model_load(handle_.get(), &source, &handle));
  return Model(handle_, handle);
}

inline Model Engine::load_model(const void* data, std::size_t size, const std::string& format) const {
  uairt_model_source source{};
  source.struct_size = sizeof(source);
  source.data = data;
  source.size = size;
  source.format = format.empty() ? nullptr : format.c_str();
  uairt_model* handle = nullptr;
  detail::check(uairt_model_load(handle_.get(), &source, &handle));
  return Model(handle_, handle);
}

inline Buffer Engine::alloc_buffer(std::size_t nbytes, Domain domain) const {
  uairt_buffer* handle = nullptr;
  detail::check(uairt_buffer_alloc(handle_.get(), nbytes, static_cast<uairt_memory_domain>(domain), &handle));
  return Buffer(handle_, handle);
}

}  // namespace uairt

#endif  // UAIRT_UAIRT_HPP
