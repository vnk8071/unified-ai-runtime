// SPDX-License-Identifier: Apache-2.0
/*
 * MNN backend (Alibaba), built as a loadable plugin.
 * v1 limits: a .mnn file from a path or from memory, host memory, static shapes. Inputs and outputs are numbered in the order of their
 * names, because MNN's session API returns them in a name-ordered map (it does not keep the order the model declared). Tensors are exchanged in the
 * layout MNN reports for them (usually NCHW for models converted from ONNX); MNN converts to its internal layout itself. A model whose
 * input shapes are not fixed needs the `input_shapes` option. device=metal, opencl, vulkan or cuda selects a GPU backend of an MNN
 * built with it, and fails if the session did not end up on it, instead of quietly running on the CPU as MNN itself would.
 * The plugin must be built with the same C++ toolchain as the MNN library it links.
 */
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <uairt/uairt_backend.h>

#include "common.h"

#define OPTION_DEVICE "device"
#define OPTION_THREADS "num_threads"
#define OPTION_PRECISION "precision"
#define OPTION_INPUT_SHAPES "input_shapes"

namespace {

const uairt_host_api* g_host;

#define fail(...) uairt_backend_fail(g_host, __VA_ARGS__)

struct Engine {
  MNNForwardType type = MNN_FORWARD_CPU;
  int threads = 0;
  MNN::BackendConfig::PrecisionMode precision = MNN::BackendConfig::Precision_Normal;
  std::vector<std::vector<int>> shapes;  // optional, one per input in name order
};

struct Io {
  std::string name;
  MNN::Tensor* tensor = nullptr;  // owned by the session
  std::vector<int> shape;
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  halide_type_t type;
  MNN::Tensor::DimensionType dim_type = MNN::Tensor::CAFFE;
  size_t nbytes = 0;
};

struct Model {
  std::unique_ptr<MNN::Interpreter, void (*)(MNN::Interpreter*)> net{nullptr, &MNN::Interpreter::destroy};
  MNN::Session* session = nullptr;
  std::vector<Io> inputs;
  std::vector<Io> outputs;
};

bool parse_shape(const std::string& text, std::vector<int>* out) {
  out->clear();
  size_t start = 0;
  while (start <= text.size()) {
    size_t end = text.find('x', start);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string part = text.substr(start, end - start);
    char* tail = nullptr;
    const long value = std::strtol(part.c_str(), &tail, 10);
    if (part.empty() || *tail != '\0' || value < 1 || value > 1 << 24) {
      return false;
    }
    out->push_back(static_cast<int>(value));
    start = end + 1;
  }
  return !out->empty();
}

uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  auto engine = std::make_unique<Engine>();
  for (size_t i = 0; i < num_options; ++i) {
    const std::string key = options[i].key;
    const std::string value = options[i].value ? options[i].value : "";
    if (key == OPTION_DEVICE) {
      if (value == "cpu") {
        engine->type = MNN_FORWARD_CPU;
      } else if (value == "metal") {
        engine->type = MNN_FORWARD_METAL;
      } else if (value == "opencl") {
        engine->type = MNN_FORWARD_OPENCL;
      } else if (value == "vulkan") {
        engine->type = MNN_FORWARD_VULKAN;
      } else if (value == "cuda") {
        engine->type = MNN_FORWARD_CUDA;
      } else {
        fail("%s must be cpu, metal, opencl, vulkan or cuda, got '%s'", OPTION_DEVICE, value.c_str());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (key == OPTION_THREADS) {
      engine->threads = std::atoi(value.c_str());
      if (engine->threads < 1) {
        fail("%s must be a positive integer", OPTION_THREADS);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (key == OPTION_PRECISION) {
      if (value == "normal") {
        engine->precision = MNN::BackendConfig::Precision_Normal;
      } else if (value == "high") {
        engine->precision = MNN::BackendConfig::Precision_High;
      } else if (value == "low") {
        engine->precision = MNN::BackendConfig::Precision_Low;
      } else if (value == "bf16") {
        engine->precision = MNN::BackendConfig::Precision_Low_BF16;
      } else {
        fail("%s must be normal, high, low or bf16, got '%s'", OPTION_PRECISION, value.c_str());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (key == OPTION_INPUT_SHAPES) {
      size_t start = 0;
      while (start <= value.size()) {
        size_t end = value.find(';', start);
        if (end == std::string::npos) {
          end = value.size();
        }
        std::vector<int> shape;
        if (!parse_shape(value.substr(start, end - start), &shape)) {
          fail("%s wants shapes such as 1x3x224x224, separated by ';', got '%s'", OPTION_INPUT_SHAPES, value.c_str());
          return UAIRT_ERR_INVALID_ARGUMENT;
        }
        engine->shapes.push_back(std::move(shape));
        start = end + 1;
      }
    } else {
      fail("unknown option '%s'", key.c_str());
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  *out_engine = engine.release();
  return UAIRT_OK;
}

void destroy_engine(void* engine) {
  delete static_cast<Engine*>(engine);
}

uairt_dtype dtype_of(halide_type_t type) {
  switch (type.code) {
    case halide_type_float:
      return type.bits == 32 ? UAIRT_DTYPE_FLOAT32 : type.bits == 16 ? UAIRT_DTYPE_FLOAT16 : UAIRT_DTYPE_UNKNOWN;
    case halide_type_bfloat:
      return type.bits == 16 ? UAIRT_DTYPE_BFLOAT16 : UAIRT_DTYPE_UNKNOWN;
    case halide_type_int:
      return type.bits == 8 ? UAIRT_DTYPE_INT8 : type.bits == 16 ? UAIRT_DTYPE_INT16
             : type.bits == 32 ? UAIRT_DTYPE_INT32 : type.bits == 64 ? UAIRT_DTYPE_INT64 : UAIRT_DTYPE_UNKNOWN;
    case halide_type_uint:
      return type.bits == 8 ? UAIRT_DTYPE_UINT8 : type.bits == 16 ? UAIRT_DTYPE_UINT16 : UAIRT_DTYPE_UNKNOWN;
    default:
      return UAIRT_DTYPE_UNKNOWN;
  }
}

bool describe_tensor(const std::string& name, MNN::Tensor* tensor, Io* io, std::string* problem) {
  io->name = name;
  io->tensor = tensor;
  io->shape = tensor->shape();
  io->type = tensor->getType();
  io->dtype = dtype_of(io->type);
  io->dim_type = tensor->getDimensionType();
  if (io->dtype == UAIRT_DTYPE_UNKNOWN) {
    *problem = "tensor '" + name + "' has a data type UAIRT does not know (" + std::to_string(io->type.code) + "/" +
               std::to_string(io->type.bits) + " bits)";
    return false;
  }
  if (io->shape.empty() || io->shape.size() > UAIRT_MAX_RANK) {
    *problem = "tensor '" + name + "' has rank " + std::to_string(io->shape.size());
    return false;
  }
  size_t elements = 1;
  for (int dim : io->shape) {
    if (dim < 1) {
      *problem = "tensor '" + name + "' has a dynamic dimension; give " OPTION_INPUT_SHAPES " (MNN reports " +
                 std::to_string(dim) + ")";
      return false;
    }
    elements *= static_cast<size_t>(dim);
  }
  io->nbytes = elements * (io->type.bits / 8);
  return true;
}

uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const Engine* engine = static_cast<Engine*>(engine_handle);
  try {
    auto model = std::make_unique<Model>();
    if (source->path) {
      if (!uairt_backend_file_exists(source->path)) {
        fail("cannot open the MNN model '%s'", source->path);
        return UAIRT_ERR_IO;
      }
      model->net.reset(MNN::Interpreter::createFromFile(source->path));
    } else {
      model->net.reset(MNN::Interpreter::createFromBuffer(source->data, source->size));
    }
    if (!model->net) {
      fail("MNN cannot read %s (is it a .mnn file?)", source->path ? source->path : "the model buffer");
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }

    MNN::ScheduleConfig config;
    config.type = engine->type;
    config.backupType = MNN_FORWARD_CPU;
    if (engine->threads > 0) {
      config.numThread = engine->threads;
    }
    MNN::BackendConfig backend;
    backend.precision = engine->precision;
    config.backendConfig = &backend;
    model->session = model->net->createSession(config);
    if (!model->session) {
      fail("MNN could not create a session for this model");
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
    if (engine->type != MNN_FORWARD_CPU) {
      int used[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
      model->net->getSessionInfo(model->session, MNN::Interpreter::BACKENDS, used);
      if (used[0] != static_cast<int>(engine->type)) {
        fail("the MNN session did not run on the requested device (this MNN may not be built with it, or it is not usable here)");
        return UAIRT_ERR_BACKEND_UNAVAILABLE;
      }
    }

    // Inputs: std::map order is by name.
    const auto& input_map = model->net->getSessionInputAll(model->session);
    if (!engine->shapes.empty()) {
      if (engine->shapes.size() != input_map.size()) {
        fail("the model has %zu input(s) but '%s' lists %zu shape(s)", input_map.size(), OPTION_INPUT_SHAPES,
             engine->shapes.size());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
      size_t index = 0;
      for (const auto& entry : input_map) {
        model->net->resizeTensor(entry.second, engine->shapes[index++]);
      }
      model->net->resizeSession(model->session);
    }
    std::string problem;
    for (const auto& entry : model->net->getSessionInputAll(model->session)) {
      Io io;
      if (!describe_tensor(entry.first, entry.second, &io, &problem)) {
        fail("%s", problem.c_str());
        return UAIRT_ERR_UNSUPPORTED;
      }
      model->inputs.push_back(std::move(io));
    }
    for (const auto& entry : model->net->getSessionOutputAll(model->session)) {
      Io io;
      if (!describe_tensor(entry.first, entry.second, &io, &problem)) {
        fail("%s", problem.c_str());
        return UAIRT_ERR_UNSUPPORTED;
      }
      model->outputs.push_back(std::move(io));
    }
    if (model->inputs.empty() || model->outputs.empty()) {
      fail("the model has no inputs or no outputs");
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
    *out_model = model.release();
    return UAIRT_OK;
  } catch (const std::exception& error) {
    fail("%s", error.what());
    return UAIRT_ERR_RUNTIME;
  }
}

void destroy_model(void* model) {
  Model* m = static_cast<Model*>(model);
  if (m->session) {
    m->net->releaseSession(m->session);
  }
  delete m;
}

size_t num_inputs(const void* model) {
  return static_cast<const Model*>(model)->inputs.size();
}

size_t num_outputs(const void* model) {
  return static_cast<const Model*>(model)->outputs.size();
}

void describe(const Io& io, uairt_tensor* out) {
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(*out);
  out->dtype = io.dtype;
  out->rank = static_cast<uint32_t>(io.shape.size());
  for (size_t i = 0; i < io.shape.size(); ++i) {
    out->dims[i] = io.shape[i];
  }
  out->domain = UAIRT_MEM_HOST;
  out->dmabuf_fd = -1;
  out->nbytes = io.nbytes;
  out->name = io.name.c_str();
}

uairt_status input_info(const void* model, size_t index, uairt_tensor* out) {
  describe(static_cast<const Model*>(model)->inputs[index], out);
  return UAIRT_OK;
}

uairt_status output_info(const void* model, size_t index, uairt_tensor* out) {
  describe(static_cast<const Model*>(model)->outputs[index], out);
  return UAIRT_OK;
}

uairt_status run(void* handle, const uairt_tensor* inputs, size_t n_in, uairt_tensor* outputs, size_t n_out) {
  Model* model = static_cast<Model*>(handle);
  try {
    for (size_t i = 0; i < n_in; ++i) {
      const Io& io = model->inputs[i];
      // A host tensor over the caller's buffer: MNN converts from it to the layout it computes in.
      std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create(io.shape, io.type, inputs[i].data, io.dim_type));
      if (!host || !io.tensor->copyFromHostTensor(host.get())) {
        fail("MNN rejected input '%s'", io.name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
    }
    if (model->net->runSession(model->session) != MNN::NO_ERROR) {
      fail("MNN failed to run the model");
      return UAIRT_ERR_RUNTIME;
    }
    for (size_t i = 0; i < n_out; ++i) {
      const Io& io = model->outputs[i];
      if (io.tensor->shape() != io.shape) {
        fail("output '%s' changed shape since load; the mnn backend needs static shapes", io.name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
      std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create(io.shape, io.type, outputs[i].data, io.dim_type));
      if (!host || !io.tensor->copyToHostTensor(host.get())) {
        fail("MNN could not read output '%s'", io.name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
    }
    return UAIRT_OK;
  } catch (const std::exception& error) {
    fail("%s", error.what());
    return UAIRT_ERR_RUNTIME;
  }
}

const uairt_backend_api* api() {
  static const uairt_backend_api kApi = [] {
    uairt_backend_api table;
    std::memset(&table, 0, sizeof(table));
    table.struct_size = sizeof(table);
    table.abi_version = UAIRT_BACKEND_ABI_VERSION;
    table.name = "mnn";
    table.supported_domains = UAIRT_MEM_HOST;
    table.create_engine = create_engine;
    table.destroy_engine = destroy_engine;
    table.load_model = load_model;
    table.destroy_model = destroy_model;
    table.num_inputs = num_inputs;
    table.num_outputs = num_outputs;
    table.input_info = input_info;
    table.output_info = output_info;
    table.run = run;
    return table;
  }();
  return &kApi;
}

}  // namespace

extern "C" UAIRT_API const uairt_backend_api* UAIRT_BACKEND_ENTRY(const uairt_host_api* host) {
  g_host = host;
  return api();
}
