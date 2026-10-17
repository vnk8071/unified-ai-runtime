// SPDX-License-Identifier: Apache-2.0
/*
 * ExecuTorch backend, built as a loadable plugin.
 * v1 limits: a .pte file loaded from a path, one method (default "forward"), tensor inputs and outputs only, static
 * shapes, contiguous (default dim order) host memory. Which hardware runs the model is decided when the .pte is
 * exported: the delegates it names (XNNPACK, CoreML, QNN, ...) must also be built into the ExecuTorch library this
 * plugin links, otherwise loading fails and the plugin does not fall back to another device.
 * The plugin must be built with the same C++ toolchain as the ExecuTorch library it links.
 */
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <uairt/uairt_backend.h>

#define OPTION_METHOD "method"

namespace {

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

const uairt_host_api* g_host;

void fail(const char* fmt, ...) {
  char message[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

struct Engine {
  std::string method = "forward";
};

struct Io {
  std::string name;
  ScalarType scalar_type = ScalarType::Float;
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  std::vector<int32_t> shape;
  size_t nbytes = 0;
};

struct Model {
  std::unique_ptr<Module> module;
  std::string method;
  std::vector<Io> inputs;
  std::vector<Io> outputs;
};

uairt_dtype to_uairt_dtype(ScalarType type) {
  switch (type) {
    case ScalarType::Float: return UAIRT_DTYPE_FLOAT32;
    case ScalarType::Half: return UAIRT_DTYPE_FLOAT16;
    case ScalarType::BFloat16: return UAIRT_DTYPE_BFLOAT16;
    case ScalarType::Char: return UAIRT_DTYPE_INT8;
    case ScalarType::Byte: return UAIRT_DTYPE_UINT8;
    case ScalarType::Short: return UAIRT_DTYPE_INT16;
    case ScalarType::UInt16: return UAIRT_DTYPE_UINT16;
    case ScalarType::Int: return UAIRT_DTYPE_INT32;
    case ScalarType::Long: return UAIRT_DTYPE_INT64;
    case ScalarType::Bool: return UAIRT_DTYPE_BOOL;
    default: return UAIRT_DTYPE_UNKNOWN;
  }
}

bool file_exists(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file) {
    std::fclose(file);
  }
  return file != nullptr;
}

uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  try {
    auto engine = std::make_unique<Engine>();
    for (size_t i = 0; i < num_options; ++i) {
      const std::string key = options[i].key;
      const std::string value = options[i].value ? options[i].value : "";
      if (key == OPTION_METHOD) {
        if (value.empty()) {
          fail("%s must not be empty", OPTION_METHOD);
          return UAIRT_ERR_INVALID_ARGUMENT;
        }
        engine->method = value;
      } else {
        fail("unknown option '%s'", key.c_str());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    }
    *out_engine = engine.release();
    return UAIRT_OK;
  } catch (const std::exception& error) {
    fail("%s", error.what());
    return UAIRT_ERR_RUNTIME;
  }
}

void destroy_engine(void* engine) {
  delete static_cast<Engine*>(engine);
}

uairt_status map_load_error(Error error, const char* what, const std::string& path) {
  switch (error) {
    case Error::NotFound:
    case Error::AccessFailed:
      fail("ExecuTorch cannot open %s of '%s': %s", what, path.c_str(), executorch::runtime::to_string(error));
      return UAIRT_ERR_IO;
    case Error::MemoryAllocationFailed:
    case Error::OutOfResources:
      fail("ExecuTorch ran out of memory loading %s of '%s': %s", what, path.c_str(),
           executorch::runtime::to_string(error));
      return UAIRT_ERR_OUT_OF_MEMORY;
    default:
      // InvalidProgram (not a .pte, or a version this library cannot read), OperatorMissing (a kernel or delegate that
      // was not built into this ExecuTorch library), NotSupported and similar all mean the artifact does not match.
      fail("ExecuTorch cannot load %s of '%s': %s (is it a .pte exported for this ExecuTorch build, with every "
           "delegate and operator it uses linked in?)",
           what, path.c_str(), executorch::runtime::to_string(error));
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
  }
}

uairt_status describe_tensor(const char* kind, size_t index, const TensorInfo& info, Io* out) {
  out->name = std::string(kind) + std::to_string(index);
  out->scalar_type = info.scalar_type();
  out->dtype = to_uairt_dtype(out->scalar_type);
  if (out->dtype == UAIRT_DTYPE_UNKNOWN) {
    fail("%s '%s' has ExecuTorch scalar type %d, which UAIRT cannot represent", kind, out->name.c_str(),
         static_cast<int>(out->scalar_type));
    return UAIRT_ERR_UNSUPPORTED;
  }
  const auto sizes = info.sizes();
  if (sizes.size() > UAIRT_MAX_RANK) {
    fail("%s '%s' has rank %zu, above UAIRT's limit of %d", kind, out->name.c_str(), sizes.size(), UAIRT_MAX_RANK);
    return UAIRT_ERR_UNSUPPORTED;
  }
  const auto dim_order = info.dim_order();
  for (size_t d = 0; d < dim_order.size(); ++d) {
    if (dim_order[d] != d) {
      fail("%s '%s' is not in the default (contiguous) dim order, for example channels-last", kind,
           out->name.c_str());
      return UAIRT_ERR_UNSUPPORTED;
    }
  }
  out->shape.assign(sizes.begin(), sizes.end());
  out->nbytes = info.nbytes();
  return UAIRT_OK;
}

uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const Engine* engine = static_cast<Engine*>(engine_handle);
  if (!source->path) {
    fail("the executorch backend loads a .pte file from a path, not from memory");
    return UAIRT_ERR_UNSUPPORTED;
  }
  try {
    const std::string path = source->path;
    if (!file_exists(path)) {
      fail("cannot open the ExecuTorch model '%s'", path.c_str());
      return UAIRT_ERR_IO;
    }
    auto model = std::make_unique<Model>();
    model->method = engine->method;
    model->module = std::make_unique<Module>(path, Module::LoadMode::File);
    Error error = model->module->load();
    if (error != Error::Ok) {
      return map_load_error(error, "the program", path);
    }
    error = model->module->load_method(model->method);
    if (error != Error::Ok) {
      return map_load_error(error, ("method '" + model->method + "'").c_str(), path);
    }
    auto meta = model->module->method_meta(model->method);
    if (!meta.ok()) {
      return map_load_error(meta.error(), ("the metadata of method '" + model->method + "'").c_str(), path);
    }
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      auto tag = meta->input_tag(i);
      auto info = meta->input_tensor_meta(i);
      if (!tag.ok() || *tag != Tag::Tensor || !info.ok()) {
        fail("input %zu of method '%s' is not a tensor; the executorch backend runs tensor inputs only", i,
             model->method.c_str());
        return UAIRT_ERR_UNSUPPORTED;
      }
      Io io;
      uairt_status status = describe_tensor("input", i, *info, &io);
      if (status != UAIRT_OK) {
        return status;
      }
      model->inputs.push_back(std::move(io));
    }
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
      auto tag = meta->output_tag(i);
      auto info = meta->output_tensor_meta(i);
      if (!tag.ok() || *tag != Tag::Tensor || !info.ok()) {
        fail("output %zu of method '%s' is not a tensor; the executorch backend returns tensor outputs only", i,
             model->method.c_str());
        return UAIRT_ERR_UNSUPPORTED;
      }
      Io io;
      uairt_status status = describe_tensor("output", i, *info, &io);
      if (status != UAIRT_OK) {
        return status;
      }
      model->outputs.push_back(std::move(io));
    }
    *out_model = model.release();
    return UAIRT_OK;
  } catch (const std::exception& error) {
    fail("%s", error.what());
    return UAIRT_ERR_RUNTIME;
  }
}

void destroy_model(void* model) {
  delete static_cast<Model*>(model);
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
    std::vector<executorch::extension::TensorPtr> views;
    std::vector<EValue> values;
    views.reserve(n_in);
    values.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
      const Io& io = model->inputs[i];
      // The tensor only views the caller's buffer; ExecuTorch does not write to inputs.
      views.push_back(executorch::extension::from_blob(
          const_cast<void*>(inputs[i].data), std::vector<executorch::aten::SizesType>(io.shape.begin(), io.shape.end()),
          io.scalar_type));
      values.emplace_back(*views.back());
    }
    auto result = model->module->execute(model->method, values);
    if (!result.ok()) {
      fail("ExecuTorch failed to run method '%s': %s", model->method.c_str(),
           executorch::runtime::to_string(result.error()));
      return UAIRT_ERR_RUNTIME;
    }
    if (result->size() != n_out) {
      fail("method '%s' returned %zu value(s), expected %zu", model->method.c_str(), result->size(), n_out);
      return UAIRT_ERR_RUNTIME;
    }
    for (size_t i = 0; i < n_out; ++i) {
      if (!(*result)[i].isTensor()) {
        fail("output '%s' is not a tensor", model->outputs[i].name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
      const auto& tensor = (*result)[i].toTensor();
      if (tensor.scalar_type() != model->outputs[i].scalar_type || tensor.nbytes() != model->outputs[i].nbytes) {
        fail("output '%s' changed type or size since load; the executorch backend needs static shapes",
             model->outputs[i].name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
      std::memcpy(outputs[i].data, tensor.const_data_ptr(), tensor.nbytes());
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
    table.name = "executorch";
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

#if defined(UAIRT_EXECUTORCH_GLOBAL_SYMBOLS) && !defined(_WIN32)
// The Metal delegate's compiled library looks up aoti_torch_* symbols in the global namespace.
extern "C" UAIRT_API const int uairt_backend_global_symbols = 1;
#endif

extern "C" UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return api();
}
