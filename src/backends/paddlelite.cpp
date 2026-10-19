// SPDX-License-Identifier: Apache-2.0
/*
 * Paddle Lite backend (Baidu's mobile and edge engine), built as a loadable plugin.
 * v1 limits: a `.nb` file (made from a Paddle model with Paddle Lite's `opt` tool) from a path or from memory, float32 tensors, static shapes,
 * host memory. A `.nb` file records no input shapes, so the `input_shapes` option is required ("1x3x224x224" for one input, entries separated
 * by ';' in the model's input order, which is the order Paddle Lite reports). Output shapes are found by running the model once on zeros at
 * load time. Inputs and outputs keep the order of the model.
 * The plugin must be built with the same C++ toolchain as the Paddle Lite library it links, and that library must have been built with
 * exceptions on (`LITE_WITH_EXCEPTION=ON`): without it a model error aborts the process instead of reaching the catch clauses below.
 */
#include <paddle_api.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <uairt/uairt_backend.h>

#include "common.h"

#define OPTION_THREADS "num_threads"
#define OPTION_POWER_MODE "power_mode"
#define OPTION_PRECISION "precision"
#define OPTION_INPUT_SHAPES "input_shapes"

namespace {

using paddle::lite_api::PaddlePredictor;

const uairt_host_api* g_host;

#define fail(...) uairt_backend_fail(g_host, __VA_ARGS__)

struct Engine {
  int threads = 0;
  paddle::lite_api::PowerMode power_mode = paddle::lite_api::LITE_POWER_NO_BIND;
  paddle::lite_api::PrecisionMode precision = paddle::lite_api::LITE_PRECISION_NORMAL;
  std::vector<std::vector<int64_t>> shapes;
};

struct Io {
  std::string name;
  std::vector<int64_t> shape;
  size_t elements = 0;
};

struct Model {
  std::shared_ptr<PaddlePredictor> predictor;
  std::vector<Io> inputs;
  std::vector<Io> outputs;
};

size_t count_elements(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

bool parse_shape(const std::string& text, std::vector<int64_t>* out) {
  out->clear();
  size_t start = 0;
  while (start <= text.size()) {
    size_t end = text.find('x', start);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string part = text.substr(start, end - start);
    char* tail = nullptr;
    const long long value = std::strtoll(part.c_str(), &tail, 10);
    if (part.empty() || *tail != '\0' || value < 1 || value > (1 << 24)) {
      return false;
    }
    out->push_back(value);
    start = end + 1;
  }
  return !out->empty() && out->size() <= UAIRT_MAX_RANK;
}

uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  auto engine = std::make_unique<Engine>();
  for (size_t i = 0; i < num_options; ++i) {
    const std::string key = options[i].key;
    const std::string value = options[i].value ? options[i].value : "";
    if (key == OPTION_THREADS) {
      engine->threads = std::atoi(value.c_str());
      if (engine->threads < 1) {
        fail("%s must be a positive integer", OPTION_THREADS);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (key == OPTION_POWER_MODE) {
      using namespace paddle::lite_api;
      if (value == "high") {
        engine->power_mode = LITE_POWER_HIGH;
      } else if (value == "low") {
        engine->power_mode = LITE_POWER_LOW;
      } else if (value == "full") {
        engine->power_mode = LITE_POWER_FULL;
      } else if (value == "no_bind") {
        engine->power_mode = LITE_POWER_NO_BIND;
      } else if (value == "rand_high") {
        engine->power_mode = LITE_POWER_RAND_HIGH;
      } else if (value == "rand_low") {
        engine->power_mode = LITE_POWER_RAND_LOW;
      } else {
        fail("%s must be high, low, full, no_bind, rand_high or rand_low, got '%s'", OPTION_POWER_MODE, value.c_str());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (key == OPTION_PRECISION) {
      if (value == "normal") {
        engine->precision = paddle::lite_api::LITE_PRECISION_NORMAL;
      } else if (value == "low") {
        engine->precision = paddle::lite_api::LITE_PRECISION_LOW;
      } else {
        fail("%s must be normal or low, got '%s'", OPTION_PRECISION, value.c_str());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    } else if (key == OPTION_INPUT_SHAPES) {
      size_t start = 0;
      while (start <= value.size()) {
        size_t end = value.find(';', start);
        if (end == std::string::npos) {
          end = value.size();
        }
        std::vector<int64_t> shape;
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
  if (engine->shapes.empty()) {
    fail("the paddlelite backend needs '%s': a .nb file records no input shapes (for example 1x3x224x224)", OPTION_INPUT_SHAPES);
    return UAIRT_ERR_INVALID_ARGUMENT;
  }
  *out_engine = engine.release();
  return UAIRT_OK;
}

void destroy_engine(void* engine) {
  delete static_cast<Engine*>(engine);
}

uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const Engine* engine = static_cast<Engine*>(engine_handle);
  try {
    paddle::lite_api::MobileConfig config;
    if (engine->threads > 0) {
      config.set_threads(engine->threads);
    }
    config.set_power_mode(engine->power_mode);
    config.set_precision_mode(engine->precision);
    if (source->path) {
      if (!uairt_backend_file_exists(source->path)) {
        fail("cannot open the Paddle Lite model '%s'", source->path);
        return UAIRT_ERR_IO;
      }
      config.set_model_from_file(source->path);
    } else {
      config.set_model_from_buffer(static_cast<const char*>(source->data), source->size);
    }

    auto model = std::make_unique<Model>();
    try {
      model->predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);
    } catch (const std::exception& error) {
      fail("Paddle Lite cannot read %s (is it a .nb file made by `opt`?): %s", source->path ? source->path : "the model buffer",
           error.what());
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
    if (!model->predictor) {
      fail("Paddle Lite could not create a predictor for %s (is it a .nb file made by `opt`?)",
           source->path ? source->path : "the model buffer");
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }

    const std::vector<std::string> input_names = model->predictor->GetInputNames();
    if (input_names.size() != engine->shapes.size()) {
      fail("the model has %zu input(s) (%s) but '%s' lists %zu shape(s)", input_names.size(),
           input_names.empty() ? "none" : input_names[0].c_str(), OPTION_INPUT_SHAPES, engine->shapes.size());
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < input_names.size(); ++i) {
      Io io;
      io.name = input_names[i];
      io.shape = engine->shapes[i];
      io.elements = count_elements(io.shape);
      model->inputs.push_back(std::move(io));
    }

    // Output shapes are not recorded: run once on zeros and read them off.
    for (size_t i = 0; i < model->inputs.size(); ++i) {
      auto tensor = model->predictor->GetInput(static_cast<int>(i));
      tensor->Resize(model->inputs[i].shape);
      std::memset(tensor->mutable_data<float>(), 0, model->inputs[i].elements * sizeof(float));
    }
    model->predictor->Run();
    const std::vector<std::string> output_names = model->predictor->GetOutputNames();
    for (size_t i = 0; i < output_names.size(); ++i) {
      auto tensor = model->predictor->GetOutput(static_cast<int>(i));
      if (tensor->precision() != paddle::lite_api::PrecisionType::kFloat) {
        fail("output '%s' is not float32 (Paddle Lite precision %d); the paddlelite backend handles float32 tensors", output_names[i].c_str(),
             static_cast<int>(tensor->precision()));
        return UAIRT_ERR_UNSUPPORTED;
      }
      Io io;
      io.name = output_names[i];
      io.shape = tensor->shape();
      if (io.shape.empty() || io.shape.size() > UAIRT_MAX_RANK) {
        fail("output '%s' has rank %zu", io.name.c_str(), io.shape.size());
        return UAIRT_ERR_UNSUPPORTED;
      }
      io.elements = count_elements(io.shape);
      model->outputs.push_back(std::move(io));
    }
    if (model->outputs.empty()) {
      fail("the model has no outputs");
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
    *out_model = model.release();
    return UAIRT_OK;
  } catch (const std::exception& error) {
    fail("the model failed on zero inputs of the shapes given in '%s': %s", OPTION_INPUT_SHAPES, error.what());
    return UAIRT_ERR_INCOMPATIBLE_MODEL;
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
  out->dtype = UAIRT_DTYPE_FLOAT32;
  out->rank = static_cast<uint32_t>(io.shape.size());
  for (size_t i = 0; i < io.shape.size(); ++i) {
    out->dims[i] = io.shape[i];
  }
  out->domain = UAIRT_MEM_HOST;
  out->dmabuf_fd = -1;
  out->nbytes = io.elements * sizeof(float);
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
      auto tensor = model->predictor->GetInput(static_cast<int>(i));
      tensor->Resize(model->inputs[i].shape);
      std::memcpy(tensor->mutable_data<float>(), inputs[i].data, model->inputs[i].elements * sizeof(float));
    }
    model->predictor->Run();
    for (size_t i = 0; i < n_out; ++i) {
      auto tensor = model->predictor->GetOutput(static_cast<int>(i));
      if (tensor->shape() != model->outputs[i].shape) {
        fail("output '%s' changed shape since load; the paddlelite backend needs static shapes", model->outputs[i].name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
      std::memcpy(outputs[i].data, tensor->data<float>(), model->outputs[i].elements * sizeof(float));
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
    table.name = "paddlelite";
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
