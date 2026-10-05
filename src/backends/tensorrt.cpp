// SPDX-License-Identifier: Apache-2.0
/*
 * NVIDIA TensorRT backend, built as a loadable plugin.
 * v1 limits: serialized engines only (.engine / .plan, built for this GPU and TensorRT version),
 * one execution context per model, static shapes, host memory or pinned (page-locked) buffers from
 * uairt_buffer_alloc (inputs and outputs are copied through device buffers the model owns). TensorRT and the CUDA toolkit come from your own installation (TENSORRT_ROOT).
 * Exceptions never cross the backend ABI.
 */
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <uairt/uairt_backend.h>

#define OPTION_DEVICE "device"

namespace {

const uairt_host_api* g_host;
thread_local std::string t_trt_message;

void fail(const char* fmt, ...) {
  char message[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      t_trt_message = message;
    }
  }
};

Logger g_logger;

bool cuda_ok(cudaError_t error, const char* what) {
  if (error == cudaSuccess) {
    return true;
  }
  fail("%s failed: %s", what, cudaGetErrorString(error));
  return false;
}

bool map_dtype(nvinfer1::DataType type, uairt_dtype* dtype, size_t* size) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: *dtype = UAIRT_DTYPE_FLOAT32; *size = 4; return true;
    case nvinfer1::DataType::kHALF: *dtype = UAIRT_DTYPE_FLOAT16; *size = 2; return true;
    case nvinfer1::DataType::kBF16: *dtype = UAIRT_DTYPE_BFLOAT16; *size = 2; return true;
    case nvinfer1::DataType::kINT8: *dtype = UAIRT_DTYPE_INT8; *size = 1; return true;
    case nvinfer1::DataType::kINT32: *dtype = UAIRT_DTYPE_INT32; *size = 4; return true;
    case nvinfer1::DataType::kINT64: *dtype = UAIRT_DTYPE_INT64; *size = 8; return true;
    case nvinfer1::DataType::kBOOL: *dtype = UAIRT_DTYPE_BOOL; *size = 1; return true;
    case nvinfer1::DataType::kUINT8: *dtype = UAIRT_DTYPE_UINT8; *size = 1; return true;
    default: return false;
  }
}

struct Engine {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  cudaStream_t stream = nullptr;
  int device = 0;
};

struct Io {
  std::string name;
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  std::vector<int64_t> dims;
  size_t nbytes = 0;
  void* device = nullptr;
};

struct Model {
  Engine* owner = nullptr;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<Io> inputs;
  std::vector<Io> outputs;

  ~Model() {
    cudaSetDevice(owner ? owner->device : 0);
    for (Io& io : inputs) cudaFree(io.device);
    for (Io& io : outputs) cudaFree(io.device);
  }
};

uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  try {
    int device = 0;
    for (size_t i = 0; i < num_options; ++i) {
      if (std::strcmp(options[i].key, OPTION_DEVICE) == 0) {
        char* end = nullptr;
        long value = options[i].value ? std::strtol(options[i].value, &end, 10) : -1;
        if (!options[i].value || *end != '\0' || value < 0) {
          fail("%s must be a CUDA device index", OPTION_DEVICE);
          return UAIRT_ERR_INVALID_ARGUMENT;
        }
        device = static_cast<int>(value);
      } else {
        fail("unknown option '%s'", options[i].key);
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    }
    auto engine = std::make_unique<Engine>();
    engine->device = device;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice") ||
        !cuda_ok(cudaStreamCreateWithFlags(&engine->stream, cudaStreamNonBlocking), "cudaStreamCreate")) {
      return UAIRT_ERR_BACKEND_UNAVAILABLE;
    }
    engine->runtime.reset(nvinfer1::createInferRuntime(g_logger));
    if (!engine->runtime) {
      cudaStreamDestroy(engine->stream);
      fail("cannot create the TensorRT runtime: %s", t_trt_message.c_str());
      return UAIRT_ERR_BACKEND_UNAVAILABLE;
    }
    *out_engine = engine.release();
    return UAIRT_OK;
  } catch (const std::exception& error) {
    fail("%s", error.what());
    return UAIRT_ERR_RUNTIME;
  }
}

void destroy_engine(void* handle) {
  Engine* engine = static_cast<Engine*>(handle);
  cudaSetDevice(engine->device);
  engine->runtime.reset();
  if (engine->stream) {
    cudaStreamDestroy(engine->stream);
  }
  delete engine;
}

bool read_file(const char* path, std::vector<char>* out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return false;
  }
  out->resize(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  return static_cast<bool>(file.read(out->data(), static_cast<std::streamsize>(out->size())));
}

// Ultralytics writes a 4-byte little-endian length and a JSON metadata block before the plan; skip it.
void skip_ultralytics_header(const void** data, size_t* size) {
  const unsigned char* bytes = static_cast<const unsigned char*>(*data);
  if (*size < 8) {
    return;
  }
  const size_t length = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (static_cast<size_t>(bytes[3]) << 24);
  if (length > 0 && length < *size - 4 && bytes[4] == '{' && bytes[3 + length] == '}') {
    *data = bytes + 4 + length;
    *size -= 4 + length;
  }
}

uairt_status describe_tensors(Model* model) {
  nvinfer1::ICudaEngine* engine = model->engine.get();
  for (int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    const nvinfer1::Dims shape = engine->getTensorShape(name);
    Io io;
    io.name = name;
    size_t element_size = 0;
    if (!map_dtype(engine->getTensorDataType(name), &io.dtype, &element_size)) {
      fail("tensor '%s' has a data type the backend does not support", name);
      return UAIRT_ERR_UNSUPPORTED;
    }
    if (shape.nbDims > UAIRT_MAX_RANK) {
      fail("tensor '%s' has rank %d, above %d", name, static_cast<int>(shape.nbDims), UAIRT_MAX_RANK);
      return UAIRT_ERR_UNSUPPORTED;
    }
    int64_t count = 1;
    for (int32_t d = 0; d < shape.nbDims; ++d) {
      if (shape.d[d] < 0) {
        fail("tensor '%s' has a dynamic dimension; export the engine with static shapes", name);
        return UAIRT_ERR_UNSUPPORTED;
      }
      io.dims.push_back(shape.d[d]);
      count *= shape.d[d];
    }
    io.nbytes = static_cast<size_t>(count) * element_size;
    if (!cuda_ok(cudaMalloc(&io.device, io.nbytes ? io.nbytes : 1), "cudaMalloc")) {
      return UAIRT_ERR_OUT_OF_MEMORY;
    }
    const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    (is_input ? model->inputs : model->outputs).push_back(std::move(io));
    Io& stored = (is_input ? model->inputs : model->outputs).back();
    if (!model->context->setTensorAddress(name, stored.device)) {
      fail("setTensorAddress failed for '%s'", name);
      return UAIRT_ERR_RUNTIME;
    }
  }
  return UAIRT_OK;
}

uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  Engine* engine = static_cast<Engine*>(engine_handle);
  if (source->format && std::strcmp(source->format, "tensorrt-engine") != 0) {
    fail("unknown model format '%s'; use a serialized engine (format \"tensorrt-engine\")", source->format);
    return UAIRT_ERR_UNSUPPORTED;
  }
  try {
    std::vector<char> file;
    const void* data = source->data;
    size_t size = source->size;
    if (source->path) {
      if (!read_file(source->path, &file)) {
        fail("cannot read model file '%s'", source->path);
        return UAIRT_ERR_IO;
      }
      data = file.data();
      size = file.size();
    }
    skip_ultralytics_header(&data, &size);
    if (!cuda_ok(cudaSetDevice(engine->device), "cudaSetDevice")) {
      return UAIRT_ERR_BACKEND_UNAVAILABLE;
    }
    auto model = std::make_unique<Model>();
    model->owner = engine;
    t_trt_message.clear();
    model->engine.reset(engine->runtime->deserializeCudaEngine(data, size));
    if (!model->engine) {
      fail("cannot deserialize the engine (it must be built for this GPU and TensorRT version): %s",
           t_trt_message.c_str());
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }
    model->context.reset(model->engine->createExecutionContext());
    if (!model->context) {
      fail("cannot create an execution context: %s", t_trt_message.c_str());
      return UAIRT_ERR_RUNTIME;
    }
    uairt_status status = describe_tensors(model.get());
    if (status != UAIRT_OK) {
      return status;
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
  out->rank = static_cast<uint32_t>(io.dims.size());
  for (size_t i = 0; i < io.dims.size(); ++i) {
    out->dims[i] = io.dims[i];
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

uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  Model* model = static_cast<Model*>(handle);
  cudaStream_t stream = model->owner->stream;
  if (!cuda_ok(cudaSetDevice(model->owner->device), "cudaSetDevice")) {
    return UAIRT_ERR_RUNTIME;
  }
  // Buffers from uairt_buffer_alloc(..., UAIRT_MEM_PINNED) are page-locked, so the driver DMAs straight from and to
  // them; plain host memory goes through the driver's slower pageable path. Both are copied in place, no staging.
  for (size_t i = 0; i < n_in; ++i) {
    if (!cuda_ok(cudaMemcpyAsync(model->inputs[i].device, inputs[i].data, model->inputs[i].nbytes,
                                 cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync (input)")) {
      cudaStreamSynchronize(stream);
      return UAIRT_ERR_RUNTIME;
    }
  }
  if (!model->context->enqueueV3(stream)) {
    cudaStreamSynchronize(stream);
    fail("TensorRT enqueue failed: %s", t_trt_message.c_str());
    return UAIRT_ERR_RUNTIME;
  }
  for (size_t i = 0; i < n_out; ++i) {
    if (!cuda_ok(cudaMemcpyAsync(outputs[i].data, model->outputs[i].device, model->outputs[i].nbytes,
                                 cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync (output)")) {
      cudaStreamSynchronize(stream);
      return UAIRT_ERR_RUNTIME;
    }
  }
  return cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize") ? UAIRT_OK : UAIRT_ERR_RUNTIME;
}

uairt_status alloc_buffer(
    void* engine_handle,
    size_t nbytes,
    uairt_memory_domain domain,
    void** out_handle,
    void** out_data,
    int32_t* out_fd) {
  Engine* engine = static_cast<Engine*>(engine_handle);
  if (domain != UAIRT_MEM_PINNED) {
    fail("the tensorrt backend allocates only pinned buffers (UAIRT_MEM_PINNED)");
    return UAIRT_ERR_UNSUPPORTED;
  }
  void* memory = nullptr;
  if (!cuda_ok(cudaSetDevice(engine->device), "cudaSetDevice") ||
      !cuda_ok(cudaHostAlloc(&memory, nbytes, cudaHostAllocDefault), "cudaHostAlloc")) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  *out_handle = memory;
  *out_data = memory;
  *out_fd = -1;
  return UAIRT_OK;
}

void free_buffer(void* engine_handle, void* handle) {
  cudaSetDevice(static_cast<Engine*>(engine_handle)->device);
  cudaFreeHost(handle);
}

const uairt_backend_api* api() {
  static const uairt_backend_api kApi = [] {
    uairt_backend_api table;
    std::memset(&table, 0, sizeof(table));
    table.struct_size = sizeof(table);
    table.abi_version = UAIRT_BACKEND_ABI_VERSION;
    table.name = "tensorrt";
    table.supported_domains = UAIRT_MEM_HOST | UAIRT_MEM_PINNED;
    table.create_engine = create_engine;
    table.destroy_engine = destroy_engine;
    table.load_model = load_model;
    table.destroy_model = destroy_model;
    table.num_inputs = num_inputs;
    table.num_outputs = num_outputs;
    table.input_info = input_info;
    table.output_info = output_info;
    table.run = run;
    table.alloc_buffer = alloc_buffer;
    table.free_buffer = free_buffer;
    return table;
  }();
  return &kApi;
}

}  // namespace

extern "C" UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return api();
}
