// SPDX-License-Identifier: Apache-2.0
/*
 * NCNN backend, built as a loadable plugin.
 * v1 limits: a .param file (with the .bin of the same stem) or a directory that holds one, float32 tensors, static
 * shapes, host memory. NCNN has no batch dimension and its .param file records no input shapes, so the `input_shapes`
 * option is required ("3x640x640" for one input, entries separated by ';' in the model's input order). Shapes follow
 * NCNN's layout: 1-d w, 2-d h x w, 3-d c x h x w, 4-d c x d x h x w. Output shapes are found by running the model once
 * on zeros at load time. device=vulkan runs on a GPU through Vulkan (NVIDIA, AMD, Intel, Apple through MoltenVK) and
 * fails if NCNN finds no usable GPU, instead of quietly running on the CPU as NCNN itself would.
 * The plugin must be built with the same C++ toolchain as the NCNN library it links.
 */
#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#include <uairt/uairt_backend.h>

#define OPTION_DEVICE "device"
#define OPTION_VULKAN_DEVICE "vulkan_device"
#define OPTION_THREADS "num_threads"
#define OPTION_FP16 "fp16"
#define OPTION_INPUT_SHAPES "input_shapes"

namespace {

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

#if NCNN_VULKAN
// ncnn::create_gpu_instance and destroy_gpu_instance are process-wide, so count the engines that use Vulkan.
std::mutex g_gpu_mutex;
int g_gpu_users = 0;
#endif

struct Engine {
  bool vulkan = false;
  int vulkan_device = 0;
  int threads = 0;
  int fp16 = -1;  // -1 keeps NCNN's default
  bool holds_gpu_instance = false;
  std::vector<std::vector<int>> shapes;

  ~Engine() {
#if NCNN_VULKAN
    if (holds_gpu_instance) {
      std::lock_guard<std::mutex> lock(g_gpu_mutex);
      if (--g_gpu_users == 0) {
        ncnn::destroy_gpu_instance();
      }
    }
#endif
  }
};

struct Io {
  std::string name;
  std::vector<int> shape;  // NCNN layout, see the file comment
  size_t elements = 0;
};

struct Model {
  ncnn::Net net;
  std::vector<Io> inputs;
  std::vector<Io> outputs;
};

size_t count_elements(const std::vector<int>& shape) {
  size_t count = 1;
  for (int dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

ncnn::Mat make_mat(const std::vector<int>& shape) {
  switch (shape.size()) {
    case 1: return ncnn::Mat(shape[0]);
    case 2: return ncnn::Mat(shape[1], shape[0]);
    case 3: return ncnn::Mat(shape[2], shape[1], shape[0]);
    default: return ncnn::Mat(shape[3], shape[2], shape[1], shape[0]);
  }
}

// Number of floats in one channel (3-d and 4-d) or in the whole mat (1-d and 2-d).
size_t plane_elements(const std::vector<int>& shape) {
  return shape.size() <= 2 ? count_elements(shape) : count_elements(shape) / static_cast<size_t>(shape[0]);
}

// Channels of a 3-d or 4-d mat are padded apart (cstep), so they are copied one by one.
const float* plane_pointer(const ncnn::Mat& mat, int q) {
  return mat.dims <= 2 ? static_cast<const float*>(mat.data) : static_cast<const float*>(mat.channel(q).data);
}

void fill_mat(ncnn::Mat& mat, const std::vector<int>& shape, const float* source) {
  const size_t plane = plane_elements(shape);
  const int planes = shape.size() <= 2 ? 1 : shape[0];
  for (int q = 0; q < planes; ++q) {
    std::memcpy(const_cast<float*>(plane_pointer(mat, q)), source + static_cast<size_t>(q) * plane,
                plane * sizeof(float));
  }
}

void read_mat(const ncnn::Mat& mat, const std::vector<int>& shape, float* destination) {
  const size_t plane = plane_elements(shape);
  const int planes = shape.size() <= 2 ? 1 : shape[0];
  for (int q = 0; q < planes; ++q) {
    std::memcpy(destination + static_cast<size_t>(q) * plane, plane_pointer(mat, q), plane * sizeof(float));
  }
}

std::vector<int> shape_of(const ncnn::Mat& mat) {
  switch (mat.dims) {
    case 1: return {mat.w};
    case 2: return {mat.h, mat.w};
    case 3: return {mat.c, mat.h, mat.w};
    default: return {mat.c, mat.d, mat.h, mat.w};
  }
}

bool parse_shapes(const std::string& text, std::vector<std::vector<int>>* out) {
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find(';', begin);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string entry = text.substr(begin, end - begin);
    begin = end + 1;
    if (entry.empty()) {
      continue;
    }
    std::vector<int> shape;
    size_t start = 0;
    while (start <= entry.size()) {
      size_t stop = entry.find('x', start);
      if (stop == std::string::npos) {
        stop = entry.size();
      }
      const std::string number = entry.substr(start, stop - start);
      char* last = nullptr;
      long value = std::strtol(number.c_str(), &last, 10);
      if (number.empty() || *last != '\0' || value <= 0) {
        fail("'%s' has an invalid dimension '%s'", OPTION_INPUT_SHAPES, number.c_str());
        return false;
      }
      shape.push_back(static_cast<int>(value));
      start = stop + 1;
    }
    if (shape.size() > 4) {
      fail("'%s' rank is above 4 (NCNN has no batch dimension)", OPTION_INPUT_SHAPES);
      return false;
    }
    out->push_back(shape);
  }
  if (out->empty()) {
    fail("option '%s' lists no shapes", OPTION_INPUT_SHAPES);
    return false;
  }
  return true;
}

uairt_status create_engine(const uairt_option* options, size_t num_options, void** out_engine) {
  try {
    auto engine = std::make_unique<Engine>();
    std::string shapes;
    for (size_t i = 0; i < num_options; ++i) {
      const std::string key = options[i].key;
      const std::string value = options[i].value ? options[i].value : "";
      if (key == OPTION_DEVICE) {
        if (value != "cpu" && value != "vulkan") {
          fail("%s must be cpu or vulkan", OPTION_DEVICE);
          return UAIRT_ERR_INVALID_ARGUMENT;
        }
        engine->vulkan = value == "vulkan";
      } else if (key == OPTION_VULKAN_DEVICE || key == OPTION_THREADS) {
        char* last = nullptr;
        long number = std::strtol(value.c_str(), &last, 10);
        if (value.empty() || *last != '\0' || number < (key == OPTION_THREADS ? 1 : 0)) {
          fail("%s must be an integer in range", key.c_str());
          return UAIRT_ERR_INVALID_ARGUMENT;
        }
        (key == OPTION_THREADS ? engine->threads : engine->vulkan_device) = static_cast<int>(number);
      } else if (key == OPTION_FP16) {
        if (value != "true" && value != "false") {
          fail("%s must be true or false", OPTION_FP16);
          return UAIRT_ERR_INVALID_ARGUMENT;
        }
        engine->fp16 = value == "true";
      } else if (key == OPTION_INPUT_SHAPES) {
        shapes = value;
      } else {
        fail("unknown option '%s'", key.c_str());
        return UAIRT_ERR_INVALID_ARGUMENT;
      }
    }
    if (shapes.empty()) {
      fail("option '%s' is required, for example %s=3x640x640 (NCNN's .param file records no input shapes)",
           OPTION_INPUT_SHAPES, OPTION_INPUT_SHAPES);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    if (!parse_shapes(shapes, &engine->shapes)) {
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    if (engine->vulkan) {
#if NCNN_VULKAN
      std::lock_guard<std::mutex> lock(g_gpu_mutex);
      if (g_gpu_users == 0 && ncnn::create_gpu_instance() != 0) {
        fail("NCNN could not create a Vulkan instance (no Vulkan driver found)");
        return UAIRT_ERR_BACKEND_UNAVAILABLE;
      }
      ++g_gpu_users;
      engine->holds_gpu_instance = true;
      const int count = ncnn::get_gpu_count();
      if (engine->vulkan_device >= count) {
        fail("Vulkan device %d is not available (NCNN found %d Vulkan device(s))", engine->vulkan_device, count);
        return UAIRT_ERR_BACKEND_UNAVAILABLE;
      }
#else
      fail("this NCNN library was built without Vulkan support (NCNN_VULKAN=OFF)");
      return UAIRT_ERR_BACKEND_UNAVAILABLE;
#endif
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

bool is_directory(const std::string& path) {
#if defined(_WIN32)
  DWORD attributes = GetFileAttributesA(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  struct stat info;
  return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

std::string find_param_in_directory(const std::string& directory) {
#if defined(_WIN32)
  WIN32_FIND_DATAA entry;
  HANDLE handle = FindFirstFileA((directory + "\\*.param").c_str(), &entry);
  if (handle == INVALID_HANDLE_VALUE) {
    return "";
  }
  std::string found = directory + "\\" + entry.cFileName;
  FindClose(handle);
  return found;
#else
  DIR* dir = opendir(directory.c_str());
  if (!dir) {
    return "";
  }
  std::string found;
  while (struct dirent* entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name.size() > 6 && name.compare(name.size() - 6, 6, ".param") == 0) {
      found = directory + "/" + name;
      break;
    }
  }
  closedir(dir);
  return found;
#endif
}

bool file_exists(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (file) {
    std::fclose(file);
  }
  return file != nullptr;
}

uairt_status load_model(void* engine_handle, const uairt_model_source* source, void** out_model) {
  const Engine* engine = static_cast<Engine*>(engine_handle);
  if (!source->path) {
    fail("the ncnn backend loads models from a path (a .param file or a directory that holds one)");
    return UAIRT_ERR_UNSUPPORTED;
  }
  try {
    std::string param = source->path;
    if (is_directory(param)) {
      const std::string directory = param;
      param = find_param_in_directory(directory);
      if (param.empty()) {
        fail("no .param model found in directory '%s'", directory.c_str());
        return UAIRT_ERR_IO;
      }
    }
    std::string weights = param;
    const std::string suffix = ".param";
    if (weights.size() > suffix.size() && weights.compare(weights.size() - suffix.size(), suffix.size(), suffix) == 0) {
      weights.replace(weights.size() - suffix.size(), suffix.size(), ".bin");
    } else {
      weights += ".bin";
    }
    if (!file_exists(param) || !file_exists(weights)) {
      fail("cannot open the NCNN model files '%s' and '%s'", param.c_str(), weights.c_str());
      return UAIRT_ERR_IO;
    }

    auto model = std::make_unique<Model>();
    model->net.opt.num_threads = engine->threads > 0 ? engine->threads : model->net.opt.num_threads;
#if NCNN_VULKAN
    model->net.opt.use_vulkan_compute = engine->vulkan;
    if (engine->vulkan) {
      model->net.set_vulkan_device(engine->vulkan_device);
    }
#endif
    if (engine->fp16 >= 0) {
      model->net.opt.use_fp16_packed = model->net.opt.use_fp16_storage = model->net.opt.use_fp16_arithmetic =
          engine->fp16 != 0;
    }
    if (model->net.load_param(param.c_str()) != 0 || model->net.load_model(weights.c_str()) != 0) {
      fail("NCNN cannot read '%s' (is it an NCNN .param/.bin pair?)", param.c_str());
      return UAIRT_ERR_INCOMPATIBLE_MODEL;
    }

    const auto& input_names = model->net.input_names();
    const auto& output_names = model->net.output_names();
    if (input_names.size() != engine->shapes.size()) {
      fail("the model has %zu input(s) but '%s' lists %zu shape(s)", input_names.size(), OPTION_INPUT_SHAPES,
           engine->shapes.size());
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < input_names.size(); ++i) {
      Io io;
      io.name = input_names[i];
      io.shape = engine->shapes[i];
      io.elements = count_elements(io.shape);
      model->inputs.push_back(std::move(io));
    }

    // NCNN does not record output shapes: run once on zeros and read them off.
    ncnn::Extractor extractor = model->net.create_extractor();
    for (const Io& io : model->inputs) {
      ncnn::Mat zeros = make_mat(io.shape);
      zeros.fill(0.f);
      extractor.input(io.name.c_str(), zeros);
    }
    for (const char* name : output_names) {
      ncnn::Mat result;
      if (extractor.extract(name, result) != 0 || result.empty()) {
        fail("the model failed on zero inputs of the shapes given in '%s' (output '%s')", OPTION_INPUT_SHAPES, name);
        return UAIRT_ERR_INCOMPATIBLE_MODEL;
      }
      if (result.elemsize != sizeof(float) || result.elempack != 1) {
        fail("output '%s' is not a plain float32 tensor", name);
        return UAIRT_ERR_UNSUPPORTED;
      }
      Io io;
      io.name = name;
      io.shape = shape_of(result);
      io.elements = count_elements(io.shape);
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

uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  Model* model = static_cast<Model*>(handle);
  try {
    ncnn::Extractor extractor = model->net.create_extractor();
    for (size_t i = 0; i < n_in; ++i) {
      ncnn::Mat mat = make_mat(model->inputs[i].shape);
      fill_mat(mat, model->inputs[i].shape, static_cast<const float*>(inputs[i].data));
      if (extractor.input(model->inputs[i].name.c_str(), mat) != 0) {
        fail("NCNN rejected input '%s'", model->inputs[i].name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
    }
    for (size_t i = 0; i < n_out; ++i) {
      ncnn::Mat result;
      if (extractor.extract(model->outputs[i].name.c_str(), result) != 0) {
        fail("NCNN failed to compute output '%s'", model->outputs[i].name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
      if (shape_of(result) != model->outputs[i].shape || result.elemsize != sizeof(float) || result.elempack != 1) {
        fail("output '%s' changed shape since load; the ncnn backend needs static shapes",
             model->outputs[i].name.c_str());
        return UAIRT_ERR_RUNTIME;
      }
      read_mat(result, model->outputs[i].shape, static_cast<float*>(outputs[i].data));
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
    table.name = "ncnn";
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

extern "C" UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return api();
}
