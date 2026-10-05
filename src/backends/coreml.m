/* SPDX-License-Identifier: Apache-2.0 */
/*
 * CoreML backend (Apple platforms), built as a loadable plugin.
 * v1 limits: multi-array inputs and outputs and fixed-size image inputs (uint8
 * [1, height, width, channels]), static shapes, host memory.
 * Models load from a path (.mlmodelc, or .mlpackage/.mlmodel which are compiled
 * on load). Inputs and outputs are ordered by name.
 */
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <uairt/uairt_backend.h>

#define OPTION_COMPUTE_UNITS "compute_units"

typedef struct {
  MLComputeUnits compute_units;
} engine_t;

typedef struct {
  char* name;
  uairt_tensor desc;
  size_t nbytes;
  size_t element_size;
  MLMultiArrayDataType ml_type;
  OSType pixel_format; /* nonzero: a CoreML image input, exposed as uint8 [1, height, width, channels] */
  size_t width;
  size_t height;
  size_t channels;
  void* shape;
  void* strides;
} io_t;

typedef struct {
  void* ml_model;
  char* compiled_path;
  io_t* inputs;
  size_t num_inputs;
  io_t* outputs;
  size_t num_outputs;
} model_t;

static const uairt_host_api* g_host;

static void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

static void fail(const char* fmt, ...) {
  char message[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(message, sizeof(message), fmt, args);
  va_end(args);
  if (g_host && g_host->set_error) {
    g_host->set_error(message);
  }
}

static void fail_error(const char* what, NSError* error) {
  fail("%s: %s", what, error ? error.localizedDescription.UTF8String : "unknown error");
}

static bool map_data_type(MLMultiArrayDataType type, uairt_dtype* dtype, size_t* size) {
  switch (type) {
    case MLMultiArrayDataTypeFloat32:
      *dtype = UAIRT_DTYPE_FLOAT32;
      *size = 4;
      return true;
    case MLMultiArrayDataTypeFloat16:
      *dtype = UAIRT_DTYPE_FLOAT16;
      *size = 2;
      return true;
    case MLMultiArrayDataTypeInt32:
      *dtype = UAIRT_DTYPE_INT32;
      *size = 4;
      return true;
    default:
      return false;
  }
}

static uairt_status create_engine(
    const uairt_option* options,
    size_t num_options,
    void** out_engine) {
  engine_t* engine = calloc(1, sizeof(*engine));
  if (!engine) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  engine->compute_units = MLComputeUnitsAll;
  for (size_t i = 0; i < num_options; ++i) {
    if (strcmp(options[i].key, OPTION_COMPUTE_UNITS) != 0) {
      fail("unknown option '%s'", options[i].key);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
    const char* value = options[i].value ? options[i].value : "";
    if (strcmp(value, "all") == 0) {
      engine->compute_units = MLComputeUnitsAll;
    } else if (strcmp(value, "cpu_only") == 0) {
      engine->compute_units = MLComputeUnitsCPUOnly;
    } else if (strcmp(value, "cpu_and_gpu") == 0) {
      engine->compute_units = MLComputeUnitsCPUAndGPU;
    } else if (strcmp(value, "cpu_and_ne") == 0) {
      engine->compute_units = MLComputeUnitsCPUAndNeuralEngine;
    } else {
      fail("compute_units must be all, cpu_only, cpu_and_gpu or cpu_and_ne, got '%s'", value);
      free(engine);
      return UAIRT_ERR_INVALID_ARGUMENT;
    }
  }
  *out_engine = engine;
  return UAIRT_OK;
}

static void destroy_engine(void* handle) {
  free(handle);
}

static void free_io(io_t* io, size_t count) {
  for (size_t i = 0; io && i < count; ++i) {
    free(io[i].name);
    if (io[i].shape) {
      CFRelease(io[i].shape);
    }
    if (io[i].strides) {
      CFRelease(io[i].strides);
    }
  }
  free(io);
}

static void destroy_model(void* handle) {
  model_t* model = handle;
  if (model->ml_model) {
    CFRelease(model->ml_model);
  }
  if (model->compiled_path) {
    [[NSFileManager defaultManager]
        removeItemAtPath:[NSString stringWithUTF8String:model->compiled_path]
                   error:nil];
    free(model->compiled_path);
  }
  free_io(model->inputs, model->num_inputs);
  free_io(model->outputs, model->num_outputs);
  free(model);
}

/* CoreML describes a fixed shape as one enumerated shape or as ranges of length one. */
static bool has_static_shape(MLMultiArrayShapeConstraint* constraint) {
  switch (constraint.type) {
    case MLMultiArrayShapeConstraintTypeUnspecified:
      return true;
    case MLMultiArrayShapeConstraintTypeEnumerated:
      return constraint.enumeratedShapes.count == 1;
    case MLMultiArrayShapeConstraintTypeRange:
      for (NSValue* range in constraint.sizeRangeForDimension) {
        if (range.rangeValue.length != 1) {
          return false;
        }
      }
      return true;
    default:
      return false;
  }
}

static bool has_static_image_size(MLImageConstraint* constraint) {
  MLImageSizeConstraint* size = constraint.sizeConstraint;
  switch (size.type) {
    case MLImageSizeConstraintTypeUnspecified:
      return true;
    case MLImageSizeConstraintTypeEnumerated:
      return size.enumeratedImageSizes.count == 1;
    case MLImageSizeConstraintTypeRange:
      return size.pixelsWideRange.length == 1 && size.pixelsHighRange.length == 1;
    default:
      return false;
  }
}

static uairt_status describe_image(MLImageConstraint* constraint, const char* kind, io_t* io) {
  switch (constraint.pixelFormatType) {
    case kCVPixelFormatType_32BGRA:
    case kCVPixelFormatType_32ARGB:
    case kCVPixelFormatType_32RGBA:
      io->channels = 4;
      break;
    case kCVPixelFormatType_OneComponent8:
      io->channels = 1;
      break;
    default:
      fail("%s '%s' has an unsupported image pixel format", kind, io->name);
      return UAIRT_ERR_UNSUPPORTED;
  }
  if (!has_static_image_size(constraint)) {
    fail("%s '%s' has a flexible image size; static sizes only for now", kind, io->name);
    return UAIRT_ERR_UNSUPPORTED;
  }
  io->pixel_format = constraint.pixelFormatType;
  io->width = (size_t)constraint.pixelsWide;
  io->height = (size_t)constraint.pixelsHigh;
  io->element_size = 1;
  io->nbytes = io->width * io->height * io->channels;
  io->desc.struct_size = sizeof(uairt_tensor);
  io->desc.domain = UAIRT_MEM_HOST;
  io->desc.dmabuf_fd = -1;
  io->desc.dtype = UAIRT_DTYPE_UINT8;
  io->desc.rank = 4;
  io->desc.name = io->name;
  io->desc.dims[0] = 1;
  io->desc.dims[1] = (int64_t)io->height;
  io->desc.dims[2] = (int64_t)io->width;
  io->desc.dims[3] = (int64_t)io->channels;
  return UAIRT_OK;
}

/* Copies tightly packed rows into a new pixel buffer, whose rows may be padded. */
static CVPixelBufferRef make_pixel_buffer(const io_t* io, const uint8_t* data) {
  CVPixelBufferRef buffer = NULL;
  if (CVPixelBufferCreate(kCFAllocatorDefault, io->width, io->height, io->pixel_format,
                          (__bridge CFDictionaryRef) @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}},
                          &buffer) != kCVReturnSuccess) {
    return NULL;
  }
  CVPixelBufferLockBaseAddress(buffer, 0);
  uint8_t* base = CVPixelBufferGetBaseAddress(buffer);
  size_t stride = CVPixelBufferGetBytesPerRow(buffer);
  size_t row = io->width * io->channels;
  for (size_t y = 0; y < io->height; ++y) {
    memcpy(base + y * stride, data + y * row, row);
  }
  CVPixelBufferUnlockBaseAddress(buffer, 0);
  return buffer;
}

static uairt_status describe_io(MLFeatureDescription* feature, const char* kind, io_t* io) {
  io->name = strdup(feature.name.UTF8String);
  if (!io->name) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  if (feature.type == MLFeatureTypeImage && strcmp(kind, "input") == 0) {
    return describe_image(feature.imageConstraint, kind, io);
  }
  if (feature.type != MLFeatureTypeMultiArray) {
    fail("%s '%s' is not a multi-array (images and other feature types are not supported yet)",
         kind, io->name);
    return UAIRT_ERR_UNSUPPORTED;
  }
  MLMultiArrayConstraint* constraint = feature.multiArrayConstraint;
  uairt_dtype dtype = UAIRT_DTYPE_UNKNOWN;
  if (!map_data_type(constraint.dataType, &dtype, &io->element_size) ||
      constraint.shape.count > UAIRT_MAX_RANK) {
    fail("%s '%s' has an unsupported data type or rank", kind, io->name);
    return UAIRT_ERR_UNSUPPORTED;
  }
  if (!has_static_shape(constraint.shapeConstraint)) {
    fail("%s '%s' has a flexible shape; static shapes only for now", kind, io->name);
    return UAIRT_ERR_UNSUPPORTED;
  }

  io->ml_type = constraint.dataType;
  io->desc.struct_size = sizeof(uairt_tensor);
  io->desc.domain = UAIRT_MEM_HOST;
  io->desc.dmabuf_fd = -1;
  io->desc.dtype = dtype;
  io->desc.rank = (uint32_t)constraint.shape.count;
  io->desc.name = io->name;
  io->nbytes = io->element_size;
  for (NSUInteger i = 0; i < constraint.shape.count; ++i) {
    io->desc.dims[i] = constraint.shape[i].longLongValue;
    io->nbytes *= (size_t)io->desc.dims[i];
  }

  NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:constraint.shape.count];
  NSInteger stride = 1;
  for (NSInteger i = (NSInteger)constraint.shape.count - 1; i >= 0; --i) {
    [strides insertObject:@(stride) atIndex:0];
    stride *= constraint.shape[(NSUInteger)i].integerValue;
  }
  io->shape = (void*)CFBridgingRetain([constraint.shape copy]);
  io->strides = (void*)CFBridgingRetain([strides copy]);
  return UAIRT_OK;
}

static uairt_status load_io(
    NSDictionary<NSString*, MLFeatureDescription*>* features,
    const char* kind,
    io_t** out_io,
    size_t* out_count) {
  NSArray<NSString*>* names = [features.allKeys sortedArrayUsingSelector:@selector(compare:)];
  io_t* io = calloc(names.count ? names.count : 1, sizeof(*io));
  if (!io) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  *out_io = io;
  *out_count = names.count;
  for (NSUInteger i = 0; i < names.count; ++i) {
    uairt_status status = describe_io(features[names[i]], kind, &io[i]);
    if (status != UAIRT_OK) {
      return status;
    }
  }
  return UAIRT_OK;
}

static bool has_suffix(const char* text, const char* suffix) {
  size_t length = strlen(text);
  size_t suffix_length = strlen(suffix);
  return length >= suffix_length && strcmp(text + length - suffix_length, suffix) == 0;
}

static NSURL* compile_model(NSURL* url, NSError** error) {
  __block NSURL* compiled = nil;
  __block NSError* compile_error = nil;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  [MLModel compileModelAtURL:url
           completionHandler:^(NSURL* result, NSError* failure) {
             compiled = result;
             compile_error = failure;
             dispatch_semaphore_signal(done);
           }];
  dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
  *error = compile_error;
  return compiled;
}

static uairt_status load_model(
    void* engine_handle,
    const uairt_model_source* source,
    void** out_model) {
  engine_t* engine = engine_handle;
  if (!source->path) {
    fail("CoreML loads models from a path, not from memory");
    return UAIRT_ERR_UNSUPPORTED;
  }
  model_t* model = calloc(1, sizeof(*model));
  if (!model) {
    return UAIRT_ERR_OUT_OF_MEMORY;
  }
  uairt_status status = UAIRT_OK;
  @autoreleasepool {
    NSString* path = [NSString stringWithUTF8String:source->path];
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
      fail("cannot open model '%s'", source->path);
      status = UAIRT_ERR_IO;
    } else {
      NSURL* url = [NSURL fileURLWithPath:path];
      NSError* error = nil;
      if (has_suffix(source->path, ".mlpackage") || has_suffix(source->path, ".mlmodel")) {
        NSURL* compiled = compile_model(url, &error);
        if (!compiled) {
          fail_error("model compilation failed", error);
          status = UAIRT_ERR_INCOMPATIBLE_MODEL;
        } else {
          model->compiled_path = strdup(compiled.path.UTF8String);
          url = compiled;
        }
      }
      if (status == UAIRT_OK) {
        MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
        configuration.computeUnits = engine->compute_units;
        MLModel* ml_model = [MLModel modelWithContentsOfURL:url
                                              configuration:configuration
                                                      error:&error];
        if (!ml_model) {
          fail_error("cannot load model", error);
          status = UAIRT_ERR_INCOMPATIBLE_MODEL;
        } else {
          model->ml_model = (void*)CFBridgingRetain(ml_model);
          status = load_io(ml_model.modelDescription.inputDescriptionsByName, "input",
                           &model->inputs, &model->num_inputs);
          if (status == UAIRT_OK) {
            status = load_io(ml_model.modelDescription.outputDescriptionsByName, "output",
                             &model->outputs, &model->num_outputs);
          }
        }
      }
    }
  }
  if (status != UAIRT_OK) {
    destroy_model(model);
    return status;
  }
  *out_model = model;
  return UAIRT_OK;
}

static size_t num_inputs(const void* handle) {
  return ((const model_t*)handle)->num_inputs;
}

static size_t num_outputs(const void* handle) {
  return ((const model_t*)handle)->num_outputs;
}

static uairt_status input_info(const void* handle, size_t index, uairt_tensor* out) {
  *out = ((const model_t*)handle)->inputs[index].desc;
  return UAIRT_OK;
}

static uairt_status output_info(const void* handle, size_t index, uairt_tensor* out) {
  *out = ((const model_t*)handle)->outputs[index].desc;
  return UAIRT_OK;
}

static MLMultiArray* wrap(const io_t* io, void* data, NSError** error) {
  return [[MLMultiArray alloc] initWithDataPointer:data
                                             shape:(__bridge NSArray<NSNumber*>*)io->shape
                                          dataType:io->ml_type
                                           strides:(__bridge NSArray<NSNumber*>*)io->strides
                                       deallocator:nil
                                             error:error];
}

/* Copies a possibly strided CoreML array into a contiguous caller buffer. */
static void copy_strided(
    const uint8_t* source,
    uint8_t* destination,
    NSArray<NSNumber*>* shape,
    NSArray<NSNumber*>* strides,
    NSUInteger dimension,
    size_t element_size) {
  NSInteger count = shape[dimension].integerValue;
  NSInteger stride = strides[dimension].integerValue;
  NSInteger inner = 1;
  for (NSUInteger i = dimension + 1; i < shape.count; ++i) {
    inner *= shape[i].integerValue;
  }
  for (NSInteger i = 0; i < count; ++i) {
    if (dimension + 1 == shape.count) {
      memcpy(destination + (size_t)i * element_size, source + (size_t)(i * stride) * element_size,
             element_size);
    } else {
      copy_strided(source + (size_t)(i * stride) * element_size,
                   destination + (size_t)(i * inner) * element_size, shape, strides,
                   dimension + 1, element_size);
    }
  }
}

static uairt_status run(
    void* handle,
    const uairt_tensor* inputs,
    size_t n_in,
    uairt_tensor* outputs,
    size_t n_out) {
  model_t* model = handle;
  __block uairt_status status = UAIRT_OK;
  @autoreleasepool {
    NSError* error = nil;
    MLModel* ml_model = (__bridge MLModel*)model->ml_model;
    NSMutableDictionary<NSString*, MLFeatureValue*>* features = [NSMutableDictionary dictionary];
    for (size_t i = 0; i < n_in && status == UAIRT_OK; ++i) {
      const io_t* io = &model->inputs[i];
      if (io->pixel_format) {
        CVPixelBufferRef pixels = make_pixel_buffer(io, inputs[i].data);
        if (!pixels) {
          fail("cannot create a pixel buffer for input '%s'", io->name);
          status = UAIRT_ERR_RUNTIME;
        } else {
          features[@(io->name)] = [MLFeatureValue featureValueWithPixelBuffer:pixels];
          CVPixelBufferRelease(pixels);
        }
        continue;
      }
      MLMultiArray* array = wrap(io, inputs[i].data, &error);
      if (!array) {
        fail_error("cannot wrap input", error);
        status = UAIRT_ERR_RUNTIME;
      } else {
        features[@(io->name)] = [MLFeatureValue featureValueWithMultiArray:array];
      }
    }
    NSMutableDictionary<NSString*, id>* backings = [NSMutableDictionary dictionary];
    for (size_t i = 0; i < n_out && status == UAIRT_OK; ++i) {
      MLMultiArray* array = wrap(&model->outputs[i], outputs[i].data, &error);
      if (!array) {
        fail_error("cannot wrap output", error);
        status = UAIRT_ERR_RUNTIME;
      } else {
        backings[@(model->outputs[i].name)] = array;
      }
    }
    id<MLFeatureProvider> result = nil;
    if (status == UAIRT_OK) {
      MLDictionaryFeatureProvider* provider =
          [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
      MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
      options.outputBackings = backings;
      result = provider ? [ml_model predictionFromFeatures:provider options:options error:&error]
                        : nil;
      if (!result) {
        fail_error("prediction failed", error);
        status = UAIRT_ERR_RUNTIME;
      }
    }
    for (size_t i = 0; i < n_out && status == UAIRT_OK; ++i) {
      const io_t* io = &model->outputs[i];
      MLMultiArray* got = [result featureValueForName:@(io->name)].multiArrayValue;
      if (!got) {
        fail("model produced no output '%s'", io->name);
        status = UAIRT_ERR_RUNTIME;
        continue;
      }
      uint8_t* destination = outputs[i].data;
      NSArray<NSNumber*>* shape = (__bridge NSArray<NSNumber*>*)io->shape;
      [got getBytesWithHandler:^(const void* bytes, NSInteger size) {
        (void)size;
        if (bytes == destination) {
          return;
        }
        if (shape.count == 0) {
          memcpy(destination, bytes, io->element_size);
        } else {
          copy_strided(bytes, destination, shape, got.strides, 0, io->element_size);
        }
      }];
    }
  }
  return status;
}

static const uairt_backend_api kApi = {
    .struct_size = sizeof(uairt_backend_api),
    .abi_version = UAIRT_BACKEND_ABI_VERSION,
    .name = "coreml",
    .supported_domains = UAIRT_MEM_HOST,
    .create_engine = create_engine,
    .destroy_engine = destroy_engine,
    .load_model = load_model,
    .destroy_model = destroy_model,
    .num_inputs = num_inputs,
    .num_outputs = num_outputs,
    .input_info = input_info,
    .output_info = output_info,
    .run = run,
};

UAIRT_API const uairt_backend_api* uairt_backend_get_api(const uairt_host_api* host) {
  g_host = host;
  return &kApi;
}
