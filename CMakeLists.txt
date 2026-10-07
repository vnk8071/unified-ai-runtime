# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)

file(READ ${CMAKE_CURRENT_LIST_DIR}/include/uairt/uairt.h UAIRT_HEADER)
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "UAIRT_VERSION_${part} ([0-9]+)" _ "${UAIRT_HEADER}")
  set(UAIRT_VERSION_${part} ${CMAKE_MATCH_1})
endforeach()

project(unified_ai_runtime
  VERSION ${UAIRT_VERSION_MAJOR}.${UAIRT_VERSION_MINOR}.${UAIRT_VERSION_PATCH}
  LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET hidden)

option(UAIRT_BUILD_TESTS "Build tests" ON)
option(UAIRT_BUILD_REFERENCE_BACKEND "Build the test-only reference backend" ON)

find_package(Threads REQUIRED)

add_library(uairt
  src/uairt.c
  src/loader.c
)
target_include_directories(uairt
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
)
target_link_libraries(uairt PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(BUILD_SHARED_LIBS)
  target_compile_definitions(uairt PRIVATE UAIRT_BUILDING_SHARED PUBLIC UAIRT_SHARED)
endif()

if(UAIRT_BUILD_REFERENCE_BACKEND)
  target_sources(uairt PRIVATE src/backends/reference.c)
  target_compile_definitions(uairt PRIVATE UAIRT_WITH_REFERENCE_BACKEND)
endif()

install(TARGETS uairt EXPORT uairtTargets ARCHIVE LIBRARY RUNTIME)
install(DIRECTORY include/uairt DESTINATION include)

option(UAIRT_BUILD_EXAMPLES "Build examples" ON)
if(UAIRT_BUILD_EXAMPLES)
  add_executable(run_model examples/run_model.c)
  target_link_libraries(run_model PRIVATE uairt)
  if(UAIRT_BUILD_REFERENCE_BACKEND)
    add_executable(quickstart examples/quickstart.c)
    target_link_libraries(quickstart PRIVATE uairt)
  endif()
endif()

option(UAIRT_BUILD_ONNXRUNTIME "Build the ONNX Runtime backend plugin" OFF)
if(UAIRT_BUILD_ONNXRUNTIME)
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
      "Directory of an ONNX Runtime release (contains include/ and lib/)")
  if(NOT ONNXRUNTIME_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.ort/include/onnxruntime_c_api.h")
    # the headers fetched by scripts/ci/fetch_onnxruntime_headers.sh for the wheel build
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.ort")
  endif()
  find_path(UAIRT_ORT_INCLUDE onnxruntime_c_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
    NO_DEFAULT_PATH)
  find_library(UAIRT_ORT_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
  if(NOT UAIRT_ORT_INCLUDE)
    message(FATAL_ERROR
      "ONNX Runtime headers not found. Set ONNXRUNTIME_ROOT to a release directory "
      "containing include/onnxruntime_c_api.h. The plugin loads the library at run time, so lib/ is optional.")
  endif()
  # The plugin loads ONNX Runtime when it loads. For a source build, remember where the library was so it is found without
  # setting UAIRT_ONNXRUNTIME_LIBRARY; a wheel build turns this off, since the build machine's path means nothing elsewhere.
  option(UAIRT_ORT_BAKE_LIBRARY_DIR "Remember the build-time ONNX Runtime directory as a search location" ON)

  add_library(uairt_backend_onnxruntime MODULE src/backends/onnxruntime.c)
  target_include_directories(uairt_backend_onnxruntime
    PRIVATE include ${UAIRT_ORT_INCLUDE})
  target_compile_definitions(uairt_backend_onnxruntime PRIVATE UAIRT_BUILDING_SHARED)
  if(UAIRT_ORT_LIBRARY AND UAIRT_ORT_BAKE_LIBRARY_DIR)
    get_filename_component(UAIRT_ORT_LIBRARY_DIR ${UAIRT_ORT_LIBRARY} DIRECTORY)
    target_compile_definitions(uairt_backend_onnxruntime PRIVATE UAIRT_ORT_LIBRARY_DIR="${UAIRT_ORT_LIBRARY_DIR}")
  endif()
  set_target_properties(uairt_backend_onnxruntime PROPERTIES
    PREFIX "lib"
    C_VISIBILITY_PRESET hidden)
  install(TARGETS uairt_backend_onnxruntime LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_COREML "Build the CoreML backend plugin (Apple platforms)" OFF)
if(UAIRT_BUILD_COREML)
  if(NOT APPLE)
    message(FATAL_ERROR "The CoreML backend needs an Apple platform.")
  endif()
  enable_language(OBJC)
  add_library(uairt_backend_coreml MODULE src/backends/coreml.m)
  target_include_directories(uairt_backend_coreml PRIVATE include)
  target_compile_options(uairt_backend_coreml PRIVATE -fobjc-arc)
  target_link_libraries(uairt_backend_coreml PRIVATE "-framework CoreML" "-framework CoreVideo" "-framework Foundation")
  set_target_properties(uairt_backend_coreml PROPERTIES PREFIX "lib" OBJC_VISIBILITY_PRESET hidden)
  install(TARGETS uairt_backend_coreml LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_QNN "Build the QNN (Qualcomm AI Runtime) backend plugin" OFF)
if(UAIRT_BUILD_QNN)
  set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Directory of your QAIRT SDK")
  find_path(UAIRT_QNN_INCLUDE QnnInterface.h
    HINTS ${QNN_SDK_ROOT}/include/QNN NO_DEFAULT_PATH)
  if(NOT UAIRT_QNN_INCLUDE)
    message(FATAL_ERROR
      "QNN headers not found. Set QNN_SDK_ROOT to your QAIRT SDK directory "
      "(it must contain include/QNN/QnnInterface.h).")
  endif()

  add_library(uairt_backend_qnn MODULE src/backends/qnn.c)
  target_include_directories(uairt_backend_qnn PRIVATE include)
  target_include_directories(uairt_backend_qnn SYSTEM PRIVATE ${UAIRT_QNN_INCLUDE})
  target_link_libraries(uairt_backend_qnn PRIVATE ${CMAKE_DL_LIBS})
  target_compile_definitions(uairt_backend_qnn PRIVATE UAIRT_BUILDING_SHARED)
  set_target_properties(uairt_backend_qnn PROPERTIES
    PREFIX "lib" C_VISIBILITY_PRESET hidden)
  install(TARGETS uairt_backend_qnn LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_TFLITE "Build the TFLite backend plugin" OFF)
if(UAIRT_BUILD_TFLITE)
  set(TFLITE_INCLUDE_DIR "$ENV{TFLITE_INCLUDE_DIR}" CACHE PATH
      "Directory containing tensorflow/lite/c/c_api.h")
  set(TFLITE_LIBRARY "$ENV{TFLITE_LIBRARY}" CACHE FILEPATH "Path to libtensorflowlite_c")
  if(NOT EXISTS "${TFLITE_INCLUDE_DIR}/tensorflow/lite/c/c_api.h" OR NOT EXISTS "${TFLITE_LIBRARY}")
    message(FATAL_ERROR
      "TFLite not found. Set TFLITE_INCLUDE_DIR (contains tensorflow/lite/c/c_api.h) and "
      "TFLITE_LIBRARY (libtensorflowlite_c) to your TFLite build.")
  endif()
  get_filename_component(UAIRT_TFLITE_LIBRARY_DIR ${TFLITE_LIBRARY} DIRECTORY)

  add_library(uairt_backend_tflite MODULE src/backends/tflite.c)
  target_include_directories(uairt_backend_tflite PRIVATE include)
  target_include_directories(uairt_backend_tflite SYSTEM PRIVATE ${TFLITE_INCLUDE_DIR})
  target_link_libraries(uairt_backend_tflite PRIVATE ${TFLITE_LIBRARY} ${CMAKE_DL_LIBS})
  target_compile_definitions(uairt_backend_tflite PRIVATE UAIRT_BUILDING_SHARED)
  set_target_properties(uairt_backend_tflite PROPERTIES
    PREFIX "lib" C_VISIBILITY_PRESET hidden BUILD_RPATH ${UAIRT_TFLITE_LIBRARY_DIR})

  # Optional QNN delegate support: needs QnnTFLiteDelegate.h from your QAIRT SDK.
  set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Directory of your QAIRT SDK")
  find_path(UAIRT_QNN_TFLITE_DELEGATE_INCLUDE QnnTFLiteDelegate.h
    HINTS ${QNN_SDK_ROOT}/include/QNN/TFLiteDelegate NO_DEFAULT_PATH)
  if(UAIRT_QNN_TFLITE_DELEGATE_INCLUDE)
    target_compile_definitions(uairt_backend_tflite PRIVATE UAIRT_TFLITE_QNN_DELEGATE)
    target_include_directories(uairt_backend_tflite SYSTEM PRIVATE ${UAIRT_QNN_TFLITE_DELEGATE_INCLUDE})
    message(STATUS "TFLite backend: QNN delegate support enabled")
  else()
    message(STATUS "TFLite backend: no QnnTFLiteDelegate.h found, QNN delegate disabled")
  endif()
  install(TARGETS uairt_backend_tflite LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_OPENVINO "Build the OpenVINO backend plugin" OFF)
if(UAIRT_BUILD_OPENVINO)
  set(OPENVINO_ROOT "$ENV{OPENVINO_ROOT}" CACHE PATH
      "OpenVINO directory: a pip install's site-packages/openvino, or a release's runtime directory")
  find_path(UAIRT_OV_INCLUDE openvino/c/openvino.h
    HINTS ${OPENVINO_ROOT}/include ${OPENVINO_ROOT}/runtime/include)
  find_library(UAIRT_OV_LIBRARY NAMES openvino_c
    HINTS ${OPENVINO_ROOT}/libs ${OPENVINO_ROOT}/lib ${OPENVINO_ROOT}/runtime/lib/intel64
          ${OPENVINO_ROOT}/runtime/lib/aarch64 ${OPENVINO_ROOT}/runtime/bin/intel64/Release)
  if(NOT UAIRT_OV_LIBRARY)
    # pip installs ship only the versioned library: libopenvino_c.so.2641, or libopenvino_c.2641.dylib on macOS
    file(GLOB UAIRT_OV_VERSIONED "${OPENVINO_ROOT}/libs/libopenvino_c.so.*"
         "${OPENVINO_ROOT}/libs/libopenvino_c.*.dylib"
         "${OPENVINO_ROOT}/runtime/lib/intel64/libopenvino_c.so.*")
    if(UAIRT_OV_VERSIONED)
      list(GET UAIRT_OV_VERSIONED 0 UAIRT_OV_LIBRARY)
    endif()
  endif()
  if(NOT UAIRT_OV_INCLUDE OR NOT UAIRT_OV_LIBRARY)
    message(FATAL_ERROR
      "OpenVINO not found. Set OPENVINO_ROOT to a directory with include/openvino/c/openvino.h and "
      "the openvino_c library (a pip install's site-packages/openvino works).")
  endif()
  get_filename_component(UAIRT_OV_LIBRARY_DIR ${UAIRT_OV_LIBRARY} DIRECTORY)

  add_library(uairt_backend_openvino MODULE src/backends/openvino.c)
  target_include_directories(uairt_backend_openvino PRIVATE include)
  target_include_directories(uairt_backend_openvino SYSTEM PRIVATE ${UAIRT_OV_INCLUDE})
  target_compile_definitions(uairt_backend_openvino PRIVATE UAIRT_BUILDING_SHARED)
  target_link_libraries(uairt_backend_openvino PRIVATE ${UAIRT_OV_LIBRARY})
  set_target_properties(uairt_backend_openvino PROPERTIES
    PREFIX "lib" C_VISIBILITY_PRESET hidden BUILD_RPATH ${UAIRT_OV_LIBRARY_DIR})
  install(TARGETS uairt_backend_openvino LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_NCNN "Build the NCNN backend plugin" OFF)
if(UAIRT_BUILD_NCNN)
  enable_language(CXX)
  set(NCNN_ROOT "$ENV{NCNN_ROOT}" CACHE PATH
      "NCNN install prefix (include/ncnn/net.h and lib/libncnn); build it with the same C++ toolchain as this plugin")
  find_path(UAIRT_NCNN_INCLUDE net.h HINTS ${NCNN_ROOT}/include/ncnn ${NCNN_ROOT}/include)
  find_library(UAIRT_NCNN_LIBRARY NAMES ncnn HINTS ${NCNN_ROOT}/lib ${NCNN_ROOT}/lib64)
  if(NOT UAIRT_NCNN_INCLUDE OR NOT UAIRT_NCNN_LIBRARY)
    message(FATAL_ERROR
      "NCNN not found. Set NCNN_ROOT to an NCNN install prefix containing include/ncnn/net.h and lib/libncnn.")
  endif()
  get_filename_component(UAIRT_NCNN_LIBRARY_DIR ${UAIRT_NCNN_LIBRARY} DIRECTORY)

  add_library(uairt_backend_ncnn MODULE src/backends/ncnn.cpp)
  target_include_directories(uairt_backend_ncnn PRIVATE include)
  target_include_directories(uairt_backend_ncnn SYSTEM PRIVATE ${UAIRT_NCNN_INCLUDE})
  target_compile_features(uairt_backend_ncnn PRIVATE cxx_std_17)
  target_compile_definitions(uairt_backend_ncnn PRIVATE UAIRT_BUILDING_SHARED)
  target_link_libraries(uairt_backend_ncnn PRIVATE ${UAIRT_NCNN_LIBRARY})
  set_target_properties(uairt_backend_ncnn PROPERTIES
    PREFIX "lib" CXX_VISIBILITY_PRESET hidden BUILD_RPATH ${UAIRT_NCNN_LIBRARY_DIR})
  install(TARGETS uairt_backend_ncnn LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_TENSORRT "Build the NVIDIA TensorRT backend plugin" OFF)
if(UAIRT_BUILD_TENSORRT)
  enable_language(CXX)
  set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH
      "Directory of your TensorRT installation (include/NvInfer.h and lib/libnvinfer)")
  find_package(CUDAToolkit REQUIRED)
  find_path(UAIRT_TRT_INCLUDE NvInfer.h
    HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu)
  find_library(UAIRT_TRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
          /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
  if(NOT UAIRT_TRT_INCLUDE OR NOT UAIRT_TRT_LIBRARY)
    message(FATAL_ERROR
      "TensorRT not found. Set TENSORRT_ROOT to your TensorRT directory "
      "(it must contain include/NvInfer.h and lib/libnvinfer).")
  endif()

  add_library(uairt_backend_tensorrt MODULE src/backends/tensorrt.cpp)
  target_include_directories(uairt_backend_tensorrt PRIVATE include)
  target_include_directories(uairt_backend_tensorrt SYSTEM PRIVATE ${UAIRT_TRT_INCLUDE})
  target_compile_features(uairt_backend_tensorrt PRIVATE cxx_std_17)
  target_compile_definitions(uairt_backend_tensorrt PRIVATE UAIRT_BUILDING_SHARED)
  target_link_libraries(uairt_backend_tensorrt PRIVATE ${UAIRT_TRT_LIBRARY} CUDA::cudart)
  set_target_properties(uairt_backend_tensorrt PROPERTIES
    PREFIX "lib" CXX_VISIBILITY_PRESET hidden BUILD_RPATH "${TENSORRT_ROOT}/lib")
  install(TARGETS uairt_backend_tensorrt LIBRARY DESTINATION lib/uairt)
endif()

option(UAIRT_BUILD_LLAMACPP "Build the llama.cpp backend plugin (GGUF language models)" OFF)
if(UAIRT_BUILD_LLAMACPP)
  enable_language(CXX)
  set(UAIRT_LLAMACPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp" CACHE PATH
      "llama.cpp source tree; the default is the pinned submodule")
  if(NOT EXISTS "${UAIRT_LLAMACPP_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR
      "llama.cpp not found at ${UAIRT_LLAMACPP_ROOT}. Run: git submodule update --init third_party/llama.cpp "
      "(or set UAIRT_LLAMACPP_ROOT to a llama.cpp checkout).")
  endif()

  # The comparison test runs llama.cpp's own llama-completion, so tools are built only when that test is enabled.
  if(DEFINED ENV{UAIRT_LLAMACPP_TEST_MODEL})
    set(UAIRT_LLAMACPP_TOOLS ON)
  else()
    set(UAIRT_LLAMACPP_TOOLS OFF)
  endif()
  set(UAIRT_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(LLAMA_BUILD_COMMON ${UAIRT_LLAMACPP_TOOLS} CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS ${UAIRT_LLAMACPP_TOOLS} CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
  set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
  if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
  endif()
  add_subdirectory(${UAIRT_LLAMACPP_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
  set(BUILD_SHARED_LIBS ${UAIRT_SAVED_BUILD_SHARED_LIBS})
  # ggml-hexagon defines its Windows driver stubs with dllexport only when this is set, which a static build never does.
  if(TARGET ggml-hexagon)
    target_compile_definitions(ggml-hexagon PRIVATE GGML_BACKEND_BUILD)
  endif()

  add_library(uairt_backend_llamacpp MODULE src/backends/llamacpp.c)
  target_include_directories(uairt_backend_llamacpp PRIVATE include)
  target_compile_definitions(uairt_backend_llamacpp PRIVATE UAIRT_BUILDING_SHARED)
  target_link_libraries(uairt_backend_llamacpp PRIVATE llama Threads::Threads)
  set_target_properties(uairt_backend_llamacpp PROPERTIES
    PREFIX "lib" C_VISIBILITY_PRESET hidden LINKER_LANGUAGE CXX)
  # Only uairt_backend_get_api may be visible: llama.cpp and ggml must not clash with another ggml in the process.
  if(APPLE)
    target_link_options(uairt_backend_llamacpp PRIVATE -Wl,-exported_symbol,_uairt_backend_get_api)
  elseif(NOT WIN32)
    target_link_options(uairt_backend_llamacpp PRIVATE -Wl,--exclude-libs,ALL)
  endif()
  install(TARGETS uairt_backend_llamacpp LIBRARY DESTINATION lib/uairt)
  if(UAIRT_BUILD_EXAMPLES)
    add_executable(llm_generate examples/llm_generate.c)
    target_link_libraries(llm_generate PRIVATE uairt)
  endif()
endif()

if(UAIRT_BUILD_TESTS)
  enable_testing()

  add_library(uairt_reference_plugin MODULE src/backends/reference.c)
  target_compile_definitions(uairt_reference_plugin PRIVATE UAIRT_REFERENCE_AS_PLUGIN UAIRT_BUILDING_SHARED)
  target_include_directories(uairt_reference_plugin PRIVATE include)
  set_target_properties(uairt_reference_plugin PROPERTIES
    PREFIX "" C_VISIBILITY_PRESET hidden)

  add_executable(test_core tests/test_core.c)
  target_link_libraries(test_core PRIVATE uairt)
  add_test(NAME core COMMAND test_core $<TARGET_FILE:uairt_reference_plugin>)

  foreach(variant 1 2 3)
    add_library(uairt_session_test_${variant} MODULE tests/session_test_plugin.c)
    target_compile_definitions(uairt_session_test_${variant} PRIVATE
      UAIRT_BUILDING_SHARED SESSION_TEST_VARIANT=${variant})
    target_include_directories(uairt_session_test_${variant} PRIVATE include)
    set_target_properties(uairt_session_test_${variant} PROPERTIES PREFIX "" C_VISIBILITY_PRESET hidden)
  endforeach()
  add_executable(test_session tests/test_session.c)
  target_link_libraries(test_session PRIVATE uairt)
  add_test(NAME session COMMAND test_session
    $<TARGET_FILE:uairt_session_test_1> $<TARGET_FILE:uairt_session_test_2> $<TARGET_FILE:uairt_session_test_3>)

  if(UAIRT_BUILD_ONNXRUNTIME)
    add_executable(test_onnxruntime tests/test_onnxruntime.c)
    target_link_libraries(test_onnxruntime PRIVATE uairt)
    add_test(NAME onnxruntime COMMAND test_onnxruntime
      $<TARGET_FILE:uairt_backend_onnxruntime>
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/add_mul.onnx
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dynamic.onnx)
  endif()

  if(UAIRT_BUILD_QNN AND DEFINED ENV{UAIRT_QNN_TEST_DLC})
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    if(DEFINED ENV{UAIRT_QNN_TARGET})
      set(UAIRT_QNN_TARGET "$ENV{UAIRT_QNN_TARGET}")
    elseif(WIN32)
      set(UAIRT_QNN_TARGET aarch64-windows-msvc)
    else()
      set(UAIRT_QNN_TARGET aarch64-ubuntu-gcc9.4)
    endif()
    set(UAIRT_QNN_DSP_ARGS "")
    if(DEFINED ENV{UAIRT_QNN_DSP_ARCH})
      set(UAIRT_QNN_DSP_ARGS --dsp-arch $ENV{UAIRT_QNN_DSP_ARCH})
    endif()
    if(DEFINED ENV{UAIRT_QNN_TEST_BACKEND})
      set(UAIRT_QNN_BACKEND "$ENV{UAIRT_QNN_TEST_BACKEND}")
    else()
      set(UAIRT_QNN_BACKEND cpu)
    endif()
    add_test(NAME qnn_${UAIRT_QNN_BACKEND}_vs_net_run COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_qnn_net_run.py
      ${QNN_SDK_ROOT} $ENV{UAIRT_QNN_TEST_DLC}
      $<TARGET_FILE:run_model> $<TARGET_FILE:uairt_backend_qnn>
      --target ${UAIRT_QNN_TARGET} --backend ${UAIRT_QNN_BACKEND} ${UAIRT_QNN_DSP_ARGS})
  endif()

  if(UAIRT_BUILD_OPENVINO)
    add_executable(test_openvino tests/test_openvino.c)
    target_link_libraries(test_openvino PRIVATE uairt)
    add_test(NAME openvino_errors COMMAND test_openvino $<TARGET_FILE:uairt_backend_openvino>
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_openvino.c ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(DEFINED ENV{UAIRT_OPENVINO_TEST_MODEL} AND DEFINED ENV{UAIRT_OPENVINO_PYTHON})
      add_test(NAME openvino_vs_openvino_python COMMAND $ENV{UAIRT_OPENVINO_PYTHON}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_openvino.py
        $<TARGET_FILE:run_model> $<TARGET_FILE:uairt_backend_openvino> $ENV{UAIRT_OPENVINO_TEST_MODEL})
    endif()
  endif()

  if(UAIRT_BUILD_NCNN)
    add_executable(test_ncnn tests/test_ncnn.c)
    target_link_libraries(test_ncnn PRIVATE uairt)
    add_test(NAME ncnn_errors COMMAND test_ncnn $<TARGET_FILE:uairt_backend_ncnn>
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(DEFINED ENV{UAIRT_NCNN_TEST_MODEL} AND DEFINED ENV{UAIRT_NCNN_TEST_SHAPES} AND DEFINED ENV{UAIRT_NCNN_PYTHON})
      add_test(NAME ncnn_vs_ncnn_python COMMAND $ENV{UAIRT_NCNN_PYTHON}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_ncnn.py
        $<TARGET_FILE:run_model> $<TARGET_FILE:uairt_backend_ncnn>
        $ENV{UAIRT_NCNN_TEST_MODEL} $ENV{UAIRT_NCNN_TEST_SHAPES})
    endif()
  endif()

  if(UAIRT_BUILD_TENSORRT)
    add_executable(test_tensorrt tests/test_tensorrt.c)
    target_link_libraries(test_tensorrt PRIVATE uairt)
    add_test(NAME tensorrt_errors COMMAND test_tensorrt $<TARGET_FILE:uairt_backend_tensorrt>
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensorrt.c)
    if(DEFINED ENV{UAIRT_TENSORRT_TEST_ENGINE} AND DEFINED ENV{UAIRT_TENSORRT_PYTHON})
      add_test(NAME tensorrt_vs_tensorrt_python COMMAND $ENV{UAIRT_TENSORRT_PYTHON}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_tensorrt.py
        $<TARGET_FILE:run_model> $<TARGET_FILE:uairt_backend_tensorrt> $ENV{UAIRT_TENSORRT_TEST_ENGINE})
    endif()
  endif()

  if(UAIRT_BUILD_COREML)
    add_executable(test_coreml tests/test_coreml.c)
    target_link_libraries(test_coreml PRIVATE uairt)
    add_test(NAME coreml_errors COMMAND test_coreml $<TARGET_FILE:uairt_backend_coreml>)
    if(DEFINED ENV{UAIRT_COREML_PYTHON})
      add_test(NAME coreml_vs_coremltools COMMAND $ENV{UAIRT_COREML_PYTHON}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_coremltools.py
        $<TARGET_FILE:run_model> $<TARGET_FILE:uairt_backend_coreml>)
    endif()
  endif()
  if(UAIRT_BUILD_LLAMACPP)
    add_executable(test_llamacpp tests/test_llamacpp.c)
    target_link_libraries(test_llamacpp PRIVATE uairt)
    add_test(NAME llamacpp_errors COMMAND test_llamacpp $<TARGET_FILE:uairt_backend_llamacpp>
      ${CMAKE_CURRENT_BINARY_DIR})
    if(DEFINED ENV{UAIRT_LLAMACPP_TEST_MODEL})
      add_test(NAME llamacpp_model COMMAND test_llamacpp $<TARGET_FILE:uairt_backend_llamacpp>
        ${CMAKE_CURRENT_BINARY_DIR} $ENV{UAIRT_LLAMACPP_TEST_MODEL})
      if(UAIRT_BUILD_EXAMPLES)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_dependencies(llm_generate llama-completion)
        add_test(NAME llamacpp_vs_llama_completion COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_llamacpp.py
          $<TARGET_FILE:llm_generate> $<TARGET_FILE:uairt_backend_llamacpp>
          $<TARGET_FILE:llama-completion> $ENV{UAIRT_LLAMACPP_TEST_MODEL})
      endif()
    endif()
  endif()

  if(BUILD_SHARED_LIBS)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pytest, numpy"
        RESULT_VARIABLE UAIRT_PYTEST_RESULT OUTPUT_QUIET ERROR_QUIET)
      if(UAIRT_PYTEST_RESULT EQUAL 0)
        add_test(NAME python_sessions COMMAND ${Python3_EXECUTABLE} -m pytest
          ${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/tests -q)
        set(UAIRT_PYTHON_TEST_ENV
          "UAIRT_LIBRARY=$<TARGET_FILE:uairt>"
          "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/bindings/python"
          "UAIRT_TEST_SESSION_PLUGIN=$<TARGET_FILE:uairt_session_test_1>")
        if(UAIRT_BUILD_LLAMACPP)
          list(APPEND UAIRT_PYTHON_TEST_ENV "UAIRT_TEST_LLAMACPP_PLUGIN=$<TARGET_FILE:uairt_backend_llamacpp>")
        endif()
        set_tests_properties(python_sessions PROPERTIES ENVIRONMENT "${UAIRT_PYTHON_TEST_ENV}")
      endif()
    endif()
  endif()

  if(UAIRT_BUILD_EXAMPLES AND UAIRT_BUILD_REFERENCE_BACKEND)
    add_test(NAME quickstart COMMAND quickstart)
  endif()

  if(UAIRT_BUILD_TFLITE)
    add_executable(test_tflite tests/test_tflite.c)
    target_link_libraries(test_tflite PRIVATE uairt)
    add_test(NAME tflite_errors COMMAND test_tflite $<TARGET_FILE:uairt_backend_tflite>)
    if(DEFINED ENV{UAIRT_TFLITE_TEST_MODEL} AND DEFINED ENV{UAIRT_TFLITE_BENCH})
      find_package(Python3 REQUIRED COMPONENTS Interpreter)
      add_test(NAME tflite_vs_bench COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_with_tflite_bench.py
        $<TARGET_FILE:run_model> $<TARGET_FILE:uairt_backend_tflite>
        $ENV{UAIRT_TFLITE_BENCH} $ENV{UAIRT_TFLITE_TEST_MODEL})
    endif()
  endif()

  include(CheckLanguage)
  check_language(CXX)
  if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    add_executable(test_cpp bindings/cpp/test_cpp.cpp)
    target_link_libraries(test_cpp PRIVATE uairt)
    if(UAIRT_BUILD_QNN)
      add_executable(run_qnn_cpp bindings/cpp/run_qnn.cpp)
      target_link_libraries(run_qnn_cpp PRIVATE uairt)
    endif()
    if(UAIRT_BUILD_ONNXRUNTIME)
      add_test(NAME cpp_binding COMMAND test_cpp $<TARGET_FILE:uairt_backend_onnxruntime>
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/add_mul.onnx)
    else()
      add_test(NAME cpp_binding COMMAND test_cpp)
    endif()
  endif()

  if(WIN32)
    find_program(UAIRT_BASH bash HINTS "$ENV{ProgramFiles}/Git/bin" "$ENV{ProgramFiles}/Git/usr/bin"
      NO_DEFAULT_PATH)
  else()
    find_program(UAIRT_BASH bash)
  endif()
  if(UAIRT_BASH)
    add_test(NAME no_vendor_files
      COMMAND ${UAIRT_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_no_vendor_files.sh)
    if(UAIRT_BUILD_LLAMACPP AND NOT WIN32)
      add_test(NAME llamacpp_symbols COMMAND ${UAIRT_BASH}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_plugin_symbols.sh $<TARGET_FILE:uairt_backend_llamacpp>)
    endif()
  endif()

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME references_scripts COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_references.py)
  endif()
endif()

# Python wheel (scikit-build-core): the core library goes into the package, and so do the plugins that can be built without a
# vendor SDK, in the package's plugins/ directory where plugin discovery looks first.
if(SKBUILD)
  # COMPONENT comes first: after an artifact kind it would apply to that kind only, and the library is a RUNTIME artifact on
  # Windows but a LIBRARY artifact on Linux and macOS.
  install(TARGETS uairt COMPONENT python LIBRARY DESTINATION uairt RUNTIME DESTINATION uairt)
  if(TARGET uairt_backend_onnxruntime)
    install(TARGETS uairt_backend_onnxruntime COMPONENT python LIBRARY DESTINATION uairt/plugins)
  endif()
endif()
