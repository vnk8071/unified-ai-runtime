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
  find_path(UAIRT_ORT_INCLUDE onnxruntime_c_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
    NO_DEFAULT_PATH)
  find_library(UAIRT_ORT_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
  if(NOT UAIRT_ORT_INCLUDE OR NOT UAIRT_ORT_LIBRARY)
    message(FATAL_ERROR
      "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to a release directory "
      "containing include/onnxruntime_c_api.h and lib/libonnxruntime.")
  endif()
  get_filename_component(UAIRT_ORT_LIBRARY_DIR ${UAIRT_ORT_LIBRARY} DIRECTORY)

  add_library(uairt_backend_onnxruntime MODULE src/backends/onnxruntime.c)
  target_include_directories(uairt_backend_onnxruntime
    PRIVATE include ${UAIRT_ORT_INCLUDE})
  target_link_libraries(uairt_backend_onnxruntime PRIVATE ${UAIRT_ORT_LIBRARY})
  target_compile_definitions(uairt_backend_onnxruntime PRIVATE UAIRT_BUILDING_SHARED)
  set_target_properties(uairt_backend_onnxruntime PROPERTIES
    PREFIX "lib"
    C_VISIBILITY_PRESET hidden
    BUILD_RPATH ${UAIRT_ORT_LIBRARY_DIR})
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
  target_link_libraries(uairt_backend_coreml PRIVATE "-framework CoreML" "-framework Foundation")
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
  endif()
endif()
