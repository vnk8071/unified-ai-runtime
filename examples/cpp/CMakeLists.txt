# SPDX-License-Identifier: Apache-2.0
# C++17 examples using the header-only wrapper in uairt.hpp. Included from the top-level CMakeLists.txt.
if(UAIRT_BUILD_REFERENCE_BACKEND)
  add_executable(quickstart_cpp quickstart.cpp)
  target_link_libraries(quickstart_cpp PRIVATE uairt_llm)
  target_include_directories(quickstart_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME example_cpp_quickstart COMMAND quickstart_cpp)
endif()
if(UAIRT_BUILD_LLAMACPP)
  add_executable(llm_stream_cpp llm_stream.cpp)
  target_link_libraries(llm_stream_cpp PRIVATE uairt_llm)
  target_include_directories(llm_stream_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
