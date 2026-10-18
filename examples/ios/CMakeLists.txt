# SPDX-License-Identifier: Apache-2.0
# An iOS app that runs UAIRT (core + the CoreML backend linked in statically) on a device. Generate an Xcode project and build:
#
#   cmake -S examples/ios -B build-ios -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DUAIRT_IOS_TEAM_ID=<your team id> \
#         -DUAIRT_IOS_BUNDLE_ID=com.example.uairt -DUAIRT_IOS_MODEL=/path/to/convnet.mlmodelc
#   xcodebuild -project build-ios/uairt_ios_example.xcodeproj -scheme UairtExample -configuration Release \
#              -destination generic/platform=iOS -allowProvisioningUpdates build
#
# See examples/ios/README.md for installing it on a device and for the model.
cmake_minimum_required(VERSION 3.21)
# CoreML ML programs need iOS 15 or later; without this the app targets the newest SDK and will not install on older phones.
set(CMAKE_OSX_DEPLOYMENT_TARGET "16.0" CACHE STRING "Minimum iOS version")
project(uairt_ios_example LANGUAGES C CXX OBJC)

set(UAIRT_IOS_TEAM_ID "" CACHE STRING "Apple developer team id used to sign the app")
set(UAIRT_IOS_BUNDLE_ID "com.example.uairt" CACHE STRING "Bundle identifier of the app")
set(UAIRT_IOS_MODEL "" CACHE FILEPATH "A compiled CoreML model directory (convnet.mlmodelc) to bundle")
set(UAIRT_IOS_DETECTOR "" CACHE FILEPATH "Optional: a compiled detector network (image in, scores and boxes out) to bundle")
set(UAIRT_IOS_DETECTOR_INPUT "" CACHE FILEPATH "Optional: the raw input bytes for the detector (BGRA, as the model's image input)")
set(UAIRT_IOS_SAMPLE_IMAGE "" CACHE FILEPATH "Optional: a JPEG to bundle as the 'Sample image' the detector can run on")

set(UAIRT_BUILD_COREML ON CACHE BOOL "" FORCE)
set(UAIRT_STATIC_BACKENDS_ONLY ON CACHE BOOL "" FORCE)
set(UAIRT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(UAIRT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. uairt)

add_executable(UairtExample MACOSX_BUNDLE main.m detector.c)
target_compile_options(UairtExample PRIVATE -fobjc-arc)
target_link_libraries(UairtExample PRIVATE uairt uairt_backend_coreml_static "-framework UIKit" "-framework Foundation"
  "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo" "-framework Accelerate" "-framework QuartzCore" "-framework CoreGraphics" "-framework PhotosUI")
set_target_properties(UairtExample PROPERTIES
  MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_LIST_DIR}/Info.plist.in
  MACOSX_BUNDLE_GUI_IDENTIFIER ${UAIRT_IOS_BUNDLE_ID}
  MACOSX_BUNDLE_BUNDLE_NAME UairtExample
  XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${UAIRT_IOS_TEAM_ID}"
  XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${UAIRT_IOS_BUNDLE_ID}
  XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2")

if(UAIRT_IOS_MODEL)
  add_custom_command(TARGET UairtExample POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UAIRT_IOS_MODEL}" "$<TARGET_BUNDLE_DIR:UairtExample>/convnet.mlmodelc")
endif()

if(UAIRT_IOS_DETECTOR AND UAIRT_IOS_DETECTOR_INPUT)
  add_custom_command(TARGET UairtExample POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UAIRT_IOS_DETECTOR}" "$<TARGET_BUNDLE_DIR:UairtExample>/detector.mlmodelc"
    COMMAND ${CMAKE_COMMAND} -E copy "${UAIRT_IOS_DETECTOR_INPUT}" "$<TARGET_BUNDLE_DIR:UairtExample>/detector_input.raw")
endif()

if(UAIRT_IOS_SAMPLE_IMAGE)
  add_custom_command(TARGET UairtExample POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${UAIRT_IOS_SAMPLE_IMAGE}" "$<TARGET_BUNDLE_DIR:UairtExample>/sample.jpg")
endif()
