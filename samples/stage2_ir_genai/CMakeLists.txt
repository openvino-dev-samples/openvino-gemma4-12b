# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Builds the Stage-2 reference C++ samples that link against the OpenVINO GenAI
# library shipped in the prebuilt package:
#   genai_facade_llm    - ov::pipeline::LLMPipeline (GenAI-compatible facade)
#   genai_vlm_deploy    - ov::genai::VLMPipeline    (genai-native reference; see NOTES.md)
#
# The package ships openvino_genai.{dll,lib} + headers but NOT a standalone
# OpenVINOGenAIConfig.cmake, so we resolve OpenVINO via its CMake config and
# link the genai library/headers from the same install prefix.
#
# Configure with -DOV_INSTALL_DIR=<repo>\install (defaults to ..\..\..\install,
# i.e. the package unzipped at the repository root).

cmake_minimum_required(VERSION 3.23)
project(gemma4_genai_samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(NOT DEFINED OV_INSTALL_DIR)
    set(OV_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../install")
endif()
get_filename_component(OV_INSTALL_DIR "${OV_INSTALL_DIR}" ABSOLUTE)
message(STATUS "Using OpenVINO install prefix: ${OV_INSTALL_DIR}")

set(OpenVINO_DIR "${OV_INSTALL_DIR}/runtime/cmake")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(_genai_inc "${OV_INSTALL_DIR}/runtime/include")
set(_genai_lib "${OV_INSTALL_DIR}/runtime/lib/intel64/Release/openvino_genai.lib")
if(NOT EXISTS "${_genai_lib}")
    message(FATAL_ERROR "openvino_genai.lib not found at ${_genai_lib}. Unzip the package into install\\ first.")
endif()

# Packaging note: the package installs GenAI-compatible *shadow* headers under
# openvino/genai/ that #include a few upstream genai headers (chat_history.hpp,
# common_types.hpp, ...) which the current package does not install. If the
# OpenVINO GenAI source tree is available, point GENAI_SRC_INCLUDE at its
# include dir to supplement; the package include dir is listed FIRST so the
# GenAI-compatible shadow headers still win.
set(_genai_supplement "")
if(NOT EXISTS "${_genai_inc}/openvino/genai/chat_history.hpp")
    if(DEFINED GENAI_SRC_INCLUDE AND EXISTS "${GENAI_SRC_INCLUDE}/openvino/genai/chat_history.hpp")
        set(_genai_supplement "${GENAI_SRC_INCLUDE}")
        message(STATUS "Supplementing missing upstream genai headers from ${GENAI_SRC_INCLUDE}")
    else()
        message(WARNING
            "Package is missing some upstream genai dev headers (e.g. chat_history.hpp).\n"
            "  Pass -DGENAI_SRC_INCLUDE=<openvino.genai>/src/cpp/include to supplement, or the C++ build may fail.\n"
            "  (The yaml_pipeline_sample.exe path in the package needs no compilation and is unaffected.)")
    endif()
endif()

# Pinned stb_image.h is vendored next to this CMakeLists (image decoding).
add_executable(genai_facade_llm genai_facade_llm.cpp)
target_include_directories(genai_facade_llm PRIVATE "${_genai_inc}" "${_genai_supplement}")
target_link_libraries(genai_facade_llm PRIVATE openvino::runtime "${_genai_lib}")

add_executable(genai_vlm_deploy genai_vlm_deploy.cpp load_image.cpp)
target_include_directories(genai_vlm_deploy PRIVATE
    "${_genai_inc}" "${_genai_supplement}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(genai_vlm_deploy PRIVATE openvino::runtime "${_genai_lib}")

message(STATUS "gemma4_genai_samples configured")
