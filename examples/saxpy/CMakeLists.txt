# Copyright 2025 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(RealmExample_saxpy)

# Only search if were building stand-alone and not as part of Realm
if(NOT TARGET Realm::Realm)
  find_package(Realm REQUIRED)
endif()

set(SAXPY_SRCS saxpy_cpu.cc)
set(SAXPY_DEFINITIONS)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  list(APPEND SAXPY_SRCS saxpy_gpu.cu)
  set(SAXPY_DEFINITIONS SAXPY_ENABLE_CUDA=1)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
  enable_language(HIP)
  list(APPEND SAXPY_SRCS saxpy_gpu.cu)
  set_source_files_properties(saxpy_gpu.cu PROPERTIES LANGUAGE HIP)
  set(SAXPY_DEFINITIONS SAXPY_ENABLE_HIP=1)
endif()

add_executable(saxpy ${SAXPY_SRCS})
target_link_libraries(saxpy Realm::Realm)
target_compile_definitions(saxpy PRIVATE ${SAXPY_DEFINITIONS})
