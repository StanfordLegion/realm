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

project(PyRealm LANGUAGES C CXX)

set(REALM_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE STRING "Source directory for realm library"
)

option(REALM_ENABLE_TESTING "Enables testing" ${PROJECT_IS_TOP_LEVEL})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
cpmfindpackage(NAME nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v2.9.2)

cpmfindpackage(
  NAME
  Realm
  SOURCE_DIR
  "${REALM_SOURCE_DIR}"
  OPTIONS
  "BUILD_SHARED_LIBS OFF"
  "REALM_ENABLE_HIP OFF"
)

if (REALM_ENABLE_TESTING)
  enable_testing()
  find_package(Pytest REQUIRED)
endif()

nanobind_add_module(realm STABLE_ABI FREE_THREADED src/realm/realm.cpp)
target_link_libraries(realm PRIVATE Realm::Realm)

nanobind_add_stub(realm_stub MODULE realm OUTPUT realm.pyi PYTHON_PATH $<TARGET_FILE_DIR:realm> MARKER_FILE py.typed DEPENDS realm)

if (REALM_ENABLE_TESTING)
  pytest_discover_tests(
    PythonTests
    TEST_PATHS tests
    LIBRARY_PATH_PREPEND $<TARGET_FILE_DIR:realm>
    PYTHON_PATH_PREPEND $<TARGET_FILE_DIR:realm>
    DEPENDS realm
  )
endif()

install(
  TARGETS realm
  COMPONENT python
  DESTINATION realm
)
install(FILES py.typed realm.pyi DESTINATION realm)