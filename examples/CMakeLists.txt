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

# This CMakeLists.txt file enumerates all the examples and creates test targets for them for the parent build system

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(TEST_RESOURCE_LOCK)
set(TEST_DEFAULT_ARGS)
set(TEST_ARGS
    ""
    CACHE STRING "Default arguments for tests"
)

if(NOT TEST_PROCESSORS)
  if(MPIEXEC_EXECUTABLE)
    set(TEST_PROCESSORS 2)
  else()
    set(TEST_PROCESSORS 1)
  endif()
endif()

if(MPIEXEC_EXECUTABLE)
  list(PREPEND TEST_LAUNCHER "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} ${TEST_PROCESSORS}
       ${MPIEXEC_PREFLAGS}
  )
  list(APPEND TEST_RESOURCE_LOCK network)
endif()

if(REALM_USE_CUDA OR REALM_USE_HIP)
  list(APPEND TEST_RESOURCE_LOCK gpu)
  list(APPEND TEST_DEFAULT_ARGS -ll:gpu 1)
  set(TEST_USE_GPU TRUE)
  set(TEST_GPU_LIBS $<TARGET_NAME_IF_EXISTS:CUDA::cudart> $<TARGET_NAME_IF_EXISTS:hip::host>)
endif()

if(REALM_USE_KOKKOS AND REALM_USE_OPENMP)
  list(APPEND TEST_DEFAULT_ARGS -ll:ocpu 1 -ll:onuma 0)
endif()

if(NOT TEST_ARGS)
  set(TEST_ARGS ${TEST_DEFAULT_ARGS})
endif()

if(REALM_USE_PYTHON)
  list(APPEND TEST_ENV "REALM_PYTHON_LIB=$<TARGET_FILE:Python3::Python>")
endif()

if(REALM_USE_UCX)
  list(APPEND TEST_LIB_PATH "$<TARGET_FILE_DIR:realm_ucp_bootstrap_mpi>")
endif()

macro(add_example_test name)
  add_subdirectory(${name})
  add_test(NAME example_${name} COMMAND $<TARGET_FILE:${name}> ${TEST_ARGS} ${${name}_ARGS})
  set_property(
    TEST example_${name} PROPERTY RESOURCE_LOCK $<REMOVE_DUPLICATES:${TEST_RESOURCE_LOCK}
                                  ${${target}_RESOURCE_LOCK}>
  )
  if(REALM_USE_UCX)
    add_dependencies(${name} realm_ucp_bootstrap_mpi)
  endif()
  list(APPEND _example_list example_${name})
endmacro()

add_example_test(saxpy)
set(stencil_ARGS -ll:cpu 4)
add_example_test(stencil)

set_tests_properties(
  ${_example_list}
  PROPERTIES TEST_LAUNCHER
             "${TEST_LAUNCHER}"
             PROCESSORS
             ${TEST_PROCESSORS}
             ENVIRONMENT
             "${TEST_ENV}"
             LABELS
             "example"
)

if(MSVC)
  set_tests_properties(
    ${_example_list} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${TEST_LIB_PATH}"
  )
else()
  set_tests_properties(
    ${_example_list} PROPERTIES ENVIRONMENT_MODIFICATION
                                "LD_LIBRARY_PATH=path_list_append:${TEST_LIB_PATH}"
  )
endif()
