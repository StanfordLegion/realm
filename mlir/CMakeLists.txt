# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(RealmMLIR LANGUAGES C CXX)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_TARGET_DEFINITIONS RealmOps.td)
mlir_tablegen(RealmDialect.h.inc -gen-dialect-decls -dialect=realm)
mlir_tablegen(RealmDialect.cpp.inc -gen-dialect-defs -dialect=realm)
mlir_tablegen(RealmEnums.h.inc -gen-enum-decls)
mlir_tablegen(RealmEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(RealmAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=realm)
mlir_tablegen(RealmAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=realm)
mlir_tablegen(RealmTypes.h.inc -gen-typedef-decls -typedefs-dialect=realm)
mlir_tablegen(RealmTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=realm)
mlir_tablegen(RealmOps.h.inc -gen-op-decls)
mlir_tablegen(RealmOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RealmMLIRIncGen)

add_library(MLIRRealmDialect RealmDialect.cpp)
add_dependencies(MLIRRealmDialect RealmMLIRIncGen)
target_link_libraries(
  MLIRRealmDialect
  PUBLIC MLIRIR MLIRSideEffectInterfaces
  PRIVATE MLIRFuncDialect
)
target_compile_features(MLIRRealmDialect PUBLIC cxx_std_17)
target_include_directories(
  MLIRRealmDialect PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}"
)
target_include_directories(MLIRRealmDialect SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
llvm_update_compile_flags(MLIRRealmDialect)

add_library(
  MLIRRealmTransforms LowerAsyncToRealm.cpp LowerGPUToRealm.cpp LowerRealmToEmitC.cpp
                      LowerRealmToLLVM.cpp SimplifyRealmDataflow.cpp
)
target_link_libraries(
  MLIRRealmTransforms
  PUBLIC MLIRRealmDialect MLIRPass
  PRIVATE MLIREmitCDialect
          MLIRFuncTransforms
          MLIRSCFTransforms
          MLIRFuncToLLVM
          MLIRArithToLLVM
          MLIRControlFlowToLLVM
          MLIRSCFToControlFlow
          MLIRAsyncDialect
          MLIRGPUDialect
          MLIRTransformUtils
)
llvm_update_compile_flags(MLIRRealmTransforms)

add_executable(realm-opt realm-opt.cpp)
target_link_libraries(
  realm-opt
  PRIVATE MLIRRealmTransforms
          MLIROptLib
          MLIRTransforms
          MLIRArithDialect
          MLIRAsyncDialect
          MLIRGPUDialect
          MLIRMemRefDialect
)
llvm_update_compile_flags(realm-opt)

include(CTest)
option(REALM_MLIR_TEST_RUNTIME "Build and run tests against an installed Realm runtime" OFF)
option(REALM_MLIR_BUILD_LLVM_TASK_EXAMPLE "Build the CPU task examples with Realm LLVM JIT" OFF)
option(REALM_MLIR_BUILD_LLVM_EXAMPLES "Build the direct LLVM CPU example" OFF)
option(REALM_MLIR_BUILD_CUDA_EXAMPLE "Build the direct LLVM CUDA example" OFF)

if(REALM_MLIR_BUILD_LLVM_TASK_EXAMPLE
   OR REALM_MLIR_BUILD_LLVM_EXAMPLES
   OR REALM_MLIR_BUILD_CUDA_EXAMPLE
   OR (BUILD_TESTING AND REALM_MLIR_TEST_RUNTIME)
)
  find_package(Realm REQUIRED CONFIG)
  find_program(
    mlir_opt
    NAMES mlir-opt
    HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED
  )
  find_program(
    mlir_translate
    NAMES mlir-translate
    HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED
  )
endif()

if(REALM_MLIR_BUILD_LLVM_EXAMPLES OR REALM_MLIR_BUILD_CUDA_EXAMPLE)
  find_program(
    llvm_llc
    NAMES llc
    HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED
  )
endif()

if(REALM_MLIR_BUILD_LLVM_TASK_EXAMPLE
   OR REALM_MLIR_BUILD_LLVM_EXAMPLES
   OR REALM_MLIR_BUILD_CUDA_EXAMPLE
)
  add_library(RealmMLIRLLVMRuntime STATIC RealmLLVMRuntime.cpp)
  target_link_libraries(RealmMLIRLLVMRuntime PUBLIC Realm::Realm)
endif()

if(BUILD_TESTING)
  add_test(NAME realm-mlir-roundtrip COMMAND realm-opt --verify-roundtrip
                                             "${CMAKE_CURRENT_SOURCE_DIR}/test/deferred-copy.mlir"
  )
  add_test(NAME realm-mlir-invalid-copy
           COMMAND realm-opt --split-input-file --verify-diagnostics
                   "${CMAKE_CURRENT_SOURCE_DIR}/test/invalid-copy.mlir"
  )
  add_test(NAME realm-mlir-invalid-index-space
           COMMAND realm-opt --split-input-file --verify-diagnostics --convert-realm-to-emitc
                   "${CMAKE_CURRENT_SOURCE_DIR}/test/invalid-index-space.mlir"
  )
  add_test(NAME realm-mlir-invalid-async-conversion
           COMMAND realm-opt --split-input-file --verify-diagnostics --convert-async-to-realm
                   "${CMAKE_CURRENT_SOURCE_DIR}/test/invalid-async-conversion.mlir"
  )
  add_test(NAME realm-mlir-invalid-gpu-conversion
           COMMAND realm-opt --split-input-file --verify-diagnostics --convert-gpu-to-realm
                   "${CMAKE_CURRENT_SOURCE_DIR}/test/invalid-gpu-conversion.mlir"
  )
  foreach(backend IN ITEMS emitc llvm)
    add_test(
      NAME realm-mlir-invalid-${backend}-lowering
      COMMAND realm-opt --split-input-file --verify-diagnostics --convert-realm-to-${backend}
              "${CMAKE_CURRENT_SOURCE_DIR}/test/invalid-${backend}-lowering.mlir"
    )
  endforeach()

  set(machine_queries "${CMAKE_CURRENT_SOURCE_DIR}/test/machine-queries.mlir")
  add_test(NAME realm-mlir-machine-query-roundtrip COMMAND realm-opt --verify-roundtrip
                                                           "${machine_queries}"
  )
  add_test(NAME realm-mlir-machine-query-lowering COMMAND realm-opt --convert-realm-to-emitc
                                                          "${machine_queries}"
  )
  find_program(
    filecheck
    NAMES FileCheck
    HINTS "${LLVM_TOOLS_BINARY_DIR}"
  )
  if(filecheck)
    add_test(
      NAME realm-mlir-allocator-order
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/allocator-order.mlir"
        "-DLLVM_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/llvm-allocator-order.mlir" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/test/check-allocator-order.cmake"
    )
    add_test(
      NAME realm-mlir-gpu-allocator-order
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/gpu-allocator-order.mlir" -DFRONTEND=gpu -P
        "${CMAKE_CURRENT_SOURCE_DIR}/test/check-frontend-lowering.cmake"
    )
    add_test(
      NAME realm-mlir-dataflow-boundaries
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/dataflow-boundaries.mlir" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/test/check-dataflow.cmake"
    )
    add_test(
      NAME realm-mlir-tooling
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/test/tooling.mlir" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/test/check-tooling.cmake"
    )
  else()
    message(STATUS "FileCheck was not found. Realm MLIR output checks are disabled.")
  endif()

  if(REALM_MLIR_TEST_RUNTIME)
    set(lowered_queries "${CMAKE_CURRENT_BINARY_DIR}/machine-queries.lowered.mlir")
    set(emitc_queries "${CMAKE_CURRENT_BINARY_DIR}/machine-queries.emitc.mlir")
    set(cpp_queries "${CMAKE_CURRENT_BINARY_DIR}/machine-queries.cpp")
    add_custom_command(
      OUTPUT "${cpp_queries}"
      BYPRODUCTS "${lowered_queries}" "${emitc_queries}"
      COMMAND realm-opt --convert-realm-to-emitc "${machine_queries}" -o "${lowered_queries}"
      COMMAND "${mlir_opt}" --convert-scf-to-emitc "${lowered_queries}" -o "${emitc_queries}"
      COMMAND "${mlir_translate}" --mlir-to-cpp "${emitc_queries}" -o "${cpp_queries}"
      DEPENDS realm-opt "${machine_queries}"
      VERBATIM
    )
    add_executable(realm-mlir-query-runtime-test test/machine-query-runtime.cpp "${cpp_queries}")
    target_include_directories(realm-mlir-query-runtime-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(realm-mlir-query-runtime-test PRIVATE Realm::Realm)
    add_test(NAME realm-mlir-machine-query-runtime COMMAND realm-mlir-query-runtime-test -ll:cpu 1
                                                           -ll:csize 16
    )
  endif()
endif()

add_subdirectory(examples)
