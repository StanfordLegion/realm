# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

set(examples
    pipeline
    fork-join
    fault-cleanup
    copy-staging
    transitive-reduction
    exclusive-reservation
    barrier-phases
    replay
)

if(REALM_MLIR_BUILD_LLVM_TASK_EXAMPLE)
  set(task_ir "${CMAKE_CURRENT_BINARY_DIR}/tasks.ll")
  set(task_source "${CMAKE_CURRENT_SOURCE_DIR}/tasks.mlir")
  add_custom_command(
    OUTPUT "${task_ir}"
    COMMAND "${mlir_translate}" --mlir-to-llvmir "${task_source}" -o "${task_ir}"
    DEPENDS "${task_source}"
    VERBATIM
  )
  set_source_files_properties("${task_ir}" PROPERTIES HEADER_FILE_ONLY TRUE)

  set(example_sources)
  foreach(example IN LISTS examples)
    set(source "${CMAKE_CURRENT_SOURCE_DIR}/${example}.mlir")
    set(optimized "${CMAKE_CURRENT_BINARY_DIR}/${example}.optimized.mlir")
    set(lowered "${CMAKE_CURRENT_BINARY_DIR}/${example}.lowered.mlir")
    set(emitc "${CMAKE_CURRENT_BINARY_DIR}/${example}.emitc.mlir")
    set(cpp "${CMAKE_CURRENT_BINARY_DIR}/${example}.cpp")
    add_custom_command(
      OUTPUT "${cpp}"
      BYPRODUCTS "${optimized}" "${lowered}" "${emitc}"
      COMMAND realm-opt --simplify-realm-dataflow "${source}" -o "${optimized}"
      COMMAND realm-opt --convert-realm-to-emitc "${optimized}" -o "${lowered}"
      COMMAND "${mlir_opt}" --convert-scf-to-emitc "${lowered}" -o "${emitc}"
      COMMAND "${mlir_translate}" --mlir-to-cpp "${emitc}" -o "${cpp}"
      DEPENDS realm-opt "${source}"
      VERBATIM
    )
    list(APPEND example_sources "${cpp}")
  endforeach()

  add_executable(realm-llvm-task main.cpp run-features.cpp ${example_sources} "${task_ir}")
  target_include_directories(realm-llvm-task PRIVATE "${PROJECT_SOURCE_DIR}")
  target_compile_definitions(realm-llvm-task PRIVATE "REALM_LLVM_TASK_IR_PATH=\"${task_ir}\"")
  target_link_libraries(realm-llvm-task PRIVATE RealmMLIRLLVMRuntime)

  if(BUILD_TESTING)
    add_test(NAME realm-mlir-llvm-task COMMAND realm-llvm-task -ll:cpu 2)
    set_tests_properties(realm-mlir-llvm-task PROPERTIES TIMEOUT 30)
  endif()
endif()

if(BUILD_TESTING)
  foreach(example IN LISTS examples)
    add_test(NAME realm-mlir-${example}-lowering
             COMMAND realm-opt --simplify-realm-dataflow --convert-realm-to-emitc
                     "${CMAKE_CURRENT_SOURCE_DIR}/${example}.mlir"
    )
  endforeach()
  add_test(NAME realm-mlir-legacy-lowering-alias
           COMMAND realm-opt --convert-realm-machine-to-emitc
                   "${CMAKE_CURRENT_SOURCE_DIR}/pipeline.mlir"
  )
  foreach(example IN ITEMS llvm-cpu llvm-cuda)
    add_test(NAME realm-mlir-${example}-lowering
             COMMAND realm-opt --convert-async-to-realm --convert-realm-to-llvm
                     "${CMAKE_CURRENT_SOURCE_DIR}/${example}.mlir"
    )
  endforeach()
  add_test(NAME realm-mlir-gpu-input-lowering
           COMMAND realm-opt --convert-gpu-to-realm --symbol-dce --convert-realm-to-llvm
                   "${CMAKE_CURRENT_SOURCE_DIR}/gpu-input.mlir"
  )

  if(filecheck)
    add_test(
      NAME realm-mlir-async-lowering
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/llvm-cpu.mlir" -DFRONTEND=async -P
        "${PROJECT_SOURCE_DIR}/test/check-frontend-lowering.cmake"
    )
    add_test(
      NAME realm-mlir-copy-staging-optimization
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/copy-staging.mlir" -P
        "${PROJECT_SOURCE_DIR}/test/check-dataflow.cmake"
    )
    add_test(
      NAME realm-mlir-transitive-reduction-optimization
      COMMAND
        "${CMAKE_COMMAND}" "-DREALM_OPT=$<TARGET_FILE:realm-opt>" "-DFILECHECK=${filecheck}"
        "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/transitive-reduction.mlir" -DCHECK_EMITC=ON -P
        "${PROJECT_SOURCE_DIR}/test/check-dataflow.cmake"
    )
  endif()
endif()

function(compile_realm_llvm_source source)
  get_filename_component(example "${source}" NAME_WE)
  get_filename_component(source "${source}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(lowered "${CMAKE_CURRENT_BINARY_DIR}/${example}.llvm.mlir")
  set(ir "${CMAKE_CURRENT_BINARY_DIR}/${example}.ll")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/${example}.o")
  add_custom_command(
    OUTPUT "${object}"
    BYPRODUCTS "${lowered}" "${ir}"
    COMMAND realm-opt ${ARGN} --convert-async-to-realm --convert-realm-to-llvm "${source}" -o
            "${lowered}"
    COMMAND "${mlir_translate}" --mlir-to-llvmir "${lowered}" -o "${ir}"
    COMMAND "${llvm_llc}" -filetype=obj -relocation-model=pic "${ir}" -o "${object}"
    DEPENDS realm-opt "${source}"
    VERBATIM
  )
  set_source_files_properties("${object}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  set(${example}_object
      "${object}"
      PARENT_SCOPE
  )
endfunction()

if(REALM_MLIR_BUILD_LLVM_EXAMPLES)
  compile_realm_llvm_source(llvm-cpu.mlir)
  add_executable(realm-llvm-cpu "${llvm-cpu_object}")
  set_target_properties(realm-llvm-cpu PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(realm-llvm-cpu PRIVATE RealmMLIRLLVMRuntime)
  if(BUILD_TESTING)
    add_test(NAME realm-mlir-llvm-cpu COMMAND realm-llvm-cpu -ll:cpu 2)
    add_test(NAME realm-mlir-llvm-cpu-profile COMMAND realm-llvm-cpu -ll:cpu 2 -realm:prof 1)
    set_tests_properties(realm-mlir-llvm-cpu realm-mlir-llvm-cpu-profile PROPERTIES TIMEOUT 30)

    compile_realm_llvm_source("${PROJECT_SOURCE_DIR}/test/llvm-allocator-order.mlir")
    add_executable(
      realm-mlir-llvm-allocator-test "${PROJECT_SOURCE_DIR}/test/llvm-allocator-runtime.cpp"
                                     "${llvm-allocator-order_object}"
    )
    target_link_libraries(realm-mlir-llvm-allocator-test PRIVATE RealmMLIRLLVMRuntime)
    add_test(NAME realm-mlir-llvm-allocator-runtime COMMAND realm-mlir-llvm-allocator-test -ll:cpu
                                                            1 -ll:csize 4
    )
    set_tests_properties(realm-mlir-llvm-allocator-runtime PROPERTIES TIMEOUT 30)
  endif()
endif()

if(REALM_MLIR_BUILD_CUDA_EXAMPLE)
  if(NOT REALM_USE_CUDA)
    message(FATAL_ERROR "The CUDA example requires a CUDA-enabled Realm installation")
  endif()
  find_package(CUDAToolkit REQUIRED)
  compile_realm_llvm_source(llvm-cuda.mlir)
  compile_realm_llvm_source(gpu-input.mlir --convert-gpu-to-realm --symbol-dce)
  set(ptx_source "${CMAKE_CURRENT_SOURCE_DIR}/cuda-kernels.mlir")
  set(ptx_ir "${CMAKE_CURRENT_BINARY_DIR}/cuda-kernels.ll")
  set(ptx "${CMAKE_CURRENT_BINARY_DIR}/cuda-kernels.ptx")
  add_custom_command(
    OUTPUT "${ptx}"
    BYPRODUCTS "${ptx_ir}"
    COMMAND "${mlir_translate}" --mlir-to-llvmir "${ptx_source}" -o "${ptx_ir}"
    COMMAND "${llvm_llc}" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_70 -mattr=+ptx63 "${ptx_ir}" -o
            "${ptx}"
    DEPENDS "${ptx_source}"
    VERBATIM
  )
  set_source_files_properties("${ptx}" PROPERTIES HEADER_FILE_ONLY TRUE)
  add_executable(realm-llvm-cuda run-cuda.cpp "${llvm-cuda_object}" "${gpu-input_object}" "${ptx}")
  target_link_libraries(realm-llvm-cuda PRIVATE RealmMLIRLLVMRuntime CUDA::cuda_driver)
  target_compile_definitions(realm-llvm-cuda PRIVATE "REALM_MLIR_CUDA_PTX=\"${ptx}\"")
  if(BUILD_TESTING)
    add_test(NAME realm-mlir-llvm-cuda COMMAND realm-llvm-cuda -ll:cpu 1 -ll:gpu 1)
    set_tests_properties(realm-mlir-llvm-cuda PROPERTIES TIMEOUT 60 RESOURCE_LOCK gpu)
  endif()
endif()
