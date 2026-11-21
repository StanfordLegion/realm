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
  target_link_libraries(realm-llvm-task PRIVATE Realm::Realm)

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

  if(filecheck)
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
