# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

foreach(backend IN ITEMS emitc llvm)
  string(TOUPPER "${backend}" prefix)
  if(backend STREQUAL "llvm")
    set(input "${LLVM_INPUT}")
  else()
    set(input "${INPUT}")
  endif()
  execute_process(
    COMMAND "${REALM_OPT}" --canonicalize --cse --simplify-realm-dataflow
            "--convert-realm-to-${backend}" "${input}"
    COMMAND "${FILECHECK}" "--check-prefix=${prefix}" "${input}" COMMAND_ERROR_IS_FATAL ANY
  )
endforeach()
