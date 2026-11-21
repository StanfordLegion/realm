# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

execute_process(
  COMMAND "${REALM_OPT}" --simplify-realm-dataflow "${INPUT}" COMMAND "${FILECHECK}" "${INPUT}"
                                                                      COMMAND_ERROR_IS_FATAL ANY
)

if(CHECK_EMITC)
  execute_process(
    COMMAND "${REALM_OPT}" --simplify-realm-dataflow --convert-realm-to-emitc "${INPUT}"
    COMMAND "${FILECHECK}" --check-prefix=EMITC "${INPUT}" COMMAND_ERROR_IS_FATAL ANY
  )
endif()
