# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

execute_process(
  COMMAND "${REALM_OPT}" "--convert-${FRONTEND}-to-realm" "${INPUT}"
  COMMAND "${FILECHECK}" --check-prefix=REALM "${INPUT}" COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${REALM_OPT}" "--convert-${FRONTEND}-to-realm" --symbol-dce --convert-realm-to-llvm
          "${INPUT}" COMMAND "${FILECHECK}" --check-prefix=LLVM "${INPUT}" COMMAND_ERROR_IS_FATAL
                             ANY
)
