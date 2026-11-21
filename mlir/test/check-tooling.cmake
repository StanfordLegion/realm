# Copyright 2026 Stanford University, NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

execute_process(
  COMMAND "${REALM_OPT}" --canonicalize --cse "${INPUT}"
  COMMAND "${FILECHECK}" --check-prefix=CANON "${INPUT}" COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND "${REALM_OPT}" --convert-realm-to-emitc "${INPUT}"
  COMMAND "${REALM_OPT}" --verify-roundtrip -
  COMMAND "${FILECHECK}" --check-prefix=LOWER "${INPUT}" COMMAND_ERROR_IS_FATAL ANY
)
