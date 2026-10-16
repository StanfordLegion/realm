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

# Tests for the gdb script embedded in librealm (src/realm/realm-gdb.py).
# Included from tests/CMakeLists.txt only when the script is embedded, gdb
# and a Python 3 interpreter are available, and REALM_BUILD_GDB_TESTS is on.
# Each test runs its own gdb session (see run_realm_gdb_tests.py); together
# they take several CPU-minutes.  Run just them with `ctest -L gdb`.

add_executable(realm_gdb_test realm_gdb_test.cc realm_gdb_test_opt.cc)
target_link_libraries(realm_gdb_test Realm::Realm)
# The tests expect particular frames, whatever the build type: unoptimized
# in realm_gdb_test.cc (no tail calls, nothing optimized out), and optimized
# in realm_gdb_test_opt.cc (values in callee-saved registers, inlining).  An
# optimized build still optimizes Realm's own frames, which is the point.
set_source_files_properties(realm_gdb_test.cc PROPERTIES COMPILE_OPTIONS "-O0;-g")
set_source_files_properties(realm_gdb_test_opt.cc PROPERTIES COMPILE_OPTIONS "-O2;-g")

set(_rgt_driver "${CMAKE_CURRENT_SOURCE_DIR}/run_realm_gdb_tests.py")
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rgt_driver}"
           "${CMAKE_CURRENT_SOURCE_DIR}/realm_gdb_tests.py"
)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${_rgt_driver}" --list
  OUTPUT_VARIABLE _rgt_names
  RESULT_VARIABLE _rgt_result
  ERROR_VARIABLE _rgt_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _rgt_result EQUAL 0)
  message(WARNING "Not testing realm-gdb.py: cannot list its tests:\n${_rgt_error}")
  return()
endif()
string(REPLACE "\n" ";" _rgt_names "${_rgt_names}")

set(_rgt_cmd
    "${Python3_EXECUTABLE}" "${_rgt_driver}" --gdb "${REALM_GDB_EXECUTABLE}" --program
    $<TARGET_FILE:realm_gdb_test> --script "${REALM_SOURCE_DIR}/realm-gdb.py"
)

# The core.* tests share one core file (most of a gigabyte), written once;
# likewise the kcore.* tests, with one the kernel writes.
set(_rgt_core "${CMAKE_CURRENT_BINARY_DIR}/realm_gdb_test.core")
set(_rgt_kcore "${CMAKE_CURRENT_BINARY_DIR}/realm_gdb_test.kcore")
foreach(_kind core kcore)
  add_test(NAME gdb.setup.${_kind} COMMAND ${_rgt_cmd} --make-${_kind} "${_rgt_${_kind}}")
  add_test(NAME gdb.cleanup.${_kind} COMMAND "${CMAKE_COMMAND}" -E rm -f "${_rgt_${_kind}}")
  set_tests_properties(
    gdb.setup.${_kind}
    PROPERTIES FIXTURES_SETUP realm_gdb_${_kind}
               LABELS "gdb"
               SKIP_RETURN_CODE 77
               TIMEOUT 600
               PROCESSORS 2
  )
  set_tests_properties(
    gdb.cleanup.${_kind} PROPERTIES FIXTURES_CLEANUP realm_gdb_${_kind} LABELS "gdb"
  )
endforeach()

foreach(_name IN LISTS _rgt_names)
  set(_kind)
  if(_name MATCHES "^(k?core)\\.")
    set(_kind "${CMAKE_MATCH_1}")
  endif()
  if(_kind)
    add_test(NAME gdb.${_name} COMMAND ${_rgt_cmd} --${_kind} "${_rgt_${_kind}}" "${_name}")
  else()
    add_test(NAME gdb.${_name} COMMAND ${_rgt_cmd} "${_name}")
  endif()
  # Each test runs gdb (which takes a few hundred MB) and the program; weigh
  # them so that `ctest -j` doesn't run one per core.
  set_tests_properties(
    gdb.${_name} PROPERTIES LABELS "gdb" SKIP_RETURN_CODE 77 TIMEOUT 600 PROCESSORS 2
  )
  if(_kind)
    set_tests_properties(gdb.${_name} PROPERTIES FIXTURES_REQUIRED realm_gdb_${_kind})
  endif()
endforeach()

# The driver itself: a test whose gdb exits badly must fail.
add_test(NAME gdb.driver_self_test COMMAND ${_rgt_cmd} --self-test)
set_tests_properties(
  gdb.driver_self_test PROPERTIES LABELS "gdb" SKIP_RETURN_CODE 77 TIMEOUT 600 PROCESSORS 2
)
# gdb's DAP (gdb >= 14), driven as an IDE would.
add_test(NAME gdb.dap COMMAND ${_rgt_cmd} --dap-test)
set_tests_properties(
  gdb.dap PROPERTIES LABELS "gdb" SKIP_RETURN_CODE 77 TIMEOUT 600 PROCESSORS 2
)

# gdb on a terminal, where it shows its prompt and repeats a command for an
# empty line (run_realm_gdb_terminal_tests.py).
set(_rgt_term_driver "${CMAKE_CURRENT_SOURCE_DIR}/run_realm_gdb_terminal_tests.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rgt_term_driver}")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${_rgt_term_driver}" --list
  OUTPUT_VARIABLE _rgt_term_names
  RESULT_VARIABLE _rgt_result
  ERROR_VARIABLE _rgt_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _rgt_result EQUAL 0)
  message(WARNING "Not testing realm-gdb.py on a terminal: cannot list its tests:\n${_rgt_error}")
  return()
endif()
string(REPLACE "\n" ";" _rgt_term_names "${_rgt_term_names}")
foreach(_name IN LISTS _rgt_term_names)
  add_test(NAME gdb.${_name} COMMAND "${Python3_EXECUTABLE}" "${_rgt_term_driver}" --gdb
                                     "${REALM_GDB_EXECUTABLE}" --program $<TARGET_FILE:realm_gdb_test> "${_name}"
  )
  set_tests_properties(
    gdb.${_name} PROPERTIES LABELS "gdb" SKIP_RETURN_CODE 77 TIMEOUT 600 PROCESSORS 2
  )
endforeach()
