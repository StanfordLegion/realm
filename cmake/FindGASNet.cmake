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

if(TARGET GASNet::GASNet)
  get_target_property(_GASNet_existing_module GASNet::GASNet GASNet_PKGCONFIG_MODULE)
  if(_GASNet_existing_module)
    set(GASNet_MODULE_NAME "${_GASNet_existing_module}")
  endif()
  set(GASNet_FOUND TRUE)
  return()
endif()

find_package(PkgConfig)
if(NOT PkgConfig_FOUND)
  return()
endif()

if(NOT GASNet_ROOT)
  set(GASNet_ROOT "$ENV{GASNET_ROOT}")
endif()

if(GASNet_ROOT)
  set(_pkg_config "$ENV{PKG_CONFIG_PATH}" "${GASNet_ROOT}/lib/pkgconfig")
  list(JOIN _pkg_config : _pkg_config_var)
  set(ENV{PKG_CONFIG_PATH} "${_pkg_config_var}")
endif()

set(_GASNet_failure_reason "")
if(GASNet_FIND_COMPONENTS)
  # Installed Realm packages specify the exact variant they linked against.
  # Use a separate prefix so pkg-config cannot reuse a caller's PkgConfig::GASNet target.
  pkg_check_modules(RealmGASNet QUIET IMPORTED_TARGET "${GASNet_FIND_COMPONENTS}")
  foreach(_GASNet_result FOUND VERSION LIBRARIES LINK_LIBRARIES LIBRARY_DIRS INCLUDE_DIRS)
    set(GASNet_${_GASNet_result} "${RealmGASNet_${_GASNet_result}}")
  endforeach()
  set(_GASNet_pkg_target PkgConfig::RealmGASNet)
  set(GASNet_MODULE_NAME "${GASNet_FIND_COMPONENTS}")
  set(_GASNet_failure_reason
      "Realm requires the ${GASNet_MODULE_NAME} pkg-config module. Set GASNet_ROOT or PKG_CONFIG_PATH to an installation providing it."
  )
else()
  if(GASNET_CONDUIT)
    if(GASNET_USE_MULTITHREADED)
      set(_GASNET_MT par)
      if(GASNET_USE_SYNC)
        set(_GASNET_MT parsync)
      endif()
    else()
      set(_GASNET_MT seq)
    endif()
    set(_GASNet_search_modules gasnet-${GASNET_CONDUIT}-${_GASNET_MT})
  else()
    # This list is in order of search priority.
    set(_GASNet_search_modules)
    foreach(
      _gasnet_conduit
      ibv
      ucx
      aries
      ofi
      gemini
      psm
      smp
      mpi
      udp
    )
      foreach(_gasnet_multithread par parsync seq)
        list(APPEND _GASNet_search_modules gasnet-${_gasnet_conduit}-${_gasnet_multithread})
      endforeach()
    endforeach()
  endif()
  pkg_search_module(GASNet ${_GASNet_search_modules} IMPORTED_TARGET GASNet)
  set(_GASNet_pkg_target PkgConfig::GASNet)
endif()

if(GASNet_FOUND)
  # GASNet wants you to use mpicc to build and link your application, so doesn't
  # include the mpi dependencies itself.  Since the MPI dependency is completely
  # contained within MPI, we really just need the MPI library, so add it as a
  # transitive dependency here.  In theory, there may be more of these cases with
  # different conduits, but we'll deal with those on a case-by-case basis
  pkg_get_variable(_needs_mpi ${GASNet_MODULE_NAME} GASNET_LD_REQUIRES_MPI)

  if(_needs_mpi STREQUAL "1")
    find_package(MPI)
    if(NOT MPI_FOUND)
      message(
        WARNING
          "MPI required for gasnet build, but not found.  Check your MPI and GASNet installation"
      )
      set(GASNet_FOUND FALSE)
      return()
    endif()
    list(APPEND _GASNet_LINK_LIBRARIES MPI::MPI_CXX)
  endif()

  # Provide a way to indicate to the caller what conduit was found
  string(REGEX REPLACE "^gasnet-(.+)-(parsync|par|seq)$" "\\1" GASNET_CONDUIT
                       "${GASNet_MODULE_NAME}"
  )

  string(TOUPPER "${GASNET_CONDUIT}" GASNET_CONDUIT_UPPER)
  set(GASNET_CONDUIT_${GASNET_CONDUIT_UPPER} TRUE)
endif()

include(FindPackageMessage)
include(FindPackageHandleStandardArgs)

if(GASNet_FOUND)
  find_package_message(
    GASNet "Found GASNet module: ${GASNet_MODULE_NAME}" "[${GASNet_LIBRARY_DIRS}]"
  )
endif()

find_package_handle_standard_args(
  GASNet
  VERSION_VAR GASNet_VERSION
  REQUIRED_VARS GASNet_FOUND GASNet_LIBRARIES GASNet_INCLUDE_DIRS GASNET_CONDUIT
                GASNET_CONDUIT_${GASNET_CONDUIT_UPPER}
  FOUND_VAR GASNet_FOUND REASON_FAILURE_MESSAGE "${_GASNet_failure_reason}"
)

if(GASNet_FOUND)
  # Export a stable dependency name even when discovery uses a different pkg-config target.
  add_library(GASNet::GASNet INTERFACE IMPORTED)
  target_link_libraries(GASNet::GASNet INTERFACE ${_GASNet_pkg_target} ${_GASNet_LINK_LIBRARIES})
  set_property(TARGET GASNet::GASNet PROPERTY GASNet_PKGCONFIG_MODULE "${GASNet_MODULE_NAME}")
  # Issues #380
  # GASNet pkgconfig adds optimization and build flags that are not necessacary
  # for linkage but override and mess with compilation.  Until GASNet provides
  # a way to remove these, we'll need to just ignore them for now
  set_property(TARGET ${_GASNet_pkg_target} PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()

mark_as_advanced(
  GASNet_INCLUDE_DIRS GASNet_LIBRARIES GASNET_CONDUIT GASNET_CONDUIT_${GASNET_CONDUIT}
)
