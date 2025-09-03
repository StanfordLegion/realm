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
  return()
endif()

find_package(PkgConfig)
if(NOT PkgConfig_FOUND)
  return()
endif()

if(GASNET_USE_MULTITHREADED)
  set(_GASNET_MT par)
  if(GASNET_USE_SYNC)
    set(_GASNET_MT ${_GASNET_MT}sync)
  endif()
else()
  set(_GASNET_MT seq)
endif()

if(NOT GASNet_ROOT)
  set(GASNet_ROOT "$ENV{GASNET_ROOT}")
endif()

if(GASNet_ROOT)
  set(_pkg_config "$ENV{PKG_CONFIG_PATH}" "${GASNet_ROOT}/lib/pkgconfig")
  list(JOIN _pkg_config : _pkg_config_var)
  set(ENV{PKG_CONFIG_PATH} "${_pkg_config_var}")
endif()

set(_GASNet_search_modules)
if(NOT GASNET_CONDUIT)
  # This list is in order of search priority.  Manually specify via GASNET_CONDUIT
  foreach(_gasnet_conduit ibv ucx aries ofi gemini psm smp mpi udp)
    foreach(_gasnet_multithread par parsync seq)
      list(APPEND _GASNet_search_modules gasnet-${_gasnet_conduit}-${_gasnet_multithread})
    endforeach()
  endforeach()
else()
  set(_GASNet_search_modules gasnet-${GASNET_CONDUIT}-${_GASNET_MT})
endif()

pkg_search_module(GASNet ${_GASNet_search_modules} IMPORTED_TARGET GASNet)

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
  if (NOT GASNET_CONDUIT)
    string(REGEX MATCH "gasnet-([a-z]+)-.*" _ "${GASNet_MODULE_NAME}")
    set(GASNET_CONDUIT ${CMAKE_MATCH_1})
  endif()

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
  REQUIRED_VARS GASNet_FOUND GASNet_LIBRARIES GASNet_INCLUDE_DIRS GASNET_CONDUIT GASNET_CONDUIT_${GASNET_CONDUIT_UPPER}
  FOUND_VAR GASNet_FOUND
)

if(GASNet_FOUND)
  add_library(GASNet::GASNet ALIAS PkgConfig::GASNet)
  string(REPLACE ";" " " _GASNet_CFLAGS_str "${GASNet_CFLAGS_OTHER}")
  string(REPLACE ";" " " _GASNet_LDFLAGS_str "${GASNet_LDFLAGS_OTHER}")
  set_target_properties(
    PkgConfig::GASNet PROPERTIES
    INTERFACE_COMPILE_OPTIONS "SHELL:${_GASNet_CFLAGS_str}"
    INTERFACE_LINK_OPTIONS "SHELL:${_GASNet_LDFLAGS_str}"
  )
  set_property(
    TARGET PkgConfig::GASNet
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES "${_GASNet_LINK_LIBRARIES}"
  )
endif()

mark_as_advanced(GASNet_INCLUDE_DIRS
                 GASNet_LIBRARIES
                 GASNET_CONDUIT
                 GASNET_CONDUIT_${GASNET_CONDUIT}
)
