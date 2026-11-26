/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// meta-header for Realm - includes all the individual pieces

// decide whether we want C and/or C++ bindings (default matches host language)
//
// each set of bindings has its own include-once ifdef armor, allowing the
//  second set of bindings to be loaded even if the first already has been
#if !defined(REALM_ENABLE_C_BINDINGS) && !defined(REALM_DISABLE_C_BINDINGS)
#ifndef __cplusplus
#define REALM_ENABLE_C_BINDINGS
#endif
#endif
#if !defined(REALM_ENABLE_CXX_BINDINGS) && !defined(REALM_DISABLE_CXX_BINDINGS)
#ifdef __cplusplus
#define REALM_ENABLE_CXX_BINDINGS
#endif
#endif

#ifdef REALM_ENABLE_C_BINDINGS
#include "realm/realm_c.h"
#endif

#ifdef REALM_ENABLE_CXX_BINDINGS
#ifndef REALM_H
#define REALM_H

#include "realm/realm_config.h"

#include "realm/profiling.h"
#include "realm/redop.h"
#include "realm/event.h"
#include "realm/reservation.h"
#include "realm/processor.h"
#include "realm/memory.h"
#include "realm/instance.h"
#include "realm/machine.h"
#include "realm/runtime.h"
#include "realm/indexspace.h"
#include "realm/codedesc.h"
#include "realm/subgraph.h"
#include "realm/module_config.h"

// module configuration structures - these headers contain only plain data
//  types and are always available, even when the corresponding module is not
//  compiled into this build of Realm, so that applications can be written
//  against them unconditionally and use Runtime::get_module_config<T>() to
//  detect (via a null return) which modules are actually present
#include "realm/core_module_config.h"
#include "realm/cuda/cuda_module_config.h"
#include "realm/hip/hip_module_config.h"
#include "realm/hdf5/hdf5_module_config.h"
#include "realm/numa/numa_module_config.h"
#include "realm/openmp/openmp_module_config.h"
#include "realm/procset/procset_module_config.h"
#include "realm/python/python_module_config.h"
#include "realm/gasnetex/gasnetex_module_config.h"
#include "realm/ucx/ucp_module_config.h"
#include "realm/mpi/mpi_module_config.h"

#ifdef REALM_USE_HDF5
#include "realm/hdf5/hdf5_access.h"
#endif

#endif // ifndef REALM_H
#endif // ifdef REALM_ENABLE_CXX_BINDINGS
