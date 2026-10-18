/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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

// configuration of the MPI network module - this header has no dependency on
//  MPI and is installed with every Realm build so that applications can refer
//  to Realm::MPIModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_MPI_MODULE_CONFIG_H
#define REALM_MPI_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  class MPIModuleConfig : public ModuleConfig {
    friend class MPIModule;

  public:
    static constexpr const char *const MODULE_NAME = "mpi";

  protected:
    MPIModuleConfig(void);

  public:
    virtual void configure_from_cmdline(std::vector<std::string> &cmdline);

  public:
    // configurations
    // size of the "global" memory (backed by an MPI RMA window) to create
    size_t cfg_global_mem_size = 0;
  };

}; // namespace Realm

#endif // ifndef REALM_MPI_MODULE_CONFIG_H
