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

// configuration of the OpenMP module - this header has no dependency on
//  OpenMP and is installed with every Realm build so that applications can
//  refer to Realm::OpenMP::OpenMPModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_OPENMP_MODULE_CONFIG_H
#define REALM_OPENMP_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  namespace OpenMP {

    class REALM_PUBLIC_API OpenMPModuleConfig : public ModuleConfig {
      friend class OpenMPModule;

    public:
      static constexpr const char *MODULE_NAME = "openmp";

      // a default-constructed object holds Realm's default settings - it may be
      //  copied freely and assigned to the object owned by the runtime (see the
      //  notes on copying in module_config.h)
      OpenMPModuleConfig(void)
        : ModuleConfig(MODULE_NAME)
      {}

      // parses this module's command-line options - invoked by the runtime via
      //  ModuleConfig::configure_from_cmdline(), not for application use
      void parse_cmdline(std::vector<std::string> &cmdline);

    public:
      // configurations
      int cfg_num_openmp_cpus = 0;
      int cfg_num_threads_per_cpu = 1;
      bool cfg_use_numa = true;
      bool cfg_fake_cpukind = false;
      size_t cfg_stack_size = 2 << 20;
    };

  }; // namespace OpenMP

}; // namespace Realm

#endif // ifndef REALM_OPENMP_MODULE_CONFIG_H
