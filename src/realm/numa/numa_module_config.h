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

// configuration of the NUMA module - installed with every Realm build so that
//  applications can refer to Realm::Numa::NumaModuleConfig unconditionally
//  (see Runtime::get_module_config<T>())

#ifndef REALM_NUMA_MODULE_CONFIG_H
#define REALM_NUMA_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  namespace Numa {

    // the resources the NumaModule discovers when its configuration is created
    //  (and reports through ModuleConfig::get_resource()) - these are not
    //  settings, so they are kept apart from them: a copy of a configuration
    //  carries them along, but assigning one configuration to another leaves
    //  the destination's discovered resources alone
    struct REALM_PUBLIC_API NumaModuleResources {
    public:
      // true once the module has successfully discovered its resources
      bool resources_discovered(void) const { return resource_discover_finished; }

    protected:
      NumaModuleResources(void) = default;
      NumaModuleResources(const NumaModuleResources &) = default;
      NumaModuleResources &operator=(const NumaModuleResources &) { return *this; }

      bool resource_discover_finished = false;
      std::vector<size_t> res_numa_mems; // the size of total memory of each numa domain.
    };

    class REALM_PUBLIC_API NumaModuleConfig : public ModuleConfig,
                                              public NumaModuleResources {
      friend class NumaModule;

    public:
      static constexpr const char *const MODULE_NAME = "numa";

      // a default-constructed object holds Realm's default settings - it may be
      //  copied freely and assigned to the object owned by the runtime (see the
      //  notes on copying in module_config.h)
      NumaModuleConfig(void)
        : ModuleConfig(MODULE_NAME)
      {}

      // parses this module's command-line options - invoked by the runtime via
      //  ModuleConfig::configure_from_cmdline(), not for application use
      void parse_cmdline(std::vector<std::string> &cmdline);

    protected:
      bool discover_resource(void);

    public:
      // configurations
      size_t cfg_numa_mem_size = 0;
      ssize_t cfg_numa_nocpu_mem_size = -1;
      int cfg_num_numa_cpus = 0;
      bool cfg_pin_memory = false;
      size_t cfg_stack_size = 2 << 20;
    };

  }; // namespace Numa

}; // namespace Realm

#endif // ifndef REALM_NUMA_MODULE_CONFIG_H
