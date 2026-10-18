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

// configuration of the ProcSet module - installed with every Realm build so
//  that applications can refer to Realm::ProcSet::ProcSetModuleConfig
//  unconditionally (see Runtime::get_module_config<T>())

#ifndef REALM_PROCSET_MODULE_CONFIG_H
#define REALM_PROCSET_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  namespace ProcSet {

    class REALM_PUBLIC_API ProcSetModuleConfig : public ModuleConfig {
      friend class ProcSetModule;

    public:
      static constexpr const char *const MODULE_NAME = "procset";

    protected:
      ProcSetModuleConfig(void);

    public:
      // parses this module's command-line options - invoked by the runtime via
      //  ModuleConfig::configure_from_cmdline(), not for application use (and
      //  deliberately not virtual - see the note in module_config.h)
      void parse_cmdline(std::vector<std::string> &cmdline);

    public:
      // configurations
      int cfg_num_mp_threads = 0; // threads per ProcSet
      int cfg_num_mp_procs = 0;   // number of ProcSets
      int cfg_num_mp_cpus = 0;    // additional cpus on any non ProcSet nodes
      size_t cfg_stack_size = 2 << 20;
    };

  }; // namespace ProcSet

}; // namespace Realm

#endif // ifndef REALM_PROCSET_MODULE_CONFIG_H
