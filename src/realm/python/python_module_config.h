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

// configuration of the Python module - this header has no dependency on
//  Python and is installed with every Realm build so that applications can
//  refer to Realm::Python::PythonModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_PYTHON_MODULE_CONFIG_H
#define REALM_PYTHON_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  namespace Python {

    class REALM_PUBLIC_API PythonModuleConfig : public ModuleConfig {
      friend class PythonModule;

    public:
      static constexpr const char *const MODULE_NAME = "python";

      // a default-constructed object holds Realm's default settings - it may be
      //  copied freely and assigned to the object owned by the runtime (see the
      //  notes on copying in module_config.h)
      PythonModuleConfig(void)
        : ModuleConfig(MODULE_NAME)
      {}

      // parses this module's command-line options - invoked by the runtime via
      //  ModuleConfig::configure_from_cmdline(), not for application use
      void parse_cmdline(std::vector<std::string> &cmdline);

    public:
      // configurations
      int cfg_num_python_cpus = 0;
      size_t cfg_stack_size = 2 << 20;
      // OpenMP threads associated with each python processor - only honored
      //  when Realm is built with OpenMP support
      int cfg_pyomp_threads = 0;
      std::vector<std::string> cfg_import_modules;
      std::vector<std::string> cfg_init_scripts;
    };

  }; // namespace Python

}; // namespace Realm

#endif // ifndef REALM_PYTHON_MODULE_CONFIG_H
