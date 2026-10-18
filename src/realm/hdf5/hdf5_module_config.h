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

// configuration of the HDF5 module - this header has no dependency on HDF5
//  and is installed with every Realm build so that applications can refer to
//  Realm::HDF5::HDF5ModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_HDF5_MODULE_CONFIG_H
#define REALM_HDF5_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  namespace HDF5 {

    class HDF5ModuleConfig : public ModuleConfig {
      friend class HDF5Module;

    public:
      static constexpr const char *const MODULE_NAME = "hdf5";

    protected:
      HDF5ModuleConfig(void);

    public:
      virtual void configure_from_cmdline(std::vector<std::string> &cmdline);

    public:
      // configurations
      bool cfg_showerrors = true;
      // number of no-longer-referenced HDF5 files to keep open for reuse
      size_t cfg_max_open_files = 0;
      // open every HDF5 file read-write, even if it is only being read
      bool cfg_force_read_write = false;
    };

  }; // namespace HDF5

}; // namespace Realm

#endif // ifndef REALM_HDF5_MODULE_CONFIG_H
