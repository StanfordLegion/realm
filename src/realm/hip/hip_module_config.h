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

// configuration of the HIP module - this header has no dependency on HIP
//  and is installed with every Realm build so that applications can refer to
//  Realm::Hip::HipModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_HIP_MODULE_CONFIG_H
#define REALM_HIP_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  namespace Hip {

    // the resources the HipModule discovers when its configuration is created
    //  (and reports through ModuleConfig::get_resource()) - these are not
    //  settings, so they are kept apart from them: a copy of a configuration
    //  carries them along, but assigning one configuration to another leaves
    //  the destination's discovered resources alone
    struct REALM_PUBLIC_API HipModuleResources {
    public:
      // true once the module has successfully discovered its resources
      bool resources_discovered(void) const { return resource_discover_finished; }

    protected:
      HipModuleResources(void) = default;
      HipModuleResources(const HipModuleResources &) = default;
      HipModuleResources &operator=(const HipModuleResources &) { return *this; }

      bool resource_discover_finished = false;
      int res_num_gpus = 0;
      size_t res_min_fbmem_size = 0;
      std::vector<size_t> res_fbmem_sizes;
    };

    class REALM_PUBLIC_API HipModuleConfig : public ModuleConfig,
                                             public HipModuleResources {
      friend class HipModule;

    public:
      static constexpr const char *const MODULE_NAME = "hip";

      // a default-constructed object holds Realm's default settings - it may be
      //  copied freely and assigned to the object owned by the runtime (see the
      //  notes on copying in module_config.h)
      HipModuleConfig(void)
        : ModuleConfig(MODULE_NAME)
      {}

      // parses this module's command-line options - invoked by the runtime via
      //  ModuleConfig::configure_from_cmdline(), not for application use
      void parse_cmdline(std::vector<std::string> &cmdline);

    protected:
      bool discover_resource(void);

    public:
      // configurations
      size_t cfg_zc_mem_size = 64 << 20, cfg_zc_ib_size = 256 << 20;
      size_t cfg_fb_mem_size = 256 << 20, cfg_fb_ib_size = 128 << 20;
      size_t cfg_uvm_mem_size = 0;
      bool cfg_use_dynamic_fb = true;
      size_t cfg_dynfb_max_size = ~size_t(0);
      int cfg_num_gpus = 0;
      std::string cfg_gpu_idxs;
      unsigned cfg_task_streams = 12, cfg_d2d_streams = 4;
      bool cfg_use_worker_threads = false, cfg_use_shared_worker = true,
           cfg_pin_sysmem = true;
      bool cfg_fences_use_callbacks = false;
      unsigned cfg_skip_gpu_count = 0;
      bool cfg_skip_busy_gpus = false;
      size_t cfg_min_avail_mem = 0;
      int cfg_task_context_sync = -1; // 0 = no, 1 = yes, -1 = default (based on hijack)
      int cfg_max_ctxsync_threads = 4;
      bool cfg_multithread_dma = false;
      size_t cfg_hostreg_limit = 1 << 30;
      int cfg_d2d_stream_priority = -1;
      bool cfg_use_hip_ipc = true;
    };

  }; // namespace Hip

}; // namespace Realm

#endif // ifndef REALM_HIP_MODULE_CONFIG_H
