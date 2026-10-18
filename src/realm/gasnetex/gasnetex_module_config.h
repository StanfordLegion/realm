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

// configuration of the GASNet-EX network module - this header has no
//  dependency on GASNet and is installed with every Realm build so that
//  applications can refer to Realm::GASNetEXModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_GASNETEX_MODULE_CONFIG_H
#define REALM_GASNETEX_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  class GASNetEXModuleConfig : public ModuleConfig {
    friend class GASNetEXModule;

  public:
    static constexpr const char *const MODULE_NAME = "gasnetex";

  protected:
    GASNetEXModuleConfig(void);

  public:
    virtual void configure_from_cmdline(std::vector<std::string> &cmdline);

  public:
    // configurations
    bool cfg_use_immediate = true;
    bool cfg_use_negotiated = true;
    long long cfg_crit_timeout = 50000; // 50us
    size_t cfg_max_medium = 0;          // 0 = use GASNet limit
    size_t cfg_max_long = 4 << 20;      // 4 MB
    bool cfg_bind_hostmem = true;
    // binding of GPU memories is only honored when Realm is built with the
    //  corresponding GPU module
    bool cfg_bind_cudamem = true;
    bool cfg_bind_hipmem = true;
    bool cfg_do_checksums = true;
    bool cfg_batch_messages = true;
    // number and size of "outbufs", used to put pkt header and/or data in
    //  registered memory for RDMA goodness - the count is raised at startup to
    //  the minimum required for the number of ranks in the job
    size_t cfg_outbuf_count = 64;
    size_t cfg_outbuf_size = 256 << 10; // 256 KB
    bool cfg_force_rma = false;
    bool cfg_use_rma_put = true; // disabled by default on GASNet releases < 2021.8.3
    int cfg_am_limit = 0;        // upper bound on AMs in flight to any given peer
  };

}; // namespace Realm

#endif // ifndef REALM_GASNETEX_MODULE_CONFIG_H
