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

// configuration of the UCX network module - this header has no dependency on
//  UCX and is installed with every Realm build so that applications can refer
//  to Realm::UCPModuleConfig unconditionally (see
//  Runtime::get_module_config<T>())

#ifndef REALM_UCP_MODULE_CONFIG_H
#define REALM_UCP_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Realm {

  class REALM_PUBLIC_API UCPModuleConfig : public ModuleConfig {
    friend class UCPModule;

  public:
    // NOTE: this is the name the network is registered (and selected via
    //  -ll:networks) under, which differs from the "ucp" name of the module
    //  object itself
    static constexpr const char *const MODULE_NAME = "ucx";

    // a default-constructed object holds Realm's default settings - it may be
    //  copied freely and assigned to the object owned by the runtime (see the
    //  notes on copying in module_config.h)
    UCPModuleConfig(void)
      : ModuleConfig(MODULE_NAME)
    {}

    // parses this module's command-line options - invoked by the runtime via
    //  ModuleConfig::configure_from_cmdline(), not for application use
    void parse_cmdline(std::vector<std::string> &cmdline);

  public:
    // configurations
    // how active messages with a remote address are delivered: "auto", "put"
    //  or "am"
    std::string cfg_am_mode = "auto";
    std::string cfg_host_nics; // comma-separated list of NICs for host memory
    bool cfg_bind_hostmem = true;
    int cfg_pollers_max = 2;
    int cfg_num_priorities = 2;
    size_t cfg_priority_size_max = 64;
    bool cfg_mpool_leakcheck = false;
    bool cfg_crc_check = true;
    bool cfg_hbuf_malloc = false;
    bool cfg_pbuf_malloc = false;
    bool cfg_use_wakeup = false;
    int cfg_prog_boff_max = 4; // progress thread maximum backoff
    int cfg_prog_itr_max = 16;
    int cfg_rdesc_rel_max = 16;
    // payload buffer memory pool
    size_t cfg_pbuf_mp_init_count = 1024;
    size_t cfg_pbuf_mp_max_count = SIZE_MAX;
    size_t cfg_pbuf_mp_max_size = 8 << 10;       // 8K
    size_t cfg_pbuf_mp_max_chunk_size = 4 << 20; // 4M
    size_t cfg_pbuf_mp_thresh = 2 << 10;         // 2K
    size_t cfg_mmp_max_obj_size = 2 << 10;       // 2K - malloc mpool max object size
    size_t cfg_fp_max = 2 << 10;                 // 2K - fast path max message size
    std::string cfg_ib_seg_size;                 // passed to UCX as IB_SEG_SIZE
    size_t cfg_outstanding_reqs_limit = 1 << 17; // 128K (count)
    std::string cfg_zcopy_thresh_host;           // passed to UCX as ZCOPY_THRESH
    std::string cfg_tls_host;                    // passed to UCX as TLS
    // GPU memory settings - only honored when Realm is built with CUDA
    bool cfg_bind_cudamem = true;
    std::string cfg_gpu_nics; // comma-separated list of NICs for GPU memory
    std::string cfg_zcopy_thresh_dev;
    std::string cfg_tls_dev;
  };

}; // namespace Realm

#endif // ifndef REALM_UCP_MODULE_CONFIG_H
