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

// configuration of the "core" module - this header is always installed, even
//  though the core module is always present, for symmetry with the other
//  module configuration headers

#ifndef REALM_CORE_MODULE_CONFIG_H
#define REALM_CORE_MODULE_CONFIG_H

#include "realm/module_config.h"

#include <string>
#include <vector>

namespace Realm {

  class HardwareTopology;

  // the resources the CoreModule discovers when its configuration is created
  //  (and reports through ModuleConfig::get_resource()) - these are not
  //  settings, so they are kept apart from them: a copy of a configuration
  //  carries them along, but assigning one configuration to another leaves
  //  the destination's discovered resources alone
  struct REALM_PUBLIC_API CoreModuleResources {
  public:
    // true once the module has successfully discovered its resources
    bool resources_discovered(void) const { return resource_discover_finished; }

  protected:
    CoreModuleResources(void) = default;
    CoreModuleResources(const CoreModuleResources &) = default;
    CoreModuleResources &operator=(const CoreModuleResources &) { return *this; }

    bool resource_discover_finished = false;
    int res_num_cpus = 0;
    size_t res_sysmem_size = 0;
  };

  // the core module provides the basic processors (cpu/util/io) and memories
  //  (system, registered, disk) used by Realm - its configuration also carries
  //  the runtime-wide settings that do not belong to any other module
  //  (background workers, dependent partitioning, the sampling profiler, ...)
  class REALM_PUBLIC_API CoreModuleConfig : public ModuleConfig,
                                            public CoreModuleResources {
    friend class CoreModule;
    friend class RuntimeImpl;

  public:
    static constexpr const char *MODULE_NAME = "core";

    // a default-constructed object holds Realm's default settings - it may be
    //  copied freely and assigned to the object owned by the runtime (see the
    //  notes on copying in module_config.h)
    CoreModuleConfig(void)
      : ModuleConfig(MODULE_NAME)
    {}

    // parses this module's command-line options - invoked by the runtime via
    //  ModuleConfig::configure_from_cmdline(), not for application use
    void parse_cmdline(std::vector<std::string> &cmdline);

  protected:
    bool discover_resource(const HardwareTopology &topo);

  public:
    // configurations
    // CoreModule
    int num_cpu_procs = 1, num_util_procs = 1, num_io_procs = 0;
    int concurrent_io_threads = 1; // Legion does not support values > 1 right now
    size_t sysmem_size = 512 << 20;
    size_t sysmem_ipc_limit = 0; // make the sysmem as shared only if share_sysmem_limit
                                 // == 0 or sysmem_size <= share_sysmem_limit
    size_t stack_size = 2 << 20;
    bool pin_util_procs = false;
    long long cpu_bgwork_timeslice = 0, util_bgwork_timeslice = 0;
    bool use_ext_sysmem = true;

    // RuntimeImpl
    // registered memory for the intermediate buffers used by (inter-node)
    //  copies - the default of ~size_t(0) is resolved during configuration to
    //  256 MB when running on multiple nodes and 64 MB otherwise
    size_t reg_ib_mem_size = ~size_t(0);
    size_t reg_mem_size = 0;
    size_t disk_mem_size = 0;
    unsigned dma_worker_threads = 0; // unused - warning on application use
    bool dma_multi_field = true;
    // event/lock tracing and per-node logging are only functional when Realm
    //  is compiled with EVENT_TRACING/LOCK_TRACING/NODE_LOGGING, but the
    //  settings are always present so that the layout of this struct does not
    //  depend on build macros that are not exported to applications
    size_t event_trace_block_size = 1 << 20;
    double event_trace_exp_arrv_rate = 1e3;
    size_t lock_trace_block_size = 1 << 20;
    double lock_trace_exp_arrv_rate = 1e2;
    std::string event_trace_file;
    std::string lock_trace_file;
    std::string prefix = ".";
    // should local proc threads get dedicated cores?
    bool dummy_reservation_ok = true;
    bool show_reservations = false;
    // are hyperthreads considered to share a physical core
    bool hyperthread_sharing = true;
    bool pin_dma_threads = false;        // unused - silently ignored on cmdline
    size_t bitset_chunk_size = 32 << 10; // 32KB
    // based on some empirical measurements, 1024 nodes seems like
    //  a reasonable cutoff for switching to twolevel nodeset bitmasks
    //  (measured on an E5-2698 v4)
    int bitset_twolevel = -1024;        // i.e. yes if > 1024 nodes
    int active_msg_handler_threads = 0; // default is none (use bgwork)
    bool active_msg_handler_bgwork = true;
    size_t replheap_size = 16 << 20;
    // size (in entries) of the LRU cache of dma transfer paths - 0 disables it
    size_t path_cache_lru_size = 0;

    // runtime-wide settings
    // if nonzero, the event loop detector complains about chains of event
    //  dependencies longer than this
    int event_loop_detection_limit = 0;
    // if true, worker threads that might have used user-level thread
    //  switching fall back to kernel threads
    bool force_kernel_threads = false;
    bool use_fast_reservation_fallback = false;
    bool use_machine_query_cache = true;
    // if true, Realm memories attempt to satisfy instance allocation requests
    //  on the basis of deferred instance destructions
    bool deferred_instance_allocation = true;
    // if true, the number and min/max/avg/stddev duration of handler per
    //  active message type is recorded and printed
    bool profile_activemsg_handlers = false;
    // the maximum time (in nanoseconds) we're willing to spend on inline
    //  active message handlers
    long long max_inline_message_time = 5000;

    // background workers
    unsigned bgwork_generic_workers = 2; // non-numa-specific workers
    unsigned bgwork_per_numa_workers = 0;
    bool bgwork_pin_generic = false;
    bool bgwork_pin_numa = false;
    size_t bgwork_worker_stacksize_in_kb = 1024;
    long long bgwork_worker_spin_interval = 0;
    long long bgwork_work_item_timeslice = 100000;

    // dependent partitioning
    int deppart_num_workers = 0; // dedicated worker threads (0 = use bgwork)
    bool deppart_disable_intersection_opt = false;
    bool deppart_worker_threads_sleep = true;
    bool deppart_allow_inline_ops = false;

    // sampling profiler - consumed when the profiler is started during
    //  Runtime::finish_configure(), so these may be modified after
    //  parse_command_line() like every other setting here
    int realmprof_nodes = 0;                        // profile nodes [0, realmprof_nodes)
    std::string realmprof_file = "realmprof_%.dat"; // '%' is replaced by the node id
    size_t realmprof_buffer_size = 1 << 20;
    size_t realmprof_sample_interval = 10000000; // 10 ms (in nanoseconds)
    // each entry is a comma-separated list of gauge name patterns to sample
    std::vector<std::string> realmprof_patterns;

    // sparsity maps
    bool report_sparsity_leaks = false;

    // barriers
    int barrier_broadcast_radix = 4;
  };

}; // namespace Realm

#endif // ifndef REALM_CORE_MODULE_CONFIG_H
