/*
 * Copyright 2025 NVIDIA Corporation
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

// UCP network module implementation for Realm

#include "realm/network.h"
#include "realm/runtime_impl.h"
#include "realm/mem_impl.h"
#include "realm/logging.h"

#include "realm/ucx/ucp_module.h"
#include "realm/ucx/ucp_internal.h"

#include <limits>

#ifdef REALM_UCX_MODULE_DYNAMIC
REGISTER_REALM_NETWORK_MODULE_DYNAMIC(Realm::UCPModule);
#endif

namespace Realm {

  Logger log_ucp("ucp");

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPModuleConfig
  //

  UCPModuleConfig::UCPModuleConfig(void)
    : ModuleConfig(MODULE_NAME, cmdline_parser_for<UCPModuleConfig>())
  {
    config_map.insert({"bind_hostmem", &cfg_bind_hostmem});
    config_map.insert({"pollers_max", &cfg_pollers_max});
    config_map.insert({"num_priorities", &cfg_num_priorities});
    config_map.insert({"priority_size_max", &cfg_priority_size_max});
    config_map.insert({"mpool_leakcheck", &cfg_mpool_leakcheck});
    config_map.insert({"crc_check", &cfg_crc_check});
    config_map.insert({"hbuf_malloc", &cfg_hbuf_malloc});
    config_map.insert({"pbuf_malloc", &cfg_pbuf_malloc});
    config_map.insert({"use_wakeup", &cfg_use_wakeup});
    config_map.insert({"prog_boff_max", &cfg_prog_boff_max});
    config_map.insert({"prog_itr_max", &cfg_prog_itr_max});
    config_map.insert({"rdesc_rel_max", &cfg_rdesc_rel_max});
    config_map.insert({"pbuf_mp_init_count", &cfg_pbuf_mp_init_count});
    config_map.insert({"pbuf_mp_max_count", &cfg_pbuf_mp_max_count});
    config_map.insert({"pbuf_mp_max_size", &cfg_pbuf_mp_max_size});
    config_map.insert({"pbuf_mp_max_chunk_size", &cfg_pbuf_mp_max_chunk_size});
    config_map.insert({"pbuf_mp_thresh", &cfg_pbuf_mp_thresh});
    config_map.insert({"mmp_max_obj_size", &cfg_mmp_max_obj_size});
    config_map.insert({"fp_max", &cfg_fp_max});
    config_map.insert({"outstanding_reqs_limit", &cfg_outstanding_reqs_limit});
    config_map.insert({"bind_cudamem", &cfg_bind_cudamem});
  }

  void UCPModuleConfig::parse_cmdline(std::vector<std::string> &cmdline)
  {
    CommandLineParser cp;
    // deferred_allocs realm test always passes -ll:gsize
    size_t deprecated_gsize = 0;
    cp.add_option_int_units("-ll:gsize", deprecated_gsize, 'm');

    cp.add_option_string("-ucx:am_mode", cfg_am_mode);

    cp.add_option_string("-ucx:host_nics", cfg_host_nics);

    cp.add_option_int("-ucx:bindhost", cfg_bind_hostmem);

    // maximum number of background work items to use for UCP polling
    cp.add_option_int("-ucx:pollers_max", cfg_pollers_max);

    // number of message priority levels to support
    cp.add_option_int("-ucx:num_priorities", cfg_num_priorities);

    // maximum message size that is sent with higher priority
    cp.add_option_int_units("-ucx:priority_size_max", cfg_priority_size_max);

    // check memory pools for leak
    cp.add_option_bool("-ucx:mpool_leakcheck", cfg_mpool_leakcheck);

    // CRC check
    cp.add_option_int("-ucx:crc_check", cfg_crc_check);

    cp.add_option_bool("-ucx:hbuf_malloc", cfg_hbuf_malloc);
    cp.add_option_bool("-ucx:pbuf_malloc", cfg_pbuf_malloc);

    // use UCX wakeup feature for progress
    cp.add_option_bool("-ucx:use_wakeup", cfg_use_wakeup);

    cp.add_option_int("-ucx:prog_boff_max", cfg_prog_boff_max);
    cp.add_option_int("-ucx:prog_itr_max", cfg_prog_itr_max);
    cp.add_option_int("-ucx:rdesc_rel_max", cfg_rdesc_rel_max);

    // pbuf mpool
    cp.add_option_int("-ucx:pb_mp_init_count", cfg_pbuf_mp_init_count);
    cp.add_option_int("-ucx:pb_mp_max_count", cfg_pbuf_mp_max_count);
    cp.add_option_int_units("-ucx:pb_mp_max_size", cfg_pbuf_mp_max_size);
    cp.add_option_int_units("-ucx:pb_mp_max_chunk_size", cfg_pbuf_mp_max_chunk_size);
    cp.add_option_int_units("-ucx:pb_mp_thresh", cfg_pbuf_mp_thresh);

    // malloc mpool
    cp.add_option_int_units("-ucx:mmp_max_obj_size", cfg_mmp_max_obj_size);

    // max size for which we try the fast path first
    cp.add_option_int_units("-ucx:fp_max", cfg_fp_max);

    cp.add_option_string("-ucx:ib_seg_size", cfg_ib_seg_size);

    cp.add_option_int("-ucx:oreqs_limit", cfg_outstanding_reqs_limit);

    cp.add_option_string("-ucx:zcopy_host", cfg_zcopy_thresh_host);
    cp.add_option_string("-ucx:tls_host", cfg_tls_host);

    // GPU memory settings are accepted whether or not CUDA support is compiled
    //  in - they are simply not used if it is not
    cp.add_option_int("-ucx:bindcuda", cfg_bind_cudamem);
    cp.add_option_string("-ucx:gpu_nics", cfg_gpu_nics);
    cp.add_option_string("-ucx:zcopy_dev", cfg_zcopy_thresh_dev);
    cp.add_option_string("-ucx:tls_dev", cfg_tls_dev);

    bool ok = cp.parse_command_line(cmdline);
    assert(ok);

    if(deprecated_gsize > 0) {
      log_ucp.fatal() << "Realm UCX backend does not provide a 'global' memory."
                      << " '-ll:gsize' not permitted";
      abort();
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class UCPModule
  //

  UCPModule::UCPModule(Realm::RuntimeImpl *runtime)
    : NetworkModule("ucp")
    , config(nullptr)
  {
    internal = new Realm::UCP::UCPInternal(this, runtime);
    assert(internal);
  }

  UCPModule::~UCPModule() { delete internal; }

  /*static*/
  NetworkModule *UCPModule::create_network_module(RuntimeImpl *runtime, int *argc,
                                                  const char ***argv)
  {
    UCPModule *mod = new UCPModule(runtime);
    assert(mod);

    bool status = mod->internal->bootstrap();
    CHKERR_JUMP(!status, "failed to create UCP network module", log_ucp, err_del_mod);

    return mod;

  err_del_mod:
    delete mod;
    return NULL;
  }

  void UCPModule::get_shared_peers(NodeSet &shared_peers)
  {
    internal->get_shared_peers(shared_peers);
  }

  ModuleConfig *UCPModule::create_module_config(RuntimeImpl *runtime)
  {
    assert(config == nullptr);
    config = new UCPModuleConfig;
    return config;
  }

  void UCPModule::parse_command_line(RuntimeImpl *runtime,
                                     std::vector<std::string> &cmdline)
  {
    assert(config != nullptr);

    //// set internal config ////
    Realm::UCP::UCPInternal::Config internal_config;

    // am-with-remote-address config
    if(config->cfg_am_mode.empty() || (config->cfg_am_mode == "auto")) {
      internal_config.am_wra_mode = Realm::UCP::AM_WITH_REMOTE_ADDR_MODE_AUTO;
    } else if(config->cfg_am_mode == "put") {
      internal_config.am_wra_mode = Realm::UCP::AM_WITH_REMOTE_ADDR_MODE_PUT;
    } else if(config->cfg_am_mode == "am") {
      internal_config.am_wra_mode = Realm::UCP::AM_WITH_REMOTE_ADDR_MODE_AM;
    } else {
      log_ucp.fatal() << "invalid mode for am with remote address " << config->cfg_am_mode
                      << ". Valid choices: auto, put, am";
      abort();
    }

    internal_config.bind_hostmem = config->cfg_bind_hostmem;
    internal_config.pollers_max = config->cfg_pollers_max;
    internal_config.num_priorities = config->cfg_num_priorities;
    internal_config.prog_boff_max = config->cfg_prog_boff_max;
    internal_config.prog_itr_max = config->cfg_prog_itr_max;
    internal_config.rdesc_rel_max = config->cfg_rdesc_rel_max;
    internal_config.mpool_leakcheck = config->cfg_mpool_leakcheck;
    internal_config.crc_check = config->cfg_crc_check;
    internal_config.hbuf_malloc = config->cfg_hbuf_malloc;
    internal_config.pbuf_malloc = config->cfg_pbuf_malloc;
    internal_config.use_wakeup = config->cfg_use_wakeup;
    internal_config.priority_size_max = config->cfg_priority_size_max;
    internal_config.fp_max = config->cfg_fp_max;
    internal_config.pbuf_mp_max_size = config->cfg_pbuf_mp_max_size;
    internal_config.pbuf_mp_max_chunk_size = config->cfg_pbuf_mp_max_chunk_size;
    internal_config.pbuf_mp_max_count = config->cfg_pbuf_mp_max_count;
    internal_config.pbuf_mp_init_count = config->cfg_pbuf_mp_init_count;
    internal_config.pbuf_mp_thresh = config->cfg_pbuf_mp_thresh;
    internal_config.mmp_max_obj_size = config->cfg_mmp_max_obj_size;
    internal_config.outstanding_reqs_limit = config->cfg_outstanding_reqs_limit;
    internal_config.ib_seg_size = config->cfg_ib_seg_size;
    internal_config.host_nics = config->cfg_host_nics;
    internal_config.zcopy_thresh_host = config->cfg_zcopy_thresh_host;
    internal_config.tls_host = config->cfg_tls_host;
#ifdef REALM_USE_CUDA
    internal_config.bind_cudamem = config->cfg_bind_cudamem;
    internal_config.gpu_nics = config->cfg_gpu_nics;
    internal_config.zcopy_thresh_dev = config->cfg_zcopy_thresh_dev;
    internal_config.tls_dev = config->cfg_tls_dev;
#endif

    if(!internal->init(internal_config)) {
      log_ucp.fatal() << "internal init failed";
      abort();
    }
  }

  void UCPModule::attach(RuntimeImpl *runtime, std::vector<NetworkSegment *> &segments)
  {
    internal->attach(segments);
  }

  void UCPModule::detach(RuntimeImpl *runtime, std::vector<NetworkSegment *> &segments)
  {
    internal->detach(segments);
    // Call finalize here because it uses other realm
    // objects (e.g., gpu) which may be destroyed before
    // the network module's destructor is called.
    internal->finalize();
  }

  void UCPModule::create_memories(RuntimeImpl *runtime) {}

  void UCPModule::barrier(void) { internal->barrier(); }

  void UCPModule::broadcast(NodeID root, const void *val_in, void *val_out, size_t bytes)
  {
    internal->broadcast(root, val_in, val_out, bytes);
  }

  void UCPModule::gather(NodeID root, const void *val_in, void *vals_out, size_t bytes)
  {
    internal->gather(root, val_in, vals_out, bytes);
  }

  void UCPModule::allgatherv(const char *val_in, size_t bytes,
                             std::vector<char> &vals_out, std::vector<size_t> &lengths)
  {
    internal->allgatherv(val_in, bytes, vals_out, lengths);
  }

  void UCPModule::sample_quiescence_state(QuiescenceState &state)
  {
    internal->sample_quiescence_state(state);
  }

  void UCPModule::quiescence_allreduce_sum(const uint64_t *local_counts,
                                           uint64_t *total_counts, size_t count)
  {
    internal->quiescence_allreduce_sum(local_counts, total_counts, count);
  }

  MemoryImpl *UCPModule::create_remote_memory(RuntimeImpl *runtime, Memory me,
                                              size_t size, Memory::Kind kind,
                                              const ByteArray &rdma_info_ba)
  {
    return new Realm::UCP::UCPRemoteMemory(runtime, me, size, kind, rdma_info_ba,
                                           internal);
  }

  IBMemory *UCPModule::create_remote_ib_memory(RuntimeImpl *runtime, Memory me,
                                               size_t size, Memory::Kind kind,
                                               const ByteArray &rdma_info_ba)
  {
    return new Realm::UCP::UCPIBMemory(runtime, me, size, kind, rdma_info_ba, internal);
  }

  ActiveMessageImpl *UCPModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const void *src_payload_addr, size_t src_payload_lines,
      size_t src_payload_line_stride, void *storage_base, size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal, target, msgid, header_size, max_payload_size, src_payload_addr,
        src_payload_lines, src_payload_line_stride, nullptr, nullptr, storage_size);
  }

  ActiveMessageImpl *UCPModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const LocalAddress &src_payload_addr, size_t src_payload_lines,
      size_t src_payload_line_stride, const RemoteAddress &dest_payload_addr,
      void *storage_base, size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal, target, msgid, header_size, max_payload_size,
        static_cast<char *>(src_payload_addr.segment->base) + src_payload_addr.offset,
        src_payload_lines, src_payload_line_stride, src_payload_addr.segment,
        &dest_payload_addr, storage_size);
  }

  ActiveMessageImpl *UCPModule::create_active_message_impl(
      NodeID target, unsigned short msgid, size_t header_size, size_t max_payload_size,
      const RemoteAddress &dest_payload_addr, void *storage_base, size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal, target, msgid, header_size, max_payload_size, nullptr, 0, 0, nullptr,
        &dest_payload_addr, storage_size);
  }

  ActiveMessageImpl *UCPModule::create_active_message_impl(
      const NodeSet &targets, unsigned short msgid, size_t header_size,
      size_t max_payload_size, const void *src_payload_addr, size_t src_payload_lines,
      size_t src_payload_line_stride, void *storage_base, size_t storage_size)
  {
    assert(storage_size >= sizeof(Realm::UCP::UCPMessageImpl));
    return new(storage_base) Realm::UCP::UCPMessageImpl(
        internal, targets, msgid, header_size, max_payload_size, src_payload_addr,
        src_payload_lines, src_payload_line_stride, storage_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target, bool with_congestion,
                                            size_t header_size)
  {
    (void)target;
    return internal->recommended_max_payload(nullptr, nullptr, nullptr, with_congestion,
                                             header_size);
  }

  size_t UCPModule::recommended_max_payload(const NodeSet &targets, bool with_congestion,
                                            size_t header_size)
  {
    (void)targets;
    return internal->recommended_max_payload(nullptr, nullptr, nullptr, with_congestion,
                                             header_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target,
                                            const RemoteAddress &dest_payload_addr,
                                            bool with_congestion, size_t header_size)
  {
    (void)target;
    return internal->recommended_max_payload(nullptr, nullptr, &dest_payload_addr,
                                             with_congestion, header_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target, const void *data,
                                            size_t bytes_per_line, size_t lines,
                                            size_t line_stride, bool with_congestion,
                                            size_t header_size)
  {
    (void)target;
    return internal->recommended_max_payload(data, nullptr, nullptr, with_congestion,
                                             header_size);
  }

  size_t UCPModule::recommended_max_payload(const NodeSet &targets, const void *data,
                                            size_t bytes_per_line, size_t lines,
                                            size_t line_stride, bool with_congestion,
                                            size_t header_size)
  {
    (void)targets;
    return internal->recommended_max_payload(data, nullptr, nullptr, with_congestion,
                                             header_size);
  }

  size_t UCPModule::recommended_max_payload(NodeID target,
                                            const LocalAddress &src_payload_addr,
                                            size_t bytes_per_line, size_t lines,
                                            size_t line_stride,
                                            const RemoteAddress &dest_payload_addr,
                                            bool with_congestion, size_t header_size)
  {
    (void)target;
    char *data =
        static_cast<char *>(src_payload_addr.segment->base) + src_payload_addr.offset;
    return internal->recommended_max_payload(
        data, src_payload_addr.segment, &dest_payload_addr, with_congestion, header_size);
  }

  size_t UCPModule::max_payload_size(size_t header_size, const void *src_payload_addr)
  {
    (void)header_size;
    (void)src_payload_addr;
    // UCX can transport messages of any size and pbuf_get can satisfy
    //  allocation requests of any size, so there is no hard upper bound to
    //  report here. Note that this is the maximum *supported* size, not the
    //  most efficient one: payloads larger than config.pbuf_mp_max_size fall
    //  off the pre-registered fast path and pay on-the-fly UCP memory
    //  registration, so callers that care about throughput should query
    //  recommended_max_payload() instead and chunk accordingly.
    return SIZE_MAX;
  }

}; // namespace Realm
