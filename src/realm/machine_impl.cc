/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
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

#include "realm/machine_impl.h"

#include <algorithm>

#include "realm/realm_assert.h"
#include "realm/logging.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/runtime_impl.h"

#include "realm/activemsg.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#ifdef REALM_ON_WINDOWS
// TODO: clean up query cache code and renable!
#define REALM_DISABLE_MACHINE_QUERY_CACHE

static int lrand48() { return rand(); }
#endif

TYPE_IS_SERIALIZABLE(Realm::NodeAnnounceTag);
TYPE_IS_SERIALIZABLE(Realm::Memory);
TYPE_IS_SERIALIZABLE(Realm::Memory::Kind);
TYPE_IS_SERIALIZABLE(Realm::Channel::SupportedPath);
TYPE_IS_SERIALIZABLE(Realm::XferDesKind);
TYPE_IS_SERIALIZABLE(Realm::Machine::ProcessorMemoryAffinity);
TYPE_IS_SERIALIZABLE(Realm::Machine::ProcessInfo);

namespace Realm {

  Logger log_machine("machine");
  Logger log_annc("announce");
  Logger log_query("query");

  template <typename KT, typename VT>
  inline void delete_map_contents(std::map<KT, VT *> &m)
  {
    for(typename std::map<KT, VT *>::const_iterator it = m.begin(); it != m.end(); ++it)
      delete it->second;
  }

  static inline bool is_local_affinity(const Machine::ProcessorMemoryAffinity &pma)
  {
    return ID(pma.p).proc_owner_node() == ID(pma.m).memory_owner_node();
  }

  static inline bool is_local_affinity(const Machine::MemoryMemoryAffinity &mma)
  {
    return ID(mma.m1).memory_owner_node() == ID(mma.m2).memory_owner_node();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineAffinityInfo<KT,AT>
  //

  template <typename KT, typename AT>
  MachineAffinityInfo<KT, AT>::MachineAffinityInfo(void)
  {}

  template <typename KT, typename AT>
  MachineAffinityInfo<KT, AT>::~MachineAffinityInfo(void)
  {
    delete_map_contents(all);
#if 0
    for(std::map<KT, AT *>::const_iterator it = all.begin();
	it != all.end();
	++it)
      delete it->second;
#endif
  }

  template <typename KT, typename AT>
  bool MachineAffinityInfo<KT, AT>::add_affinity(KT key, const AT &aff, bool is_local)
  {
    // look up existing affinity
    AT *&ptr = all[key];
    if(ptr == 0) {
      // create a new entry
      ptr = new AT(aff);
      if(is_local)
        local[key] = ptr;
    } else {
      // see if it's an exact match
      if((ptr->bandwidth == aff.bandwidth) && (ptr->latency == aff.latency))
        return false; // no change

      ptr->bandwidth = aff.bandwidth;
      ptr->latency = aff.latency;
    }

    // maybe update the best set
    if(best.empty()) {
      // easy case - we're best by default
      best[key] = ptr;
    } else {
      // all existing ones are equal so just look at the first
      AT *cur_best = best.begin()->second;
      if(cur_best->bandwidth > aff.bandwidth) {
        // better ones exist - do nothing
      } else if(cur_best->bandwidth < aff.bandwidth) {
        // this is better than existing - erase those and replace with us
        best.clear();
        best[key] = ptr;
      } else {
        // equal to existing ones - add ourselves
        best[key] = ptr;
      }
    }

    return true; // something changed
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineProcInfo
  //

  MachineProcInfo::MachineProcInfo(Processor _p)
    : p(_p)
  {}

  MachineProcInfo::~MachineProcInfo(void) {}

  bool MachineProcInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma)
  {
    log_machine.info() << "ProcInfo: adding proc-mem affinity " << pma.p << " -> "
                       << pma.m << " with bandwidth " << pma.bandwidth << " and latency "
                       << pma.latency;
    REALM_ASSERT(pma.bandwidth > 0);
    REALM_ASSERT(pma.latency > 0);
    bool is_local = is_local_affinity(pma);
    return pmas.add_affinity(pma.m, pma, is_local);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineMemInfo
  //

  MachineMemInfo::MachineMemInfo(Memory _m)
    : m(_m)
  {}

  MachineMemInfo::~MachineMemInfo(void) {}

  bool MachineMemInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma)
  {
    bool is_local = is_local_affinity(pma);
    return pmas.add_affinity(pma.p, pma, is_local);
  }

  bool MachineMemInfo::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity &mma)
  {
    bool is_local = is_local_affinity(mma);

    if(mma.m1 == m)
      return mmas_out.add_affinity(mma.m2, mma, is_local);

    if(mma.m2 == m)
      return mmas_in.add_affinity(mma.m1, mma, is_local);

    // shouldn't have been called!
    abort();
    return false;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineNodeInfo
  //

  MachineNodeInfo::MachineNodeInfo(int _node, RuntimeImpl *_runtime_impl)
    : node(_node)
    , runtime_impl(_runtime_impl)
  {}

  MachineNodeInfo::~MachineNodeInfo(void)
  {
    delete_map_contents(procs);
    delete_map_contents(mems);
    // TODO: assert process_info is valid after adding machine announcement
    if(process_info) {
      delete process_info;
    }
    runtime_impl = nullptr;
  }

  bool MachineNodeInfo::add_processor(Processor p)
  {
    REALM_ASSERT(node == NodeID(ID(p).proc_owner_node()));
    MachineProcInfo *&ptr = procs[p];
    // TODO: see if anything changed?
    if(ptr != 0)
      return false;

    ptr = new MachineProcInfo(p);

    return true;
  }

  bool MachineNodeInfo::add_memory(Memory m)
  {
    REALM_ASSERT(node == NodeID(ID(m).memory_owner_node()));
    MachineMemInfo *&ptr = mems[m];
    // TODO: see if anything changed?
    if(ptr != 0)
      return false;

    ptr = new MachineMemInfo(m);

    return true;
  }

  bool MachineNodeInfo::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma)
  {
    bool changed = false;

    if(NodeID(ID(pma.p).proc_owner_node()) == node) {
      MachineProcInfo *mpi = procs[pma.p];
      REALM_ASSERT(mpi != 0);
      if(mpi->add_proc_mem_affinity(pma))
        changed = true;
    }

    if(NodeID(ID(pma.m).memory_owner_node()) == node) {
      MachineMemInfo *mmi = mems[pma.m];
      REALM_ASSERT(mmi != 0);
      if(mmi->add_proc_mem_affinity(pma))
        changed = true;
    }

    return changed;
  }

  bool MachineNodeInfo::add_mem_mem_affinity(const Machine::MemoryMemoryAffinity &mma)
  {
    bool changed = false;

    if(NodeID(ID(mma.m1).memory_owner_node()) == node) {
      MachineMemInfo *mmi = mems[mma.m1];
      REALM_ASSERT(mmi != 0);
      if(mmi->add_mem_mem_affinity(mma))
        changed = true;
    }

    if(NodeID(ID(mma.m2).memory_owner_node()) == node) {
      MachineMemInfo *mmi = mems[mma.m2];
      REALM_ASSERT(mmi != 0);
      if(mmi->add_mem_mem_affinity(mma))
        changed = true;
    }

    return changed;
  }

  bool MachineNodeInfo::add_process_info(const Machine::ProcessInfo &proc_info)
  {
    if(!process_info) {
      process_info = new Machine::ProcessInfo();
      *process_info = proc_info;
      return true;
    } else {
      return false;
    }
  }

  void MachineNodeInfo::update_kind_maps(void)
  {
    proc_by_kind.clear();
    for(std::map<Processor, MachineProcInfo *>::const_iterator it = procs.begin();
        it != procs.end(); ++it) {
      Processor::Kind proc_kind = runtime_impl->get_processor_impl(it->first)->kind;
      proc_by_kind[proc_kind][it->first] = it->second;
    }

    mem_by_kind.clear();
    for(std::map<Memory, MachineMemInfo *>::const_iterator it = mems.begin();
        it != mems.end(); ++it) {
      Memory::Kind mem_kind = MemoryImpl::get_memory_kind(runtime_impl, it->first);
      mem_by_kind[mem_kind][it->first] = it->second;
    }
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine
  //

  /*static*/ Machine Machine::get_machine(void)
  {
    return Machine(get_runtime()->machine);
  }

  size_t Machine::get_address_space_count(void) const { return Network::max_node_id + 1; }

  void Machine::get_all_memories(std::set<Memory> &mset) const
  {
    return (static_cast<MachineImpl *>(impl))->get_all_memories(mset);
  }

  void Machine::get_all_processors(std::set<Processor> &pset) const
  {
    return (static_cast<MachineImpl *>(impl))->get_all_processors(pset);
  }

  void Machine::get_local_processors(std::set<Processor> &pset) const
  {
    return (static_cast<MachineImpl *>(impl))->get_local_processors(pset);
  }

  void Machine::get_local_processors_by_kind(std::set<Processor> &pset,
                                             Processor::Kind kind) const
  {
    return (static_cast<MachineImpl *>(impl))->get_local_processors_by_kind(pset, kind);
  }

  // Return the set of memories visible from a processor
  void Machine::get_visible_memories(Processor p, std::set<Memory> &mset,
                                     bool local_only /*= true*/) const
  {
    return (static_cast<MachineImpl *>(impl))->get_visible_memories(p, mset, local_only);
  }

  // Return the set of memories visible from a memory
  void Machine::get_visible_memories(Memory m, std::set<Memory> &mset,
                                     bool local_only /*= true*/) const
  {
    return (static_cast<MachineImpl *>(impl))->get_visible_memories(m, mset, local_only);
  }

  // Return the set of processors which can all see a given memory
  void Machine::get_shared_processors(Memory m, std::set<Processor> &pset,
                                      bool local_only /*= true*/) const
  {
    return (static_cast<MachineImpl *>(impl))->get_shared_processors(m, pset, local_only);
  }

  bool Machine::get_process_info(Processor p, ProcessInfo *info) const
  {
    return (static_cast<MachineImpl *>(impl))->get_process_info(p, info);
  }

  bool Machine::has_affinity(Processor p, Memory m,
                             AffinityDetails *details /*= 0*/) const
  {
    return (static_cast<MachineImpl *>(impl))->has_affinity(p, m, details);
  }

  bool Machine::has_affinity(Memory m1, Memory m2, AffinityDetails *details /*= 0*/) const
  {
    return (static_cast<MachineImpl *>(impl))->has_affinity(m1, m2, details);
  }

  int Machine::get_proc_mem_affinity(
      std::vector<Machine::ProcessorMemoryAffinity> &result,
      Processor restrict_proc /*= Processor::NO_PROC*/,
      Memory restrict_memory /*= Memory::NO_MEMORY*/, bool local_only /*= true*/) const
  {
    return (static_cast<MachineImpl *>(impl))
        ->get_proc_mem_affinity(result, restrict_proc, restrict_memory, local_only);
  }

  int Machine::get_mem_mem_affinity(std::vector<Machine::MemoryMemoryAffinity> &result,
                                    Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
                                    Memory restrict_mem2 /*= Memory::NO_MEMORY*/,
                                    bool local_only /*= true*/) const
  {
    return (static_cast<MachineImpl *>(impl))
        ->get_mem_mem_affinity(result, restrict_mem1, restrict_mem2, local_only);
  }

  void Machine::add_subscription(MachineUpdateSubscriber *subscriber)
  {
    (static_cast<MachineImpl *>(impl))->add_subscription(subscriber);
  }

  void Machine::remove_subscription(MachineUpdateSubscriber *subscriber)
  {
    (static_cast<MachineImpl *>(impl))->remove_subscription(subscriber);
  }

  namespace Config {
    bool use_machine_query_cache = true;
  };

  ////////////////////////////////////////////////////////////////////////
  //
  // class MachineImpl
  //

  MachineImpl *machine_singleton = nullptr;

  MachineImpl::MachineImpl(RuntimeImpl *_runtime_impl)
    : runtime_impl(_runtime_impl)
  {
    REALM_ASSERT(machine_singleton == nullptr);
    machine_singleton = this;
  }

  MachineImpl::~MachineImpl(void)
  {
    REALM_ASSERT(machine_singleton == this);
    machine_singleton = nullptr;
    delete_map_contents(nodeinfos);
  }

  RuntimeImpl *MachineImpl::get_runtime_impl(void) const { return runtime_impl; }

#ifndef REALM_SKIP_INTERNODE_AFFINITIES
  static bool allows_internode_copies(Memory::Kind kind, bool &send_ok, bool &recv_ok,
                                      bool &is_reg)
  {
    send_ok = false;
    recv_ok = false;
    is_reg = false;
    switch(kind) {
      // these can be the source of a RemoteWrite, but it's non-ideal
    case Memory::SYSTEM_MEM:
    case Memory::Z_COPY_MEM:
    {
      send_ok = true;
      recv_ok = true;
      break;
    }

      // these can be the target of a RemoteWrite message, but
      //  not the source
    case Memory::GPU_FB_MEM:
    case Memory::DISK_MEM:
    case Memory::HDF_MEM:
    case Memory::FILE_MEM:
    {
      recv_ok = true;
      break;
    }

    case Memory::REGDMA_MEM:
    {
      send_ok = true;
      recv_ok = true;
      is_reg = true;
      break;
    }

    default:
      break;
    }

    return (send_ok || recv_ok);
  }
#endif

  void MachineImpl::parse_node_announce_data(int node_id, const void *args, size_t arglen,
                                             bool remote)
  {
    StripedRWLock::AutoWriterLock al(mutex);

    REALM_ASSERT(node_id <= Network::max_node_id);
    Node &n = runtime_impl->nodes[node_id];

    Serialization::FixedBufferDeserializer fbd(args, arglen);
    bool ok = true;
    while(ok && (fbd.bytes_left() > 0)) {
      NodeAnnounceTag tag;
      if(!(fbd >> tag)) {
        log_annc.fatal() << "unexpected end of input";
        abort();
      }

      switch(tag) {
      case NODE_ANNOUNCE_PROC:
      {
        Processor p;
        Processor::Kind kind;
        int num_cores;
        ok = (ok && (fbd >> p) && (fbd >> kind) && (fbd >> num_cores));
        if(ok) {
          REALM_ASSERT(NodeID(ID(p).proc_owner_node()) == node_id);
          log_annc.debug() << "adding proc " << p << " (kind = " << kind
                           << " num_cores = " << num_cores << ")";
          if(remote) {
            RemoteProcessor *proc = new RemoteProcessor(runtime_impl, p, kind, num_cores);
            if(n.processors.size() <= ID(p).proc_proc_idx())
              n.processors.resize(ID(p).proc_proc_idx() + 1, 0);
            n.processors[ID(p).proc_proc_idx()] = proc;
          }
        }
        break;
      }

      case NODE_ANNOUNCE_MEM:
      {
        Memory m;
        Memory::Kind kind;
        size_t size;
        bool has_rdma_info = false;
        ByteArray rdma_info;
        ok = (ok && (fbd >> m) && (fbd >> kind) && (fbd >> size) &&
              (fbd >> has_rdma_info));
        if(has_rdma_info)
          ok = ok && (fbd >> rdma_info);
        if(ok) {
          REALM_ASSERT(NodeID(ID(m).memory_owner_node()) == node_id);
          log_annc.debug() << "adding memory " << m << " (kind = " << kind
                           << ", size = " << size << ", has_rdma = " << has_rdma_info
                           << ")";
          if(remote) {
            MemoryImpl *mem;
            if(has_rdma_info) {
              mem = Network::get_network(node_id)->create_remote_memory(
                  runtime_impl, m, size, kind, rdma_info);
            } else {
              mem = new RemoteMemory(runtime_impl, m, size, kind);
            }
            if(n.memories.size() >= ID(m).memory_mem_idx())
              n.memories.resize(ID(m).memory_mem_idx() + 1, 0);
            n.memories[ID(m).memory_mem_idx()] = mem;

#ifndef REALM_SKIP_INTERNODE_AFFINITIES
            {
              // manufacture affinities for remote writes
              // acceptable local sources: SYSTEM, Z_COPY, REGDMA (bonus)
              // acceptable remote targets: SYSTEM, Z_COPY, GPU_FB, DISK, HDF, FILE,
              // REGDMA (bonus)
              bool rem_send, rem_recv, rem_reg;
              if(!allows_internode_copies(kind, rem_send, rem_recv, rem_reg))
                continue;
            }
#endif
          }
        }
        break;
      }

      case NODE_ANNOUNCE_IB_MEM:
      {
        Memory m;
        Memory::Kind kind;
        size_t size;
        bool has_rdma_info = false;
        ByteArray rdma_info;
        ok = (ok && (fbd >> m) && (fbd >> kind) && (fbd >> size) &&
              (fbd >> has_rdma_info));
        if(has_rdma_info)
          ok = ok && (fbd >> rdma_info);
        if(ok) {
          REALM_ASSERT(NodeID(ID(m).memory_owner_node()) == node_id);
          log_annc.debug() << "adding ib memory " << m << " (kind = " << kind
                           << ", size = " << size << ", has_rdma = " << has_rdma_info
                           << ")";
          if(remote) {
            IBMemory *ibmem;
            if(has_rdma_info) {
              ibmem = Network::get_network(node_id)->create_remote_ib_memory(
                  runtime_impl, m, size, kind, rdma_info);
            } else {
              ibmem = new IBMemory(runtime_impl, m, size, MemoryImpl::MKIND_REMOTE, kind,
                                   0, 0);
            }

            if(n.ib_memories.size() >= ID(m).memory_mem_idx())
              n.ib_memories.resize(ID(m).memory_mem_idx() + 1, 0);
            n.ib_memories[ID(m).memory_mem_idx()] = ibmem;
          }
        }
        break;
      }

      case NODE_ANNOUNCE_PMA:
      {
        Machine::ProcessorMemoryAffinity pma;
        ok = (ok && (fbd >> pma));
        if(ok) {
          log_annc.debug() << "adding affinity " << pma.p << " -> " << pma.m
                           << " (bw = " << pma.bandwidth << ", latency = " << pma.latency
                           << ")";

          add_proc_mem_affinity(pma, true /*lock held*/);
          // proc_mem_affinities.push_back(pma);
        }
        break;
      }

      case NODE_ANNOUNCE_DMA_CHANNEL:
      {
        RemoteChannelInfo *rci = RemoteChannelInfo::deserialize_new(fbd);
        if(rci) {
          size_t num_redops = 0;
          RemoteChannel *rc = rci->create_remote_channel();
          delete rci;

          fbd >> num_redops;
          for(size_t i = 0; i < num_redops; i++) {
            ReductionOpID redop_id = 0;
            fbd >> redop_id;
            rc->register_redop(redop_id);
          }

          log_annc.debug() << "adding channel: " << *rc;
          REALM_ASSERT(rc->node == node_id);
          if(remote)
            runtime_impl->add_dma_channel(rc);
          else
            delete rc; // don't actually need it
        }
        break;
      }

      case NODE_ANNOUNCE_PROCESS_INFO:
      {
        Machine::ProcessInfo process_info;
        ok = (ok && (fbd >> process_info));
        if(ok) {
          add_process_info(node_id, process_info, true);
        }
        break;
      }

      default:
        log_annc.fatal() << "unknown tag: " << tag;
        abort();
      }
    }

    REALM_ASSERT(ok && (fbd.bytes_left() == 0));
  }

  void MachineImpl::get_all_memories(std::set<Memory> &mset) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin();
        it != proc_mem_affinities.end(); it++) {
      mset.insert((*it).m);
    }
#else
    for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
        it != nodeinfos.end(); ++it)
      for(std::map<Memory, MachineMemInfo *>::const_iterator it2 =
              it->second->mems.begin();
          it2 != it->second->mems.end(); ++it2)
        mset.insert(it2->first);
#endif
  }

  void MachineImpl::get_all_processors(std::set<Processor> &pset) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin();
        it != proc_mem_affinities.end(); it++) {
      pset.insert((*it).p);
    }
#else
    for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
        it != nodeinfos.end(); ++it)
      for(std::map<Processor, MachineProcInfo *>::const_iterator it2 =
              it->second->procs.begin();
          it2 != it->second->procs.end(); ++it2)
        pset.insert(it2->first);
#endif
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(int node) const
  {
    std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.find(node);
    if(it != nodeinfos.end())
      return it->second;
    else
      return nullptr;
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(Processor p) const
  {
    return get_nodeinfo(ID(p).proc_owner_node());
  }

  inline MachineNodeInfo *MachineImpl::get_nodeinfo(Memory m) const
  {
    return get_nodeinfo(ID(m).memory_owner_node());
  }

  void MachineImpl::get_local_processors(std::set<Processor> &pset) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin();
        it != proc_mem_affinities.end(); it++) {
      Processor p = (*it).p;
      if(ID(p).proc_owner_node() == Network::my_node_id)
        pset.insert(p);
    }
#else
    const MachineNodeInfo *mynode = get_nodeinfo(Network::my_node_id);
    REALM_ASSERT(mynode != nullptr);
    for(std::map<Processor, MachineProcInfo *>::const_iterator it = mynode->procs.begin();
        it != mynode->procs.end(); ++it)
      pset.insert(it->first);
#endif
  }

  void MachineImpl::get_local_processors_by_kind(std::set<Processor> &pset,
                                                 Processor::Kind kind) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin();
        it != proc_mem_affinities.end(); it++) {
      Processor p = (*it).p;
      Processor::Kind p_kind =
          ProcessorImpl::get_processor_kind(machine->get_runtime_impl(), p);
      if((ID(p).proc_owner_node() == Network::my_node_id) && (p_kind == kind))
        pset.insert(p);
    }
#else
    const MachineNodeInfo *mynode = get_nodeinfo(Network::my_node_id);
    REALM_ASSERT(mynode != nullptr);
    std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::const_iterator it =
        mynode->proc_by_kind.find(kind);
    if(it != mynode->proc_by_kind.end())
      for(std::map<Processor, MachineProcInfo *>::const_iterator it2 = it->second.begin();
          it2 != it->second.end(); ++it2)
        pset.insert(it2->first);
#endif
  }

  // Return the set of memories visible from a processor
  void MachineImpl::get_visible_memories(Processor p, std::set<Memory> &mset,
                                         bool local_only) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin();
        it != proc_mem_affinities.end(); it++) {
      size_t m_capacity = MemoryImpl::get_memory_size(runtime_impl, (*it).m);
      if((*it).p == p && m_capacity > 0 && (!local_only || is_local_affinity(*it)))
        mset.insert((*it).m);
    }
#else
    const MachineNodeInfo *ni = get_nodeinfo(p);
    if(!ni)
      return;
    std::map<Processor, MachineProcInfo *>::const_iterator it = ni->procs.find(p);
    if(it == ni->procs.end())
      return;
    const std::map<Memory, Machine::ProcessorMemoryAffinity *> &pmas =
        (local_only ? it->second->pmas.local : it->second->pmas.all);
    for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 =
            pmas.begin();
        it2 != pmas.end(); ++it2)
      mset.insert(it2->first);
#endif
  }

  // Return the set of memories visible from a memory
  void MachineImpl::get_visible_memories(Memory m, std::set<Memory> &mset,
                                         bool local_only) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::MemoryMemoryAffinity>::const_iterator it =
            mem_mem_affinities.begin();
        it != mem_mem_affinities.end(); it++) {
      if(local_only && !is_local_affinity(*it))
        continue;

      size_t m2_capacity = MemoryImpl::get_memory_size(runtime_impl, (*it).m2);
      if((*it).m1 == m && m2_capacity > 0)
        mset.insert((*it).m2);

      size_t m1_capacity = MemoryImpl::get_memory_size(runtime_impl, (*it).m1);
      if((*it).m2 == m && m1_capacity > 0)
        mset.insert((*it).m1);
    }
#else
    const MachineNodeInfo *ni = get_nodeinfo(m);
    if(!ni)
      return;
    std::map<Memory, MachineMemInfo *>::const_iterator it = ni->mems.find(m);
    if(it == ni->mems.end())
      return;
    // handle both directions for now
    {
      const std::map<Memory, Machine::MemoryMemoryAffinity *> &mmas =
          (local_only ? it->second->mmas_out.local : it->second->mmas_out.all);
      for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
              mmas.begin();
          it2 != mmas.end(); ++it2)
        mset.insert(it2->first);
    }
    {
      const std::map<Memory, Machine::MemoryMemoryAffinity *> &mmas =
          (local_only ? it->second->mmas_in.local : it->second->mmas_in.all);
      for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
              mmas.begin();
          it2 != mmas.end(); ++it2)
        mset.insert(it2->first);
    }
#endif
  }

  // Return the set of processors which can all see a given memory
  void MachineImpl::get_shared_processors(Memory m, std::set<Processor> &pset,
                                          bool local_only) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
#ifdef USE_OLD_AFFINITIES
    for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
            proc_mem_affinities.begin();
        it != proc_mem_affinities.end(); it++) {
      if(((*it).m == m) && (!local_only || is_local_affinity(*it)))
        pset.insert((*it).p);
    }
#else
    const MachineNodeInfo *ni = get_nodeinfo(m);
    if(!ni)
      return;
    std::map<Memory, MachineMemInfo *>::const_iterator it = ni->mems.find(m);
    if(it == ni->mems.end())
      return;
    const std::map<Processor, Machine::ProcessorMemoryAffinity *> &pmas =
        (local_only ? it->second->pmas.local : it->second->pmas.all);
    for(std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator it2 =
            pmas.begin();
        it2 != pmas.end(); ++it2)
      pset.insert(it2->first);
#endif
  }

  bool MachineImpl::get_process_info(Processor p, Machine::ProcessInfo *info) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
    MachineNodeInfo *node_info = get_nodeinfo(p);
    if(!node_info) {
      return false;
    }
    if(!node_info->process_info) {
      log_machine.warning("Process info is not set");
      return false;
    }
    *info = *(node_info->process_info);
    return true;
  }

  bool MachineImpl::has_affinity(Processor p, Memory m,
                                 Machine::AffinityDetails *details /*= 0*/) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
    // indexed lookup: node -> processor -> memory
    const MachineNodeInfo *ni = get_nodeinfo(p);
    if(!ni)
      return false;
    std::map<Processor, MachineProcInfo *>::const_iterator it = ni->procs.find(p);
    if(it == ni->procs.end())
      return false;
    std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 =
        it->second->pmas.all.find(m);
    if(it2 == it->second->pmas.all.end())
      return false;
    if(details) {
      details->bandwidth = it2->second->bandwidth;
      details->latency = it2->second->latency;
    }
    return true;
  }

  bool MachineImpl::has_affinity(Memory m1, Memory m2,
                                 Machine::AffinityDetails *details /*= 0*/) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
    // indexed lookup: node -> memory -> memory (must not insert into the maps)
    const MachineNodeInfo *ni = get_nodeinfo(m1);
    if(!ni)
      return false;
    std::map<Memory, MachineMemInfo *>::const_iterator it = ni->mems.find(m1);
    if(it == ni->mems.end())
      return false;
    std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
        it->second->mmas_out.all.find(m2);
    if(it2 == it->second->mmas_out.all.end())
      return false;
    if(details) {
      details->bandwidth = it2->second->bandwidth;
      details->latency = it2->second->latency;
    }
    return true;
  }

  int MachineImpl::get_proc_mem_affinity(
      std::vector<Machine::ProcessorMemoryAffinity> &result,
      Processor restrict_proc /*= Processor::NO_PROC*/,
      Memory restrict_memory /*= Memory::NO_MEMORY*/, bool local_only /*= true*/) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
    return get_proc_mem_affinity_unlocked(result, restrict_proc, restrict_memory,
                                          local_only);
  }

  int MachineImpl::get_proc_mem_affinity_unlocked(
      std::vector<Machine::ProcessorMemoryAffinity> &result, Processor restrict_proc,
      Memory restrict_memory, bool local_only) const
  {
    int count = 0;
    result.clear();

    {
#ifdef USE_OLD_AFFINITIES
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              proc_mem_affinities.begin();
          it != proc_mem_affinities.end(); it++) {
        if(restrict_proc.exists() && ((*it).p != restrict_proc))
          continue;
        if(restrict_memory.exists() && ((*it).m != restrict_memory))
          continue;
        if(local_only && !is_local_affinity(*it))
          continue;
        result.push_back(*it);
        count++;
      }
#else
      if(restrict_proc.exists()) {
        const MachineNodeInfo *np = get_nodeinfo(restrict_proc);
        if(!np)
          return 0;
        std::map<Processor, MachineProcInfo *>::const_iterator it =
            np->procs.find(restrict_proc);
        if(it == np->procs.end())
          return 0;
        const MachineProcInfo *mpi = it->second;
        const std::map<Memory, Machine::ProcessorMemoryAffinity *> &pmas =
            (local_only ? mpi->pmas.local : mpi->pmas.all);

        if(restrict_memory.exists()) {
          std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 =
              pmas.find(restrict_memory);
          if(it2 != pmas.end()) {
            result.push_back(*(it2->second));
            count++;
          }
        } else {
          for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it2 =
                  pmas.begin();
              it2 != pmas.end(); ++it2) {
            result.push_back(*(it2->second));
            count++;
          }
        }
      } else {
        if(restrict_memory.exists()) {
          const MachineNodeInfo *nm = get_nodeinfo(restrict_memory);
          if(!nm)
            return 0;
          std::map<Memory, MachineMemInfo *>::const_iterator it =
              nm->mems.find(restrict_memory);
          if(it == nm->mems.end())
            return 0;
          const MachineMemInfo *mmi = it->second;
          const std::map<Processor, Machine::ProcessorMemoryAffinity *> &pmas =
              (local_only ? mmi->pmas.local : mmi->pmas.all);

          for(std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator
                  it2 = pmas.begin();
              it2 != pmas.end(); ++it2) {
            result.push_back(*(it2->second));
            count++;
          }
        } else {
          // lookup of every single affinity - blech
          for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
              it != nodeinfos.end(); ++it)
            for(std::map<Processor, MachineProcInfo *>::const_iterator it2 =
                    it->second->procs.begin();
                it2 != it->second->procs.end(); ++it2) {
              const MachineProcInfo *mpi = it2->second;
              const std::map<Memory, Machine::ProcessorMemoryAffinity *> &pmas =
                  (local_only ? mpi->pmas.local : mpi->pmas.all);

              for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator
                      it2 = pmas.begin();
                  it2 != pmas.end(); ++it2) {
                result.push_back(*(it2->second));
                count++;
              }
            }
        }
      }
#endif
    }

    return count;
  }

  int MachineImpl::get_mem_mem_affinity(
      std::vector<Machine::MemoryMemoryAffinity> &result,
      Memory restrict_mem1 /*= Memory::NO_MEMORY*/,
      Memory restrict_mem2 /*= Memory::NO_MEMORY*/, bool local_only /*= true*/) const
  {
    StripedRWLock::AutoReaderLock al(mutex);
    return get_mem_mem_affinity_unlocked(result, restrict_mem1, restrict_mem2,
                                         local_only);
  }

  int MachineImpl::get_mem_mem_affinity_unlocked(
      std::vector<Machine::MemoryMemoryAffinity> &result, Memory restrict_mem1,
      Memory restrict_mem2, bool local_only) const
  {
    // Clear the vector in case it has old data
    result.clear();

    int count = 0;

    {
      if(restrict_mem1.exists()) {
        const MachineNodeInfo *nm1 = get_nodeinfo(restrict_mem1);
        if(!nm1)
          return 0;
        std::map<Memory, MachineMemInfo *>::const_iterator it =
            nm1->mems.find(restrict_mem1);
        if(it == nm1->mems.end())
          return 0;
        const MachineMemInfo *mmi = it->second;
        const std::map<Memory, Machine::MemoryMemoryAffinity *> &mmas =
            (local_only ? mmi->mmas_out.local : mmi->mmas_out.all);

        if(restrict_mem2.exists()) {
          std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
              mmas.find(restrict_mem2);
          if(it2 != mmas.end()) {
            result.push_back(*(it2->second));
            count++;
          }
        } else {
          for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
                  mmas.begin();
              it2 != mmas.end(); ++it2) {
            result.push_back(*(it2->second));
            count++;
          }
        }
      } else {
        if(restrict_mem2.exists()) {
          const MachineNodeInfo *nm2 = get_nodeinfo(restrict_mem2);
          if(!nm2)
            return 0;
          std::map<Memory, MachineMemInfo *>::const_iterator it =
              nm2->mems.find(restrict_mem2);
          if(it == nm2->mems.end())
            return 0;
          const MachineMemInfo *mmi = it->second;
          const std::map<Memory, Machine::MemoryMemoryAffinity *> &mmas =
              (local_only ? mmi->mmas_in.local : mmi->mmas_in.all);

          for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
                  mmas.begin();
              it2 != mmas.end(); ++it2) {
            result.push_back(*(it2->second));
            count++;
          }
        } else {
          // lookup of every single affinity - blech
          for(std::map<NodeID, MachineNodeInfo *>::const_iterator it = nodeinfos.begin();
              it != nodeinfos.end(); ++it)
            for(std::map<Memory, MachineMemInfo *>::const_iterator it2 =
                    it->second->mems.begin();
                it2 != it->second->mems.end(); ++it2) {
              const MachineMemInfo *mmi = it2->second;
              const std::map<Memory, Machine::MemoryMemoryAffinity *> &mmas =
                  (local_only ? mmi->mmas_out.local : mmi->mmas_out.all);

              for(std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it2 =
                      mmas.begin();
                  it2 != mmas.end(); ++it2) {
                result.push_back(*(it2->second));
                count++;
              }
            }
        }
      }
    }

    return count;
  }

  void MachineImpl::add_proc_mem_affinity(const Machine::ProcessorMemoryAffinity &pma,
                                          bool lock_held /*= false*/)
  {
    if(!lock_held)
      mutex.wrlock();

    proc_mem_affinities.push_back(pma);

    int np = ID(pma.p).proc_owner_node();
    int mp = ID(pma.m).memory_owner_node();
    {
      MachineNodeInfo *&ptr = nodeinfos[np];
      if(!ptr)
        ptr = new MachineNodeInfo(np, runtime_impl);
      ptr->add_processor(pma.p);
      if(np == mp)
        ptr->add_memory(pma.m);
      ptr->add_proc_mem_affinity(pma);
    }
    if(np != mp) {
      MachineNodeInfo *&ptr = nodeinfos[mp];
      if(!ptr)
        ptr = new MachineNodeInfo(mp, runtime_impl);
      ptr->add_memory(pma.m);
      ptr->add_proc_mem_affinity(pma);
    }
    invalidate_query_caches();
    if(!lock_held)
      mutex.wrunlock();
  }

  void MachineImpl::add_subscription(Machine::MachineUpdateSubscriber *subscriber)
  {
    StripedRWLock::AutoWriterLock al(mutex);
    subscribers.insert(subscriber);
  }

  void MachineImpl::remove_subscription(Machine::MachineUpdateSubscriber *subscriber)
  {
    StripedRWLock::AutoWriterLock al(mutex);
    subscribers.erase(subscriber);
  }

  static bool get_memcpy_affinity(const RuntimeImpl *runtime_impl,
                                  Machine::MemoryMemoryAffinity &affinity,
                                  Realm::Memory src_mem, Realm::Memory dst_mem)
  {
    bool found_channel = false;
    affinity.m1 = src_mem;
    affinity.m2 = dst_mem;
    affinity.bandwidth = 0;
    affinity.latency = UINT_MAX;
    Realm::Node &src_node = runtime_impl->nodes[ID(src_mem).memory_owner_node()];
    // Find the best channel to satisfy a basic copy and track it's bandwidth
    // and latency.
    for(Channel *channel : src_node.dma_channels) {
      unsigned bandwidth, latency;
      if(channel->supports_path(ChannelCopyInfo{src_mem, dst_mem}, 0, 0, 0, 1, nullptr,
                                nullptr, nullptr, &bandwidth, &latency) != 0) {
        affinity.bandwidth = std::max(bandwidth, affinity.bandwidth);
        affinity.latency = std::min(latency, affinity.latency);
        found_channel = true;
      }
    }
    return found_channel;
  }

  void MachineImpl::update_kind_maps(void)
  {
    StripedRWLock::AutoWriterLock al(mutex);
    for(std::pair<const int, Realm::MachineNodeInfo *> &node_info : nodeinfos) {
      node_info.second->update_kind_maps();
    }
    invalidate_query_caches();
  }

  void MachineImpl::enumerate_mem_mem_affinities(void)
  {
    StripedRWLock::AutoWriterLock al(mutex);
    // This only enumerates the local-local, local-remote, and remote-local
    // memory affinities to save on cache space
    Realm::MachineNodeInfo *src_nodeInfo = nodeinfos[Network::my_node_id];
    if(src_nodeInfo == nullptr) {
      src_nodeInfo = new Realm::MachineNodeInfo(Network::my_node_id, runtime_impl);
      nodeinfos[Network::my_node_id] = src_nodeInfo;
    }
    for(MemoryImpl *src_mem : runtime_impl->nodes[Network::my_node_id].memories) {
      src_nodeInfo->add_memory(src_mem->me);
      Realm::MachineMemInfo *src_memInfo = src_nodeInfo->mems[src_mem->me];
      for(Realm::NodeID dst_idx = 0; dst_idx <= Network::max_node_id; dst_idx++) {
        Realm::MachineNodeInfo *dst_nodeInfo = nodeinfos[dst_idx];
        if(dst_nodeInfo == nullptr) {
          dst_nodeInfo = new Realm::MachineNodeInfo(dst_idx, runtime_impl);
          nodeinfos[dst_idx] = dst_nodeInfo;
        }
        for(MemoryImpl *dst_mem : runtime_impl->nodes[dst_idx].memories) {
          Machine::MemoryMemoryAffinity affinity;
          dst_nodeInfo->add_memory(dst_mem->me);
          Realm::MachineMemInfo *dst_memInfo = dst_nodeInfo->mems[dst_mem->me];
          // Add the in and out edges for both the source and destination meminfos
          if(get_memcpy_affinity(runtime_impl, affinity, src_mem->me, dst_mem->me)) {
            src_memInfo->add_mem_mem_affinity(affinity);
            dst_memInfo->add_mem_mem_affinity(affinity);
          }
          if(get_memcpy_affinity(runtime_impl, affinity, dst_mem->me, src_mem->me)) {
            src_memInfo->add_mem_mem_affinity(affinity);
            dst_memInfo->add_mem_mem_affinity(affinity);
          }
        }
      }
    }
    invalidate_query_caches();
  }

  void MachineImpl::add_process_info(int node_id,
                                     const Machine::ProcessInfo &process_info,
                                     bool lock_held /*= false*/)
  {
    if(!lock_held) {
      mutex.wrlock();
    }
    MachineNodeInfo *&ptr = nodeinfos[node_id];
    if(!ptr) {
      ptr = new MachineNodeInfo(node_id, runtime_impl);
    }
    if(!ptr->add_process_info(process_info)) {
      log_machine.fatal("Failed to add process info because it already exists");
      abort();
    }
    if(!lock_held) {
      mutex.wrunlock();
    }
  }

  void MachineImpl::invalidate_query_caches()
  {
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    // bump the generations: every published snapshot and every list a query
    // is attached to becomes stale, and the next query that can use the cache
    // rebuilds it. Nothing to lock - readers only ever compare generations.
    MemoryQueryImpl::cache_invalid_count.fetch_add(1);
    ProcessorQueryImpl::cache_invalid_count.fetch_add(1);
#endif
  }

  void cleanup_query_caches()
  {
    // runtime shutdown: the snapshots go away, so no query may be executed
    //  after this (a query object may still be destroyed - that does not
    //  touch the list it is attached to)
    MemoryQueryImpl::kind_cache.cleanup();
    ProcessorQueryImpl::kind_cache.cleanup();
    ProcessorQueryImpl::affinity_cache.cleanup();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class QueryCacheSlot<S>
  //

  template <typename S>
  const S *QueryCacheSlot<S>::install(S *built, unsigned live_generation)
  {
    AutoLock<Mutex> al(mutex);
    const S *cur = current.load();
    if(cur && (cur->generation == built->generation)) {
      delete built; // somebody beat us to it - nobody has seen ours
      return cur;
    }
    if(built->generation != live_generation) {
      delete built; // the machine changed while this was being built
      return nullptr;
    }
    // 'cur' (if any) is older, but queries may still point into it
    if(cur)
      retired.push_back(cur);
    current.store_release(built);
    return built;
  }

  template <typename S>
  void QueryCacheSlot<S>::cleanup(void)
  {
    AutoLock<Mutex> al(mutex);
    const S *cur = current.exchange(nullptr);
    delete cur;
    for(const S *s : retired)
      delete s;
    retired.clear();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine::ProcessorQuery
  //

  Machine::ProcessorQuery::ProcessorQuery(const Machine &m)
  {
    impl = new ProcessorQueryImpl(m);
  }

  Machine::ProcessorQuery::ProcessorQuery(const ProcessorQuery &q)
  {
    ((ProcessorQueryImpl *)(q.impl))->add_reference();
    impl = q.impl;
  }

  Machine::ProcessorQuery::~ProcessorQuery(void)
  {
    ((ProcessorQueryImpl *)impl)->remove_reference();
  }

  Machine::ProcessorQuery &Machine::ProcessorQuery::operator=(const ProcessorQuery &q)
  {
    if(impl != q.impl) {
      ((ProcessorQueryImpl *)impl)->remove_reference();
      ((ProcessorQueryImpl *)(q.impl))->add_reference();
      impl = q.impl;
    }
    return *this;
  }

  Machine::ProcessorQuery &Machine::ProcessorQuery::only_kind(Processor::Kind kind)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_kind(kind);
    return *this;
  }

  Machine::ProcessorQuery &Machine::ProcessorQuery::local_address_space(void)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(Network::my_node_id);
    return *this;
  }

  Machine::ProcessorQuery &Machine::ProcessorQuery::same_address_space_as(Processor p)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(ID(p).proc_owner_node());
    return *this;
  }

  Machine::ProcessorQuery &Machine::ProcessorQuery::same_address_space_as(Memory m)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ((ProcessorQueryImpl *)impl)->restrict_to_node(ID(m).proc_owner_node());
    return *this;
  }

  Machine::ProcessorQuery &
  Machine::ProcessorQuery::has_affinity_to(Memory m, unsigned min_bandwidth /*= 0*/,
                                           unsigned max_latency /*= 0*/)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    // we have a cached version of the map, record it
    ((ProcessorQueryImpl *)impl)
        ->add_predicate(new ProcessorHasAffinityPredicate(m, min_bandwidth, max_latency));
    // query type can use one of the cached maps
    if(!min_bandwidth && !max_latency)
      ((ProcessorQueryImpl *)impl)->set_cached_mem(m);
    else
      ((ProcessorQueryImpl *)impl)->reset_cached_mem();
    return *this;
  }

  Machine::ProcessorQuery &
  Machine::ProcessorQuery::best_affinity_to(Memory m, int bandwidth_weight /*= 1*/,
                                            int latency_weight /*= 0*/)
  {
    impl = ((ProcessorQueryImpl *)impl)->writeable_reference();
    ProcessorQueryImpl *impl_ptr = ((ProcessorQueryImpl *)impl);

    // Warn if overwriting a different target
    if(impl_ptr->best_affinity_cost.has_value() &&
       impl_ptr->best_affinity_cost->memory != m) {
      log_query.warning() << "ProcessorQuery: best_affinity_to() called with different "
                          << "memory targets - using most recent (Memory " << m.id << ")";
    }

    // Store the best affinity cost function
    impl_ptr->best_affinity_cost =
        ProcessorQueryImpl::BestAffinityCostFn{m, bandwidth_weight, latency_weight};

    impl_ptr->reset_cached_mem();
    // any result computed so far (e.g. the plain by-kind list) no longer
    //  describes this query
    impl_ptr->reset_cached_list();
    return *this;
  }

  size_t Machine::ProcessorQuery::count(void) const
  {
    return ((ProcessorQueryImpl *)impl)->count_matches();
  }

  Processor Machine::ProcessorQuery::first(void) const
  {
    return ((ProcessorQueryImpl *)impl)->first_match();
  }

  Processor Machine::ProcessorQuery::next(Processor after) const
  {
    if(Config::use_machine_query_cache)
      return ((ProcessorQueryImpl *)impl)->cache_next(after);
    return ((ProcessorQueryImpl *)impl)->next_match(after);
  }

  Processor Machine::ProcessorQuery::random(void) const
  {
    return ((ProcessorQueryImpl *)impl)->random_match();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Machine::MemoryQuery
  //

  Machine::MemoryQuery::MemoryQuery(const Machine &m) { impl = new MemoryQueryImpl(m); }

  Machine::MemoryQuery::MemoryQuery(const MemoryQuery &q)
  {
    ((MemoryQueryImpl *)(q.impl))->add_reference();
    impl = q.impl;
  }

  Machine::MemoryQuery::~MemoryQuery(void)
  {
    ((MemoryQueryImpl *)impl)->remove_reference();
  }

  Machine::MemoryQuery &Machine::MemoryQuery::operator=(const MemoryQuery &q)
  {
    if(impl != q.impl) {
      ((MemoryQueryImpl *)impl)->remove_reference();
      ((MemoryQueryImpl *)(q.impl))->add_reference();
      impl = q.impl;
    }
    return *this;
  }

  Machine::MemoryQuery &Machine::MemoryQuery::only_kind(Memory::Kind kind)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_kind(kind);
    return *this;
  }

  Machine::MemoryQuery &Machine::MemoryQuery::local_address_space(void)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(Network::my_node_id);
    return *this;
  }

  Machine::MemoryQuery &Machine::MemoryQuery::same_address_space_as(Processor p)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(ID(p).proc_owner_node());
    return *this;
  }

  Machine::MemoryQuery &Machine::MemoryQuery::same_address_space_as(Memory m)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_to_node(ID(m).memory_owner_node());
    return *this;
  }

  Machine::MemoryQuery &
  Machine::MemoryQuery::has_affinity_to(Memory m, unsigned min_bandwidth /*= 0*/,
                                        unsigned max_latency /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)
        ->add_predicate(new MemoryHasMemAffinityPredicate(m, min_bandwidth, max_latency));
    return *this;
  }

  Machine::MemoryQuery &
  Machine::MemoryQuery::best_affinity_to(Memory m, int bandwidth_weight /*= 1*/,
                                         int latency_weight /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    MemoryQueryImpl *impl_ptr = ((MemoryQueryImpl *)impl);

    // Warn if overwriting a different target
    if(impl_ptr->best_mem_affinity_cost.has_value() &&
       impl_ptr->best_mem_affinity_cost->memory != m) {
      log_query.warning()
          << "MemoryQuery: best_affinity_to(Memory) called with different "
          << "memory targets - using most recent (Memory " << m.id << ")";
    }

    // Store the best memory affinity cost function
    impl_ptr->best_mem_affinity_cost =
        MemoryQueryImpl::BestMemAffinityCostFn{m, bandwidth_weight, latency_weight};

    // any result computed so far no longer describes this query
    impl_ptr->reset_cached_list();
    return *this;
  }

  Machine::MemoryQuery &
  Machine::MemoryQuery::has_affinity_to(Processor p, unsigned min_bandwidth /*= 0*/,
                                        unsigned max_latency /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)
        ->add_predicate(
            new MemoryHasProcAffinityPredicate(p, min_bandwidth, max_latency));
    ((MemoryQueryImpl *)impl)->set_affinity_proc(p);
    return *this;
  }

  Machine::MemoryQuery &
  Machine::MemoryQuery::best_affinity_to(Processor p, int bandwidth_weight /*= 1*/,
                                         int latency_weight /*= 0*/)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    MemoryQueryImpl *impl_ptr = ((MemoryQueryImpl *)impl);

    // Warn if overwriting a different target
    if(impl_ptr->best_proc_affinity_cost.has_value() &&
       impl_ptr->best_proc_affinity_cost->proc != p) {
      log_query.warning()
          << "MemoryQuery: best_affinity_to(Processor) called with different "
          << "processor targets - using most recent (Processor " << p.id << ")";
    }

    // Store the best processor affinity cost function
    impl_ptr->best_proc_affinity_cost =
        MemoryQueryImpl::BestProcAffinityCostFn{p, bandwidth_weight, latency_weight};

    // any result computed so far no longer describes this query
    impl_ptr->reset_cached_list();
    return *this;
  }

  Machine::MemoryQuery &Machine::MemoryQuery::has_capacity(size_t min_bytes)
  {
    impl = ((MemoryQueryImpl *)impl)->writeable_reference();
    ((MemoryQueryImpl *)impl)->restrict_by_capacity(min_bytes);
    return *this;
  }

  size_t Machine::MemoryQuery::count(void) const
  {
    return ((MemoryQueryImpl *)impl)->count_matches();
  }

  Memory Machine::MemoryQuery::first(void) const
  {
    return ((MemoryQueryImpl *)impl)->first_match();
  }

  Memory Machine::MemoryQuery::next(Memory after) const
  {
    if(Config::use_machine_query_cache)
      return ((MemoryQueryImpl *)impl)->cache_next(after);
    return ((MemoryQueryImpl *)impl)->next_match(after);
  }

  Memory Machine::MemoryQuery::random(void) const
  {
    return ((MemoryQueryImpl *)impl)->random_match();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorHasAffinityPredicate
  //

  ProcessorHasAffinityPredicate::ProcessorHasAffinityPredicate(Memory _memory,
                                                               unsigned _min_bandwidth,
                                                               unsigned _max_latency)
    : memory(_memory)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  ProcQueryPredicate *ProcessorHasAffinityPredicate::clone(void) const
  {
    return new ProcessorHasAffinityPredicate(memory, min_bandwidth, max_latency);
  }

  bool ProcessorHasAffinityPredicate::matches_predicate(const MachineImpl *machine,
                                                        Processor thing,
                                                        const MachineProcInfo *info) const
  {
#ifdef USE_OLD_AFFINITIES
    Machine::AffinityDetails details;
    if(!machine->has_affinity(thing, memory, &details))
      return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth))
      return false;
    if((max_latency != 0) && (details.latency > max_latency))
      return false;
    return true;
#else
    REALM_ASSERT(info != 0);
    std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it =
        info->pmas.all.find(memory);
    if(it == info->pmas.all.end())
      return false;
    if((min_bandwidth != 0) && (it->second->bandwidth < min_bandwidth))
      return false;
    if((max_latency != 0) && (it->second->latency > max_latency))
      return false;
    return true;
#endif
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorBestAffinityPredicate
  //

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcessorQueryImpl
  //
  atomic<unsigned int> ProcessorQueryImpl::cache_invalid_count(0);
  QueryCacheSlot<ProcessorQueryImpl::KindCache> ProcessorQueryImpl::kind_cache;
  QueryCacheSlot<ProcessorQueryImpl::AffinityCache> ProcessorQueryImpl::affinity_cache;

  ProcessorQueryImpl::ProcessorQueryImpl(const Machine &_machine)
    : ProcessorQueryImpl(static_cast<const MachineImpl *>(_machine.impl))
  {}

  ProcessorQueryImpl::ProcessorQueryImpl(const MachineImpl *_machine_impl)
    : references(1)
    , machine(_machine_impl)
    , is_restricted_node(false)
    , is_restricted_kind(false)
    , cached_mem(Memory::NO_MEMORY)
    , is_cached_mem(false)
    , attached(nullptr)
  {}

  // a copy is only ever made to refine a shared query (writeable_reference),
  //  and every refinement drops the attached list, so it is not copied: the
  //  source's best-affinity lists stay with the source
  ProcessorQueryImpl::ProcessorQueryImpl(const ProcessorQueryImpl &copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted_node(copy_from.is_restricted_node)
    , restricted_node_id(copy_from.restricted_node_id)
    , is_restricted_kind(copy_from.is_restricted_kind)
    , restricted_kind(copy_from.restricted_kind)
    , best_affinity_cost(copy_from.best_affinity_cost)
    , cached_mem(copy_from.cached_mem)
    , is_cached_mem(copy_from.is_cached_mem)
    , attached(nullptr)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<ProcQueryPredicate *>::const_iterator it =
            copy_from.predicates.begin();
        it != copy_from.predicates.end(); it++)
      predicates.push_back((*it)->clone());
  }

  ProcessorQueryImpl::~ProcessorQueryImpl(void)
  {
    REALM_ASSERT(references.load() == 0);
    for(std::vector<ProcQueryPredicate *>::iterator it = predicates.begin();
        it != predicates.end(); it++)
      delete *it;
  }

  void ProcessorQueryImpl::add_reference(void) { references.fetch_add(1); }

  void ProcessorQueryImpl::remove_reference(void)
  {
    // acquire/release: another holder's last use of this object must be
    //  complete before it is deleted (or, below, modified in place)
    int left = references.fetch_sub_acqrel(1) - 1;
    if(left == 0)
      delete this;
  }

  ProcessorQueryImpl *ProcessorQueryImpl::writeable_reference(void)
  {
    // we hold a reference, so a count of 1 means there are no others
    if(references.load_acquire() == 1) {
      return this;
    } else {
      ProcessorQueryImpl *copy = new ProcessorQueryImpl(*this);
      remove_reference();
      return copy;
    }
  }

  void ProcessorQueryImpl::restrict_to_node(int new_node_id)
  {
    // attempts to restrict to two different nodes results in no possible match
    if(is_restricted_node && (new_node_id != restricted_node_id)) {
      restricted_node_id = -1;
    } else {
      is_restricted_node = true;
      restricted_node_id = new_node_id;
    }
    // any constraint to processor query will invalidate cached list
    reset_cached_list();
  }

  void ProcessorQueryImpl::restrict_to_kind(Processor::Kind new_kind)
  {
    // attempts to restrict to two different kind results in no possible match
    // (use node restriction to enforce this)
    if(is_restricted_kind && (new_kind != restricted_kind)) {
      is_restricted_node = true;
      restricted_node_id = -1;
    } else {
      is_restricted_kind = true;
      restricted_kind = new_kind;
    }
    reset_cached_list();
  }

  void ProcessorQueryImpl::add_predicate(ProcQueryPredicate *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
    reset_cached_list();
  }

  /*static*/ Processor ProcessorQueryImpl::next(const std::vector<Processor> &list,
                                                Processor after)
  {
    // cached lists are sorted by id, so the successor is a binary search
    std::vector<Processor>::const_iterator it =
        std::upper_bound(list.begin(), list.end(), after);
    return (it != list.end()) ? *it : Processor::NO_PROC;
  }

  const ProcessorQueryImpl::CachedList *
  ProcessorQueryImpl::attach(const CachedList *list) const
  {
    attached.store_release(list);
    return list;
  }

  const ProcessorQueryImpl::CachedList *ProcessorQueryImpl::attached_list() const
  {
    const CachedList *list = attached.load_acquire();
    if(list && (list->generation == cache_invalid_count.load()))
      return list;
    return nullptr;
  }

  void ProcessorQueryImpl::reset_cached_list()
  {
    // the caller holds the only reference to this query (see
    //  writeable_reference), so nobody is reading this concurrently
    attached.store(nullptr);
  }

  // Returns the shared list that can serve this query, or null when the
  // shared cache does not cover it: node-restricted queries, best-affinity
  // queries, and predicated queries other than the single memory-affinity
  // predicate walk the machine directly. Reads nothing but the current
  // snapshot unless that is stale, in which case a new one is built (under
  // the machine's reader lock) and published.
  const ProcessorQueryImpl::CachedList *ProcessorQueryImpl::cached_list() const
  {
    const CachedList *list = attached_list();
    if(list)
      return list;
    if(!is_restricted_kind || is_restricted_node || best_affinity_cost.has_value() ||
       (!predicates.empty() && !is_cached_mem) ||
       (unsigned(restricted_kind) >= NUM_PROCESSOR_KINDS))
      return nullptr;

    // the generation is read before the snapshot: a snapshot stamped with it
    //  describes the machine as of this generation or later
    const unsigned generation = cache_invalid_count.load();
    if(!is_cached_mem) {
      const KindCache *kc = kind_cache.get(generation);
      if(!kc)
        kc = build_kind_cache(generation);
      if(!kc)
        return nullptr; // the machine changed underneath us; walk it instead
      return attach(&kc->by_kind[restricted_kind]);
    } else {
      const AffinityCache *ac = affinity_cache.get(generation);
      if(!ac)
        ac = build_affinity_cache(generation);
      if(!ac)
        return nullptr;
      const std::map<Memory, CachedList> &by_mem = ac->by_kind_and_mem[restricted_kind];
      std::map<Memory, CachedList>::const_iterator it = by_mem.find(cached_mem);
      return attach((it != by_mem.end()) ? &it->second : &ac->none);
    }
  }

  const ProcessorQueryImpl::KindCache *
  ProcessorQueryImpl::build_kind_cache(unsigned generation) const
  {
    KindCache *kc = new KindCache;
    kc->generation = generation;
    for(unsigned k = 0; k < NUM_PROCESSOR_KINDS; k++)
      kc->by_kind[k].generation = generation;
    {
      StripedRWLock::AutoReaderLock al(machine->mutex);
      for(std::map<NodeID, MachineNodeInfo *>::const_iterator it =
              machine->nodeinfos.begin();
          it != machine->nodeinfos.end(); ++it)
        for(std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::
                const_iterator it2 = it->second->proc_by_kind.begin();
            it2 != it->second->proc_by_kind.end(); ++it2) {
          if(unsigned(it2->first) >= NUM_PROCESSOR_KINDS)
            continue;
          std::vector<Processor> &items = kc->by_kind[it2->first].items;
          for(std::map<Processor, MachineProcInfo *>::const_iterator it3 =
                  it2->second.begin();
              it3 != it2->second.end(); ++it3)
            items.push_back(it3->first);
        }
    }
    // nodes and the processors within a node are visited in id order, and
    //  ids sort by node first, so the lists are sorted - next() relies on it
    for(unsigned k = 0; k < NUM_PROCESSOR_KINDS; k++)
      std::sort(kc->by_kind[k].items.begin(), kc->by_kind[k].items.end());
    return kind_cache.install(kc, cache_invalid_count.load());
  }

  const ProcessorQueryImpl::AffinityCache *
  ProcessorQueryImpl::build_affinity_cache(unsigned generation) const
  {
    AffinityCache *ac = new AffinityCache;
    ac->generation = generation;
    ac->none.generation = generation;
    {
      // every affinity of every processor, as ProcessorHasAffinityPredicate
      //  sees them (not just the same-node ones)
      StripedRWLock::AutoReaderLock al(machine->mutex);
      for(std::map<NodeID, MachineNodeInfo *>::const_iterator it =
              machine->nodeinfos.begin();
          it != machine->nodeinfos.end(); ++it)
        for(std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::
                const_iterator it2 = it->second->proc_by_kind.begin();
            it2 != it->second->proc_by_kind.end(); ++it2) {
          if(unsigned(it2->first) >= NUM_PROCESSOR_KINDS)
            continue;
          std::map<Memory, CachedList> &by_mem = ac->by_kind_and_mem[it2->first];
          for(std::map<Processor, MachineProcInfo *>::const_iterator it3 =
                  it2->second.begin();
              it3 != it2->second.end(); ++it3) {
            const std::map<Memory, Machine::ProcessorMemoryAffinity *> &pmas =
                it3->second->pmas.all;
            for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator
                    it4 = pmas.begin();
                it4 != pmas.end(); ++it4)
              by_mem[it4->first].items.push_back(it3->first);
          }
        }
    }
    for(unsigned k = 0; k < NUM_PROCESSOR_KINDS; k++)
      for(std::map<Memory, CachedList>::iterator it = ac->by_kind_and_mem[k].begin();
          it != ac->by_kind_and_mem[k].end(); ++it) {
        it->second.generation = generation;
        std::sort(it->second.items.begin(), it->second.items.end());
      }
    return affinity_cache.install(ac, cache_invalid_count.load());
  }

  bool ProcessorQueryImpl::cached_query(Processor &pval, QueryType q) const
  {
    if(!Config::use_machine_query_cache)
      return false;
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    const CachedList *clist = cached_list();
    if(!clist)
      return false;
    const std::vector<Processor> &items = clist->items;
    switch(q) {
    case QUERY_FIRST:
      pval = items.empty() ? Processor::NO_PROC : items.front();
      break;
    case QUERY_RANDOM:
      pval = items.empty() ? Processor::NO_PROC : items[lrand48() % items.size()];
      break;
    default:
      log_query.fatal() << "unexpected cached query type " << int(q);
      abort();
    }
    return true;
#else
    return false;
#endif
  }

  bool ProcessorQueryImpl::cached_query(Processor p, Processor &pval)
  {
    if(!Config::use_machine_query_cache)
      return false;
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    // if the machine changed since this query last touched the cache, this
    //  attaches to the rebuilt list and we continue walking from 'p' in it
    const CachedList *clist = cached_list();
    if(!clist)
      return false;
    pval = next(clist->items, p);
    return true;
#else
    return false;
#endif
  }

  bool ProcessorQueryImpl::cached_query(size_t &count) const
  {
    if(!Config::use_machine_query_cache)
      return false;
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    const CachedList *clist = cached_list();
    if(!clist)
      return false;
    count = clist->items.size();
    return true;
#else
    return false;
#endif
  }

  const ProcessorQueryImpl::CachedList *ProcessorQueryImpl::best_affinity_list() const
  {
    const CachedList *list = attached_list();
    if(list)
      return list;
    // copies of this query may be building concurrently: one builds, the
    //  rest find its list
    AutoLock<UnfairMutex> al(build_mutex);
    list = attached_list();
    if(list)
      return list;
    std::unique_ptr<CachedList> built(new CachedList);
    // read the generation before walking the machine, so a change during the
    //  walk makes the result stale and the next call rebuilds it
    built->generation = cache_invalid_count.load();
    build_best_affinity_list(built->items);
    list = built.get();
    own_lists.push_back(std::move(built));
    return attach(list);
  }

  void ProcessorQueryImpl::build_best_affinity_list(std::vector<Processor> &result) const
  {
    std::optional<int> best_score;
    std::vector<Processor> candidates_with_best_score;

    // the whole walk reads the machine model
    StripedRWLock::AutoReaderLock al(machine->mutex);

    // Iterate through all candidates matching other predicates
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node) {
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    } else {
      it = machine->nodeinfos.begin();
    }

    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id)) {
        break;
      }

      const std::map<Processor, MachineProcInfo *> *plist = 0;
      if(is_restricted_kind) {
        std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::const_iterator
            it2 = it->second->proc_by_kind.find(restricted_kind);
        if(it2 != it->second->proc_by_kind.end()) {
          plist = &(it2->second);
        } else {
          plist = 0;
        }
      } else {
        plist = &(it->second->procs);
      }

      if(plist) {
        std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
        std::vector<Machine::ProcessorMemoryAffinity> best_affinities;
        while(it2 != plist->end()) {
          bool ok = true;
          // Check all non-best-affinity predicates
          for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end()); it3++) {
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          }

          if(ok) {
            // Calculate affinity score for this candidate
            if(best_affinities.empty()) {
              machine->get_proc_mem_affinity_unlocked(best_affinities, Processor::NO_PROC,
                                                      best_affinity_cost->memory,
                                                      true /*local_only*/);
            }
            for(const Machine::ProcessorMemoryAffinity &affinity : best_affinities) {
              if(affinity.p == it2->first) {
                int score = (affinity.bandwidth * best_affinity_cost->bandwidth_weight) +
                            (affinity.latency * best_affinity_cost->latency_weight);
                if(!best_score || score > *best_score) {
                  best_score = score;
                  candidates_with_best_score.clear();
                  candidates_with_best_score.push_back(it2->first);
                } else if(score == *best_score) {
                  candidates_with_best_score.push_back(it2->first);
                }
                break;
              }
            }
          }
          ++it2;
        }
      }
      ++it;
    }

    // only the best candidates are the result (in machine order, i.e. sorted)
    result = std::move(candidates_with_best_score);
  }

  Processor ProcessorQueryImpl::first_match(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Processor p = (*it).p;
        if(is_restricted_node &&
           (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
          continue;
        Processor::Kind p_kind =
            ProcessorImpl::get_processor_kind(machine->get_runtime_impl(), p);
        if(is_restricted_kind && (p_kind != restricted_kind))
          continue;
        bool ok = true;
        for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, p);
        if(ok && (!lowest.exists() || (p.id < lowest.id)))
          lowest = p;
      }
    }
    return lowest;
#else

    // If best affinity is specified, we need to collect all candidates
    // and filter by best affinity cost
    if(best_affinity_cost.has_value()) {
      const std::vector<Processor> &best = best_affinity_list()->items;
      return best.empty() ? Processor::NO_PROC : best.front();
    }

    // optimize if restricted kind without predicates and restricted_node attached to the
    // query
    Processor pval = Processor::NO_PROC;
    if(cached_query(pval, QUERY_FIRST))
      return pval;

    // general case where restricted_node_id or predicates are defined
    StripedRWLock::AutoReaderLock al(machine->mutex);
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Processor, MachineProcInfo *> *plist = 0;
      if(is_restricted_kind) {
        std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::const_iterator
            it2 = it->second->proc_by_kind.find(restricted_kind);
        if(it2 != it->second->proc_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->procs);

      if(plist) {
        // if restricted node id is set or is_restricted_kind && predicates.size() == 0
        if((is_restricted_node || is_restricted_kind) && predicates.empty() &&
           !plist->empty())
          return plist->begin()->first;

        std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
        while(it2 != plist->end()) {
          bool ok = true;
          for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok)
            return it2->first;

          // try next processor (if it exists)
          ++it2;
        }
      }

      // try the next node (if it exists)
      ++it;
    }
    return Processor::NO_PROC;
#endif
  }

  Processor ProcessorQueryImpl::next_match(Processor after)
  {
    if(!after.exists())
      return Processor::NO_PROC;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return Processor::NO_PROC;
    Processor lowest = Processor::NO_PROC;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Processor p = (*it).p;
        if(p.id <= after.id)
          continue;
        if(is_restricted_node &&
           (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
          continue;
        Processor::Kind p_kind =
            ProcessorImpl::get_processor_kind(machine->get_runtime_impl(), p);
        if(is_restricted_kind && (p_kind != restricted_kind))
          continue;
        bool ok = true;
        for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, p);
        if(ok && (!lowest.exists() || (p.id < lowest.id)))
          lowest = p;
      }
    }
    return lowest;
#else
    // If best affinity is specified, the results should already be in the Level 2 cache
    // from first_match(). Just iterate through the cached list.
    if(best_affinity_cost.has_value()) {
      // Find the next processor in the (sorted) results after the given one
      return next(best_affinity_list()->items, after);
    }

    StripedRWLock::AutoReaderLock al(machine->mutex);
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).proc_owner_node());
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::const_iterator
            it2 = it->second->proc_by_kind.find(restricted_kind);
        if(it2 != it->second->proc_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->procs);

      if(plist) {
        std::map<Processor, MachineProcInfo *>::const_iterator it2;
        // same node?  if so, skip past ones we've done
        if(it->first == NodeID(ID(after).proc_owner_node()))
          it2 = plist->upper_bound(after);
        else
          it2 = plist->begin();
        while(it2 != plist->end()) {
          bool ok = true;
          for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok)
            return it2->first;

          // try next processor (if it exists)
          ++it2;
        }
      }

      // try the next node (if it exists)
      ++it;
    }
    return Processor::NO_PROC;
#endif
  }

  // cache the set of valid processors if required and return next
  Processor ProcessorQueryImpl::cache_next(Processor after)
  {

    log_query.debug("cache_next: processor input id =  %llx\n", after.id);
    Processor pval = Processor::NO_PROC;
    if(cached_query(after, pval))
      return pval;
    else
      // caches may be invalidated in the middle of a query
      return next_match(after);
  }

  size_t ProcessorQueryImpl::count_matches(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return 0;
    std::set<Processor> pset;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Processor p = (*it).p;
        if(is_restricted_node &&
           (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
          continue;
        Processor::Kind p_kind =
            ProcessorImpl::get_processor_kind(machine->get_runtime_impl(), p);
        if(is_restricted_kind && (p_kind != restricted_kind))
          continue;
        bool ok = true;
        for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, p);
        if(ok)
          pset.insert((*it).p);
      }
    }
    return pset.size();
#else
    // If best affinity is specified, use the Level 2 cache
    if(best_affinity_cost.has_value()) {
      return best_affinity_list()->items.size();
    }

    size_t count = 0;
    if(cached_query(count))
      return count;

    StripedRWLock::AutoReaderLock al(machine->mutex);
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::const_iterator
            it2 = it->second->proc_by_kind.find(restricted_kind);
        if(it2 != it->second->proc_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->procs);

      if(plist) {
        // no predicates: the whole list matches; a node-restricted query is
        // done, an unrestricted one must keep summing over the other nodes
        if(predicates.empty()) {
          if(is_restricted_node)
            return plist->size();
          count += plist->size();
          ++it;
          continue;
        }

        std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
        while(it2 != plist->end()) {
          bool ok = true;
          for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok)
            count += 1;

          // continue to next processor (if it exists)
          ++it2;
        }
      }

      // continue to the next node (if it exists)
      ++it;
    }
    return count;
#endif
  }

  Processor ProcessorQueryImpl::random_match(void) const
  {
    Processor chosen = Processor::NO_PROC;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return Processor::NO_PROC;
    int count = 0;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Processor p = (*it).p;
        if(is_restricted_node &&
           (ID(p).proc_owner_node() != (unsigned)restricted_node_id))
          continue;
        Processor::Kind p_kind =
            ProcessorImpl::get_processor_kind(machine->get_runtime_impl(), p);
        if(is_restricted_kind && (p_kind != restricted_kind))
          continue;
        bool ok = true;
        for(std::vector<ProcQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, p);
        if(ok) {
          count++;
          if((count == 1) || ((lrand48() % count) == 0))
            chosen = p;
        }
      }
    }
#else
    // If best affinity is specified, the results should be in the Level 2 cache
    // Pick randomly from those
    if(best_affinity_cost.has_value()) {
      const std::vector<Processor> &best = best_affinity_list()->items;
      if(!best.empty()) {
        size_t idx = lrand48() % best.size();
        return best[idx];
      }
      return Processor::NO_PROC;
    }

    // optimize if restricted kind without predicates and restricted_node attached to the
    // query
    Processor pval = Processor::NO_PROC;
    if(cached_query(pval, QUERY_RANDOM))
      return pval;
    StripedRWLock::AutoReaderLock al(machine->mutex);
    int count = 0;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Processor, MachineProcInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Processor::Kind, std::map<Processor, MachineProcInfo *>>::const_iterator
            it2 = it->second->proc_by_kind.find(restricted_kind);
        if(it2 != it->second->proc_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->procs);

      if(plist) {
        std::map<Processor, MachineProcInfo *>::const_iterator it2 = plist->begin();
        while(it2 != plist->end()) {
          bool ok = true;
          for(std::vector<ProcQueryPredicate *>::const_iterator it3 = predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok) {
            count++;
            if((count == 1) || ((lrand48() % count) == 0))
              chosen = it2->first;
          }

          // continue to next processor (if it exists)
          ++it2;
        }
      }

      // continue to the next node (if it exists)
      ++it;
    }
#endif
    return chosen;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryHasProcAffinityPredicate
  //

  MemoryHasProcAffinityPredicate::MemoryHasProcAffinityPredicate(Processor _proc,
                                                                 unsigned _min_bandwidth,
                                                                 unsigned _max_latency)
    : proc(_proc)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  MemoryQueryPredicate *MemoryHasProcAffinityPredicate::clone(void) const
  {
    return new MemoryHasProcAffinityPredicate(proc, min_bandwidth, max_latency);
  }

  bool MemoryHasProcAffinityPredicate::matches_predicate(const MachineImpl *machine,
                                                         Memory thing,
                                                         const MachineMemInfo *info) const
  {
#ifdef USE_OLD_AFFINITIES
    Machine::AffinityDetails details;
    if(!machine->has_affinity(proc, thing, &details))
      return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth))
      return false;
    if((max_latency != 0) && (details.latency > max_latency))
      return false;
    return true;
#else
    REALM_ASSERT(info != 0);
    std::map<Processor, Machine::ProcessorMemoryAffinity *>::const_iterator it =
        info->pmas.all.find(proc);
    if(it == info->pmas.all.end())
      return false;
    if((min_bandwidth != 0) && (it->second->bandwidth < min_bandwidth))
      return false;
    if((max_latency != 0) && (it->second->latency > max_latency))
      return false;
    return true;
#endif
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryHasMemAffinityPredicate
  //

  MemoryHasMemAffinityPredicate::MemoryHasMemAffinityPredicate(Memory _memory,
                                                               unsigned _min_bandwidth,
                                                               unsigned _max_latency)
    : memory(_memory)
    , min_bandwidth(_min_bandwidth)
    , max_latency(_max_latency)
  {}

  MemoryQueryPredicate *MemoryHasMemAffinityPredicate::clone(void) const
  {
    return new MemoryHasMemAffinityPredicate(memory, min_bandwidth, max_latency);
  }

  bool MemoryHasMemAffinityPredicate::matches_predicate(const MachineImpl *machine,
                                                        Memory thing,
                                                        const MachineMemInfo *info) const
  {
#ifdef USE_OLD_AFFINITIES
    Machine::AffinityDetails details;
    if(!machine->has_affinity(memory, thing, &details))
      return false;
    if((min_bandwidth != 0) && (details.bandwidth < min_bandwidth))
      return false;
    if((max_latency != 0) && (details.latency > max_latency))
      return false;
    return true;
#else
    REALM_ASSERT(info != 0);
    std::map<Memory, Machine::MemoryMemoryAffinity *>::const_iterator it =
        info->mmas_out.all.find(memory);
    if(it == info->mmas_out.all.end())
      return false;
    if((min_bandwidth != 0) && (it->second->bandwidth < min_bandwidth))
      return false;
    if((max_latency != 0) && (it->second->latency > max_latency))
      return false;
    return true;
#endif
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryBestProcAffinityPredicate
  //

  ////////////////////////////////////////////////////////////////////////
  //
  // class MemoryQueryImpl
  //
  atomic<unsigned int> MemoryQueryImpl::cache_invalid_count(0);
  QueryCacheSlot<MemoryQueryImpl::KindCache> MemoryQueryImpl::kind_cache;

  MemoryQueryImpl::MemoryQueryImpl(const Machine &_machine)
    : MemoryQueryImpl(static_cast<const MachineImpl *>(_machine.impl))
  {}

  MemoryQueryImpl::MemoryQueryImpl(const MachineImpl *_machine_impl)
    : references(1)
    , machine(_machine_impl)
    , is_restricted_node(false)
    , is_restricted_kind(false)
    , restricted_min_capacity(0)
    , attached(nullptr)
    , affinity_proc(Processor::NO_PROC)
  {}

  // see ProcessorQueryImpl's copy constructor
  MemoryQueryImpl::MemoryQueryImpl(const MemoryQueryImpl &copy_from)
    : references(1)
    , machine(copy_from.machine)
    , is_restricted_node(copy_from.is_restricted_node)
    , restricted_node_id(copy_from.restricted_node_id)
    , is_restricted_kind(copy_from.is_restricted_kind)
    , restricted_kind(copy_from.restricted_kind)
    , restricted_min_capacity(copy_from.restricted_min_capacity)
    , attached(nullptr)
    , affinity_proc(copy_from.affinity_proc)
    , best_proc_affinity_cost(copy_from.best_proc_affinity_cost)
    , best_mem_affinity_cost(copy_from.best_mem_affinity_cost)
  {
    predicates.reserve(copy_from.predicates.size());
    for(std::vector<MemoryQueryPredicate *>::const_iterator it =
            copy_from.predicates.begin();
        it != copy_from.predicates.end(); it++)
      predicates.push_back((*it)->clone());
  }

  MemoryQueryImpl::~MemoryQueryImpl(void)
  {
    REALM_ASSERT(references.load() == 0);
    for(std::vector<MemoryQueryPredicate *>::iterator it = predicates.begin();
        it != predicates.end(); it++)
      delete *it;
  }

  void MemoryQueryImpl::add_reference(void) { references.fetch_add(1); }

  void MemoryQueryImpl::remove_reference(void)
  {
    // see ProcessorQueryImpl::remove_reference
    int left = references.fetch_sub_acqrel(1) - 1;
    if(left == 0)
      delete this;
  }

  MemoryQueryImpl *MemoryQueryImpl::writeable_reference(void)
  {
    // we hold a reference, so a count of 1 means there are no others
    if(references.load_acquire() == 1) {
      return this;
    } else {
      MemoryQueryImpl *copy = new MemoryQueryImpl(*this);
      remove_reference();
      return copy;
    }
  }

  void MemoryQueryImpl::restrict_to_node(int new_node_id)
  {
    // attempts to restrict to two different nodes results in no possible match
    if(is_restricted_node && (new_node_id != restricted_node_id)) {
      restricted_node_id = -1;
    } else {
      is_restricted_node = true;
      restricted_node_id = new_node_id;
    }
    reset_cached_list();
  }

  void MemoryQueryImpl::restrict_to_kind(Memory::Kind new_kind)
  {
    // attempts to restrict to two different kind results in no possible match
    // (use node restriction to enforce this)
    if(is_restricted_kind && (new_kind != restricted_kind)) {
      is_restricted_node = true;
      restricted_node_id = -1;
    } else {
      is_restricted_kind = true;
      restricted_kind = new_kind;
    }
    reset_cached_list();
  }

  void MemoryQueryImpl::restrict_by_capacity(size_t new_min_bytes)
  {
    restricted_min_capacity = std::max(restricted_min_capacity, new_min_bytes);
    reset_cached_list();
  }

  void MemoryQueryImpl::add_predicate(MemoryQueryPredicate *pred)
  {
    // a writer is always unique, so no need for mutexes
    predicates.push_back(pred);
    if(predicates.size() > 1)
      affinity_proc = Processor::NO_PROC;
    reset_cached_list();
  }

  /*static*/ Memory MemoryQueryImpl::next(const std::vector<Memory> &list, Memory after)
  {
    // cached lists are sorted by id, so the successor is a binary search
    std::vector<Memory>::const_iterator it =
        std::upper_bound(list.begin(), list.end(), after);
    return (it != list.end()) ? *it : Memory::NO_MEMORY;
  }

  const MemoryQueryImpl::CachedList *MemoryQueryImpl::attach(const CachedList *list) const
  {
    attached.store_release(list);
    return list;
  }

  const MemoryQueryImpl::CachedList *MemoryQueryImpl::attached_list() const
  {
    const CachedList *list = attached.load_acquire();
    if(list && (list->generation == cache_invalid_count.load()))
      return list;
    return nullptr;
  }

  void MemoryQueryImpl::reset_cached_list()
  {
    // see ProcessorQueryImpl::reset_cached_list
    attached.store(nullptr);
  }

  void MemoryQueryImpl::set_affinity_proc(Processor p)
  {
    affinity_proc = (predicates.size() == 1) ? p : Processor::NO_PROC;
  }

  // the caller holds the machine's reader lock (and keeps holding it while it
  //  uses the returned map)
  const std::map<Memory, Machine::ProcessorMemoryAffinity *> *
  MemoryQueryImpl::anchor_affinities(Processor p, bool local_only) const
  {
    std::map<NodeID, MachineNodeInfo *>::const_iterator nit =
        machine->nodeinfos.find(ID(p).proc_owner_node());
    if(nit == machine->nodeinfos.end())
      return NULL;
    std::map<Processor, MachineProcInfo *>::const_iterator pit =
        nit->second->procs.find(p);
    if(pit == nit->second->procs.end())
      return NULL;
    return local_only ? &(pit->second->pmas.local) : &(pit->second->pmas.all);
  }

  // the caller holds the machine's reader lock
  bool MemoryQueryImpl::candidate_matches(Memory m) const
  {
    const NodeID node = ID(m).memory_owner_node();
    if(is_restricted_node && (node != NodeID(restricted_node_id)))
      return false;
    std::map<NodeID, MachineNodeInfo *>::const_iterator nit =
        machine->nodeinfos.find(node);
    if(nit == machine->nodeinfos.end())
      return false;
    const MachineMemInfo *info = NULL;
    if(is_restricted_kind) {
      std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator kit =
          nit->second->mem_by_kind.find(restricted_kind);
      if(kit == nit->second->mem_by_kind.end())
        return false;
      std::map<Memory, MachineMemInfo *>::const_iterator mit = kit->second.find(m);
      if(mit == kit->second.end())
        return false;
      info = mit->second;
    } else {
      std::map<Memory, MachineMemInfo *>::const_iterator mit = nit->second->mems.find(m);
      if(mit == nit->second->mems.end())
        return false;
      info = mit->second;
    }
    if((restricted_min_capacity > 0) &&
       (MemoryImpl::get_memory_size(machine->get_runtime_impl(), m) <
        restricted_min_capacity))
      return false;
    for(std::vector<MemoryQueryPredicate *>::const_iterator it = predicates.begin();
        it != predicates.end(); ++it)
      if(!(*it)->matches_predicate(machine, m, info))
        return false;
    return true;
  }

  // Returns the shared list that can serve this query, or null when the
  // shared cache does not cover it (see ProcessorQueryImpl::cached_list).
  const MemoryQueryImpl::CachedList *MemoryQueryImpl::cached_list() const
  {
    const CachedList *list = attached_list();
    if(list)
      return list;
    if(!is_restricted_kind || is_restricted_node || !predicates.empty() ||
       (restricted_min_capacity != 0) || best_proc_affinity_cost.has_value() ||
       best_mem_affinity_cost.has_value() ||
       (unsigned(restricted_kind) >= NUM_MEMORY_KINDS))
      return nullptr;

    const unsigned generation = cache_invalid_count.load();
    const KindCache *kc = kind_cache.get(generation);
    if(!kc)
      kc = build_kind_cache(generation);
    if(!kc)
      return nullptr; // the machine changed underneath us; walk it instead
    return attach(&kc->by_kind[restricted_kind]);
  }

  const MemoryQueryImpl::KindCache *
  MemoryQueryImpl::build_kind_cache(unsigned generation) const
  {
    KindCache *kc = new KindCache;
    kc->generation = generation;
    for(unsigned k = 0; k < NUM_MEMORY_KINDS; k++)
      kc->by_kind[k].generation = generation;
    {
      StripedRWLock::AutoReaderLock al(machine->mutex);
      for(std::map<NodeID, MachineNodeInfo *>::const_iterator it =
              machine->nodeinfos.begin();
          it != machine->nodeinfos.end(); ++it)
        for(std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator
                it2 = it->second->mem_by_kind.begin();
            it2 != it->second->mem_by_kind.end(); ++it2) {
          if(unsigned(it2->first) >= NUM_MEMORY_KINDS)
            continue;
          std::vector<Memory> &items = kc->by_kind[it2->first].items;
          for(std::map<Memory, MachineMemInfo *>::const_iterator it3 =
                  it2->second.begin();
              it3 != it2->second.end(); ++it3)
            items.push_back(it3->first);
        }
    }
    for(unsigned k = 0; k < NUM_MEMORY_KINDS; k++)
      std::sort(kc->by_kind[k].items.begin(), kc->by_kind[k].items.end());
    return kind_cache.install(kc, cache_invalid_count.load());
  }

  bool MemoryQueryImpl::cached_query(Memory &mval, QueryType q) const
  {
    if(!Config::use_machine_query_cache)
      return false;
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    const CachedList *clist = cached_list();
    if(!clist)
      return false;
    const std::vector<Memory> &items = clist->items;
    switch(q) {
    case QUERY_FIRST:
      mval = items.empty() ? Memory::NO_MEMORY : items.front();
      break;
    case QUERY_RANDOM:
      mval = items.empty() ? Memory::NO_MEMORY : items[lrand48() % items.size()];
      break;
    default:
      log_query.fatal() << "unexpected cached query type " << int(q);
      abort();
    }
    return true;
#else
    return false;
#endif
  }

  bool MemoryQueryImpl::cached_query(Memory m, Memory &mval)
  {
    if(!Config::use_machine_query_cache)
      return false;
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    const CachedList *clist = cached_list();
    if(!clist)
      return false;
    mval = next(clist->items, m);
    return true;
#else
    return false;
#endif
  }

  bool MemoryQueryImpl::cached_query(size_t &count) const
  {
    if(!Config::use_machine_query_cache)
      return false;
#ifndef REALM_DISABLE_MACHINE_QUERY_CACHE
    const CachedList *clist = cached_list();
    if(!clist)
      return false;
    count = clist->items.size();
    return true;
#else
    return false;
#endif
  }

  const MemoryQueryImpl::CachedList *MemoryQueryImpl::best_affinity_list() const
  {
    // see ProcessorQueryImpl::best_affinity_list
    const CachedList *list = attached_list();
    if(list)
      return list;
    AutoLock<UnfairMutex> al(build_mutex);
    list = attached_list();
    if(list)
      return list;
    std::unique_ptr<CachedList> built(new CachedList);
    built->generation = cache_invalid_count.load();
    build_best_affinity_list(built->items);
    list = built.get();
    own_lists.push_back(std::move(built));
    return attach(list);
  }

  void MemoryQueryImpl::build_best_affinity_list(std::vector<Memory> &result) const
  {
    // the whole walk reads the machine model
    StripedRWLock::AutoReaderLock al(machine->mutex);

    if(best_proc_affinity_cost.has_value()) {
      // only memories with a local affinity to the processor can score, so walk
      // that processor's affinity map instead of every memory of the kind
      std::optional<int> best_score;
      std::vector<Memory> candidates;
      std::vector<Machine::MemoryMemoryAffinity> mem_affinities;
      bool mem_affinities_fetched = false;
      const std::map<Memory, Machine::ProcessorMemoryAffinity *> *pmas =
          anchor_affinities(best_proc_affinity_cost->proc, true /*local_only*/);
      if(pmas) {
        for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it =
                pmas->begin();
            it != pmas->end(); ++it) {
          if(!candidate_matches(it->first))
            continue;
          int score =
              (it->second->bandwidth * best_proc_affinity_cost->bandwidth_weight) +
              (it->second->latency * best_proc_affinity_cost->latency_weight);
          if(best_mem_affinity_cost.has_value()) {
            if(!mem_affinities_fetched) {
              machine->get_mem_mem_affinity_unlocked(
                  mem_affinities, best_mem_affinity_cost->memory, Memory::NO_MEMORY,
                  true /*local_only*/);
              mem_affinities_fetched = true;
            }
            bool found = false;
            for(const Machine::MemoryMemoryAffinity &a : mem_affinities) {
              if(a.m2 == it->first) {
                score += (a.bandwidth * best_mem_affinity_cost->bandwidth_weight) +
                         (a.latency * best_mem_affinity_cost->latency_weight);
                found = true;
                break;
              }
            }
            if(!found)
              continue;
          }
          if(!best_score || (score > *best_score)) {
            best_score = score;
            candidates.clear();
            candidates.push_back(it->first);
          } else if(score == *best_score) {
            candidates.push_back(it->first);
          }
        }
      }
      // the affinity map is keyed by memory, so this is already sorted
      result = std::move(candidates);
      return;
    }

    std::optional<int> best_score;
    std::vector<Memory> candidates_with_best_score;

    // Iterate through all candidates matching other predicates
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node) {
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    } else {
      it = machine->nodeinfos.begin();
    }

    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id)) {
        break;
      }

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator it2 =
            it->second->mem_by_kind.find(restricted_kind);
        if(it2 != it->second->mem_by_kind.end()) {
          plist = &(it2->second);
        } else {
          plist = 0;
        }
      } else {
        plist = &(it->second->mems);
      }

      if(plist) {
        std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
        std::vector<Machine::ProcessorMemoryAffinity> best_proc_affinities;
        std::vector<Machine::MemoryMemoryAffinity> best_mem_affinities;
        while(it2 != plist->end()) {
          size_t it2_capacity =
              MemoryImpl::get_memory_size(machine->get_runtime_impl(), it2->first);
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2_capacity >= restricted_min_capacity));
          // Check all non-best-affinity predicates
          for(std::vector<MemoryQueryPredicate *>::const_iterator it3 =
                  predicates.begin();
              ok && (it3 != predicates.end()); it3++) {
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          }

          if(ok) {
            // Calculate affinity score for this candidate
            int score = 0;
            bool has_required_affinity = true;
            // Add processor affinity score if specified
            if(best_proc_affinity_cost.has_value()) {
              if(best_proc_affinities.empty()) {
                machine->get_proc_mem_affinity_unlocked(
                    best_proc_affinities, best_proc_affinity_cost->proc,
                    Memory::NO_MEMORY, true /*local_only*/);
              }
              bool found_proc_affinity = false;
              for(const Machine::ProcessorMemoryAffinity &affinity :
                  best_proc_affinities) {
                if(affinity.m == it2->first) {
                  score +=
                      (affinity.bandwidth * best_proc_affinity_cost->bandwidth_weight) +
                      (affinity.latency * best_proc_affinity_cost->latency_weight);
                  found_proc_affinity = true;
                  break;
                }
              }
              // If we require proc affinity but didn't find it, skip this candidate
              if(!found_proc_affinity) {
                has_required_affinity = false;
              }
            }

            // Add memory affinity score if specified
            if(has_required_affinity && best_mem_affinity_cost.has_value()) {
              if(best_mem_affinities.empty()) {
                machine->get_mem_mem_affinity_unlocked(
                    best_mem_affinities, best_mem_affinity_cost->memory,
                    Memory::NO_MEMORY, true /*local_only*/);
              }
              bool found_mem_affinity = false;
              for(const Machine::MemoryMemoryAffinity &affinity : best_mem_affinities) {
                if(affinity.m2 == it2->first) {
                  score +=
                      (affinity.bandwidth * best_mem_affinity_cost->bandwidth_weight) +
                      (affinity.latency * best_mem_affinity_cost->latency_weight);
                  found_mem_affinity = true;
                  break;
                }
              }
              // If we require mem affinity but didn't find it, skip this candidate
              if(!found_mem_affinity) {
                has_required_affinity = false;
              }
            }

            // Only consider this candidate if it has all required affinities
            if(has_required_affinity) {
              if(!best_score || score > *best_score) {
                best_score = score;
                candidates_with_best_score.clear();
                candidates_with_best_score.push_back(it2->first);
              } else if(score == *best_score) {
                candidates_with_best_score.push_back(it2->first);
              }
            }
          }
          ++it2;
        }
      }
      ++it;
    }

    // only the best candidates are the result (in machine order, i.e. sorted)
    result = std::move(candidates_with_best_score);
  }

  Memory MemoryQueryImpl::first_match(void) const
  {
#if USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Memory m = (*it).m;
        if(is_restricted_node &&
           (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
          continue;
        Memory::Kind m_kind = MemoryImpl::get_memory_kind(machine->get_runtime_impl(), m);
        if(is_restricted_kind && (m_kind != restricted_kind))
          continue;
        size_t m_capacity = MemoryImpl::get_memory_size(machine->get_runtime_impl(), m);
        if((restricted_min_capacity > 0) && (m_capacity < restricted_min_capacity))
          continue;
        bool ok = true;
        for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, m);
        if(ok && (!lowest.exists() || (m.id < lowest.id)))
          lowest = m;
      }
    }
    return lowest;
#else

    // If best affinity is specified, we need to collect all candidates
    // and filter by best affinity cost
    if(best_proc_affinity_cost.has_value() || best_mem_affinity_cost.has_value()) {
      const std::vector<Memory> &best = best_affinity_list()->items;
      return best.empty() ? Memory::NO_MEMORY : best.front();
    }

    if(affinity_proc.exists()) {
      StripedRWLock::AutoReaderLock al(machine->mutex);
      const std::map<Memory, Machine::ProcessorMemoryAffinity *> *pmas =
          anchor_affinities(affinity_proc, false /*!local_only*/);
      if(pmas)
        for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it =
                pmas->begin();
            it != pmas->end(); ++it)
          if(candidate_matches(it->first))
            return it->first;
      return Memory::NO_MEMORY;
    }

    Memory mval = Memory::NO_MEMORY;
    if(cached_query(mval, QUERY_FIRST))
      return mval;

    StripedRWLock::AutoReaderLock al(machine->mutex);
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator it2 =
            it->second->mem_by_kind.find(restricted_kind);
        if(it2 != it->second->mem_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->mems);

      if(plist) {
        if(is_restricted_node && predicates.empty() && (restricted_min_capacity == 0) &&
           !plist->empty())
          return plist->begin()->first;

        std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
        while(it2 != plist->end()) {
          size_t it2_capacity =
              MemoryImpl::get_memory_size(machine->get_runtime_impl(), it2->first);
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2_capacity >= restricted_min_capacity));
          for(std::vector<MemoryQueryPredicate *>::const_iterator it3 =
                  predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok)
            return it2->first;

          // try next memory (if it exists)
          ++it2;
        }
      }

      // try the next node (if it exists)
      ++it;
    }
    return Memory::NO_MEMORY;
#endif
  }

  Memory MemoryQueryImpl::next_match(Memory after) const
  {
    if(!after.exists())
      return Memory::NO_MEMORY;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return Memory::NO_MEMORY;
    Memory lowest = Memory::NO_MEMORY;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Memory m = (*it).m;
        if(m.id <= after.id)
          continue;
        if(is_restricted_node &&
           (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
          continue;
        Memory::Kind m_kind = MemoryImpl::get_memory_kind(machine->get_runtime_impl(), m);
        if(is_restricted_kind && (m_kind != restricted_kind))
          continue;
        size_t m_capacity = MemoryImpl::get_memory_size(machine->get_runtime_impl(), m);
        if((restricted_min_capacity > 0) && (m_capacity < restricted_min_capacity))
          continue;
        bool ok = true;
        for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, m);
        if(ok && (!lowest.exists() || (m.id < lowest.id)))
          lowest = m;
      }
    }
    return lowest;
#else
    // If best affinity is specified, the results should already be in the Level 2 cache
    // from first_match(). Just iterate through the cached list.
    if(best_proc_affinity_cost.has_value() || best_mem_affinity_cost.has_value()) {
      // Find the next memory in the (sorted) results after the given one
      return next(best_affinity_list()->items, after);
    }

    StripedRWLock::AutoReaderLock al(machine->mutex);
    if(affinity_proc.exists()) {
      const std::map<Memory, Machine::ProcessorMemoryAffinity *> *pmas =
          anchor_affinities(affinity_proc, false /*!local_only*/);
      if(pmas)
        for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it =
                pmas->upper_bound(after);
            it != pmas->end(); ++it)
          if(candidate_matches(it->first))
            return it->first;
      return Memory::NO_MEMORY;
    }

    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    // start where we left off
    it = machine->nodeinfos.find(ID(after).memory_owner_node());
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator it2 =
            it->second->mem_by_kind.find(restricted_kind);
        if(it2 != it->second->mem_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->mems);

      if(plist) {
        std::map<Memory, MachineMemInfo *>::const_iterator it2;
        // same node?  if so, skip past ones we've done
        if(it->first == NodeID(ID(after).memory_owner_node()))
          it2 = plist->upper_bound(after);
        else
          it2 = plist->begin();
        while(it2 != plist->end()) {
          size_t it2_capacity =
              MemoryImpl::get_memory_size(machine->get_runtime_impl(), it2->first);
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2_capacity >= restricted_min_capacity));
          for(std::vector<MemoryQueryPredicate *>::const_iterator it3 =
                  predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok)
            return it2->first;

          // try next memory (if it exists)
          ++it2;
        }
      }

      // try the next node (if it exists)
      ++it;
    }
    return Memory::NO_MEMORY;
#endif
  }

  Memory MemoryQueryImpl::cache_next(Memory after)
  {
    log_query.debug("cache_next: memory input id =  %llx\n", after.id);
    Memory mval = Memory::NO_MEMORY;
    if(cached_query(after, mval))
      return mval;
    else
      // caches may be invalidated in the middle of a query
      return next_match(after);
  }

  size_t MemoryQueryImpl::count_matches(void) const
  {
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return 0;
    std::set<Memory> pset;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Memory m = (*it).m;
        if(is_restricted_node &&
           (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
          continue;
        Memory::Kind m_kind = MemoryImpl::get_memory_kind(machine->get_runtime_impl(), m);
        if(is_restricted_kind && (m_kind != restricted_kind))
          continue;
        size_t m_capacity = MemoryImpl::get_memory_size(machine->get_runtime_impl(), m);
        if((restricted_min_capacity > 0) && (m_capacity < restricted_min_capacity))
          continue;
        bool ok = true;
        for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, m);
        if(ok)
          pset.insert((*it).m);
      }
    }
    return pset.size();
#else
    // If best affinity is specified, use the Level 2 cache
    if(best_proc_affinity_cost.has_value() || best_mem_affinity_cost.has_value()) {
      return best_affinity_list()->items.size();
    }

    if(affinity_proc.exists()) {
      StripedRWLock::AutoReaderLock al(machine->mutex);
      size_t count = 0;
      const std::map<Memory, Machine::ProcessorMemoryAffinity *> *pmas =
          anchor_affinities(affinity_proc, false /*!local_only*/);
      if(pmas)
        for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it =
                pmas->begin();
            it != pmas->end(); ++it)
          if(candidate_matches(it->first))
            count++;
      return count;
    }

    size_t count = 0;
    if(cached_query(count))
      return count;
    StripedRWLock::AutoReaderLock al(machine->mutex);
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator it2 =
            it->second->mem_by_kind.find(restricted_kind);
        if(it2 != it->second->mem_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->mems);

      if(plist) {
        // if predicates is empty then increment count by size of plist
        if(!predicates.size() && restricted_min_capacity == 0) {
          count = count + plist->size();
        }
        // else iterate over all relevant memories on this node and match the predicates
        else {
          std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
          while(it2 != plist->end()) {
            size_t it2_capacity =
                MemoryImpl::get_memory_size(machine->get_runtime_impl(), it2->first);
            bool ok = ((restricted_min_capacity == 0) ||
                       (it2_capacity >= restricted_min_capacity));
            for(std::vector<MemoryQueryPredicate *>::const_iterator it3 =
                    predicates.begin();
                ok && (it3 != predicates.end()); it3++)
              ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
            if(ok)
              count += 1;

            // continue to next memory (if it exists)
            ++it2;
          }
        }
      }

      // continue to the next node (if it exists)
      ++it;
    }
    return count;
#endif
  }

  Memory MemoryQueryImpl::random_match(void) const
  {
    Memory chosen = Memory::NO_MEMORY;
#ifdef USE_OLD_AFFINITIES
    if(is_restricted_node && (restricted_node_id < 0))
      return Memory::NO_MEMORY;
    int count = 0;
    {
      // problem with nested locks here...
      // AutoLock<> al(machine->mutex);
      for(std::vector<Machine::ProcessorMemoryAffinity>::const_iterator it =
              machine->proc_mem_affinities.begin();
          it != machine->proc_mem_affinities.end(); it++) {
        Memory m = (*it).m;
        if(is_restricted_node &&
           (ID(m).memory_owner_node() != (unsigned)restricted_node_id))
          continue;
        Memory::Kind m_kind = MemoryImpl::get_memory_kind(machine->get_runtime_impl(), m);
        if(is_restricted_kind && (m_kind != restricted_kind))
          continue;
        size_t m_capacity = MemoryImpl::get_memory_size(machine->get_runtime_impl(), m);
        if((restricted_min_capacity > 0) && (m_capacity < restricted_min_capacity))
          continue;
        bool ok = true;
        for(std::vector<MemoryQueryPredicate *>::const_iterator it2 = predicates.begin();
            ok && (it2 != predicates.end()); it2++)
          ok &= (*it2)->matches_predicate(machine, m);
        if(ok) {
          count++;
          if((count == 1) || ((lrand48() % count) == 0))
            chosen = m;
        }
      }
    }
#else
    // If best affinity is specified, the results should be in the Level 2 cache
    // Pick randomly from those
    if(best_proc_affinity_cost.has_value() || best_mem_affinity_cost.has_value()) {
      const std::vector<Memory> &best = best_affinity_list()->items;
      if(!best.empty()) {
        size_t idx = lrand48() % best.size();
        return best[idx];
      }
      return Memory::NO_MEMORY;
    }

    if(affinity_proc.exists()) {
      StripedRWLock::AutoReaderLock al(machine->mutex);
      Memory chosen = Memory::NO_MEMORY;
      int count = 0;
      const std::map<Memory, Machine::ProcessorMemoryAffinity *> *pmas =
          anchor_affinities(affinity_proc, false /*!local_only*/);
      if(pmas)
        for(std::map<Memory, Machine::ProcessorMemoryAffinity *>::const_iterator it =
                pmas->begin();
            it != pmas->end(); ++it)
          if(candidate_matches(it->first)) {
            count++;
            if((count == 1) || ((lrand48() % count) == 0))
              chosen = it->first;
          }
      return chosen;
    }

    Memory mval = Memory::NO_MEMORY;
    if(cached_query(mval, QUERY_RANDOM))
      return mval;

    StripedRWLock::AutoReaderLock al(machine->mutex);
    size_t count = 0;
    std::map<NodeID, MachineNodeInfo *>::const_iterator it;
    if(is_restricted_node)
      it = machine->nodeinfos.lower_bound(restricted_node_id);
    else
      it = machine->nodeinfos.begin();
    while(it != machine->nodeinfos.end()) {
      if(is_restricted_node && (it->first != restricted_node_id))
        break;

      const std::map<Memory, MachineMemInfo *> *plist;
      if(is_restricted_kind) {
        std::map<Memory::Kind, std::map<Memory, MachineMemInfo *>>::const_iterator it2 =
            it->second->mem_by_kind.find(restricted_kind);
        if(it2 != it->second->mem_by_kind.end())
          plist = &(it2->second);
        else
          plist = 0;
      } else
        plist = &(it->second->mems);

      if(plist) {
        std::map<Memory, MachineMemInfo *>::const_iterator it2 = plist->begin();
        while(it2 != plist->end()) {
          size_t it2_capacity =
              MemoryImpl::get_memory_size(machine->get_runtime_impl(), it2->first);
          bool ok = ((restricted_min_capacity == 0) ||
                     (it2_capacity >= restricted_min_capacity));
          for(std::vector<MemoryQueryPredicate *>::const_iterator it3 =
                  predicates.begin();
              ok && (it3 != predicates.end()); it3++)
            ok = (*it3)->matches_predicate(machine, it2->first, it2->second);
          if(ok) {
            count++;
            if((count == 1) || ((lrand48() % count) == 0))
              chosen = it2->first;
          }

          // continue to next memory (if it exists)
          ++it2;
        }
      }

      // continue to the next node (if it exists)
      ++it;
    }
#endif
    return chosen;
  }

}; // namespace Realm
