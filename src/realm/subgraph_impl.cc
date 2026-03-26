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

// Realm subgraph implementation

#include "realm/subgraph_impl.h"
#include "realm/event.h"
#include "realm/memory.h"
#include "realm/network.h"
#include "realm/proc_impl.h"
#include "realm/runtime_impl.h"
#include "realm/subgraph.h"
#include "realm/tasks.h"

namespace Realm {

  Logger log_subgraph("subgraph");

  ////////////////////////////////////////////////////////////////////////
  //
  // class Subgraph

  /*static*/ const Subgraph Subgraph::NO_SUBGRAPH = {/* zero-initialization */};

  /*static*/ Event Subgraph::create_subgraph(Subgraph &subgraph,
                                             const SubgraphDefinition &defn,
                                             const ProfilingRequestSet &prs,
                                             Event wait_on /*= Event::NO_EVENT*/)
  {
    NodeID target_node = Network::my_node_id;
    SubgraphImpl *impl =
        get_runtime()->local_subgraph_free_lists[target_node]->alloc_entry();
    impl->me.subgraph_creator_node() = Network::my_node_id;
    subgraph = impl->me.convert<Subgraph>();

    impl->defn = new SubgraphDefinition(defn);

    // no handling of preconditions or profiling yet
    assert(wait_on.has_triggered());
    assert(prs.empty());

    if(impl->compile()) {
      log_subgraph.info() << "created: subgraph=" << subgraph
                          << " ops=" << impl->interpreted_schedule.size();
      return Event::NO_EVENT;
    } else {
      // fatal error for now - once we have profiling, return a poisoned event
      //  if there was a profiling request for OperationStatus
      log_subgraph.fatal() << "subgraph compilation failed";
      abort();
    }
  }

  Event Subgraph::destroy(Event wait_on /*= Event::NO_EVENT*/) const
  {
    NodeID owner = ID(*this).subgraph_owner_node();

    log_subgraph.info() << "destroy: subgraph=" << *this << " wait_on=" << wait_on;

    if(owner == Network::my_node_id) {
      SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(*this);

      // Help the user out here to make deletion of Subgraphs only finish
      // when all pending instantiations are done.
      if(subgraph->defn->concurrency_mode == SubgraphDefinition::INSTANTIATION_ORDER &&
         subgraph->defn->execution_mode == SubgraphDefinition::COMPILED) {
        AutoLock<Mutex> al(subgraph->instantiation_lock);
        wait_on =
            Event::merge_events(wait_on, subgraph->previous_instantiation_completion,
                                subgraph->previous_cleanup_completion);
      }

      if(wait_on.has_triggered()) {
        subgraph->destroy();
        return Event::NO_EVENT;
      } else {
        UserEvent done = UserEvent::create_user_event();
        subgraph->deferred_destroy.defer(subgraph, wait_on, done);
        return done;
      }
    } else {
      // TODO (rohany): Forward a UserEvent through this active message.
      ActiveMessage<SubgraphDestroyMessage> amsg(owner);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg.commit();
      return Event::NO_EVENT;
    }
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
                              const ProfilingRequestSet &prs,
                              Event wait_on /*= Event::NO_EVENT*/,
                              int priority_adjust /*= 0*/) const
  {
    NodeID target_node = ID(*this).subgraph_owner_node();

    Event finish_event = GenEventImpl::create_genevent()->current_event();

    log_subgraph.info() << "instantiate: subgraph=" << *this << " before=" << wait_on
                        << " after=" << finish_event;

    if(target_node == Network::my_node_id) {
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs, empty_span() /*preconditions*/,
                        empty_span() /*postconditions*/, wait_on, finish_event,
                        priority_adjust);
    } else {
      Serialization::ByteCountSerializer bcs;
      {
        bool ok = (bcs.append_bytes(args, arglen) && (bcs << span<const Event>()) &&
                   (bcs << span<const Event>()) && (bcs << prs));
        assert(ok);
      }
      size_t msglen = bcs.bytes_used();
      ActiveMessage<SubgraphInstantiateMessage> amsg(target_node, msglen);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg->finish_event = finish_event;
      amsg->arglen = arglen;
      amsg->priority_adjust = priority_adjust;
      {
        amsg.add_payload(args, arglen);
        bool ok = ((amsg << span<const Event>()) && (amsg << span<const Event>()) &&
                   (amsg << prs));
        assert(ok);
      }
      amsg.commit();
    }
    return finish_event;
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
                              const ProfilingRequestSet &prs,
                              const std::vector<Event> &preconditions,
                              std::vector<Event> &postconditions,
                              Event wait_on /*= Event::NO_EVENT*/,
                              int priority_adjust /*= 0*/) const
  {
    NodeID target_node = ID(*this).subgraph_owner_node();

    Event finish_event = GenEventImpl::create_genevent()->current_event();

    // need to pre-create all the postcondition events too
    for(size_t i = 0; i < postconditions.size(); i++)
      postconditions[i] = GenEventImpl::create_genevent()->current_event();

    log_subgraph.info() << "instantiate: subgraph=" << *this << " before=" << wait_on
                        << " after=" << finish_event
                        << " preconds=" << PrettyVector<Event>(preconditions)
                        << " postconds=" << PrettyVector<Event>(postconditions);

    if(target_node == Network::my_node_id) {
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs, preconditions, postconditions, wait_on,
                        finish_event, priority_adjust);
    } else {
      Serialization::ByteCountSerializer bcs;
      {
        bool ok = (bcs.append_bytes(args, arglen) && (bcs << preconditions) &&
                   (bcs << postconditions) && (bcs << prs));
        assert(ok);
      }
      size_t msglen = bcs.bytes_used();
      ActiveMessage<SubgraphInstantiateMessage> amsg(target_node, msglen);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg->finish_event = finish_event;
      amsg->arglen = arglen;
      amsg->priority_adjust = priority_adjust;
      {
        amsg.add_payload(args, arglen);
        bool ok = ((amsg << preconditions) && (amsg << postconditions) && (amsg << prs));
        assert(ok);
      }
      amsg.commit();
    }
    return finish_event;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphImpl

  SubgraphImpl::SubgraphImpl()
    : me(Subgraph::NO_SUBGRAPH)
  {}

  SubgraphImpl::~SubgraphImpl() {}

  void SubgraphImpl::init(ID _me, int _owner)
  {
    me = _me;
    assert(NodeID(me.subgraph_owner_node()) == NodeID(_owner));
  }

  static bool
  has_interpolation(const std::vector<SubgraphDefinition::Interpolation> &interpolations,
                    unsigned first_interp, unsigned num_interps,
                    SubgraphDefinition::Interpolation::TargetKind target_kind,
                    unsigned target_index)
  {
    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation &it = interpolations[first_interp + i];
      if((it.target_kind == target_kind) && (it.target_index == target_index))
        return true;
    }
    return false;
  }

  class InterpolationScratchHelper {
  public:
    template <unsigned N>
    InterpolationScratchHelper(char (&prealloc)[N], size_t _needed)
      : needed(_needed)
      , used(0)
    {
      if(needed > N) {
        need_free = true;
        base = static_cast<char *>(malloc(N));
        assert(base != 0);
      } else {
        need_free = false;
        base = prealloc;
      }
    }

    ~InterpolationScratchHelper()
    {
      if(need_free)
        free(base);
    }

    void *next(size_t bytes)
    {
      void *p = base + used;
      used += bytes;
      assert(used <= needed);
      return p;
    }

  protected:
    size_t needed, used;
    bool need_free;
    char *base;
  };

  // performs any necessary interpolations, making a copy of the destination
  //  in the supplied scratch memory if needed, and returns a pointer to either
  //  the original if no changes were made or the scratch if the copy was
  //  performed
  static const void *
  do_interpolation(const std::vector<SubgraphDefinition::Interpolation> &interpolations,
                   unsigned first_interp, unsigned num_interps,
                   SubgraphDefinition::Interpolation::TargetKind target_kind,
                   unsigned target_index, const void *srcdata, size_t srclen,
                   const void *dstdata, size_t dstlen,
                   InterpolationScratchHelper &scratch_helper)
  {
    void *scratch_buffer = 0;
    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation &it = interpolations[first_interp + i];
      if((it.target_kind != target_kind) || (it.target_index != target_index))
        continue;

      // match - make the copy if we haven't already
      if(scratch_buffer == 0) {
        scratch_buffer = scratch_helper.next(dstlen);
        memcpy(scratch_buffer, dstdata, dstlen);
      }

      assert((it.offset + it.bytes) <= srclen);
      if(it.redop_id == 0) {
        // overwrite
        assert((it.target_offset + it.bytes) <= dstlen);
        memcpy(reinterpret_cast<char *>(scratch_buffer) + it.target_offset,
               reinterpret_cast<const char *>(srcdata) + it.offset, it.bytes);
      } else {
        const ReductionOpUntyped *redop =
            get_runtime()->reduce_op_table.get(it.redop_id, 0);
        assert((it.target_offset + redop->sizeof_lhs) <= dstlen);
        (redop->cpu_apply_excl_fn)(reinterpret_cast<char *>(scratch_buffer) +
                                       it.target_offset,
                                   0, reinterpret_cast<const char *>(srcdata) + it.offset,
                                   0, 1 /*count*/, redop->userdata);
      }
    }

    return ((scratch_buffer != 0) ? scratch_buffer : dstdata);
  }

  // a typed version for interpolating small values
  template <typename T>
  static T
  do_interpolation(const std::vector<SubgraphDefinition::Interpolation> &interpolations,
                   unsigned first_interp, unsigned num_interps,
                   SubgraphDefinition::Interpolation::TargetKind target_kind,
                   unsigned target_index, const void *srcdata, size_t srclen, T dstdata)
  {
    T val = dstdata;

    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation &it = interpolations[first_interp + i];
      if((it.target_kind != target_kind) || (it.target_index != target_index))
        continue;

      assert((it.offset + it.bytes) <= srclen);
      if(it.redop_id == 0) {
        // overwrite
        assert((it.target_offset + it.bytes) <= sizeof(T));
        memcpy(reinterpret_cast<char *>(&val) + it.target_offset,
               reinterpret_cast<const char *>(srcdata) + it.offset, it.bytes);
      } else {
        const ReductionOpUntyped *redop =
            get_runtime()->reduce_op_table.get(it.redop_id, 0);
        assert((it.target_offset + redop->sizeof_lhs) <= sizeof(T));
        (redop->cpu_apply_excl_fn)(reinterpret_cast<char *>(&val) + it.target_offset, 0,
                                   reinterpret_cast<const char *>(srcdata) + it.offset, 0,
                                   1 /*count*/, redop->userdata);
      }
    }

    return val;
  }

  class SortInterpolationsByKindAndIndex {
  public:
    bool operator()(const SubgraphDefinition::Interpolation &a,
                    const SubgraphDefinition::Interpolation &b) const
    {
      // ignore bottom 8 bits of interpolation kinds so we're just looking
      //  at the operation kind
      unsigned a_opkind = a.target_kind >> 8;
      unsigned b_opkind = b.target_kind >> 8;
      return ((a_opkind < b_opkind) ||
              ((a_opkind == b_opkind) && (a.target_index < b.target_index)));
    }
  };

  bool SubgraphImpl::compile(void)
  {

    // Some kind of initial checks for compilation and execution. Many of these
    // checks will be relaxed as the compiled subgraph implementation proceeds.
    if(defn->execution_mode == SubgraphDefinition::COMPILED) {
      // We currently are not going to support the SERIALIZABLE and CONCURRENT concurrency
      // modes.
      if(defn->concurrency_mode != SubgraphDefinition::ONE_SHOT &&
         defn->concurrency_mode != SubgraphDefinition::INSTANTIATION_ORDER) {
        log_subgraph.error() << "compiled subgraphs are only supported for one-shot or "
                                "instantiation-order concurrency modes";
        return false;
      }

      // In the version of the compiled subgraph path that lands first, we're only
      // going to support subgraphs that contain tasks running on the CPU and no
      // features like external pre/post-conditions, copies, barrier arrivals, etc.
      for(auto &task : defn->tasks) {
        if(task.proc.kind() != Processor::LOC_PROC) {
          log_subgraph.error()
              << "compiled subgraphs are currently only supported for LOC_PROC tasks";
          return false;
        }
        // The tasks should also be on this node.
        if(NodeID(task.proc.address_space()) != Network::my_node_id) {
          log_subgraph.error() << "compiled subgraphs are currently only supported for "
                                  "tasks running on the local node";
          return false;
        }
        if(task.priority != 0) {
          log_subgraph.error()
              << "compiled subgraphs do not currently support tasks with priorities";
          return false;
        }
        if(!task.prs.empty()) {
          log_subgraph.error() << "compiled subgraphs do not currently support tasks "
                                  "with profiling requests";
          return false;
        }
      }
      if(!defn->copies.empty()) {
        log_subgraph.error() << "compiled subgraphs do not currently support copies";
        return false;
      }
      if(!defn->arrivals.empty()) {
        log_subgraph.error()
            << "compiled subgraphs do not currently support barrier arrivals";
        return false;
      }
      if(!defn->instantiations.empty()) {
        log_subgraph.error() << "compiled subgraphs do not currently support recursive "
                                "subgraph instantiations";
        return false;
      }
      if(!defn->acquires.empty()) {
        log_subgraph.error()
            << "compiled subgraphs do not currently support reservation acquires";
        return false;
      }
      if(!defn->releases.empty()) {
        log_subgraph.error()
            << "compiled subgraphs do not currently support reservation releases";
        return false;
      }
      if(!defn->interpolations.empty()) {
        log_subgraph.error()
            << "compiled subgraphs do not currently support interpolations";
        return false;
      }

      // The dependencies between operations should only be between operations
      // in the subgraph, no external dependencies supported in the first land.
      for(auto &dependency : defn->dependencies) {
        if(dependency.src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND ||
           dependency.src_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND ||
           dependency.tgt_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND ||
           dependency.tgt_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND) {
          log_subgraph.error()
              << "compiled subgraphs do not currently support external dependencies";
          return false;
        }
      }

      log_subgraph.info() << "Compiling Realm Subgraph.";
    }

    typedef std::pair<SubgraphDefinition::OpKind, unsigned> OpInfo;
    typedef std::map<OpInfo, unsigned> TopoMap;
    TopoMap toposort;

    unsigned nextval = 0;
    for(unsigned i = 0; i < defn->tasks.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_TASK, i)] = nextval++;
    for(unsigned i = 0; i < defn->copies.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_COPY, i)] = nextval++;
    for(unsigned i = 0; i < defn->arrivals.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_ARRIVAL, i)] = nextval++;
    for(unsigned i = 0; i < defn->instantiations.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_INSTANTIATION, i)] = nextval++;
    for(unsigned i = 0; i < defn->acquires.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_ACQUIRE, i)] = nextval++;
    for(unsigned i = 0; i < defn->releases.size(); i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_RELEASE, i)] = nextval++;
    unsigned total_ops = nextval;

    // for subgraph instantiations, we need to do a pass over the dependencies
    //  to see which ports are used
    std::vector<unsigned> inst_pre_max_port(defn->instantiations.size(), 0);
    std::vector<unsigned> inst_post_max_port(defn->instantiations.size(), 0);

    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it =
            defn->dependencies.begin();
        it != defn->dependencies.end(); ++it) {
      if(it->src_op_kind == SubgraphDefinition::OPKIND_INSTANTIATION) {
        inst_post_max_port[it->src_op_index] =
            std::max(inst_post_max_port[it->src_op_index], it->src_op_port);
      } else
        assert(it->src_op_port == 0);

      if(it->tgt_op_kind == SubgraphDefinition::OPKIND_INSTANTIATION) {
        inst_pre_max_port[it->tgt_op_index] =
            std::max(inst_pre_max_port[it->tgt_op_index], it->tgt_op_port);
      } else
        assert(it->tgt_op_port == 0);
    }

    // sort by performing passes over dependency list...
    // any dependency whose target is before the source is resolved by
    //  moving the target to be after everybody
    // takes at most depth (<= N) passes unless there are loops
    bool converged = false;
    for(unsigned i = 0; !converged && (i < total_ops); i++) {
      converged = true;
      for(std::vector<SubgraphDefinition::Dependency>::const_iterator it =
              defn->dependencies.begin();
          it != defn->dependencies.end(); ++it) {
        // external pre/post-conditions are always satisfied
        if(it->src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND)
          continue;
        if(it->tgt_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND)
          continue;

        TopoMap::const_iterator src =
            toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
        assert(src != toposort.end());

        TopoMap::iterator tgt =
            toposort.find(std::make_pair(it->tgt_op_kind, it->tgt_op_index));
        assert(tgt != toposort.end());

        if(src->second > tgt->second) {
          tgt->second = nextval++;
          converged = false;
        }
      }
    }
    if(!converged) {
      log_subgraph.error() << "subgraph sort did not converge - has a cycle?";
      return false;
    }

    // re-compact the ordering indices
    unsigned curval = 0;
    while(curval < total_ops) {
      TopoMap::iterator best = toposort.end();
      for(TopoMap::iterator it = toposort.begin(); it != toposort.end(); ++it)
        if((it->second >= curval) &&
           ((best == toposort.end()) || (best->second > it->second)))
          best = it;
      assert(best != toposort.end());
      best->second = curval++;
    }

    // if there are any external postconditions, add them to the end of the
    //  toposort
    unsigned num_ext_postcond = 0;
    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it =
            defn->dependencies.begin();
        it != defn->dependencies.end(); ++it) {
      if(it->tgt_op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND)
        continue;
      if(it->tgt_op_index >= num_ext_postcond)
        num_ext_postcond = it->tgt_op_index + 1;
    }
    for(unsigned i = 0; i < num_ext_postcond; i++)
      toposort[std::make_pair(SubgraphDefinition::OPKIND_EXT_POSTCOND, i)] = total_ops++;

    interpreted_schedule.resize(total_ops);
    for(TopoMap::const_iterator it = toposort.begin(); it != toposort.end(); ++it) {
      interpreted_schedule[it->second].op_kind = it->first.first;
      interpreted_schedule[it->second].op_index = it->first.second;
    }

    // sort the interpolations so that each operation has a compact range
    //  to iterate through
    std::sort(defn->interpolations.begin(), defn->interpolations.end(),
              SortInterpolationsByKindAndIndex());
    for(std::vector<SubgraphScheduleEntry>::iterator it = interpreted_schedule.begin();
        it != interpreted_schedule.end(); ++it) {
      // binary search to find an interpolation for this operation
      unsigned lo = 0;
      unsigned hi = defn->interpolations.size();
      while(true) {
        if(lo >= hi) {
          // search failed - no interpolations
          it->first_interp = it->num_interps = 0;
          break;
        }
        unsigned mid = (lo + hi) >> 1;
        int mid_opkind = defn->interpolations[mid].target_kind >> 8;
        if(it->op_kind < mid_opkind) {
          hi = mid;
        } else if(it->op_kind > mid_opkind) {
          lo = mid + 1;
        } else {
          if(it->op_index < defn->interpolations[mid].target_index) {
            hi = mid;
          } else if(it->op_index > defn->interpolations[mid].target_index) {
            lo = mid + 1;
          } else {
            // found a value - now scan linearly up and down for full range
            lo = mid;
            while((lo > 0) &&
                  ((defn->interpolations[lo - 1].target_kind >> 8) == it->op_kind) &&
                  (defn->interpolations[lo - 1].target_index == it->op_index))
              lo--;
            hi = mid + 1;
            while((hi < defn->interpolations.size()) &&
                  ((defn->interpolations[hi].target_kind >> 8) == it->op_kind) &&
                  (defn->interpolations[hi].target_index == it->op_index))
              hi++;
            it->first_interp = lo;
            it->num_interps = hi - lo;
            break;
          }
        }
      }
    }

    // also sanity-check that any interpolation using a reduction op has it
    //  defined and sizes match up
    for(std::vector<SubgraphDefinition::Interpolation>::iterator it =
            defn->interpolations.begin();
        it != defn->interpolations.end(); ++it) {
      if(it->redop_id != 0) {
        const ReductionOpUntyped *redop =
            get_runtime()->reduce_op_table.get(it->redop_id, 0);
        if(redop == 0) {
          log_subgraph.error() << "no reduction op registered for ID " << it->redop_id;
          return false;
        }
        if(redop->sizeof_rhs != it->bytes) {
          log_subgraph.error() << "reduction op size mismatch";
          return false;
        }
      }
    }

    // Perform final event analysis before partitioning the schedule.
    num_final_events = 0;
    for(std::vector<SubgraphScheduleEntry>::iterator it = interpreted_schedule.begin();
        it != interpreted_schedule.end(); ++it) {
      if(it->op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) {
        // We'll clear this later if we find our contribution to the final
        //  event is done transitively
        it->is_final_event = true;
        num_final_events++;
      } else
        it->is_final_event = false;
    }

    // Any operation that points into a final event is not a final event.
    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it =
            defn->dependencies.begin();
        it != defn->dependencies.end(); ++it) {
      if(it->src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND)
        continue;
      TopoMap::const_iterator src =
          toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
      // If we are depending on port 0 of another node and we're not an
      //  external postcondition, then the preceeding node is not final.
      if((it->src_op_port == 0) &&
         (it->tgt_op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) &&
         (interpreted_schedule[src->second].is_final_event)) {
        interpreted_schedule[src->second].is_final_event = false;
        num_final_events--;
      }
    }

    // Compile the subgraph. This will also eventually involve splitting the subgraph
    // into a compiled component and an interpreted component. Some of the analysis
    // for interpreted subgraphs will be done only on the interpreted component.
    if(defn->execution_mode == SubgraphDefinition::COMPILED) {
      // Separate the dynamic and static parts of the schedule.
      std::vector<SubgraphScheduleEntry> static_schedule;
      std::vector<SubgraphScheduleEntry> dynamic_schedule;
      // Maintain reverse mappings operations into their positions in
      // the corresponding schedules.
      std::map<OpInfo, unsigned> static_op_to_index;
      std::map<OpInfo, unsigned> dynamic_op_to_index;

      // Clear toposort because we will produce a new toposort of leftover operations
      // that can't be part of the compiled graph. In the current phase this should be
      // empty, but will be extended in future work.
      toposort.clear();
      for(auto &it : interpreted_schedule) {
        OpInfo key = std::make_pair(it.op_kind, it.op_index);

        // We should only be compiling tasks right now, but I'll keep
        // the code structure to make it clear what will happen in the
        // near future.
        assert(it.op_kind == SubgraphDefinition::OPKIND_TASK);
        if(it.op_kind == SubgraphDefinition::OPKIND_TASK) {
          static_op_to_index[key] = static_schedule.size();
          static_schedule.push_back(it);
          // If we're moving an operation into the static part of the schedule,
          // its contribution to the final event in the subgraph will be done
          // separately, so remove it from the "dynamic" set of final events.
          if(it.is_final_event) {
            num_final_events--;
          }
        } else {
          toposort[key] = dynamic_schedule.size();
          // This is a little redundant with toposort, but toposort is used in
          // other parts of the code and it's easier to read if there is symmetry
          // between the static and dynamic parts of the schedule.
          dynamic_op_to_index[key] = dynamic_schedule.size();
          dynamic_schedule.push_back(it);
        }
      }
      // Now, schedule is only the dynamic schedule.
      interpreted_schedule = dynamic_schedule;

      // Construct the compiled_subgraph_operations vector.
      // TODO (rohany): Something to investigate is potentially sorting
      //  compiled_subgraph_operations by some key so that there is more
      //  locality in data accessed by each processor / background worker.
      compiled_subgraph_operations.resize(static_schedule.size());
      for(unsigned i = 0; i < static_schedule.size(); i++) {
        compiled_subgraph_operations[i] = SubgraphOperationDesc(
            static_schedule[i].op_kind, static_schedule[i].op_index,
            static_schedule[i].is_final_event,
            // TODO (rohany): In future work, we'll handle tasks that launch
            //  asynchronous work items.
            false /* is_async */
        );
      }

      // There are two main phases of subgraph compilation. The first is
      // to construct data structures for each processor to manage the pending
      // work quickly. The second (and will be implemented in the future) is
      // to do something similar for all background work items.

      // Perform a group-by on the dependencies list to have quick access
      // to the incoming and outgoing edges for each operation.
      std::map<OpInfo, std::vector<OpInfo>> incoming_edges;
      std::map<OpInfo, std::vector<OpInfo>> outgoing_edges;
      for(auto &it : defn->dependencies) {
        OpInfo src = std::make_pair(it.src_op_kind, it.src_op_index);
        OpInfo tgt = std::make_pair(it.tgt_op_kind, it.tgt_op_index);
        incoming_edges[tgt].push_back(src);
        outgoing_edges[src].push_back(tgt);
      }

      // Set up the incoming and outgoing edges for each operation.
      std::vector<std::vector<EdgeInfo>> op_incoming_edges(static_schedule.size());
      std::vector<std::vector<EdgeInfo>> op_outgoing_edges(static_schedule.size());
      operation_precondition_counters.resize(static_schedule.size());
      for(unsigned i = 0; i < static_schedule.size(); i++) {
        OpInfo desc =
            std::make_pair(static_schedule[i].op_kind, static_schedule[i].op_index);
        for(auto &it : incoming_edges[desc]) {
          // For now, all incoming edges should be part of the static schedule. A future
          // improvement will allow edges to cross the boundary between the static and
          // dynamic components of the graph.
          auto it2 = static_op_to_index.find(it);
          assert(it2 != static_op_to_index.end());
          op_incoming_edges[i].push_back(EdgeInfo(it2->second));
        }
        for(auto &it : outgoing_edges[desc]) {
          // Same comment as above.
          auto it2 = static_op_to_index.find(it);
          assert(it2 != static_op_to_index.end());
          op_outgoing_edges[i].push_back(EdgeInfo(it2->second));
        }
        operation_precondition_counters[i] = op_incoming_edges[i].size();
      }

      // Collect all processors used in this subgraph (which may be none).
      // To avoid indirections later, we'll map processors to indices.
      for(auto &task : defn->tasks) {
        if(processor_to_index.find(task.proc) == processor_to_index.end()) {
          processor_to_index[task.proc] = subgraph_processors.size();
          subgraph_processors.push_back(task.proc);
          LocalTaskProcessor *ltp = dynamic_cast<LocalTaskProcessor *>(
              get_runtime()->get_processor_impl(task.proc));
          assert(ltp != nullptr);
          subgraph_processor_impls.push_back(ltp);
        }
      }

      // Collect the tasks per processor. We'll use this information
      // to construct arenas per processor to place lightweight queues for ready tasks.
      std::map<Processor, std::vector<unsigned>> tasks_per_processor;
      for(unsigned i = 0; i < static_schedule.size(); i++) {
        const SubgraphScheduleEntry &it = static_schedule[i];
        switch(it.op_kind) {
        case SubgraphDefinition::OPKIND_TASK:
        {
          const SubgraphDefinition::TaskDesc &td = defn->tasks[it.op_index];
          tasks_per_processor[td.proc].push_back(i);
          break;
        }
        default:
        {
          assert(false);
          break;
        }
        }
      }

      // Construct the initial queue for each processor. This will contain all operations
      // that do not have any preconditions, followed by a default value for all other
      // entries in the queue.
      std::vector<std::vector<int64_t>> processor_queues(subgraph_processors.size());
      initial_queue_entry_counts.resize(subgraph_processors.size());
      for(unsigned i = 0; i < subgraph_processors.size(); i++) {
        Processor proc = subgraph_processors[i];
        const std::vector<unsigned> &tasks = tasks_per_processor.at(proc);
        processor_queues[i] =
            std::vector<int64_t>(tasks.size(), SUBGRAPH_EMPTY_QUEUE_ENTRY);
        unsigned idx = 0;
        for(auto task_idx : tasks) {
          // If the task has no incoming edges, we can add it to the queue
          // directly. The queue will contain indices into compiled_subgraph_operations.
          const SubgraphScheduleEntry &task = static_schedule[task_idx];
          OpInfo desc = std::make_pair(task.op_kind, task.op_index);
          if(incoming_edges[desc].size() == 0) {
            processor_queues[i][idx++] = task_idx;
          }
        }
        initial_queue_entry_counts[i] = idx;
      }

      // Flatten the nested metadata produced from the compilation process.
      operation_incoming_edges = op_incoming_edges;
      operation_outgoing_edges = op_outgoing_edges;
      initial_processor_queues = processor_queues;
    }

    // Once the subgraph compilation has completed, the subgraph has also been
    // partitioned into a compiled component and an interpreted component. Once this
    // has been done, we can finish the calculation of intermediate events. Note that
    // instantiations can produce more than one intermediate event.
    num_intermediate_events = 0;
    for(std::vector<SubgraphScheduleEntry>::iterator it = interpreted_schedule.begin();
        it != interpreted_schedule.end(); ++it) {
      it->intermediate_event_base = num_intermediate_events;
      if(it->op_kind == SubgraphDefinition::OPKIND_INSTANTIATION)
        it->intermediate_event_count = inst_post_max_port[it->op_index] + 1;
      else if(it->op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND)
        it->intermediate_event_count = 1;
      else
        it->intermediate_event_count = 0;
      num_intermediate_events += it->intermediate_event_count;
    }

    for(std::vector<SubgraphDefinition::Dependency>::const_iterator it =
            defn->dependencies.begin();
        it != defn->dependencies.end(); ++it) {
      TopoMap::const_iterator tgt =
          toposort.find(std::make_pair(it->tgt_op_kind, it->tgt_op_index));
      // If we can't find the target, that means the target is part
      // of the static schedule, and will be handled by that logic.
      if(tgt == toposort.end()) {
        continue;
      }
      assert(tgt != toposort.end());

      switch(it->src_op_kind) {
      case SubgraphDefinition::OPKIND_EXT_PRECOND:
      {
        // external preconditions are encoded as negative indices
        int idx = -1 - (int)(it->src_op_index);
        interpreted_schedule[tgt->second].preconditions.push_back(
            std::make_pair(it->tgt_op_port, idx));
        break;
      }

      default:
      {
        TopoMap::const_iterator src =
            toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
        // Same story for the source edge.
        if(src == toposort.end())
          continue;
        unsigned ev_idx =
            interpreted_schedule[src->second].intermediate_event_base + it->src_op_port;
        interpreted_schedule[tgt->second].preconditions.push_back(
            std::make_pair(it->tgt_op_port, ev_idx));
        break;
      }
      }
    }

    // Now sort the preconditions for each entry - allows us to group by port
    // and also notice duplicates.
    max_preconditions = 1; // have to count global precondition when needed
    for(std::vector<SubgraphScheduleEntry>::iterator it = interpreted_schedule.begin();
        it != interpreted_schedule.end(); ++it) {
      if(it->preconditions.empty())
        continue;

      std::sort(it->preconditions.begin(), it->preconditions.end());
      // look for duplicates past the first event
      size_t num_unique = 1;
      for(size_t i = 1; i < it->preconditions.size(); i++)
        if(it->preconditions[i] != it->preconditions[num_unique - 1]) {
          if(num_unique < i)
            it->preconditions[num_unique] = it->preconditions[i];
          num_unique++;
        }
      if(num_unique < it->preconditions.size())
        it->preconditions.resize(num_unique);
      if(num_unique >= max_preconditions)
        max_preconditions = num_unique + 1;
    }

    return true;
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
                                 const ProfilingRequestSet &prs,
                                 span<const Event> preconditions,
                                 span<const Event> postconditions, Event start_event,
                                 Event finish_event, int priority_adjust)
  {
    // Some more initial checking for compiled subgraphs.
    if(defn->execution_mode == SubgraphDefinition::COMPILED) {
      if(!prs.empty()) {
        log_subgraph.error() << "compiled subgraphs do not currently support profiling";
        assert(false);
      }
      if(!preconditions.empty()) {
        log_subgraph.error()
            << "compiled subgraphs do not currently support external preconditions";
        assert(false);
      }
      if(!postconditions.empty()) {
        log_subgraph.error()
            << "compiled subgraphs do not currently support external postconditions";
        assert(false);
      }

      if(defn->concurrency_mode == SubgraphDefinition::INSTANTIATION_ORDER) {
        // Since we're an instantiation-order subgraph, we get to assume that
        // subgraph executions are happening in-order. So, make sure that we're
        // the only launcher of this subgraph.
        instantiation_lock.lock();
        // Next, make sure that we're starting only when the previous instantiation
        // has completed.
        start_event = Event::merge_events(start_event, previous_instantiation_completion);
      }
    }

    // Handle extra execution setup required for launching a compiled subgraph.
    UserEvent static_finish_event = UserEvent::NO_USER_EVENT;
    UserEvent cleanup_done_event = UserEvent::NO_USER_EVENT;
    if(defn->execution_mode == SubgraphDefinition::COMPILED) {
      static_finish_event = UserEvent::create_user_event();
      // Allocate the state for this subgraph replay.
      SubgraphExecutionState *exec_state =
          new SubgraphExecutionState(this, args, arglen, static_finish_event);

      // Issue the static portion of the subgraph.
      SubgraphWorkLauncher::launch_or_defer(exec_state, start_event);
      // Register a cleanup operation for when this subgraph execution
      // is complete.
      SubgraphInstantiationCleanup *cleanup =
          new SubgraphInstantiationCleanup(exec_state, cleanup_done_event);
      EventImpl::add_waiter(finish_event, cleanup);
    }

    // we precomputed the number of intermediate events we need, so put them
    //  on the stack
    Event *intermediate_events =
        static_cast<Event *>(alloca(num_intermediate_events * sizeof(Event)));
    size_t cur_intermediate_events = 0;

    // we've also computed how many events will contribute to the finish
    //  event, so we can arm the merger as we go
    GenEventImpl *event_impl = 0;
    // num_final_events may be zero if all the final events were sucked into
    // the static part of the subgraph. So we'll arm a finish event no matter
    // what and include the contribution of the static component. Include
    // a +1 to num_final_events to account for the static component.
    event_impl = get_genevent_impl(finish_event);
    event_impl->merger.prepare_merger(finish_event, false /*!ignore_faults*/,
                                      num_final_events + 1);
    event_impl->merger.add_precondition(static_finish_event);

    Event *preconds = static_cast<Event *>(alloca(max_preconditions * sizeof(Event)));

    for(std::vector<SubgraphScheduleEntry>::const_iterator it =
            interpreted_schedule.begin();
        it != interpreted_schedule.end(); ++it) {
      // assemble precondition
      size_t num_preconds = 0;
      bool need_global_precond = start_event.exists();

      size_t pc_idx = 0;
      while(pc_idx < it->preconditions.size()) {
        // if we see something for a nonzero port, save those for later
        if(it->preconditions[pc_idx].first != 0)
          break;

        if(it->preconditions[pc_idx].second >= 0) {
          // this is a dependency on another operation
          assert(unsigned(it->preconditions[pc_idx].second) < cur_intermediate_events);
          preconds[num_preconds++] =
              intermediate_events[it->preconditions[pc_idx].second];
          // we get the global precondition transitively...
          need_global_precond = false;
        } else {
          // external precondition
          int idx = -1 - it->preconditions[pc_idx].second;
          if((idx < int(preconditions.size())) && preconditions[idx].exists())
            preconds[num_preconds++] = preconditions[idx];
        }

        pc_idx++;
      }
      if(need_global_precond)
        preconds[num_preconds++] = start_event;

      assert(num_preconds <= max_preconditions);

      // for external postconditions, merge the preconditions directly into the
      //  returned event
      if(it->op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND) {
        // only bother if the caller wanted the event
        if(it->op_index < postconditions.size()) {
          Event post_event = postconditions[it->op_index];
          if(num_preconds > 0) {
            GenEventImpl *post_impl = get_genevent_impl(post_event);
            post_impl->merger.prepare_merger(post_event, false /*!ignore_faults*/,
                                             num_preconds);
            for(size_t i = 0; i < num_preconds; i++)
              post_impl->merger.add_precondition(preconds[i]);
            post_impl->merger.arm_merger();
          } else
            GenEventImpl::trigger(post_event, false /*!poisoned*/);
        }
        continue;
      }

      span<const Event> s(preconds, num_preconds);
      Event pre = GenEventImpl::merge_events(s, false);
#if 0
      Event pre = GenEventImpl::merge_events(make_span<const Event>(preconds,
								    num_preconds),
					     false /*!ignore_faults*/);
#endif
      // scratch buffer used for interpolations
      const size_t SCRATCH_SIZE = 1024;
      char interp_scratch[SCRATCH_SIZE];

      Event e = Event::NO_EVENT;

      switch(it->op_kind) {
      case SubgraphDefinition::OPKIND_TASK:
      {
        const SubgraphDefinition::TaskDesc &td = defn->tasks[it->op_index];
        Processor proc = td.proc;
        Processor::TaskFuncID task_id = td.task_id;
        int priority = td.priority;

        size_t scratch_needed = 0;
        if(has_interpolation(defn->interpolations, it->first_interp, it->num_interps,
                             SubgraphDefinition::Interpolation::TARGET_TASK_ARGS,
                             it->op_index))
          scratch_needed += td.args.size();

        InterpolationScratchHelper ish(interp_scratch, scratch_needed);

        const void *task_args = do_interpolation(
            defn->interpolations, it->first_interp, it->num_interps,
            SubgraphDefinition::Interpolation::TARGET_TASK_ARGS, it->op_index, args,
            arglen, td.args.base(), td.args.size(), ish);

        e = proc.spawn(task_id, task_args, td.args.size(), td.prs, pre,
                       priority + priority_adjust);
        intermediate_events[cur_intermediate_events++] = e;
        break;
      }

      case SubgraphDefinition::OPKIND_COPY:
      {
        const SubgraphDefinition::CopyDesc &cd = defn->copies[it->op_index];
        e = cd.space.copy(cd.srcs, cd.dsts, cd.prs, pre);
        intermediate_events[cur_intermediate_events++] = e;
        break;
      }

      case SubgraphDefinition::OPKIND_ARRIVAL:
      {
        const SubgraphDefinition::ArrivalDesc &ad = defn->arrivals[it->op_index];

        InterpolationScratchHelper ish(interp_scratch, ad.reduce_value.size());

        Barrier b =
            do_interpolation(defn->interpolations, it->first_interp, it->num_interps,
                             SubgraphDefinition::Interpolation::TARGET_ARRIVAL_BARRIER,
                             it->op_index, args, arglen, ad.barrier);
        const void *red_val = do_interpolation(
            defn->interpolations, it->first_interp, it->num_interps,
            SubgraphDefinition::Interpolation::TARGET_ARRIVAL_VALUE, it->op_index, args,
            arglen, ad.reduce_value.base(), ad.reduce_value.size(), ish);
        unsigned count = ad.count;
        b.arrive(count, pre, red_val, ad.reduce_value.size());

        // "finish event" is precondition
        intermediate_events[cur_intermediate_events++] = e = pre;
        break;
      }

      case SubgraphDefinition::OPKIND_ACQUIRE:
      {
        const SubgraphDefinition::AcquireDesc &ad = defn->acquires[it->op_index];
        Reservation rsrv = ad.rsrv;
        unsigned mode = ad.mode;
        bool excl = ad.exclusive;
        e = rsrv.acquire(mode, excl, pre);
        intermediate_events[cur_intermediate_events++] = e;
        break;
      }

      case SubgraphDefinition::OPKIND_RELEASE:
      {
        const SubgraphDefinition::ReleaseDesc &rd = defn->releases[it->op_index];
        Reservation rsrv = rd.rsrv;
        rsrv.release(pre);
        // "finish event" is precondition
        intermediate_events[cur_intermediate_events++] = e = pre;
        break;
      }

      case SubgraphDefinition::OPKIND_INSTANTIATION:
      {
        const SubgraphDefinition::InstantiationDesc &id =
            defn->instantiations[it->op_index];
        Subgraph sg_inner = id.subgraph;
        int priority_adjust = id.priority_adjust;

        size_t scratch_needed = 0;
        if(has_interpolation(defn->interpolations, it->first_interp, it->num_interps,
                             SubgraphDefinition::Interpolation::TARGET_INSTANCE_ARGS,
                             it->op_index))
          scratch_needed += id.args.size();

        InterpolationScratchHelper ish(interp_scratch, scratch_needed);

        const void *inst_args = do_interpolation(
            defn->interpolations, it->first_interp, it->num_interps,
            SubgraphDefinition::Interpolation::TARGET_INSTANCE_ARGS, it->op_index, args,
            arglen, id.args.base(), id.args.size(), ish);

        // TODO: avoid dynamic allocation?
        std::vector<Event> inst_preconds, inst_postconds;

        // how many preconditions do we need to form?
        unsigned num_inst_preconds =
            (it->preconditions.empty() ? 0 : it->preconditions.rbegin()->first);
        // log_subgraph.print() << "inst_preconds = " << num_inst_preconds;
        if(num_inst_preconds > 0) {
          inst_preconds.resize(num_inst_preconds);
          for(unsigned i = 0; i < num_inst_preconds; i++) {
            std::vector<Event> evs;
            // continue scanning preconditions where the previous scan(s) stopped
            while((pc_idx < it->preconditions.size()) &&
                  (it->preconditions[pc_idx].first == (i + 1))) {
              if(it->preconditions[pc_idx].second >= 0) {
                // this is a dependency on another operation
                assert(unsigned(it->preconditions[pc_idx].second) <
                       cur_intermediate_events);
                evs.push_back(intermediate_events[it->preconditions[pc_idx].second]);
              } else {
                // external precondition
                int idx = -1 - it->preconditions[pc_idx].second;
                if((idx < int(preconditions.size())) && preconditions[idx].exists())
                  evs.push_back(preconditions[idx]);
              }

              pc_idx++;
            }

            inst_preconds[i] = GenEventImpl::merge_events(evs, false /*!ignore_faults*/);
          }
        }

        if(it->intermediate_event_count > 1) {
          // log_subgraph.print() << "inst_postconds = " << (it->intermediate_event_count
          // - 1);
          inst_postconds.resize(it->intermediate_event_count - 1);
        }

        e = sg_inner.instantiate(inst_args, id.args.size(), id.prs, inst_preconds,
                                 inst_postconds, pre, priority_adjust);

        intermediate_events[cur_intermediate_events] = e;
        if(it->intermediate_event_count > 1)
          memcpy(&intermediate_events[cur_intermediate_events + 1], inst_postconds.data(),
                 (it->intermediate_event_count - 1) * sizeof(Event));
        cur_intermediate_events += it->intermediate_event_count;
        break;
      }

      default:
        assert(0);
      }

      // contribute to the final event if we need to
      if(it->is_final_event)
        event_impl->merger.add_precondition(e);
    }

    // sanity-check that we counted right
    assert(cur_intermediate_events == num_intermediate_events);

    // If we compiled a part of the subgraph or had some finish events
    // in the dynamic portion, then we need to arm the merger. Otherwise,
    // final event is ready to trigger as-is.
    if(num_final_events > 0 || defn->execution_mode == SubgraphDefinition::COMPILED) {
      event_impl->merger.arm_merger();
    } else {
      GenEventImpl::trigger(finish_event, false /*!poisoned*/);
    }

    if(defn->concurrency_mode == SubgraphDefinition::INSTANTIATION_ORDER &&
       defn->execution_mode == SubgraphDefinition::COMPILED) {
      // Chain the finish event of this subgraph launch through
      // the state of this subgraph for future launches.
      previous_instantiation_completion = finish_event;
      previous_cleanup_completion =
          Event::merge_events(previous_cleanup_completion, cleanup_done_event);
      instantiation_lock.unlock();
    }
  }

  void SubgraphImpl::destroy(void)
  {
    delete defn;
    interpreted_schedule.clear();

    subgraph_processors.clear();
    subgraph_processor_impls.clear();
    processor_to_index.clear();
    compiled_subgraph_operations.clear();
    operation_incoming_edges.clear();
    operation_outgoing_edges.clear();
    operation_precondition_counters.clear();
    initial_processor_queues.clear();
    initial_queue_entry_counts.clear();

    // TODO: when we create subgraphs on remote nodes, send a message to the
    //  creator node so they can add it to their free list
    NodeID creator_node = ID(me).subgraph_creator_node();
    assert(creator_node == Network::my_node_id);
    NodeID owner_node = ID(me).subgraph_owner_node();
    assert(owner_node == Network::my_node_id);

    get_runtime()->local_subgraph_free_lists[owner_node]->free_entry(this);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphImpl::DeferredDestroy
  //

  void SubgraphImpl::DeferredDestroy::defer(SubgraphImpl *_subgraph, Event wait_on,
                                            UserEvent _to_trigger)
  {
    subgraph = _subgraph;
    to_trigger = _to_trigger;
    EventImpl::add_waiter(wait_on, this);
  }

  void SubgraphImpl::DeferredDestroy::event_triggered(bool poisoned, TimeLimit work_until)
  {
    assert(!poisoned);
    subgraph->destroy();
    to_trigger.trigger();
  }

  void SubgraphImpl::DeferredDestroy::print(std::ostream &os) const
  {
    os << "deferred subgraph destruction: subgraph=" << subgraph->me;
  }

  Event SubgraphImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphInstantiateMessage

  /*static*/ void
  SubgraphInstantiateMessage::handle_message(NodeID sender,
                                             const SubgraphInstantiateMessage &msg,
                                             const void *data, size_t datalen)
  {
    SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(msg.subgraph);
    span<const Event> preconditions, postconditions;
    ProfilingRequestSet prs;

    Serialization::FixedBufferDeserializer fbd(data, datalen);
    fbd.extract_bytes(
        0, msg.arglen); // skip over instantiation args - we'll access those directly
    bool ok = ((fbd >> preconditions) && (fbd >> postconditions));
    if(ok && (fbd.bytes_left() > 0))
      ok = (fbd >> prs);
    assert(ok);

    subgraph->instantiate(data, msg.arglen, prs, preconditions, postconditions,
                          msg.wait_on, msg.finish_event, msg.priority_adjust);
  }

  ActiveMessageHandlerReg<SubgraphInstantiateMessage>
      subgraph_instantiate_message_handler;

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphDestroyMessage

  /*static*/ void
  SubgraphDestroyMessage::handle_message(NodeID sender, const SubgraphDestroyMessage &msg,
                                         const void *data, size_t datalen)
  {
    SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(msg.subgraph);

    if(msg.wait_on.has_triggered())
      subgraph->destroy();
    else
      subgraph->deferred_destroy.defer(subgraph, msg.wait_on);
  }

  ActiveMessageHandlerReg<SubgraphDestroyMessage> subgraph_destroy_message_handler;

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphWorkLauncher
  //
  ////////////////////////////////////////////////////////////////////////

  SubgraphWorkLauncher::SubgraphWorkLauncher(SubgraphExecutionState *subgraph)
    : subgraph(subgraph)
  {}

  /*static*/ void SubgraphWorkLauncher::launch_or_defer(SubgraphExecutionState *subgraph,
                                                        Event wait_on)
  {
    if(!wait_on.exists() || wait_on.has_triggered()) {
      // If there isn't a precondition, or the precondition is already triggered,
      // then just launch the subgraph.
      SubgraphWorkLauncher(subgraph).launch();
    } else {
      // Otherwise, allocate the launcher to start up the subgraph. It will
      // clean itself up.
      SubgraphWorkLauncher *launcher = new SubgraphWorkLauncher(subgraph);
      EventImpl::add_waiter(wait_on, launcher);
    }
  }

  void SubgraphWorkLauncher::launch()
  {
    // Install the subrgraph onto the target processors. In the future
    // this method will also handle starting background work items with
    // no preconditions.
    for(auto &proc : subgraph->subgraph->subgraph_processor_impls) {
      proc->enqueue_subgraph(subgraph);
    }
  }

  void SubgraphWorkLauncher::event_triggered(bool poisoned, TimeLimit work_until)
  {
    // Launch the subgraph and clean up after ourselves.
    launch();
    delete this;
  }

  void SubgraphWorkLauncher::print(std::ostream &os) const
  {
    os << "SubgraphWorkLauncher: subgraph=" << subgraph->subgraph->me;
  }

  Event SubgraphWorkLauncher::get_finish_event(void) const { return Event::NO_EVENT; }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphInstantiationCleanup
  //
  ////////////////////////////////////////////////////////////////////////

  SubgraphInstantiationCleanup::SubgraphInstantiationCleanup(
      SubgraphExecutionState *subgraph, UserEvent to_trigger)
    : subgraph(subgraph)
    , to_trigger(to_trigger)
  {}

  void SubgraphInstantiationCleanup::cleanup()
  {
    delete subgraph;
    // We'll have more work to do here once we add more features to
    // the compiled subgraph implementation.
    to_trigger.trigger();
  }

  void SubgraphInstantiationCleanup::event_triggered(bool poisoned, TimeLimit work_until)
  {
    // Defer the cleanup to a background worker so we don't make this
    // event waiter wake unnecessarily expensive.
    get_runtime()->subgraph_resource_reaper.enqueue_cleanup(this);
  }

  void SubgraphInstantiationCleanup::print(std::ostream &os) const
  {
    os << "SubgraphInstantiationCleanup: subgraph=" << subgraph->subgraph->me;
  }

  Event SubgraphInstantiationCleanup::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphResourceReaper
  //
  ////////////////////////////////////////////////////////////////////////

  SubgraphResourceReaper::SubgraphResourceReaper()
    : BackgroundWorkItem("SubgraphResourceReaper")
  {}

  void SubgraphResourceReaper::enqueue_cleanup(SubgraphInstantiationCleanup *item)
  {
    {
      AutoLock<> al(mutex);
      pending_cleanups.push(item);
    }
    make_active();
  }

  bool SubgraphResourceReaper::do_work(TimeLimit work_until)
  {
    size_t left = 0;
    SubgraphInstantiationCleanup *item = nullptr;
    {
      AutoLock<> al(mutex);
      if(!pending_cleanups.empty()) {
        item = pending_cleanups.front();
        pending_cleanups.pop();
        left = pending_cleanups.size();
      }
    }
    if(!item)
      return false;

    item->cleanup();
    delete item;
    return left > 0;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // struct SubgraphExecutionState
  //
  ////////////////////////////////////////////////////////////////////////

  SubgraphExecutionState::SubgraphExecutionState(SubgraphImpl *subgraph,
                                                 const void *_args, size_t arglen,
                                                 UserEvent finish_event)
    : subgraph(subgraph)
    , args(nullptr)
    , arglen(arglen)
    , finish_counter(0)
    , finish_event(finish_event)
    , preconditions(nullptr)
    , processor_queues(nullptr)
  {
    // Make a copy of the arguments passed to the subgraph.
    if(_args != nullptr && arglen > 0) {
      args = malloc(arglen);
      memcpy(args, _args, arglen);
    }

    // preconditions and queues are initialized as malloc'd data rather than
    // vectors because we don't want to pay for the constructor of atomic<>
    // on every element, which is going to do an atomic store at location 
    // instead of the cheaper memcpy we want to do instead. This means that
    // we have to manage that memory ourselves.

    // TODO (rohany): Can we assume that int32_t is enough to store the
    //  number of preconditions held for each operation?
    // Allocate a fresh copy of the preconditions array and copy the
    // pre-computed precondition counters into it.
    static_assert(sizeof(int64_t) == sizeof(atomic<int64_t>));
    size_t precondition_ctr_bytes =
        sizeof(atomic<int64_t>) * subgraph->operation_precondition_counters.size();
    preconditions = static_cast<atomic<int64_t> *>(malloc(precondition_ctr_bytes));
    memcpy(preconditions, subgraph->operation_precondition_counters.data(),
           precondition_ctr_bytes);

    // Next, create a fresh queue for each processor.
    size_t queue_bytes =
        sizeof(atomic<int64_t>) * subgraph->initial_processor_queues.data.size();
    processor_queues = static_cast<atomic<int64_t> *>(malloc(queue_bytes));
    memcpy(processor_queues, subgraph->initial_processor_queues.data.data(), queue_bytes);

    // Allocate the finish counter. All processors will decrement this counter
    // upon finishing their work. When asynchronous work is implemented, this
    // counter will also include asynchronous finish items to ensure that the
    // subgraph completion event is only triggered after all subgraph work
    // is completed.
    {
      int64_t count = subgraph->subgraph_processors.size();
      // In the future, we'll increment counter to include asynchronous
      // finish items, asynchronous background work items, and
      // profiling responses.
      finish_counter.store(count);
    }

    processor_state.resize(subgraph->subgraph_processors.size());
    for(size_t i = 0; i < subgraph->subgraph_processors.size(); i++) {
      processor_state[i].queue_back.store(subgraph->initial_queue_entry_counts[i]);
    }
  }

  SubgraphExecutionState::~SubgraphExecutionState()
  {
    if(args != nullptr) {
      free(args);
    }
    free(preconditions);
    free(processor_queues);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcSubgraphExecutor
  //
  ////////////////////////////////////////////////////////////////////////

  ProcSubgraphExecutor::ProcSubgraphExecutor(Processor _proc,
                                             ThreadedTaskScheduler *_scheduler)
    : current_subgraph(nullptr)
    , proc(_proc)
    , proc_index(-1)
    , queue_front(-1)
    , scheduler(_scheduler)
  {}

  ProcSubgraphExecutor::~ProcSubgraphExecutor() {}

  bool ProcSubgraphExecutor::execute_subgraph_work()
  {
    // If we don't currently have a subgraph, try to acquire one.
    if(!has_active_subgraph()) {
      if(!try_acquire_subgraph()) {
        return false;
      }

      // TODO (rohany): This is going to be trickier when we actually have
      //  a real context to manage that messes with thread-local state and
      //  we allow for subgraph tasks to get pre-empted and other (either kernel
      //  or user-threads) will then start interacting with this executor. We'll
      //  punt on this for now though.
      // We definitely have some work now. Since we just acquired a subgraph,
      // push the execution context for this subgraph.
      push_subgraph_execution_context();
    }
    assert(current_subgraph != nullptr && proc_index != -1 && queue_front >= 0);
    SubgraphImpl *subgraph_impl = current_subgraph->subgraph;
    LocalTaskProcessor *proc_impl = subgraph_impl->subgraph_processor_impls[proc_index];

    // Find the offset into the big processor queue for this processor.
    int64_t queue_proc_offset =
        subgraph_impl->initial_processor_queues.offsets[proc_index];
    // The next slot to read in the queue is the current queue_front value.
    int64_t queue_index = queue_proc_offset + queue_front;

    // We also need to make sure that queue_front here is within the range of what
    // locations are allowed for this processor. What could happen if we don't is the
    // following:
    // 1. This thread gets the last task that this processor is assigned in the subgraph.
    // 2. The thread runs the task and waits on an event to go to sleep.
    // 3. The scheduler spawns/wakes another thread to run other work in the scheduler and
    //    enters this function and looks at the next place in the queue which actually
    //    another processor's queue and starts running work for the other processor.
    if(queue_index >= subgraph_impl->initial_processor_queues.offsets[proc_index + 1]) {
      return false;
    }

    atomic<int64_t> &queue_slot = current_subgraph->processor_queues[queue_index];
    // Try to get a piece of work from the queue.
    int64_t queue_entry = queue_slot.load_acquire();
    if(queue_entry == SUBGRAPH_EMPTY_QUEUE_ENTRY) {
      // In this case, we didn't find any work to do. The prototype
      // implementation of the subgraph compilation had the option to
      // spin in the scheduler here until work appeared, which we can
      // investigate later if it becomes important for performance.
      return false;
    }

    // If we're here, that means we actually found some work to do, so
    // let's execute it. Before we start that, we need to bump the
    // next_queue_slot pointer for this processor. We need to do this
    // so that if we go to sleep running the acquired task, the next thread
    // woken up by the scheduler will pick a different task to run instead
    // of the same task that we would have just gone to sleep running. We'll
    // also pull this onto the stack to avoid it changing from underneath us.
    int64_t next_queue_front = queue_front++;

    // Find the operation to run.
    const SubgraphImpl::SubgraphOperationDesc &op_desc =
        subgraph_impl->compiled_subgraph_operations[queue_entry];
    // Processors should only be running tasks.
    assert(op_desc.op_kind == SubgraphDefinition::OPKIND_TASK);
    const SubgraphDefinition::TaskDesc &task_desc =
        subgraph_impl->defn->tasks[op_desc.op_index];
    // TODO (rohany): In future work, support interpolations.

    // Set thread-local state before starting the task.
    ThreadLocal::current_processor = proc;
    Thread *thread = Thread::self();
    thread->start_subgraph_task_execution();

    // We can't hold the lock while executing tasks.
    scheduler->lock.unlock();
    // Execute the task.
    proc_impl->execute_task(task_desc.task_id, task_desc.args);
    // Re-acquire the scheduler lock.
    scheduler->lock.lock();

    // Restore thread-local state after the task.
    thread->stop_subgraph_task_execution();
    ThreadLocal::current_processor = Processor::NO_PROC;

    // Trigger the out-bound dependencies of this operation.
    const auto &outgoing_edges = subgraph_impl->operation_outgoing_edges;
    for(unsigned i = outgoing_edges.offsets[queue_entry];
        i < outgoing_edges.offsets[queue_entry + 1]; i++) {
      const SubgraphImpl::EdgeInfo &edge_info = outgoing_edges.data[i];
      const SubgraphImpl::SubgraphOperationDesc &target_op =
          subgraph_impl->compiled_subgraph_operations[edge_info.index];
      // We should only be dealing with tasks in the current implementation.
      assert(target_op.op_kind == SubgraphDefinition::OPKIND_TASK);
      const SubgraphDefinition::TaskDesc &target_task_desc =
          subgraph_impl->defn->tasks[target_op.op_index];

      // TODO (rohany): When there are more cases to handle here, we'll want to
      //  extract this logic into a centralized helper function.
      // Decrement the precondition counter for this operation. If we are
      // final dependency, then we need to add the operation to the target
      // processor's work queue. This operation is analagous to "sending the
      // target actor a message" in the view of the compiled subgraph as a
      // translation to an actor-based programming model.
      atomic<int64_t> &trigger = current_subgraph->preconditions[edge_info.index];
      int64_t remaining = trigger.fetch_sub_acqrel(1) - 1;
      if(remaining == 0) {
        // Get a slot to insert the next task for the target processor at.
        auto target_proc_index =
            current_subgraph->subgraph->processor_to_index.at(target_task_desc.proc);
        auto target_queue_slot = current_subgraph->processor_state[target_proc_index]
                                     .queue_back.fetch_add_acqrel(1);
        // Turn the target_queue_slot into a global index into the processor_queues array.
        target_queue_slot =
            target_proc_index +
            subgraph_impl->initial_processor_queues.offsets[target_proc_index];
        current_subgraph->processor_queues[target_queue_slot].store_release(
            edge_info.index);
        // Notify the target processor's scheduler that there might be new work.
        subgraph_impl->subgraph_processor_impls[target_proc_index]
            ->notify_scheduler_of_new_work();
      }
    }

    // If we hit the end of the queue for this processor, then we can potentially start
    // the cleanup process for this subgraph.
    if(subgraph_impl->initial_processor_queues.offsets[proc_index] + next_queue_front ==
       subgraph_impl->initial_processor_queues.offsets[proc_index + 1]) {
      // Clear the execution context for this subgraph.
      pop_subgraph_execution_context();

      // Decrement the finish counter for this subgraph. If this processor
      // was the last one to finish, then we can trigger the finish event.
      // Note that in the future, there may be pending asynchronous work that
      // is also contributing to the finish counter, so even if all processors finish
      // the counter is still not 0.
      int64_t finish_count = current_subgraph->finish_counter.fetch_sub_acqrel(1) - 1;
      if(finish_count == 0) {
        // We can't hold the scheduler lock while we trigger the event.
        scheduler->lock.unlock();
        current_subgraph->finish_event.trigger();
        scheduler->lock.lock();
      }

      // Now that we're done with all the work, we can release this subgraph from
      // the current processor.
      release_subgraph();
    }

    // Work was done, so return true.
    return true;
  }

  bool ProcSubgraphExecutor::try_acquire_subgraph()
  {
    // Peek at the top of the pending subgraphs queue. If taking
    // this reader lock in the case that we don't have contention is
    // expensive, we can pivot to a work-counter based implementation
    // that skips this entire check if it is known that no subgraphs
    // have been enqueued since the last check.
    RWLock::AutoReaderLock al(pending_subgraphs_lock);
    if(!pending_subgraphs.empty()) {
      current_subgraph = pending_subgraphs.front();
      pending_subgraphs.pop();
      proc_index = current_subgraph->subgraph->processor_to_index.at(proc);
      queue_front = 0;
      return true;
    }
    return false;
  }

  void ProcSubgraphExecutor::release_subgraph()
  {
    // Subgraph release just unsets some fields in the ProcSubgraphExecutor.
    // The subgraph data itself will be cleaned up by separate processes.
    current_subgraph = nullptr;
    proc_index = -1;
    queue_front = -1;
  }

  void ProcSubgraphExecutor::push_subgraph_execution_context()
  {
    // This is currently a no-op.
  }

  void ProcSubgraphExecutor::pop_subgraph_execution_context()
  {
    // This is currently a no-op.
  }

  void ProcSubgraphExecutor::enqueue_subgraph(SubgraphExecutionState *subgraph)
  {
    {
      RWLock::AutoWriterLock al(pending_subgraphs_lock);
      pending_subgraphs.push(subgraph);
    }
    // Notify the scheduler that there is new work available.
    scheduler->notify_of_new_work();
  }

}; // namespace Realm
