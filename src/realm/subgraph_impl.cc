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
#include "realm/runtime_impl.h"
#include "realm/proc_impl.h"
#include "realm/idx_impl.h"
#include "realm/transfer/transfer.h"

#include <unordered_set>

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
                          << " ops=" << impl->schedule.size();
      return Event::NO_EVENT;
    } else {
      // fatal error for now - once we have profiling, return a poisoned event
      //  if there was a profiling request for OperationStatus
      log_subgraph.fatal() << "subgraph compilation failed";
      abort();
    }
  }

  void Subgraph::destroy(Event wait_on /*= Event::NO_EVENT*/) const
  {
    NodeID owner = ID(*this).subgraph_owner_node();

    log_subgraph.info() << "destroy: subgraph=" << *this << " wait_on=" << wait_on;

    if(owner == Network::my_node_id) {
      SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(*this);

      if(wait_on.has_triggered())
        subgraph->destroy();
      else
        subgraph->deferred_destroy.defer(subgraph, wait_on);
    } else {
      ActiveMessage<SubgraphDestroyMessage> amsg(owner);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg.commit();
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

  bool SubgraphImpl::task_has_async_effects(SubgraphDefinition::TaskDesc& task) {
    switch (task.proc.kind()) {
      case Processor::TOC_PROC: {
        // TODO (rohany): Query an API in the future.
        return true;
      }
      default:
        return false;
    }
  }

  bool SubgraphImpl::copy_has_async_effects(unsigned op_idx) {
    // We can't just look at the source and destination of the copy,
    // as the GPU's copy engines can even be used sometimes for H2H
    // copies. So just look at the XD and see if it has async effects.
    auto xd = planned_copy_xds[op_idx];
    assert(xd);
    return xd->launches_async_work();
  }

  bool SubgraphImpl::task_respects_async_effects(SubgraphDefinition::TaskDesc& src, SubgraphDefinition::TaskDesc& dst) {
    // TODO (rohany): Do more detailed analysis in the future.
    return src.proc.kind() == Processor::TOC_PROC && dst.proc.kind() == Processor::TOC_PROC;
  }

  bool SubgraphImpl::operation_respects_async_effects(
    SubgraphDefinition::OpKind src_op_kind, unsigned src_op_idx,
    SubgraphDefinition::OpKind dst_op_kind, unsigned dst_op_idx
  ) {
    switch (dst_op_kind) {
      case SubgraphDefinition::OPKIND_TASK: {
        switch (src_op_kind) {
          case SubgraphDefinition::OPKIND_TASK: {
            return task_respects_async_effects(defn->tasks[src_op_idx], defn->tasks[dst_op_idx]);
          }
          case SubgraphDefinition::OPKIND_COPY: {
            // TODO (rohany): I don't know the right way that this API
            //  could look yet, so for now, a "copy being async" means
            //  that it pushes work onto a GPU. So the check here is
            //  just copy is async + task is on GPU.
            return copy_has_async_effects(src_op_idx) && task_has_async_effects(defn->tasks[dst_op_idx]);
          }
          default:
            return false;
        }
      }
      case SubgraphDefinition::OPKIND_COPY: {
        // See the comment above in the COPY->TASK case. We'll pretend
        // an async copy is really a copy serviced by the GPU.
        switch (src_op_kind) {
          case SubgraphDefinition::OPKIND_TASK: {
            return copy_has_async_effects(dst_op_idx) && task_has_async_effects(defn->tasks[src_op_idx]);
          }
          case SubgraphDefinition::OPKIND_COPY: {
            return copy_has_async_effects(src_op_idx) && copy_has_async_effects(dst_op_idx);
          }
          default:
            return false;
        }
      }
      default:
        return false;
    }
  }

  bool SubgraphImpl::operation_has_async_effects(SubgraphDefinition::OpKind op_kind, unsigned op_idx) {
    switch (op_kind) {
      case SubgraphDefinition::OPKIND_TASK: {
        return task_has_async_effects(defn->tasks[op_idx]);
      }
      case SubgraphDefinition::OPKIND_COPY: {
        return copy_has_async_effects(op_idx);
      }
      case SubgraphDefinition::OPKIND_EXT_PRECOND: [[fallthrough]];
      case SubgraphDefinition::OPKIND_ARRIVAL: {
        return false;
      }
      default: {
        assert(false);
        return false;
      }
    }
  }

  bool is_plannable_copy(SubgraphDefinition::CopyDesc& copy) {
    auto node = Network::my_node_id;
    bool allowed = true;
    for(auto &src : copy.srcs) {
      auto mem = src.inst.get_location();
      if(mem.address_space() != node)
        allowed = false;
    }
    for(auto &dst : copy.dsts) {
      auto mem = dst.inst.get_location();
      if(mem.address_space() != node)
        allowed = false;
    }
    return allowed;
  }

  class MockXDFactory : public XferDesFactory {
  public:
    MockXDFactory() {}
    virtual bool needs_release() override { return false; }
    virtual void create_xfer_des(uintptr_t dma_op,
                                 NodeID launch_node,
                                 NodeID target_node,
                                 XferDesID guid,
                                 const std::vector<XferDesPortInfo>& inputs_info,
                                 const std::vector<XferDesPortInfo>& outputs_info,
                                 int priority,
                                 XferDesRedopInfo redop_info,
                                 const void *fill_data, size_t fill_size,
                                 size_t fill_total) override {
      assert(target_node == Network::my_node_id);
      assert(launch_node == Network::my_node_id);
      auto sxdf = dynamic_cast<SimpleXferDesFactory*>(factory);
      assert(sxdf != nullptr);
      LocalChannel *c = reinterpret_cast<LocalChannel *>(sxdf->channel);
      xd = c->create_xfer_des(dma_op, launch_node, guid,
                              inputs_info, outputs_info,
                              priority, redop_info,
                              fill_data, fill_size, fill_total);
    }
    XferDesFactory* factory = nullptr;
    XferDes* xd = nullptr;
  };

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
        base = static_cast<char *>(malloc(_needed));
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

  void
  do_interpolation_inline(const std::vector<SubgraphDefinition::Interpolation> &interpolations,
                          unsigned first_interp, unsigned num_interps,
                          SubgraphDefinition::Interpolation::TargetKind target_kind,
                          unsigned target_index,
                          const void *srcdata, size_t srclen,
                          void *dstdata, size_t dstlen)
  {
    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation &it = interpolations[first_interp + i];
      if ((it.target_kind != target_kind) || (it.target_index != target_index))
        continue;

      assert((it.offset + it.bytes) <= srclen);
      if(it.redop_id == 0) {
        // overwrite
        assert((it.target_offset + it.bytes) <= dstlen);
        memcpy(reinterpret_cast<char *>(dstdata) + it.target_offset,
               reinterpret_cast<const char *>(srcdata) + it.offset, it.bytes);
      } else {
        assert(false);
        // const ReductionOpUntyped *redop =
        //     get_runtime()->reduce_op_table.get(it.redop_id, 0);
        // assert((it.target_offset + redop->sizeof_lhs) <= dstlen);
        // (redop->cpu_apply_excl_fn)(reinterpret_cast<char *>(scratch_buffer) +
        //                            it.target_offset,
        //                            0, reinterpret_cast<const char *>(srcdata) + it.offset,
        //                            0, 1 /*count*/, redop->userdata);
      }
    }
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

    ModuleConfig *core = Runtime::get_runtime().get_module_config("core");
    bool rt_opt = false;
    assert(core->get_property("static_subgraph_opt", rt_opt) == REALM_SUCCESS);
    opt = rt_opt && defn->concurrency_mode == SubgraphDefinition::INSTANTIATION_ORDER;
    if (opt)
      log_subgraph.info() << "Performing static subgraph optimizations.";

    typedef std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, unsigned> TopoMap;
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

    schedule.resize(total_ops);
    for(TopoMap::const_iterator it = toposort.begin(); it != toposort.end(); ++it) {
      schedule[it->second].op_kind = it->first.first;
      schedule[it->second].op_index = it->first.second;
    }

    // Maintain a separate collection of tasks to interpolation metadata.
    // We'll use this when extracting the interpolation metadata for the
    // statically scheduled graph component.
    std::map<std::pair<SubgraphDefinition::OpKind, size_t>, std::pair<size_t, size_t>> op_to_interpolation_data;

    // sort the interpolations so that each operation has a compact range
    //  to iterate through
    std::sort(defn->interpolations.begin(), defn->interpolations.end(),
              SortInterpolationsByKindAndIndex());
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
        it != schedule.end(); ++it) {
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
            op_to_interpolation_data[{it->op_kind, it->op_index}] = {lo, hi-lo};
            // log_subgraph.print() << "search (" << it->op_kind << "," << it->op_index <<
            // ") -> " << lo << " .. " << hi;
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

    // Perform "final event" analysis before partitioning the schedule.
    num_final_events = 0;
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
        it != schedule.end(); ++it) {
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
      if (it->src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND)
        continue;
      TopoMap::const_iterator src =
          toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
      // If we are depending on port 0 of another node and we're not an
      //  external postcondition, then the preceeding node is not final.
      if((it->src_op_port == 0) &&
         (it->tgt_op_kind != SubgraphDefinition::OPKIND_EXT_POSTCOND) &&
         (schedule[src->second].is_final_event)) {
        schedule[src->second].is_final_event = false;
        num_final_events--;
      }
    }

    // Separate the schedules into "static" and "dynamic" portions of
    // the original schedule.
    // Construct a separate schedule for operations that are not
    // included in the "statically-optimized" portion of the graph.
    std::vector<SubgraphScheduleEntry> dynamic_schedule;
    std::vector<SubgraphScheduleEntry> static_schedule;

    std::vector<SubgraphDefinition::OpKind> allowed_kinds;
    allowed_kinds.push_back(SubgraphDefinition::OPKIND_TASK);
    allowed_kinds.push_back(SubgraphDefinition::OPKIND_ARRIVAL);
    allowed_kinds.push_back(SubgraphDefinition::OPKIND_EXT_PRECOND);
    // TODO (rohany): Comment ...
    std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, unsigned> bgwork_schedule;

    if (opt) {
      // Reset the toposort for the new schedule.
      toposort.clear();
      for (auto& it : schedule) {
        auto key = std::make_pair(it.op_kind, it.op_index);
        // Copies require special handling, as they don't go into the
        // per-processor local schedules.
        if (it.op_kind == SubgraphDefinition::OPKIND_COPY && is_plannable_copy(defn->copies[it.op_index])) {
          bgwork_schedule[key] = bgwork_items.size();
          bgwork_items.push_back(it);
          if (it.is_final_event)
            num_final_events--;
          continue;
        }

        // Operations we know how to handle go into the static schedule.
        if (std::find(allowed_kinds.begin(), allowed_kinds.end(), it.op_kind) != allowed_kinds.end()) {
            static_schedule.push_back(it);
          // If we're moving an operation into the static part of the schedule,
          // its contribution to the final event in the subgraph will be done
          // separately, so remove it from the "dynamic" set of final events.
          if (it.is_final_event)
            num_final_events--;
        } else {
          toposort[{it.op_kind, it.op_index}] = dynamic_schedule.size();
          dynamic_schedule.push_back(it);
        }
      }

      // The schedule is now only the dynamic portion.
      schedule = dynamic_schedule;
    }

    // count number of intermediate events - instantiations can produce more
    //  than one
    num_intermediate_events = 0;
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
        it != schedule.end(); ++it) {
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
      if (tgt == toposort.end()) {
        continue;
      }
      assert(tgt != toposort.end());

      switch(it->src_op_kind) {
      case SubgraphDefinition::OPKIND_EXT_PRECOND: {
        // external preconditions are encoded as negative indices
        int idx = -1 - (int)(it->src_op_index);
        schedule[tgt->second].preconditions.push_back(
            std::make_pair(it->tgt_op_port, idx));
        break;
      }

      default: {
        TopoMap::const_iterator src =
            toposort.find(std::make_pair(it->src_op_kind, it->src_op_index));
        // Same story for the source edge.
        if (src == toposort.end())
          continue;
        unsigned ev_idx = schedule[src->second].intermediate_event_base + it->src_op_port;
        schedule[tgt->second].preconditions.push_back(
            std::make_pair(it->tgt_op_port, ev_idx));
        break;
      }
      }
    }

    // now sort the preconditions for each entry - allows us to group by port
    //  and also notice duplicates
    max_preconditions = 1; // have to count global precondition when needed
    for(std::vector<SubgraphScheduleEntry>::iterator it = schedule.begin();
        it != schedule.end(); ++it) {
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

    std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, std::vector<std::pair<SubgraphDefinition::OpKind, unsigned>>> incoming_edges;
    std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, std::vector<std::pair<SubgraphDefinition::OpKind, unsigned>>> outgoing_edges;
    // Perform a group-by on the dependencies list.
    for (auto& it : defn->dependencies) {
      // Add the incoming edge.
      auto skey = std::make_pair(it.src_op_kind, it.src_op_index);
      auto tkey = std::make_pair(it.tgt_op_kind, it.tgt_op_index);
      incoming_edges[tkey].push_back(skey);
      outgoing_edges[skey].push_back(tkey);
    }

    // If we're doing optimization, make sure that the graph only
    // contains operations we know about.
    if (opt) {
      std::vector<SubgraphDefinition::OpKind> disallowed_edges;
      disallowed_edges.push_back(SubgraphDefinition::OPKIND_INSTANTIATION);
      for (auto& it : incoming_edges) {
        // For now, tasks should only depend on allowed operations.
        for (auto& edge : it.second) {
          assert(std::find(disallowed_edges.begin(), disallowed_edges.end(), edge.first) == disallowed_edges.end());
        }
      }
      // Each node's outgoing edge should also to an allowed operation.
      for (auto& it : outgoing_edges) {
        for (auto& edge : it.second) {
          assert(std::find(disallowed_edges.begin(), disallowed_edges.end(), edge.first) == disallowed_edges.end());
        }
      }
    }

    {
      planned_copy_xds.resize(defn->copies.size(), nullptr);
      for (auto& it : bgwork_items) {
        if (it.op_kind != SubgraphDefinition::OPKIND_COPY)
          continue;
        auto& copy = defn->copies[it.op_index];
        planned_copy_xds[it.op_index] = analyze_copy(copy);
      }
    }

    std::map<Processor, int32_t> proc_to_index;
    std::map<Processor, std::vector<std::pair<SubgraphDefinition::OpKind, unsigned>>> local_schedules;
    std::map<Processor, std::vector<OpMeta>> local_op_meta;
    std::map<Processor, std::vector<int32_t>> local_incoming;
    std::map<Processor, std::vector<std::vector<CompletionInfo>>> local_outgoing;
    std::map<Processor, std::vector<std::vector<CompletionInfo>>> async_effect_triggers;
    std::map<Processor, std::vector<std::vector<CompletionInfo>>> async_incoming_events;
    // Gather the interpolations into compacted lists for each processor.
    std::map<Processor, std::vector<std::vector<SubgraphDefinition::Interpolation>>> local_interpolations;
    // Figure out which operations each external precondition
    // is going to trigger.
    std::map<unsigned, std::vector<CompletionInfo>> external_preconditions;

    // Set up the processor -> id mapping.
    for (auto& it : static_schedule) {
      switch (it.op_kind) {
        case SubgraphDefinition::OPKIND_TASK: {
          auto& task = defn->tasks[it.op_index];
          // During iteration, also initialize the mapping
          // of processor IDs to local indices.
          if (proc_to_index.find(task.proc) == proc_to_index.end()) {
            proc_to_index[task.proc] = int32_t(proc_to_index.size());
            all_procs.push_back(task.proc);
            auto pimpl = get_runtime()->get_processor_impl(task.proc);
            auto lp = dynamic_cast<LocalTaskProcessor*>(pimpl);
            assert(lp);
            all_proc_impls.push_back(lp);
          }
          break;
        }
        case SubgraphDefinition::OPKIND_COPY: [[fallthrough]];
        case SubgraphDefinition::OPKIND_ARRIVAL:
          break;
        default:
          assert(false);
      }
    }

    // Remember where we placed all of our operations.
    std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, size_t> operation_indices;
    std::map<std::pair<SubgraphDefinition::OpKind, unsigned>, int32_t> operation_procs;

    // For now, round-robin arrivals across the available
    // processors. In the future, we should put an arrival on the
    // likely last processor to trigger it.
    size_t arrival_proc_idx = 0;

    for (auto& it : static_schedule) {
      auto desc = std::make_pair(it.op_kind, it.op_index);
      switch (it.op_kind) {
        case SubgraphDefinition::OPKIND_TASK: {
          auto& task = defn->tasks[it.op_index];
          operation_indices[desc] = local_schedules[task.proc].size();
          operation_procs[desc] = proc_to_index[task.proc];
          local_schedules[task.proc].push_back(desc);
          // Comment for here and below. It's OK that incoming_edges
          // can contain edges that point from the dynamic partition
          // into the static partition. The total count of preconditions
          // is still valid, as the precondition from the other partition
          // still has to be respected.
          local_incoming[task.proc].push_back(int32_t(incoming_edges[desc].size()));
          OpMeta meta;
          meta.is_final_event = it.is_final_event;
          meta.is_async = task_has_async_effects(task);
          local_op_meta[task.proc].push_back(meta);
          break;
        }
        case SubgraphDefinition::OPKIND_ARRIVAL: {
          auto proc = all_procs[arrival_proc_idx++ % all_procs.size()];
          operation_indices[desc] = local_schedules[proc].size();
          operation_procs[desc] = proc_to_index[proc];
          local_schedules[proc].push_back(desc);
          local_incoming[proc].push_back(int32_t(incoming_edges[desc].size()));
          OpMeta meta;
          meta.is_final_event = it.is_final_event;
          meta.is_async = false;
          local_op_meta[proc].push_back(meta);
          break;
        }
        default:
          assert(false);
      }
    }

    // Initialize the record of static->dynamic edges.
    static_to_dynamic_counts = std::vector<int32_t>(schedule.size(), 0);

    // Also initialize preconditions of bgwork items.
    bgwork_preconditions = std::vector<int32_t>(bgwork_items.size(), 0);
    bgwork_postconditions = std::vector<std::vector<CompletionInfo>>(bgwork_items.size());
    bgwork_async_preconditions = std::vector<std::vector<CompletionInfo>>(bgwork_items.size());
    bgwork_async_postconditions = std::vector<std::vector<CompletionInfo>>(bgwork_items.size());
    for (size_t i = 0; i < bgwork_items.size(); i++) {
      auto& it = bgwork_items[i];
      auto key = std::make_pair(it.op_kind, it.op_index);
      bgwork_preconditions[i] = int32_t(incoming_edges[key].size());

      auto& outgoing = bgwork_postconditions[i];
      for (auto& edge : outgoing_edges[key]) {
        if (bgwork_schedule.find(edge) != bgwork_schedule.end()) {
          auto idx = bgwork_schedule.at(edge);
          CompletionInfo info(-1, idx, CompletionInfo::EdgeKind::BGWORK_TO_BGWORK);
          outgoing.push_back(info);
        } else if (toposort.find(edge) != toposort.end()) {
          auto idx = toposort.at(edge);
          static_to_dynamic_counts[idx]++;
          CompletionInfo info(-1, idx, CompletionInfo::EdgeKind::BGWORK_TO_DYNAMIC);
          outgoing.push_back(info);
        } else {
          CompletionInfo info(operation_procs[edge], operation_indices[edge], CompletionInfo::EdgeKind::BGWORK_TO_STATIC);
          outgoing.push_back(info);
        }
      }

      // TODO (rohany): This is some unfortunate code duplication, where we
      //  basically have to do the same thing as below ...
      if (operation_has_async_effects(it.op_kind, it.op_index)) {

        // Remember the processor that we need to return the
        // async events to.
        if (it.op_kind == SubgraphDefinition::OPKIND_COPY) {
          auto proc = planned_copy_xds[it.op_index]->get_async_event_proc();
          bgwork_async_event_procs[proc].push_back(i);
        }

        for (auto& edge : outgoing_edges[key]) {
          // Similar to below, see if the edge is to the dynamic partition.
          if (toposort.find(edge) != toposort.end()) {
            auto idx = toposort.at(edge);
            static_to_dynamic_counts[idx]++;
            bgwork_async_postconditions[i].emplace_back(-1, idx, CompletionInfo::EdgeKind::BGWORK_TO_DYNAMIC);
            continue;
          }

          // Respects case.
          if (operation_respects_async_effects(it.op_kind, it.op_index, edge.first, edge.second)) {
            // If the child operation knows how to sequence itself
            // after us without blocking, then we don't have to worry.
            continue;
          }

          auto bgwork_it = bgwork_schedule.find(edge);
          if (bgwork_it != bgwork_schedule.end()) {
            bgwork_preconditions[bgwork_it->second]++;
            bgwork_async_postconditions[i].emplace_back(-1, bgwork_it->second, CompletionInfo::EdgeKind::BGWORK_TO_BGWORK);
          } else {
            auto procidx = operation_procs[edge];
            auto schedidx = operation_indices[edge];
            local_incoming[all_procs[procidx]][schedidx]++;
            // Remember the processors that this operation will have to trigger.
            bgwork_async_postconditions[i].emplace_back(procidx, schedidx, CompletionInfo::EdgeKind::BGWORK_TO_STATIC);
          }
        }
        for (auto& edge : incoming_edges[key]) {
          // Respects case. See the comments below.
          if (!operation_has_async_effects(edge.first, edge.second) ||
              !operation_respects_async_effects(edge.first, edge.second, it.op_kind, it.op_index)) {
            continue;
          }
          // Enqueue case.
          auto bgwork_it = bgwork_schedule.find(edge);
          if (bgwork_it != bgwork_schedule.end()) {
            bgwork_async_preconditions[i].emplace_back(-1, bgwork_it->second, CompletionInfo::EdgeKind::BGWORK_TO_BGWORK);
          } else {
            bgwork_async_preconditions[i].emplace_back(operation_procs[edge], operation_indices[edge], CompletionInfo::EdgeKind::STATIC_TO_BGWORK);
          }
        }
      }
    }

    // TODO (rohany): Perform a big refactor of this code that unifies
    //  the background work items with the normal schedule. Since we have
    //  the dynamic queue etc, we don't need things to be assigned solely
    //  to processors etc and have it as rigid as it is right now. We can
    //  do it "badly" once, and then fix it the next time.
    // Now that all tasks have been added, fill out per-task information
    // like outgoing neighbors and interpolations.
    for (auto& it : static_schedule) {
      auto key = std::make_pair(it.op_kind, it.op_index);
      auto proc = all_procs[operation_procs[key]];

      std::vector<CompletionInfo> outgoing;
      for (auto& edge : outgoing_edges[key]) {
        if (bgwork_schedule.find(edge) != bgwork_schedule.end()) {
          auto idx = bgwork_schedule.at(edge);
          CompletionInfo info(-1, idx, CompletionInfo::EdgeKind::STATIC_TO_BGWORK);
          outgoing.push_back(info);
        } else if (toposort.find(edge) != toposort.end()) {
          // NOTE: External postconditions are already included in the dynamic
          //  schedule, and wouldn't be mapped into the static schedule. So
          //  they are included implicitly in this case.
          auto idx = toposort.at(edge);
          static_to_dynamic_counts[idx]++;
          CompletionInfo info(-1, idx, CompletionInfo::EdgeKind::STATIC_TO_DYNAMIC);
          outgoing.push_back(info);
        } else {
          CompletionInfo info(operation_procs[edge], operation_indices[edge], CompletionInfo::EdgeKind::STATIC_TO_STATIC);
          outgoing.push_back(info);
        }
      }
      local_outgoing[proc].push_back(outgoing);

      // Handle tasks with potentially asynchronous behavior.
      if (operation_has_async_effects(it.op_kind, it.op_index)) {
        // If an operation is asynchronous, then we might need extra
        // bookkeeping for dependent operations that can only start once
        // the asynchronous effects are complete.
        std::vector<CompletionInfo> to_trigger;
        for (auto& edge : outgoing_edges[key]) {
          // If the consumer of this operation is outside the static
          // partition of the graph, then it is opaque to us, and we have
          // to wait for all effects of this task to complete before
          // it can start. Note that external postconditions are also
          // included by this case.
          if (toposort.find(edge) != toposort.end()) {
            auto idx = toposort.at(edge);
            static_to_dynamic_counts[idx]++;
            to_trigger.emplace_back(-1, idx, CompletionInfo::EdgeKind::STATIC_TO_DYNAMIC);
            continue;
          }

          if (operation_respects_async_effects(it.op_kind, it.op_index, edge.first, edge.second)) {
            // If the child operation knows how to sequence itself
            // after us without blocking, then we don't have to worry.
            continue;
          }

          // Otherwise, make the dependent operation have to wait on the
          // control code of this operation completing, _and_ its
          // asynchronous effect.
          auto bgwork_it = bgwork_schedule.find(edge);
          if (bgwork_it != bgwork_schedule.end()) {
            bgwork_preconditions[bgwork_it->second]++;
            to_trigger.emplace_back(-1, bgwork_it->second, CompletionInfo::EdgeKind::STATIC_TO_BGWORK);
          } else {
            auto procidx = operation_procs[edge];
            auto schedidx = operation_indices[edge];
            local_incoming[all_procs[procidx]][schedidx]++;
            // Remember the processors that this operation will have to trigger.
            to_trigger.emplace_back(procidx, schedidx, CompletionInfo::EdgeKind::STATIC_TO_STATIC);
          }
        }
        async_effect_triggers[proc].push_back(to_trigger);
        // Additionally, we need to "wait" on the right "tokens" of
        // incoming asynchronous operations.
        std::vector<CompletionInfo> will_trigger;
        for (auto& edge : incoming_edges[key]) {
          if (!operation_has_async_effects(edge.first, edge.second) ||
              !operation_respects_async_effects(edge.first, edge.second, it.op_kind, it.op_index)) {
            // There are two cases in which we don't need to worry about
            // asynchronous predecessors.
            // 1) The predecessor is not asynchronous.
            // 2) We don't know how to handle async effects. In this case,
            //    we've already registered ourselves as dependent on the
            //    asynchronous effects completing.
            continue;
          }
          // If we do know how to handle the incoming async effect, then
          // remember the async predecessor tasks to wait on.
          auto bgwork_it = bgwork_schedule.find(edge);
          if (bgwork_it != bgwork_schedule.end()) {
            will_trigger.emplace_back(-1, bgwork_it->second, CompletionInfo::EdgeKind::BGWORK_TO_STATIC);
          } else {
            will_trigger.emplace_back(operation_procs[edge], operation_indices[edge], CompletionInfo::EdgeKind::STATIC_TO_STATIC);
          }
        }
        async_incoming_events[proc].push_back(will_trigger);
      } else {
        // Otherwise, initialize the metadata structures with null data.
        async_effect_triggers[proc].push_back({});
        async_incoming_events[proc].push_back({});
      }

      std::vector<SubgraphDefinition::Interpolation> interps;
      auto& meta = op_to_interpolation_data[key];
      for (size_t i = meta.first; i < meta.first + meta.second; i++) {
        interps.push_back(defn->interpolations[i]);
      }
      local_interpolations[proc].push_back(interps);
    }

    // Preallocate a set of vectors for external operations
    // to trigger upon completion.
    dynamic_to_static_triggers.resize(schedule.size());
    // Perform external precondition analysis.
    unsigned max_ext_precond_id = 0;
    for (auto& it : static_schedule) {
      auto key = std::make_pair(it.op_kind, it.op_index);
      CompletionInfo info(operation_procs[key], operation_indices[key], CompletionInfo::EdgeKind::DYNAMIC_TO_STATIC);
      for (auto& edge : incoming_edges[key]) {
        if (edge.first == SubgraphDefinition::OPKIND_EXT_PRECOND) {
          external_preconditions[edge.second].push_back(info);
          max_ext_precond_id = std::max(max_ext_precond_id, edge.second);
        } else if (toposort.find(edge) != toposort.end()) {
          // In this case, we have an incoming edge from
          // the dynamic partition into the static partition.
          dynamic_to_static_triggers[toposort.at(edge)].to_trigger.push_back(info);
        }
      }
    }
    // Do the same analysis for bgwork items.
    for (size_t i = 0; i < bgwork_items.size(); i++) {
      auto& it = bgwork_items[i];
      auto key = std::make_pair(it.op_kind, it.op_index);
      CompletionInfo info(-1, i, CompletionInfo::EdgeKind::DYNAMIC_TO_BGWORK);
      for (auto& edge : incoming_edges[key]) {
        if (edge.first == SubgraphDefinition::OPKIND_EXT_PRECOND) {
          external_preconditions[edge.second].push_back(info);
          max_ext_precond_id = std::max(max_ext_precond_id, edge.second);
        } else if (toposort.find(edge) != toposort.end()) {
          // In this case, we have an incoming edge from
          // the dynamic partition into the static partition.
          dynamic_to_static_triggers[toposort.at(edge)].to_trigger.push_back(info);
        }
      }
    }

    // Collapse the metadata for each external waiter.
    if (!external_preconditions.empty()) {
      external_precondition_info.resize(max_ext_precond_id + 1);
      for (size_t i = 0; i < max_ext_precond_id + 1; i++) {
        external_precondition_info[i].to_trigger = external_preconditions[i];
      }
    }

    // Construct the initial "queue" for each processor. This will contain
    // all operations that do not have any preconditions, and then a default
    // value for all other spaces in the queue.
    std::map<Processor, std::vector<int64_t>> processor_queues;
    initial_queue_entry_count = std::vector<uint64_t>(all_procs.size());
    for (size_t i = 0; i < all_procs.size(); i++) {
      auto proc = all_procs[i];
      auto& preds = local_incoming[proc];
      std::vector<int64_t> queue(preds.size(), SUBGRAPH_EMPTY_QUEUE_ENTRY);
      size_t idx = 0;
      for (size_t j = 0; j < preds.size(); j++) {
        if (preds[j] == 0) {
          // Record that the operation at index `j` of this
          // processor is ready to run.
          queue[idx++] = j;
        }
      }
      processor_queues[proc] = queue;
      initial_queue_entry_count[i] = idx;
    }

    bgwork_finish_events = 0;
    for (size_t i = 0; i < bgwork_items.size(); i++) {
      // Collect the bgwork items that should be triggered immediately
      // upon subgraph replay starting.
      if (bgwork_preconditions[i] == 0)
        bgwork_items_without_preconditions.push_back(i);
      // Also collect the number of bgwork items the subgraph
      // replay needs to finish before being considered done.
      if (bgwork_items[i].is_final_event) {
        bgwork_finish_events++;
        if (operation_has_async_effects(bgwork_items[i].op_kind, bgwork_items[i].op_index)) {
          bgwork_finish_events++;
        }
      }
    }

    // Identify how many asynchronous operations will contribute to
    // the subgraph's completion event.
    async_finish_events = std::vector<int32_t>(all_procs.size(), 0);
    for (auto& it : static_schedule) {
      if (it.is_final_event && operation_has_async_effects(it.op_kind, it.op_index)) {
        auto proc = operation_procs.at({it.op_kind, it.op_index});
        async_finish_events[proc]++;
      }
    }
    // TODO (rohany): This feels like an big hack, but I should
    //  probably unify the async items per proc with "anything async
    //  happening not in the task launch side of the world".
    if (opt) {
      assert(!all_procs.empty());
      async_finish_events[0] += bgwork_finish_events;
    }

    // Flatten the operations.
    operations = FlattenedProcMap<std::pair<SubgraphDefinition::OpKind, unsigned>>(all_procs, local_schedules);
    operation_meta = FlattenedProcMap<OpMeta>(all_procs, local_op_meta);

    // Collapse the per-processor queues into a single flattened vector.
    initial_queue_state.reserve(operations.data.size());
    for (size_t i = 0; i < all_procs.size(); i++) {
      for (auto& v : processor_queues[all_procs[i]]) {
        initial_queue_state.push_back(v);
      }
    }

    // Completion information next.
    completion_infos = FlattenedProcTaskMap<CompletionInfo>(all_procs, local_outgoing);

    // Interpolation information.
    interpolations = FlattenedProcTaskMap<SubgraphDefinition::Interpolation>(all_procs, local_interpolations);

    // Finally, the precondition array.
    original_preconditions = FlattenedProcMap<int32_t>(all_procs, local_incoming);

    // Aggregate asynchronous pre- and post-condition metadata.
    async_outgoing_infos = FlattenedProcTaskMap<CompletionInfo>(all_procs, async_effect_triggers);
    async_incoming_infos = FlattenedProcTaskMap<CompletionInfo>(all_procs, async_incoming_events);

    return true;
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
                                 const ProfilingRequestSet &prs,
                                 span<const Event> preconditions,
                                 span<const Event> postconditions, Event start_event,
                                 Event finish_event, int priority_adjust)
  {
    // If we're running with the static scheduling optimization, then
    // there's extra work to be done. We have to preemptively
    // declare some data set by the optimization path though.
    GenEventImpl *event_impl = get_genevent_impl(finish_event);
    UserEvent* dynamic_events = nullptr;
    ExternalPreconditionTriggerer* dynamic_precondition_waiters = nullptr;
    atomic<int32_t>* precondition_ctrs = nullptr;
    ProcSubgraphReplayState* state = nullptr;
    if (opt) {
      // We're going to return early before completing the interpolations,
      // so we need to make a copy of the argument buffer to read from.
      void* copied_args = nullptr;
      if (args != nullptr && arglen > 0) {
        copied_args = malloc(arglen);
        memcpy(copied_args, args, arglen);
      }

      // Create a fresh copy of the preconditions array.
      static_assert(sizeof(int32_t) == sizeof(atomic<int32_t>));
      size_t precondition_ctr_bytes = sizeof(atomic<int32_t>) * original_preconditions.data.size();
      precondition_ctrs = (atomic<int32_t>*)malloc(precondition_ctr_bytes);
      memcpy(precondition_ctrs, original_preconditions.data.data(), precondition_ctr_bytes);

      // Allocate the data structures necessary to let control
      // flow from the static partition to the dynamic one.
      size_t dynamic_precondition_ctr_bytes = sizeof(atomic<int32_t>) * static_to_dynamic_counts.size();
      auto dynamic_precondition_ctrs = (atomic<int32_t>*)malloc(dynamic_precondition_ctr_bytes);
      memcpy(dynamic_precondition_ctrs, static_to_dynamic_counts.data(), dynamic_precondition_ctr_bytes);
      size_t dynamic_event_bytes = sizeof(UserEvent) * static_to_dynamic_counts.size();
      dynamic_events = (UserEvent*)malloc(dynamic_event_bytes);
      for (size_t i = 0; i < static_to_dynamic_counts.size(); i++) {
        if (static_to_dynamic_counts[i] > 0)
          dynamic_events[i] = UserEvent::create_user_event();
      }

      // TODO (rohany): Free this.
      size_t bgwork_precondition_ctr_bytes = sizeof(atomic<int32_t>) * bgwork_preconditions.size();
      auto bgwork_precondition_ctrs = (atomic<int32_t>*)malloc(bgwork_precondition_ctr_bytes);
      memcpy(bgwork_precondition_ctrs, bgwork_preconditions.data(), bgwork_precondition_ctr_bytes);

      // Create a fresh queue vector for this instantiation of the graph.
      static_assert(sizeof(int64_t) == sizeof(atomic<int64_t>));
      size_t queue_bytes = sizeof(atomic<int64_t>) * initial_queue_state.size();
      auto queue = (atomic<int64_t>*)malloc(queue_bytes);
      memcpy(queue, initial_queue_state.data(), queue_bytes);

      // TODO (rohany): In the future, these allocations could be sized more
      //  tightly, rather than for each operation.
      auto async_operation_events = (void**)malloc(sizeof(void*) * operations.data.size());
      auto async_operation_effect_triggerers = (AsyncGPUWorkTriggerer*)malloc(sizeof(AsyncGPUWorkTriggerer) * operations.data.size());
      auto async_bgwork_events = (void**)malloc(sizeof(void*) * bgwork_items.size());

      // Initialize a counter that all processors will decrement upon completion.
      auto finish_counter = (atomic<int32_t>*)malloc(sizeof(atomic<int32_t>));
      finish_counter->store_release(all_procs.size());
      auto static_finish_event = UserEvent::create_user_event();

      // Arm the result merger with the completion event, and
      // the final events from the dynamic portion.
      event_impl->merger.prepare_merger(finish_event, false /*!ignore_faults*/,
                                     num_final_events + 1);
      event_impl->merger.add_precondition(static_finish_event);

      // Separate the installation and state creation steps
      // to enable deferring the launch of the subgraph.
      state = new ProcSubgraphReplayState[all_procs.size()];
      for (size_t i = 0; i < all_procs.size(); i++) {
        state[i].all_proc_states = state;
        state[i].subgraph = this;
        state[i].finish_counter = finish_counter;
        state[i].finish_event = static_finish_event;
        state[i].proc_index = int32_t(i);
        state[i].args = copied_args;
        state[i].arglen = arglen;
        state[i].preconditions = precondition_ctrs;
        state[i].dynamic_preconditions = dynamic_precondition_ctrs;
        state[i].dynamic_events = dynamic_events;
        state[i].bgwork_preconditions = bgwork_precondition_ctrs;
        state[i].async_operation_events = async_operation_events;
        state[i].async_operation_effect_triggerers = async_operation_effect_triggerers;
        state[i].async_bgwork_events = async_bgwork_events;
        state[i].has_pending_async_work = async_finish_events[i] > 0;
        state[i].pending_async_count.store(async_finish_events[i]);
        state[i].queue = queue;
        state[i].next_queue_slot.store(operations.proc_offsets[i] + initial_queue_entry_count[i]);
      }

      // Since we have a fresh precondition vector, we can
      // just queue up the external precondition waiters
      // to start directly on the new data.
      auto external_precondition_waiters = (ExternalPreconditionTriggerer*)malloc(sizeof(ExternalPreconditionTriggerer) * external_precondition_info.size());
      for (size_t i = 0; i < external_precondition_info.size(); i++) {
        // Note that the user may not provide all the preconditions
        // they said they would.
        Event precond = Event::NO_EVENT;
        if (i < preconditions.size())
          precond = preconditions[i];
        auto meta = &external_precondition_info[i];
        auto waiter = new (&external_precondition_waiters[i]) ExternalPreconditionTriggerer(this, meta, precondition_ctrs, state);
        if (!precond.exists() || precond.has_triggered()) {
          waiter->trigger();
        } else {
          EventImpl::add_waiter(precond, waiter);
        }
      }

      // Also allocate precondition waiters for external events to
      // trigger from dynamic operations into static the static
      // subgraph. We'll lazily initialize it as we issue operations
      // from the dynamic portion of the subgraph.
      dynamic_precondition_waiters = (ExternalPreconditionTriggerer*)(malloc(sizeof(ExternalPreconditionTriggerer) * dynamic_to_static_triggers.size()));

      // If there's nothing to defer, start the subgraph right away.
      if (!start_event.exists() || start_event.has_triggered()) {
        SubgraphWorkLauncher(state, this).launch();
      } else {
        // Otherwise, allocate a launcher and queue
        // it up to be launched. The launcher will
        // clean itself up.
        auto launcher = new SubgraphWorkLauncher(state, this);
        EventImpl::add_waiter(start_event, launcher);
      }

      // Issue a cleanup background work item. The item
      // will clean itself up.
      auto cleanup = new InstantiationCleanup(
        all_procs.size(),
        state,
        copied_args,
        precondition_ctrs,
        queue,
        external_precondition_waiters,
        dynamic_precondition_waiters,
        async_operation_events,
        async_operation_effect_triggerers,
        async_bgwork_events,
        dynamic_precondition_ctrs,
        dynamic_events,
        finish_counter
      );
      EventImpl::add_waiter(finish_event, cleanup);
    } else {
      event_impl->merger.prepare_merger(finish_event, false /*!ignore_faults*/, num_final_events);
    }

    // we precomputed the number of intermediate events we need, so put them
    //  on the stack
    Event *intermediate_events =
        static_cast<Event *>(alloca(num_intermediate_events * sizeof(Event)));
    size_t cur_intermediate_events = 0;

    // Allocate one extra precondition slot for triggers from the
    // static subgraph into the dynamic subgraph.
    Event *preconds = static_cast<Event *>(alloca((max_preconditions + 1) * sizeof(Event)));

    for(std::vector<SubgraphScheduleEntry>::const_iterator it = schedule.begin();
        it != schedule.end(); ++it) {
      size_t sched_idx = it - schedule.begin();
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
      if (static_to_dynamic_counts[sched_idx] > 0 && dynamic_events)
        preconds[num_preconds++] = dynamic_events[sched_idx];

      assert(num_preconds <= max_preconditions + 1);

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

      // The resulting event may need to contribute to
      // triggering some work in the static component
      // of the subgraph.
      if (opt && !dynamic_to_static_triggers[sched_idx].to_trigger.empty()) {
        auto meta = &dynamic_to_static_triggers[sched_idx];
        auto waiter = new (&dynamic_precondition_waiters[sched_idx]) ExternalPreconditionTriggerer(this, meta, precondition_ctrs, state);
        if (!e.exists() || e.has_triggered()) {
          waiter->trigger();
        } else {
          EventImpl::add_waiter(e, waiter);
        }
      }
    }

    // sanity-check that we counted right
    assert(cur_intermediate_events == num_intermediate_events);

    if (opt) {
      event_impl->merger.arm_merger();
    } else {
      if(num_final_events > 0) {
        event_impl->merger.arm_merger();
      } else {
        GenEventImpl::trigger(finish_event, false /*!poisoned*/);
      }
    }
  }

  void SubgraphImpl::destroy(void)
  {
    delete defn;
    schedule.clear();

    // Clean up all state used for the static portion of the subgraph.
    all_procs.clear();
    all_proc_impls.clear();
    async_finish_events.clear();
    operations.clear();
    operation_meta.clear();
    completion_infos.clear();
    original_preconditions.clear();
    async_outgoing_infos.clear();
    async_incoming_infos.clear();
    interpolations.clear();
    initial_queue_state.clear();
    initial_queue_entry_count.clear();
    external_precondition_info.clear();
    dynamic_to_static_triggers.clear();
    static_to_dynamic_counts.clear();

    bgwork_items.clear();
    for (auto xd : planned_copy_xds) {
      if (xd) {
        xd->remove_reference();
      }
    }
    planned_copy_xds.clear();
    bgwork_preconditions.clear();
    bgwork_postconditions.clear();
    bgwork_items_without_preconditions.clear();
    bgwork_async_postconditions.clear();
    bgwork_async_preconditions.clear();
    bgwork_async_event_procs.clear();

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

  void SubgraphImpl::DeferredDestroy::defer(SubgraphImpl *_subgraph, Event wait_on)
  {
    subgraph = _subgraph;
    EventImpl::add_waiter(wait_on, this);
  }

  void SubgraphImpl::DeferredDestroy::event_triggered(bool poisoned, TimeLimit work_until)
  {
    assert(!poisoned);
    subgraph->destroy();
  }

  void SubgraphImpl::DeferredDestroy::print(std::ostream &os) const
  {
    os << "deferred subgraph destruction: subgraph=" << subgraph->me;
  }

  Event SubgraphImpl::DeferredDestroy::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  void SubgraphImpl::InstantiationCleanup::print(std::ostream &os) const {
    os << "deferred instantiation cleanup";
  }

  SubgraphImpl::InstantiationCleanup::InstantiationCleanup(
    size_t _num_procs,
    ProcSubgraphReplayState* _state,
    void* _args,
    atomic<int32_t>* _preconds,
    atomic<int64_t>* _queue,
    ExternalPreconditionTriggerer* _external_preconds,
    ExternalPreconditionTriggerer* _dynamic_preconds,
    void** _async_operation_events,
    AsyncGPUWorkTriggerer* _async_operation_event_triggerers,
    void** _async_bgwork_events,
    atomic<int32_t>* _dynamic_precond_counters,
    UserEvent* _dynamic_events,
    atomic<int32_t>* _finish_counter
  ) : num_procs(_num_procs), state(_state), args(_args), preconds(_preconds), queue(_queue),
      external_preconds(_external_preconds), dynamic_preconds(_dynamic_preconds),
      async_operation_events(_async_operation_events), async_operation_event_triggerers(_async_operation_event_triggerers),
      async_bgwork_events(_async_bgwork_events),
      dynamic_precond_counters(_dynamic_precond_counters), dynamic_events(_dynamic_events), finish_counter(_finish_counter), EventWaiter() {}

  void SubgraphImpl::InstantiationCleanup::event_triggered(bool poisoned, Realm::TimeLimit work_until) {
    // Before we free async_operation_events, make sure that we return all
    // GPU events appropriately.
    assert(num_procs > 0);
    auto sg = state[0].subgraph;
    std::vector<void*> tokens;
    for (size_t proc = 0; proc < num_procs; proc++) {
      auto pimpl = sg->all_proc_impls[proc];
      for (size_t j = sg->operations.proc_offsets[proc]; j < sg->operations.proc_offsets[proc + 1]; j++) {
        if (async_operation_events[j] != nullptr)
          tokens.push_back(async_operation_events[j]);
      }
      for (auto idx : sg->bgwork_async_event_procs[pimpl]) {
        assert(async_bgwork_events[idx] != nullptr);
        tokens.push_back(async_bgwork_events[idx]);
      }
      pimpl->return_subgraph_async_tokens(tokens);
      tokens.clear();
    }

    delete[] state;
    free(args);
    free(preconds);
    free(queue);
    free(external_preconds);
    free(dynamic_preconds);
    free(async_operation_events);
    free(async_operation_event_triggerers);
    free(async_bgwork_events);
    free(dynamic_precond_counters);
    free(dynamic_events);
    free(finish_counter);
    // Delete ourselves after freeing the resource. This pattern
    // is used by other event waiters, so it's reasonable to replicate it here.
    delete this;
  }

  Event SubgraphImpl::InstantiationCleanup::get_finish_event() const {
    return Event::NO_EVENT;
  }

  SubgraphImpl::ExternalPreconditionTriggerer::ExternalPreconditionTriggerer(
    SubgraphImpl* _subgraph,
    ExternalPreconditionMeta* _meta,
    atomic<int32_t>* _preconditions,
    ProcSubgraphReplayState* _all_proc_states
  ) : subgraph(_subgraph), meta(_meta), preconditions(_preconditions), all_proc_states(_all_proc_states), EventWaiter() {}

  void SubgraphImpl::ExternalPreconditionTriggerer::trigger() {
    for (auto& info : meta->to_trigger) {
      trigger_subgraph_operation_completion(all_proc_states, info, true /* incr_counter */, nullptr /* ctx */);
    }
  }

  void SubgraphImpl::ExternalPreconditionTriggerer::event_triggered(bool poisoned, Realm::TimeLimit work_until) {
    trigger();
  }

  void SubgraphImpl::ExternalPreconditionTriggerer::print(std::ostream &os) const {
    os << "External precondition triggerer";
  }

  Event SubgraphImpl::ExternalPreconditionTriggerer::get_finish_event() const {
    return Event::NO_EVENT;
  }

  SubgraphImpl::AsyncGPUWorkTriggerer::AsyncGPUWorkTriggerer(
    ProcSubgraphReplayState* _all_proc_states,
    span<Realm::SubgraphImpl::CompletionInfo> _infos,
    atomic<int32_t> *_preconditions,
    atomic<int32_t>* _final_ev_counter
  ) : all_proc_states(_all_proc_states), infos(_infos), preconditions(_preconditions), final_ev_counter(_final_ev_counter) {}

  void SubgraphImpl::AsyncGPUWorkTriggerer::request_completed() {
    for (size_t i = 0; i < infos.size(); i++) {
      auto& info = infos[i];
      trigger_subgraph_operation_completion(all_proc_states, info, true /* incr_counter */, nullptr /* ctx */);
    }
    // Also see if we need to go trigger the completion event.
    if (final_ev_counter) {
      int32_t remaining = final_ev_counter->fetch_sub_acqrel(1) - 1;
      if (remaining == 0) {
        maybe_trigger_subgraph_final_completion_event(all_proc_states[0]);
      }
    }
  }


  SubgraphImpl::SubgraphWorkLauncher::SubgraphWorkLauncher(Realm::ProcSubgraphReplayState *_state,
                                                           Realm::SubgraphImpl *_subgraph)
                                                           : state(_state), subgraph(_subgraph), EventWaiter() {}

  void SubgraphImpl::SubgraphWorkLauncher::event_triggered(bool poisoned, Realm::TimeLimit work_until) {
    launch();
    delete this;
  }

  void SubgraphImpl::SubgraphWorkLauncher::launch() {
    // Install each processor's replay state.
    for (size_t i = 0; i < subgraph->all_proc_impls.size(); i++) {
      subgraph->all_proc_impls[i]->install_subgraph_replay(&state[i]);
    }
    // Start up any background work items without any preconditions.
    auto sg = state[0].subgraph;
    for (auto idx : sg->bgwork_items_without_preconditions) {
      launch_async_bgwork_item(state, idx);
    }
  }

  void SubgraphImpl::SubgraphWorkLauncher::print(std::ostream &os) const {
    os << "Subgraph work launcher";
  }

  Event SubgraphImpl::SubgraphWorkLauncher::get_finish_event() const {
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

  void trigger_subgraph_operation_completion(
      ProcSubgraphReplayState* all_proc_states,
      const SubgraphImpl::CompletionInfo& info,
      bool incr_counter,
      SubgraphTriggerContext* ctx
  ) {
    switch (info.kind) {
      case SubgraphImpl::CompletionInfo::BGWORK_TO_STATIC: [[fallthrough]];
      case SubgraphImpl::CompletionInfo::EdgeKind::STATIC_TO_STATIC: [[fallthrough]];
      case SubgraphImpl::CompletionInfo::EdgeKind::DYNAMIC_TO_STATIC: {
        auto& state = all_proc_states[info.proc];
        auto subgraph = state.subgraph;
        auto& trigger = state.preconditions[subgraph->original_preconditions.proc_offsets[info.proc] + info.index];
        // Decrement the counter. fetch_sub returns the value before
        // the decrement, so if it was 1, then we've done the final trigger.
        // If we were the final trigger, we need to also add the triggered
        // operation into the target processors queue.
        int32_t remaining = trigger.fetch_sub_acqrel(1) - 1;
        if (remaining == 0) {
          // Get a slot to insert at.
          auto index = state.next_queue_slot.fetch_add(1);
          // Write out the value.
          state.queue[index].store_release(int64_t(info.index));

          // If requested to increment the scheduler's counter, do so.
          if (incr_counter) {
            auto lp = subgraph->all_proc_impls[info.proc];
            lp->sched->work_counter.increment_counter();
          }
        }
        break;
      }
      case SubgraphImpl::CompletionInfo::BGWORK_TO_DYNAMIC: [[fallthrough]];
      case SubgraphImpl::CompletionInfo::EdgeKind::STATIC_TO_DYNAMIC: {
        // If we're in the static-to-dynamic case, the CompletionInfo
        // edge doesn't have a processor associated with it, so pick
        // the first one (arbitrarily).
        auto& state = all_proc_states[0];
        auto& trigger = state.dynamic_preconditions[info.index];
        int32_t remaining = trigger.fetch_sub_acqrel(1) - 1;
        if (remaining == 0) {
          // Need some help from the caller about allowing the surrounding
          // context to make an event trigger.
          if (ctx) { ctx->enter(); }
          state.dynamic_events[info.index].trigger();
          if (ctx) { ctx->exit(); }
        }
        break;
      }
      case SubgraphImpl::CompletionInfo::STATIC_TO_BGWORK: [[fallthrough]];
      case SubgraphImpl::CompletionInfo::DYNAMIC_TO_BGWORK: [[fallthrough]];
      case SubgraphImpl::CompletionInfo::BGWORK_TO_BGWORK: {
        // The *-bgwork cases also don't have an associated processor.
        auto& state = all_proc_states[0];
        auto& trigger = state.bgwork_preconditions[info.index];
        int32_t remaining = trigger.fetch_sub_acqrel(1) - 1;
        if (remaining == 0) {
          // To be safe, also use the context to ensure the
          // appropriate locks are dropped.
          if (ctx) { ctx->enter(); }
          launch_async_bgwork_item(all_proc_states, info.index);
          if (ctx) { ctx->exit(); }
        }
        break;
      }
      default:
        assert(false);
    }
  }

  void maybe_trigger_subgraph_final_completion_event(ProcSubgraphReplayState& state) {
    // Trigger check for the final event. All processors
    // share the same finish counter and finish event, so any
    // entry of all_proc_states can be passed.
    int32_t remaining = state.finish_counter->fetch_sub_acqrel(1) - 1;
    if (remaining == 0) {
      state.finish_event.trigger();
    }
  }

  void launch_async_bgwork_item(ProcSubgraphReplayState* all_proc_states, unsigned index) {
    auto subgraph = all_proc_states[0].subgraph;
    auto& item = subgraph->bgwork_items[index];
    assert(item.op_kind == SubgraphDefinition::OPKIND_COPY);
    auto xd = subgraph->planned_copy_xds[item.op_index];
    assert(xd);
    xd->reset();
    xd->subgraph_replay_state = all_proc_states;
    xd->subgraph_index = index;
    // Add a reference before enqueing the XD, as each pass through the pipeline
    // removes a reference upon completion.
    xd->add_reference();
    xd->channel->enqueue_ready_xd(xd);
  }

  XferDes* SubgraphImpl::analyze_copy(SubgraphDefinition::CopyDesc& copy) {
    TransferDesc* td = copy.space.impl->make_transfer_desc(copy.srcs, copy.dsts);
    // Just constructing a TransferDescriptor should perform all the
    // copy analysis inline. It could be deferred, but if all the instances
    // have been created already, then the analysis happens right away.
    // Assert that this is true.
    // TODO (rohany): Have to have some sort of filter for the copies that
    //  we actually analyze in this manner.
    assert(td->analysis_complete.load() && td->analysis_successful);
    // For now, the copies we're analyzing should only ever
    //  be one-hop copies inside this node. After this works, we
    //  can consider extending the analysis to cache plans for
    //  multi-hop copies through the network.
    TransferGraph& graph = td->graph;
    assert(graph.xd_nodes.size() == 1);
    assert(graph.ib_edges.empty());
    // Now, spoof the XDFactory inside each node of the transfer
    // graph so that we can extract the XD's.
    std::vector<MockXDFactory> factories(graph.xd_nodes.size());
    for (size_t i = 0; i < graph.xd_nodes.size(); i++) {
      factories[i].factory = graph.xd_nodes[i].factory;
      graph.xd_nodes[i].factory = &factories[i];
    }

    // It looks like we will have to make a new GenEvent for each copy,
    // but it's not the end of the world.
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();
    TransferOperation *op = new TransferOperation(*td, Event::NO_EVENT, finish_event,
                                                  ID(ev).event_generation(), 0 /* priority */);
    // Actually create the XD's for the copy. We have to call
    // TransferOperation::allocate_ibs(), which then falls through
    // to call TransferOperation::create_xds().
    op->allocate_ibs();

    // After TransferOperation::create_xds(), our mock factory should
    // have been invoked to handle the creation.
    assert(factories[0].xd != nullptr);
    // We need to make the TransferOperation think that it is finished.
    // The SubgraphImpl will take over the reference to the XD.
    assert(op->xd_trackers.size() == 1);
    op->xd_trackers[0]->mark_finished(true /* successful */);

    // At this point, we don't need a reference to the transfer descriptor anymore.
    td->remove_reference();

    auto xd = factories[0].xd;
    // Detach the xd from its DMA op.
    xd->dma_op = 0;
    return xd;
  }
}; // namespace Realm
