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
#include "realm/event_impl.h"
#include "realm/network.h"
#include "realm/numa/numasysif.h"
#include "realm/proc_impl.h"
#include "realm/runtime_impl.h"
#include "realm/tasks.h"
#include "realm/timers.h"
#include "realm/transfer/transfer.h"
#include "realm/idx_impl.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef __linux__
#include <sched.h>
#endif

namespace Realm {

  Logger log_subgraph("subgraph");

  namespace {
    // Unsupported features and misuse are hard errors: a subgraph that ran
    // with different semantics than requested would be worse than one that
    // refuses to run.
    [[noreturn]] void subgraph_fatal(ID me, const std::string &what)
    {
      log_subgraph.fatal() << "subgraph " << me << ": " << what;
      abort();
    }
#define SUBGRAPH_FATAL(me, expr)                                                         \
  do {                                                                                   \
    std::ostringstream _ss;                                                              \
    _ss << expr;                                                                         \
    subgraph_fatal((me), _ss.str());                                                     \
  } while(0)

    size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

    const char *op_kind_name(SubgraphDefinition::OpKind kind)
    {
      switch(kind) {
      case SubgraphDefinition::OPKIND_TASK:
        return "task";
      case SubgraphDefinition::OPKIND_COPY:
        return "copy";
      case SubgraphDefinition::OPKIND_ARRIVAL:
        return "barrier arrival";
      case SubgraphDefinition::OPKIND_INSTANTIATION:
        return "subgraph instantiation";
      case SubgraphDefinition::OPKIND_ACQUIRE:
        return "reservation acquire";
      case SubgraphDefinition::OPKIND_RELEASE:
        return "reservation release";
      case SubgraphDefinition::OPKIND_EXT_PRECOND:
        return "external precondition";
      case SubgraphDefinition::OPKIND_EXT_POSTCOND:
        return "external postcondition";
      case SubgraphDefinition::OPKIND_COLL_PRECOND:
        return "collective precondition";
      case SubgraphDefinition::OPKIND_COLL_POSTCOND:
        return "collective postcondition";
      default:
        return "invalid";
      }
    }
  } // namespace

  ////////////////////////////////////////////////////////////////////////
  //
  // class Subgraph
  //

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

    if(!wait_on.has_triggered())
      SUBGRAPH_FATAL(impl->me, "deferred creation (create_subgraph with an untriggered "
                               "wait_on) is not implemented");
    if(!prs.empty())
      SUBGRAPH_FATAL(impl->me, "profiling requests on create_subgraph are not implemented");

    impl->compile();
    log_subgraph.info() << "created: subgraph=" << subgraph
                        << " ops=" << impl->compiled.ops.size()
                        << " procs=" << impl->compiled.procs.size()
                        << " domains=" << impl->compiled.domains.size();
    return Event::NO_EVENT;
  }

  Event Subgraph::destroy(Event wait_on /*= Event::NO_EVENT*/) const
  {
    NodeID owner = ID(*this).subgraph_owner_node();

    log_subgraph.info() << "destroy: subgraph=" << *this << " wait_on=" << wait_on;

    if(owner == Network::my_node_id) {
      SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(*this);
      return subgraph->request_destroy(wait_on);
    } else {
      UserEvent done = UserEvent::create_user_event();
      ActiveMessage<SubgraphDestroyMessage> amsg(owner);
      amsg->subgraph = *this;
      amsg->wait_on = wait_on;
      amsg->to_trigger = done;
      amsg.commit();
      return done;
    }
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
                              const ProfilingRequestSet &prs,
                              Event wait_on /*= Event::NO_EVENT*/,
                              int priority_adjust /*= 0*/) const
  {
    return instantiate(args, arglen, prs, SubgraphInstantiationProfiling(), wait_on,
                       priority_adjust);
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
                              const ProfilingRequestSet &prs,
                              const std::vector<Event> &preconditions,
                              std::vector<Event> &postconditions,
                              Event wait_on /*= Event::NO_EVENT*/,
                              int priority_adjust /*= 0*/) const
  {
    return instantiate(args, arglen, prs, SubgraphInstantiationProfiling(), preconditions,
                       postconditions, wait_on, priority_adjust);
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
                              const ProfilingRequestSet &prs,
                              const SubgraphInstantiationProfiling &profiling,
                              Event wait_on /*= Event::NO_EVENT*/,
                              int priority_adjust /*= 0*/) const
  {
    NodeID target_node = ID(*this).subgraph_owner_node();

    Event finish_event = GenEventImpl::create_genevent()->current_event();

    log_subgraph.info() << "instantiate: subgraph=" << *this << " before=" << wait_on
                        << " after=" << finish_event;

    if(target_node == Network::my_node_id) {
      SubgraphImpl *impl = get_runtime()->get_subgraph_impl(*this);
      impl->instantiate(args, arglen, prs, profiling, empty_span() /*preconditions*/,
                        empty_span() /*postconditions*/, wait_on, finish_event,
                        priority_adjust);
    } else {
      Serialization::ByteCountSerializer bcs;
      {
        bool ok = (bcs.append_bytes(args, arglen) && (bcs << span<const Event>()) &&
                   (bcs << span<const Event>()) && (bcs << prs) && (bcs << profiling));
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
                   (amsg << prs) && (amsg << profiling));
        assert(ok);
      }
      amsg.commit();
    }
    return finish_event;
  }

  Event Subgraph::instantiate(const void *args, size_t arglen,
                              const ProfilingRequestSet &prs,
                              const SubgraphInstantiationProfiling &profiling,
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
      impl->instantiate(args, arglen, prs, profiling, preconditions, postconditions,
                        wait_on, finish_event, priority_adjust);
    } else {
      Serialization::ByteCountSerializer bcs;
      {
        bool ok = (bcs.append_bytes(args, arglen) && (bcs << preconditions) &&
                   (bcs << postconditions) && (bcs << prs) && (bcs << profiling));
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
        bool ok = ((amsg << preconditions) && (amsg << postconditions) && (amsg << prs) &&
                   (amsg << profiling));
        assert(ok);
      }
      amsg.commit();
    }
    return finish_event;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // struct CompiledSubgraph
  //

  void CompiledSubgraph::clear()
  {
    ops.clear();
    num_direct_ops = 0;
    task_ops.clear();
    copy_ops.clear();
    for(TransferDesc *plan : copy_plans)
      if(plan)
        plan->remove_reference();
    copy_plans.clear();
    any_task_profiling = false;
    roots.clear();
    procs.clear();
    domains.clear();
    proc_index.clear();
    successors.clear();
    late_successors.clear();
    token_waits.clear();
    num_async_ops = 0;
    postconds_of.clear();
    inputs.clear();
    postconds.clear();
    interps.clear();
    input_words = 0;
    op_inputs.clear();
    postcond_inputs.clear();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphImpl
  //

  SubgraphImpl::SubgraphImpl()
    : me(Subgraph::NO_SUBGRAPH)
    , next_free(nullptr)
    , defn(nullptr)
  {}

  SubgraphImpl::~SubgraphImpl() {}

  void SubgraphImpl::init(ID _me, int _owner)
  {
    me = _me;
    assert(NodeID(me.subgraph_owner_node()) == NodeID(_owner));
  }

  void SubgraphImpl::compile(void)
  {
    const SubgraphDefinition &d = *defn;
    typedef SubgraphDefinition::OpKind OpKind;
    typedef SubgraphDefinition::Interpolation Interpolation;

    // ---- validation: everything the compiled implementation cannot run yet
    if((d.concurrency_mode != SubgraphDefinition::ONE_SHOT) &&
       (d.concurrency_mode != SubgraphDefinition::INSTANTIATION_ORDER))
      SUBGRAPH_FATAL(me, "concurrency modes SERIALIZABLE and CONCURRENT are not "
                         "implemented; use ONE_SHOT or INSTANTIATION_ORDER");
    for(size_t i = 0; i < d.copies.size(); i++) {
      const SubgraphDefinition::CopyDesc &cd = d.copies[i];
      if(!cd.space.impl)
        SUBGRAPH_FATAL(me, "copy " << i << " has no index space");
      if(cd.srcs.empty() || (cd.srcs.size() != cd.dsts.size()))
        SUBGRAPH_FATAL(me, "copy " << i << " has " << cd.srcs.size() << " sources and "
                                   << cd.dsts.size()
                                   << " destinations; they must match and not be empty");
      for(size_t f = 0; f < cd.srcs.size(); f++) {
        const int si = cd.srcs[f].indirect_index, di = cd.dsts[f].indirect_index;
        if((si >= int(cd.indirects.size())) || (di >= int(cd.indirects.size())))
          SUBGRAPH_FATAL(me, "copy " << i << " field " << f
                                     << " refers to an indirection that was not added "
                                        "(see CopyDesc::add_indirection)");
      }
    }
    if(!d.instantiations.empty())
      SUBGRAPH_FATAL(me, "nested subgraph instantiations are not implemented ("
                             << d.instantiations.size() << " in definition)");
    if(!d.acquires.empty() || !d.releases.empty())
      SUBGRAPH_FATAL(me, "reservation acquires and releases are not implemented");

    for(size_t i = 0; i < d.tasks.size(); i++) {
      const SubgraphDefinition::TaskDesc &t = d.tasks[i];
      if(!t.proc.exists())
        SUBGRAPH_FATAL(me, "task " << i << " has no processor");
      if(NodeID(t.proc.address_space()) != Network::my_node_id)
        SUBGRAPH_FATAL(me, "task " << i << " runs on " << t.proc
                                   << ", which belongs to another node; only local "
                                      "tasks are implemented");
      if(t.priority != 0)
        SUBGRAPH_FATAL(me, "task " << i << " has priority " << t.priority
                                   << "; per-task priorities are not implemented (use "
                                      "the instantiation priority)");
    }

    // interpolations: which arrivals get their barrier from the arguments?
    std::vector<bool> arrival_barrier_interpolated(d.arrivals.size(), false);
    for(size_t i = 0; i < d.interpolations.size(); i++) {
      const Interpolation &ip = d.interpolations[i];
      size_t target_size = 0;
      switch(ip.target_kind) {
      case Interpolation::TARGET_TASK_ARGS:
        if(ip.target_index >= d.tasks.size())
          SUBGRAPH_FATAL(me, "interpolation " << i << " targets task " << ip.target_index
                                              << ", which does not exist");
        target_size = d.tasks[ip.target_index].args.size();
        break;
      case Interpolation::TARGET_ARRIVAL_BARRIER:
        if(ip.target_index >= d.arrivals.size())
          SUBGRAPH_FATAL(me, "interpolation " << i << " targets arrival " << ip.target_index
                                              << ", which does not exist");
        if((ip.target_offset != 0) || (ip.bytes != sizeof(Barrier)) || (ip.redop_id != 0))
          SUBGRAPH_FATAL(me, "interpolation " << i
                                              << " must overwrite the whole barrier of "
                                                 "arrival "
                                              << ip.target_index);
        arrival_barrier_interpolated[ip.target_index] = true;
        target_size = sizeof(Barrier);
        break;
      case Interpolation::TARGET_ARRIVAL_VALUE:
        if(ip.target_index >= d.arrivals.size())
          SUBGRAPH_FATAL(me, "interpolation " << i << " targets arrival " << ip.target_index
                                              << ", which does not exist");
        target_size = d.arrivals[ip.target_index].reduce_value.size();
        break;
      case Interpolation::TARGET_INSTANCE_ARGS:
        SUBGRAPH_FATAL(me, "interpolation " << i
                                            << " targets a nested instantiation, which is "
                                               "not implemented");
      default:
        SUBGRAPH_FATAL(me, "interpolation " << i << " has an invalid target kind");
      }
      if(ip.redop_id == 0) {
        if((ip.target_offset + ip.bytes) > target_size)
          SUBGRAPH_FATAL(me, "interpolation " << i << " writes past its target ("
                                              << ip.target_offset << "+" << ip.bytes << " > "
                                              << target_size << ")");
      } else {
        const ReductionOpUntyped *redop =
            get_runtime()->reduce_op_table.get(ip.redop_id, nullptr);
        if(!redop)
          SUBGRAPH_FATAL(me, "interpolation " << i << " uses unknown reduction op "
                                              << ip.redop_id);
        if(ip.bytes != redop->sizeof_rhs)
          SUBGRAPH_FATAL(me, "interpolation " << i << " provides " << ip.bytes
                                              << " bytes but reduction op " << ip.redop_id
                                              << " expects " << redop->sizeof_rhs);
        if((ip.target_offset + redop->sizeof_lhs) > target_size)
          SUBGRAPH_FATAL(me, "interpolation " << i << " reduces past its target");
      }
    }
    for(size_t i = 0; i < d.arrivals.size(); i++)
      if(!d.arrivals[i].barrier.exists() && !arrival_barrier_interpolated[i])
        SUBGRAPH_FATAL(me, "arrival " << i << " has no barrier and no interpolation "
                                         "providing one");

    // dependencies
    unsigned num_inputs = 0, num_postconds = 0;
    for(size_t i = 0; i < d.dependencies.size(); i++) {
      const SubgraphDefinition::Dependency &dep = d.dependencies[i];
      auto check_index = [&](OpKind k, unsigned idx, const char *role) {
        size_t limit = 0;
        switch(k) {
        case SubgraphDefinition::OPKIND_TASK:
          limit = d.tasks.size();
          break;
        case SubgraphDefinition::OPKIND_ARRIVAL:
          limit = d.arrivals.size();
          break;
        case SubgraphDefinition::OPKIND_EXT_PRECOND:
        case SubgraphDefinition::OPKIND_EXT_POSTCOND:
          return; // any index: defines how many there are
        case SubgraphDefinition::OPKIND_COLL_PRECOND:
        case SubgraphDefinition::OPKIND_COLL_POSTCOND:
          SUBGRAPH_FATAL(me, "dependency " << i
                                           << " uses collective conditions, which are not "
                                              "implemented");
        default:
          SUBGRAPH_FATAL(me, "dependency " << i << " " << role << " is a "
                                           << op_kind_name(k)
                                           << ", which is not implemented");
        }
        if(idx >= limit)
          SUBGRAPH_FATAL(me, "dependency " << i << " " << role << " refers to "
                                           << op_kind_name(k) << " " << idx
                                           << ", but the definition has " << limit);
      };
      check_index(dep.src_op_kind, dep.src_op_index, "source");
      check_index(dep.tgt_op_kind, dep.tgt_op_index, "target");
      if(dep.src_op_kind == SubgraphDefinition::OPKIND_ARRIVAL)
        SUBGRAPH_FATAL(me, "dependency " << i << " has arrival " << dep.src_op_index
                                         << " as its source, but arrivals have no outputs");
      if(dep.src_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND)
        SUBGRAPH_FATAL(me, "dependency " << i
                                         << " has an external postcondition as its source");
      if(dep.tgt_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND)
        SUBGRAPH_FATAL(me, "dependency " << i
                                         << " has an external precondition as its target");
      if((dep.src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND) &&
         (dep.tgt_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND))
        SUBGRAPH_FATAL(me, "dependency " << i
                                         << " connects an external precondition directly "
                                            "to an external postcondition");
      if((dep.src_op_port != 0) || (dep.tgt_op_port != 0))
        SUBGRAPH_FATAL(me, "dependency " << i
                                         << " uses a nonzero port; ports are not "
                                            "implemented");
      if(dep.src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND)
        num_inputs = std::max(num_inputs, dep.src_op_index + 1);
      if(dep.tgt_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND)
        num_postconds = std::max(num_postconds, dep.tgt_op_index + 1);
    }

    // ---- processors and NUMA domains
    const size_t ntasks = d.tasks.size();
    std::vector<Processor> procs_seen;
    std::unordered_map<Processor, LocalTaskProcessor *> proc_impls;
    std::unordered_map<Processor, int> proc_nodes;
    for(size_t i = 0; i < ntasks; i++) {
      Processor p = d.tasks[i].proc;
      if(proc_impls.count(p))
        continue;
      LocalTaskProcessor *impl =
          dynamic_cast<LocalTaskProcessor *>(get_runtime()->get_processor_impl(p));
      if(!impl || !impl->supports_subgraph_tasks())
        SUBGRAPH_FATAL(me, "task " << i << " runs on " << p << ", a processor of kind "
                                   << p.kind()
                                   << "; only CPU (LOC_PROC) and CUDA GPU (TOC_PROC) "
                                      "tasks are implemented");
      procs_seen.push_back(p);
      proc_impls[p] = impl;
      proc_nodes[p] = impl->numa_node();
    }
    std::stable_sort(procs_seen.begin(), procs_seen.end(),
                     [&](Processor a, Processor b) { return proc_nodes[a] < proc_nodes[b]; });

    CompiledSubgraph &c = compiled;
    c.clear();
    std::vector<int> domain_nodes; // sorted unique
    for(Processor p : procs_seen) {
      int node = proc_nodes[p];
      if(domain_nodes.empty() || (domain_nodes.back() != node))
        domain_nodes.push_back(node);
    }
    if(domain_nodes.empty())
      domain_nodes.push_back(-1); // no tasks: one block for everything else
    c.domains.resize(domain_nodes.size());
    for(size_t i = 0; i < domain_nodes.size(); i++)
      c.domains[i].numa_node = domain_nodes[i];

    c.procs.resize(procs_seen.size());
    for(size_t i = 0; i < procs_seen.size(); i++) {
      CompiledSubgraph::Proc &cp = c.procs[i];
      cp.proc = procs_seen[i];
      cp.impl = proc_impls[cp.proc];
      cp.domain = int32_t(std::lower_bound(domain_nodes.begin(), domain_nodes.end(),
                                           proc_nodes[cp.proc]) -
                          domain_nodes.begin());
      cp.first_op = cp.num_ops = cp.queue_offset = cp.tail_offset = cp.inputs_offset = 0;
      cp.remaining_offset = 0;
      cp.initial_ready = 0;
      cp.pending_inputs = 0;
      cp.async = cp.impl->subgraph_tasks_are_async();
      cp.first_async = cp.num_async = 0;
      c.proc_index[cp.proc] = uint32_t(i);
    }

    // ---- operations: each processor's tasks, then directly launched ones
    std::vector<uint32_t> task_to_op(ntasks), arrival_to_op(d.arrivals.size()),
        copy_to_op(d.copies.size());
    auto new_op = [&](OpKind kind, unsigned index, int32_t proc, int32_t domain) {
      CompiledSubgraph::Op op;
      op.kind = kind;
      op.index = index;
      op.proc = proc;
      op.counter_domain = domain;
      op.counter_offset = 0;
      op.args_domain = -1;
      op.args_offset = op.args_size = 0;
      op.async = op.deferred = false;
      op.async_index = -1;
      c.ops.push_back(op);
      return uint32_t(c.ops.size() - 1);
    };
    c.num_async_ops = 0;
    for(size_t pi = 0; pi < c.procs.size(); pi++) {
      CompiledSubgraph::Proc &cp = c.procs[pi];
      cp.first_op = uint32_t(c.ops.size());
      cp.first_async = c.num_async_ops;
      for(size_t i = 0; i < ntasks; i++) {
        if(d.tasks[i].proc != cp.proc)
          continue;
        task_to_op[i] = new_op(SubgraphDefinition::OPKIND_TASK, unsigned(i), int32_t(pi),
                               cp.domain);
        // the task must be registered already: how it runs depends on it
        const unsigned flags = cp.impl->subgraph_task_flags(d.tasks[i].task_id);
        if(!(flags & LocalTaskProcessor::SUBGRAPH_TASK_REGISTERED))
          SUBGRAPH_FATAL(me, "task " << i << " uses task id " << d.tasks[i].task_id
                                     << ", which is not registered on " << cp.proc);
        CompiledSubgraph::Op &op = c.ops.back();
        op.async = cp.async;
        op.deferred =
            cp.async && ((flags & LocalTaskProcessor::SUBGRAPH_TASK_DEFERRED_EFFECTS) != 0);
        if(cp.async)
          op.async_index = int32_t(c.num_async_ops++);
      }
      cp.num_ops = uint32_t(c.ops.size()) - cp.first_op;
      cp.num_async = c.num_async_ops - cp.first_async;
    }
    for(size_t i = 0; i < d.arrivals.size(); i++)
      arrival_to_op[i] = new_op(SubgraphDefinition::OPKIND_ARRIVAL, unsigned(i), -1, 0);
    for(size_t i = 0; i < d.copies.size(); i++)
      copy_to_op[i] = new_op(SubgraphDefinition::OPKIND_COPY, unsigned(i), -1, 0);
    c.num_direct_ops = uint32_t(d.arrivals.size() + d.copies.size());
    c.task_ops = task_to_op;
    c.copy_ops = copy_to_op;
    c.any_task_profiling = false;
    for(const SubgraphDefinition::TaskDesc &t : d.tasks)
      c.any_task_profiling = c.any_task_profiling || !t.prs.empty();
    const size_t n = c.ops.size();
    auto op_of = [&](OpKind k, unsigned idx) {
      switch(k) {
      case SubgraphDefinition::OPKIND_TASK:
        return task_to_op[idx];
      case SubgraphDefinition::OPKIND_COPY:
        return copy_to_op[idx];
      default:
        return arrival_to_op[idx];
      }
    };

    // ---- copy plans: built and analyzed once, replayed by every instantiation
    // (analysis is not thread-safe, and concurrent instantiations would
    //  otherwise race to perform it)
    c.copy_plans.assign(d.copies.size(), nullptr);
    for(size_t i = 0; i < d.copies.size(); i++) {
      const SubgraphDefinition::CopyDesc &cd = d.copies[i];
      std::vector<CopySrcDstField> dsts = cd.dsts;
      if(cd.redop_id != 0)
        for(CopySrcDstField &f : dsts)
          if(f.redop_id == 0)
            f.set_redop(cd.redop_id, cd.red_fold);
      TransferDesc *plan = cd.space.impl->make_transfer_desc(cd.srcs, dsts, cd.indirects,
                                                             cd.prs);
      if(!plan)
        SUBGRAPH_FATAL(me, "copy " << i << " has an indirection for a different index "
                                      "space type than the copy's");
      c.copy_plans[i] = plan;
      std::vector<Event> preconditions;
      plan->check_analysis_preconditions(preconditions);
      if(!preconditions.empty()) {
        // instance metadata, typically for remote instances
        Event e = Event::merge_events(preconditions);
        if(!e.has_triggered()) {
          log_subgraph.info() << "copy " << i << " of subgraph " << me
                              << ": waiting for instance metadata before compiling";
          e.wait();
        }
      }
      while(!plan->analyze(TimeLimit::relative(1000000000LL /*1 s*/)))
        ;
    }

    // ---- edges (deduplicated), inputs, postconditions
    std::vector<std::vector<uint32_t>> succ(n), pred(n), feeds(n);
    std::vector<std::vector<uint32_t>> input_targets(num_inputs), postcond_sources(num_postconds);
    std::vector<std::vector<uint32_t>> direct_inputs(n); // external inputs gating an op directly
    for(const SubgraphDefinition::Dependency &dep : d.dependencies) {
      if(dep.src_op_kind == SubgraphDefinition::OPKIND_EXT_PRECOND) {
        uint32_t t = op_of(dep.tgt_op_kind, dep.tgt_op_index);
        input_targets[dep.src_op_index].push_back(t);
        direct_inputs[t].push_back(dep.src_op_index);
      } else if(dep.tgt_op_kind == SubgraphDefinition::OPKIND_EXT_POSTCOND) {
        uint32_t src = op_of(dep.src_op_kind, dep.src_op_index);
        postcond_sources[dep.tgt_op_index].push_back(src);
        feeds[src].push_back(dep.tgt_op_index);
      } else {
        uint32_t src = op_of(dep.src_op_kind, dep.src_op_index);
        uint32_t t = op_of(dep.tgt_op_kind, dep.tgt_op_index);
        if(src == t)
          SUBGRAPH_FATAL(me, op_kind_name(dep.src_op_kind) << " " << dep.src_op_index
                                                           << " depends on itself");
        succ[src].push_back(t);
        pred[t].push_back(src);
      }
    }
    auto dedupe = [](std::vector<uint32_t> &v) {
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    for(size_t i = 0; i < n; i++) {
      dedupe(succ[i]);
      dedupe(pred[i]);
      dedupe(feeds[i]);
      dedupe(direct_inputs[i]);
    }
    for(auto &v : input_targets)
      dedupe(v);
    for(auto &v : postcond_sources)
      dedupe(v);
    for(size_t j = 0; j < num_postconds; j++)
      if(postcond_sources[j].empty())
        SUBGRAPH_FATAL(me, "external postcondition " << j << " has no sources");

    // topological order (also the cycle check) and transitive input sets
    std::vector<uint32_t> topo;
    {
      std::vector<uint32_t> indeg(n), ready;
      for(size_t i = 0; i < n; i++) {
        indeg[i] = uint32_t(pred[i].size());
        if(indeg[i] == 0)
          ready.push_back(uint32_t(i));
      }
      while(!ready.empty()) {
        uint32_t o = ready.back();
        ready.pop_back();
        topo.push_back(o);
        for(uint32_t sx : succ[o])
          if(--indeg[sx] == 0)
            ready.push_back(sx);
      }
      if(topo.size() != n)
        SUBGRAPH_FATAL(me, "the dependencies contain a cycle ("
                               << (n - topo.size()) << " operations can never run)");
    }
    c.input_words = (num_inputs + 63) / 64;
    c.op_inputs.assign(n * c.input_words, 0);
    for(uint32_t o : topo) {
      uint64_t *bits = c.op_inputs.data() + size_t(o) * c.input_words;
      for(uint32_t in : direct_inputs[o])
        bits[in / 64] |= (uint64_t(1) << (in % 64));
      for(uint32_t p : pred[o]) {
        const uint64_t *pb = c.op_inputs.data() + size_t(p) * c.input_words;
        for(size_t w = 0; w < c.input_words; w++)
          bits[w] |= pb[w];
      }
    }
    c.postcond_inputs.assign(num_postconds * c.input_words, 0);
    for(size_t j = 0; j < num_postconds; j++)
      for(uint32_t src : postcond_sources[j])
        for(size_t w = 0; w < c.input_words; w++)
          c.postcond_inputs[j * c.input_words + w] |=
              c.op_inputs[size_t(src) * c.input_words + w];
    c.inputs.resize(num_inputs);
    for(size_t in = 0; in < num_inputs; in++)
      c.inputs[in].targets = input_targets[in];
    for(size_t pi = 0; pi < c.procs.size(); pi++) {
      CompiledSubgraph::Proc &cp = c.procs[pi];
      std::vector<uint64_t> acc(c.input_words, 0);
      for(uint32_t o = cp.first_op; o < cp.first_op + cp.num_ops; o++)
        for(size_t w = 0; w < c.input_words; w++)
          acc[w] |= c.op_inputs[size_t(o) * c.input_words + w];
      for(size_t in = 0; in < num_inputs; in++)
        if(acc[in / 64] & (uint64_t(1) << (in % 64))) {
          c.inputs[in].procs.push_back(uint32_t(pi));
          cp.pending_inputs++;
        }
    }
    for(size_t i = 0; i < n; i++)
      if(pred[i].empty())
        c.roots.push_back(uint32_t(i));
    // Successors of an asynchronous operation wait for its work to complete,
    // except tasks on the same processor after a deferred-effects operation:
    // they start when its function returns and order their work after it
    // with the token it leaves behind.
    std::vector<std::vector<uint32_t>> early_succ(n), late_succ(n), token_wait(n);
    for(size_t i = 0; i < n; i++) {
      const CompiledSubgraph::Op &src = c.ops[i];
      for(uint32_t t : succ[i]) {
        const CompiledSubgraph::Op &tgt = c.ops[t];
        if(!src.async) {
          early_succ[i].push_back(t);
        } else if(src.deferred && (tgt.kind == SubgraphDefinition::OPKIND_TASK) &&
                  (tgt.proc == src.proc)) {
          early_succ[i].push_back(t);
          token_wait[t].push_back(uint32_t(src.async_index));
        } else {
          late_succ[i].push_back(t);
        }
      }
    }
    c.successors = FlattenedSparseMatrix<uint32_t>(early_succ);
    c.late_successors = FlattenedSparseMatrix<uint32_t>(late_succ);
    c.token_waits = FlattenedSparseMatrix<uint32_t>(token_wait);
    c.postconds_of = FlattenedSparseMatrix<uint32_t>(feeds);

    // ---- counter placement: with the predecessors when they share a domain
    auto domain_of_op = [&](uint32_t o) {
      return (c.ops[o].proc >= 0) ? c.procs[c.ops[o].proc].domain : int32_t(0);
    };
    auto common_domain = [&](const std::vector<uint32_t> &writers, int32_t fallback) {
      if(writers.empty())
        return fallback;
      int32_t dom = domain_of_op(writers[0]);
      for(uint32_t w : writers)
        if(domain_of_op(w) != dom)
          return fallback;
      return dom;
    };
    for(size_t i = 0; i < n; i++)
      c.ops[i].counter_domain = common_domain(pred[i], domain_of_op(uint32_t(i)));
    c.postconds.resize(num_postconds);
    for(size_t j = 0; j < num_postconds; j++) {
      c.postconds[j].num_sources = uint32_t(postcond_sources[j].size());
      c.postconds[j].counter_domain = common_domain(postcond_sources[j], 0);
      c.postconds[j].counter_offset = 0;
    }

    // ---- interpolations, grouped per operation
    for(const Interpolation &ip : d.interpolations) {
      CompiledSubgraph::Interp it;
      it.op = (ip.target_kind == Interpolation::TARGET_TASK_ARGS)
                  ? task_to_op[ip.target_index]
                  : arrival_to_op[ip.target_index];
      it.src_offset = ip.offset;
      it.bytes = ip.bytes;
      // arrival argument copies hold the barrier followed by the reduce value
      it.dst_offset = (ip.target_kind == Interpolation::TARGET_ARRIVAL_VALUE)
                          ? sizeof(Barrier) + ip.target_offset
                          : ip.target_offset;
      it.redop_id = ip.redop_id;
      c.interps.push_back(it);
    }
    std::stable_sort(c.interps.begin(), c.interps.end(),
                     [](const CompiledSubgraph::Interp &x, const CompiledSubgraph::Interp &y) {
                       return x.op < y.op;
                     });
    std::vector<bool> interpolated(n, false);
    for(const CompiledSubgraph::Interp &it : c.interps)
      interpolated[it.op] = true;

    // ---- block layout per domain and initial images
    std::vector<size_t> off(c.domains.size(), 0);
    for(size_t i = 0; i < n; i++) {
      CompiledSubgraph::Op &op = c.ops[i];
      op.counter_offset = uint32_t(off[op.counter_domain]);
      off[op.counter_domain] += sizeof(int64_t);
    }
    for(CompiledSubgraph::Postcond &pc : c.postconds) {
      pc.counter_offset = uint32_t(off[pc.counter_domain]);
      off[pc.counter_domain] += sizeof(int64_t);
    }
    for(size_t dm = 0; dm < c.domains.size(); dm++)
      off[dm] = round_up(off[dm], SUBGRAPH_CACHE_LINE_BYTES);
    for(CompiledSubgraph::Proc &cp : c.procs) {
      cp.queue_offset = uint32_t(off[cp.domain]);
      off[cp.domain] += round_up(size_t(cp.num_ops) * sizeof(int64_t),
                                 SUBGRAPH_CACHE_LINE_BYTES);
      // tail, pending-inputs and remaining-operations counters share one
      // line that is written once per operation at most
      cp.tail_offset = uint32_t(off[cp.domain]);
      cp.inputs_offset = cp.tail_offset + uint32_t(sizeof(uint64_t));
      cp.remaining_offset = cp.inputs_offset + uint32_t(sizeof(int64_t));
      off[cp.domain] += SUBGRAPH_CACHE_LINE_BYTES;
    }
    for(size_t i = 0; i < n; i++) {
      if(!interpolated[i])
        continue;
      CompiledSubgraph::Op &op = c.ops[i];
      op.args_domain = domain_of_op(uint32_t(i));
      op.args_size = (op.kind == SubgraphDefinition::OPKIND_TASK)
                         ? uint32_t(d.tasks[op.index].args.size())
                         : uint32_t(sizeof(Barrier) + d.arrivals[op.index].reduce_value.size());
      op.args_offset = uint32_t(off[op.args_domain]);
      off[op.args_domain] += round_up(op.args_size, sizeof(uint64_t));
    }
    for(size_t dm = 0; dm < c.domains.size(); dm++) {
      CompiledSubgraph::Domain &dom = c.domains[dm];
      dom.bytes = std::max(round_up(off[dm], SUBGRAPH_CACHE_LINE_BYTES),
                           SUBGRAPH_CACHE_LINE_BYTES);
      dom.image.assign(dom.bytes, 0);
    }
    auto put = [&](int32_t dm, uint32_t offset, const void *src, size_t bytes) {
      memcpy(c.domains[dm].image.data() + offset, src, bytes);
    };
    for(size_t i = 0; i < n; i++) {
      const CompiledSubgraph::Op &op = c.ops[i];
      // predecessors plus the implicit start input for roots
      int64_t count = int64_t(pred[i].size()) + int64_t(direct_inputs[i].size()) +
                      (pred[i].empty() ? 1 : 0);
      put(op.counter_domain, op.counter_offset, &count, sizeof(count));
      if(interpolated[i]) {
        if(op.kind == SubgraphDefinition::OPKIND_TASK) {
          put(op.args_domain, op.args_offset, d.tasks[op.index].args.base(),
              d.tasks[op.index].args.size());
        } else {
          const SubgraphDefinition::ArrivalDesc &ad = d.arrivals[op.index];
          put(op.args_domain, op.args_offset, &ad.barrier, sizeof(Barrier));
          put(op.args_domain, op.args_offset + uint32_t(sizeof(Barrier)),
              ad.reduce_value.base(), ad.reduce_value.size());
        }
      }
    }
    for(const CompiledSubgraph::Postcond &pc : c.postconds) {
      int64_t count = pc.num_sources;
      put(pc.counter_domain, pc.counter_offset, &count, sizeof(count));
    }
    for(CompiledSubgraph::Proc &cp : c.procs) {
      std::vector<char> &img = c.domains[cp.domain].image;
      int64_t *slots = reinterpret_cast<int64_t *>(img.data() + cp.queue_offset);
      for(uint32_t k = 0; k < cp.num_ops; k++)
        slots[k] = SUBGRAPH_EMPTY_QUEUE_ENTRY;
      // queues start empty: roots are released by start()
      uint64_t tail = 0;
      memcpy(img.data() + cp.tail_offset, &tail, sizeof(tail));
      int64_t inputs = cp.pending_inputs;
      memcpy(img.data() + cp.inputs_offset, &inputs, sizeof(inputs));
      int64_t remaining = cp.num_ops;
      memcpy(img.data() + cp.remaining_offset, &remaining, sizeof(remaining));
    }

    for(size_t dm = 0; dm < c.domains.size(); dm++)
      log_subgraph.info() << "subgraph " << me << ": domain " << dm << " numa_node="
                          << c.domains[dm].numa_node << " bytes=" << c.domains[dm].bytes;
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
                                 const ProfilingRequestSet &prs,
                                 const SubgraphInstantiationProfiling &profiling,
                                 span<const Event> preconditions,
                                 span<const Event> postconditions, Event start_event,
                                 Event finish_event, int priority_adjust)
  {
    if(!prs.empty())
      SUBGRAPH_FATAL(me, "profiling requests on the instantiation itself are not "
                         "implemented; use SubgraphInstantiationProfiling for per-"
                         "operation requests");
    if(preconditions.size() != compiled.inputs.size())
      SUBGRAPH_FATAL(me, "instantiated with " << preconditions.size()
                                              << " external preconditions, but the "
                                                 "definition declares "
                                              << compiled.inputs.size());
    if(postconditions.size() != compiled.postconds.size())
      SUBGRAPH_FATAL(me, "instantiated with " << postconditions.size()
                                              << " external postconditions, but the "
                                                 "definition declares "
                                              << compiled.postconds.size());

    {
      AutoLock<> al(lifecycle_lock);
      if(destroy_requested)
        SUBGRAPH_FATAL(me, "instantiated after destroy");
      outstanding_instantiations++;
      if(defn->concurrency_mode == SubgraphDefinition::INSTANTIATION_ORDER) {
        // Instantiations run in order: this one starts once the previous one
        // has finished, and the next one will wait for this one.
        start_event = Event::merge_events(start_event, previous_instantiation_completion);
        previous_instantiation_completion = finish_event;
      }
    }

    // priority_adjust is the priority of this instantiation as a whole
    SubgraphExecutionState *state =
        new SubgraphExecutionState(this, finish_event, priority_adjust, postconditions);
    state->interpolate(args, arglen);
    state->setup_profiling(profiling);
    // Release the execution state once the instantiation has finished.
    EventImpl::add_waiter(finish_event, new SubgraphInstantiationCleanup(state));
    // Deliver external inputs; nothing can run before start() releases the roots.
    for(size_t i = 0; i < preconditions.size(); i++) {
      Event e = preconditions[i];
      bool poisoned = false;
      if(!e.exists() || e.has_triggered_faultaware(poisoned))
        state->input_triggered(uint32_t(i), poisoned);
      else
        EventImpl::add_waiter(e, new SubgraphInputWaiter(state, uint32_t(i)));
    }
    // Start once the precondition is satisfied.
    SubgraphWorkLauncher::launch_or_defer(state, start_event);
  }

  void SubgraphImpl::destroy(void)
  {
    delete defn;
    defn = nullptr;
    compiled.clear();

    {
      AutoLock<> al(block_pool_lock);
      for(std::vector<char *> &blocks : block_pool)
        free_blocks(blocks);
      block_pool.clear();
    }

    previous_instantiation_completion = Event::NO_EVENT;
    outstanding_instantiations = 0;
    destroy_requested = false;
    destroy_wait_on = Event::NO_EVENT;
    destroy_done = UserEvent::NO_USER_EVENT;

    // TODO: when we create subgraphs on remote nodes, send a message to the
    //  creator node so they can add it to their free list
    NodeID creator_node = ID(me).subgraph_creator_node();
    assert(creator_node == Network::my_node_id);
    NodeID owner_node = ID(me).subgraph_owner_node();
    assert(owner_node == Network::my_node_id);

    get_runtime()->local_subgraph_free_lists[owner_node]->free_entry(this);
  }

  Event SubgraphImpl::request_destroy(Event wait_on)
  {
    {
      AutoLock<> al(lifecycle_lock);
      if(destroy_requested)
        SUBGRAPH_FATAL(me, "destroyed twice");
      destroy_requested = true;
      if(outstanding_instantiations > 0) {
        // The last instantiation to release its state completes the destroy.
        destroy_wait_on = wait_on;
        destroy_done = UserEvent::create_user_event();
        return destroy_done;
      }
    }
    return complete_destroy(wait_on, UserEvent::NO_USER_EVENT);
  }

  Event SubgraphImpl::complete_destroy(Event wait_on, UserEvent done)
  {
    if(wait_on.has_triggered()) {
      destroy();
      if(done.exists()) {
        done.trigger();
        return done;
      }
      return Event::NO_EVENT;
    }
    if(!done.exists())
      done = UserEvent::create_user_event();
    deferred_destroy.defer(this, wait_on, done);
    return done;
  }

  void SubgraphImpl::instantiation_released(void)
  {
    Event wait_on;
    UserEvent done;
    {
      AutoLock<> al(lifecycle_lock);
      assert(outstanding_instantiations > 0);
      outstanding_instantiations--;
      if((outstanding_instantiations > 0) || !destroy_requested)
        return;
      wait_on = destroy_wait_on;
      done = destroy_done;
    }
    complete_destroy(wait_on, done);
  }

  // Blocks are placed on the processors' NUMA nodes, which is page-granular
  // and costs system calls, so they are allocated once and recycled.
  void SubgraphImpl::acquire_blocks(std::vector<char *> &blocks)
  {
    {
      AutoLock<> al(block_pool_lock);
      if(!block_pool.empty()) {
        blocks.swap(block_pool.back());
        block_pool.pop_back();
        return;
      }
    }
    static const bool numa_available = numasysif_numa_available();
    blocks.resize(compiled.domains.size(), nullptr);
    for(size_t dm = 0; dm < compiled.domains.size(); dm++) {
      const CompiledSubgraph::Domain &dom = compiled.domains[dm];
      void *p = nullptr;
      if(numa_available && (dom.numa_node >= 0))
        p = numasysif_alloc_mem(dom.numa_node, dom.bytes, false /*!pin*/);
      if(!p) {
        if(posix_memalign(&p, SUBGRAPH_CACHE_LINE_BYTES, dom.bytes) != 0)
          p = nullptr;
      }
      if(!p)
        SUBGRAPH_FATAL(me, "failed to allocate " << dom.bytes
                                                 << " bytes of instantiation state");
      blocks[dm] = static_cast<char *>(p);
    }
  }

  void SubgraphImpl::release_blocks(std::vector<char *> &blocks)
  {
    AutoLock<> al(block_pool_lock);
    block_pool.emplace_back();
    block_pool.back().swap(blocks);
  }

  void SubgraphImpl::free_blocks(std::vector<char *> &blocks)
  {
    static const bool numa_available = numasysif_numa_available();
    for(size_t dm = 0; dm < blocks.size(); dm++) {
      const CompiledSubgraph::Domain &dom = compiled.domains[dm];
      bool freed = false;
      if(numa_available && (dom.numa_node >= 0))
        freed = numasysif_free_mem(dom.numa_node, blocks[dm], dom.bytes);
      if(!freed)
        free(blocks[dm]);
    }
    blocks.clear();
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
    if(poisoned)
      SUBGRAPH_FATAL(subgraph->me, "destroy precondition was poisoned");
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
  //

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
    SubgraphInstantiationProfiling profiling;
    bool ok = ((fbd >> preconditions) && (fbd >> postconditions));
    if(ok && (fbd.bytes_left() > 0))
      ok = (fbd >> prs);
    if(ok && (fbd.bytes_left() > 0))
      ok = (fbd >> profiling);
    assert(ok);

    subgraph->instantiate(data, msg.arglen, prs, profiling, preconditions, postconditions,
                          msg.wait_on, msg.finish_event, msg.priority_adjust);
  }

  ActiveMessageHandlerReg<SubgraphInstantiateMessage>
      subgraph_instantiate_message_handler;

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphDestroyMessage
  //

  /*static*/ void
  SubgraphDestroyMessage::handle_message(NodeID sender, const SubgraphDestroyMessage &msg,
                                         const void *data, size_t datalen)
  {
    SubgraphImpl *subgraph = get_runtime()->get_subgraph_impl(msg.subgraph);
    msg.to_trigger.trigger(subgraph->request_destroy(msg.wait_on));
  }

  ActiveMessageHandlerReg<SubgraphDestroyMessage> subgraph_destroy_message_handler;

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphWorkLauncher
  //

  SubgraphWorkLauncher::SubgraphWorkLauncher(SubgraphExecutionState *_state)
    : state(_state)
  {}

  /*static*/ void SubgraphWorkLauncher::launch_or_defer(SubgraphExecutionState *state,
                                                        Event wait_on)
  {
    bool poisoned = false;
    if(!wait_on.exists() || wait_on.has_triggered_faultaware(poisoned)) {
      launch(state, poisoned);
    } else {
      // The launcher deletes itself once the precondition triggers.
      EventImpl::add_waiter(wait_on, new SubgraphWorkLauncher(state));
    }
  }

  /*static*/ void SubgraphWorkLauncher::launch(SubgraphExecutionState *state,
                                               bool poisoned)
  {
    const CompiledSubgraph &c = state->subgraph->compiled;
    if(poisoned) {
      // Nothing runs; the finish event and every postcondition are
      // poisoned and cleanup proceeds as usual.
      log_subgraph.info() << "poisoned precondition: subgraph=" << state->subgraph->me;
      for(Event pc : state->postconditions)
        GenEventImpl::trigger(pc, true /*poisoned*/);
      GenEventImpl::trigger(state->finish_event, true /*poisoned*/);
      return;
    }
    if(c.ops.empty()) {
      // Nothing to wait for.
      for(Event pc : state->postconditions)
        GenEventImpl::trigger(pc, false /*!poisoned*/);
      GenEventImpl::trigger(state->finish_event, false /*!poisoned*/);
      return;
    }
    for(const CompiledSubgraph::Proc &p : c.procs)
      p.impl->enqueue_subgraph(state);
    state->start();
  }

  void SubgraphWorkLauncher::event_triggered(bool poisoned, TimeLimit work_until)
  {
    launch(state, poisoned);
    delete this;
  }

  void SubgraphWorkLauncher::print(std::ostream &os) const
  {
    os << "SubgraphWorkLauncher: subgraph=" << state->subgraph->me;
  }

  Event SubgraphWorkLauncher::get_finish_event(void) const { return Event::NO_EVENT; }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphInstantiationCleanup
  //

  SubgraphInstantiationCleanup::SubgraphInstantiationCleanup(
      SubgraphExecutionState *_state)
    : state(_state)
  {}

  void SubgraphInstantiationCleanup::cleanup()
  {
    SubgraphImpl *subgraph = state->get_subgraph();
    delete state;
    state = nullptr;
    // After the state is gone: this may complete a pending destroy.
    subgraph->instantiation_released();
  }

  void SubgraphInstantiationCleanup::event_triggered(bool poisoned, TimeLimit work_until)
  {
    // Defer to a background worker to keep the event trigger path cheap.
    get_runtime()->subgraph_resource_reaper.enqueue_cleanup(this);
  }

  void SubgraphInstantiationCleanup::print(std::ostream &os) const
  {
    os << "SubgraphInstantiationCleanup: state=" << static_cast<void *>(state);
  }

  Event SubgraphInstantiationCleanup::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphInputWaiter
  //

  SubgraphInputWaiter::SubgraphInputWaiter(SubgraphExecutionState *_state, uint32_t _input)
    : state(_state)
    , input(_input)
  {}

  void SubgraphInputWaiter::event_triggered(bool poisoned, TimeLimit work_until)
  {
    state->input_triggered(input, poisoned);
    delete this;
  }

  void SubgraphInputWaiter::print(std::ostream &os) const
  {
    os << "SubgraphInputWaiter: subgraph=" << state->get_subgraph()->me
       << " input=" << input;
  }

  Event SubgraphInputWaiter::get_finish_event(void) const { return Event::NO_EVENT; }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphResourceReaper
  //

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
  // class SubgraphExecutionState
  //

  SubgraphExecutionState::SubgraphExecutionState(SubgraphImpl *_subgraph,
                                                 Event _finish_event, int _priority,
                                                 span<const Event> _postconditions)
    : subgraph(_subgraph)
    , finish_counter(int64_t(_subgraph->compiled.procs.size()) +
                     int64_t(_subgraph->compiled.num_direct_ops))
    , finish_event(_finish_event)
    , postconditions(_postconditions.data(), _postconditions.data() + _postconditions.size())
    , poisoned_inputs(_subgraph->compiled.input_words)
    , poisoned(false)
    , priority(_priority)
  {
    for(auto &w : poisoned_inputs)
      w.store(0);
    subgraph->acquire_blocks(blocks);
    const CompiledSubgraph &c = subgraph->compiled;
    for(size_t dm = 0; dm < c.domains.size(); dm++)
      memcpy(blocks[dm], c.domains[dm].image.data(), c.domains[dm].bytes);
    if(c.num_async_ops > 0) {
      async_ops.resize(c.num_async_ops);
      tokens.assign(c.num_async_ops, nullptr);
      for(size_t i = 0; i < c.ops.size(); i++)
        if(c.ops[i].async_index >= 0) {
          async_ops[c.ops[i].async_index].state = this;
          async_ops[c.ops[i].async_index].op = uint32_t(i);
        }
    }
  }

  SubgraphExecutionState::~SubgraphExecutionState()
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(const CompiledSubgraph::Proc &p : c.procs)
      if(p.num_async > 0)
        p.impl->release_subgraph_tokens(tokens.data() + p.first_async, p.num_async);
    subgraph->release_blocks(blocks);
  }

  atomic<int64_t> &SubgraphExecutionState::counter(uint32_t op) const
  {
    const CompiledSubgraph::Op &o = subgraph->compiled.ops[op];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[o.counter_domain] +
                                                o.counter_offset);
  }

  atomic<int64_t> *SubgraphExecutionState::queue(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return reinterpret_cast<atomic<int64_t> *>(blocks[p.domain] + p.queue_offset);
  }

  atomic<uint64_t> &SubgraphExecutionState::tail(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return *reinterpret_cast<atomic<uint64_t> *>(blocks[p.domain] + p.tail_offset);
  }

  atomic<int64_t> &SubgraphExecutionState::pending_inputs(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[p.domain] + p.inputs_offset);
  }

  atomic<int64_t> &SubgraphExecutionState::remaining(uint32_t proc) const
  {
    const CompiledSubgraph::Proc &p = subgraph->compiled.procs[proc];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[p.domain] + p.remaining_offset);
  }

  atomic<int64_t> &SubgraphExecutionState::postcond_counter(uint32_t pc) const
  {
    const CompiledSubgraph::Postcond &p = subgraph->compiled.postconds[pc];
    return *reinterpret_cast<atomic<int64_t> *>(blocks[p.counter_domain] +
                                                p.counter_offset);
  }

  ByteArrayRef SubgraphExecutionState::op_args(uint32_t op) const
  {
    const CompiledSubgraph::Op &o = subgraph->compiled.ops[op];
    if(o.args_domain >= 0)
      return ByteArrayRef(blocks[o.args_domain] + o.args_offset, o.args_size);
    if(o.kind == SubgraphDefinition::OPKIND_TASK) {
      const ByteArray &a = subgraph->defn->tasks[o.index].args;
      return ByteArrayRef(a.base(), a.size());
    }
    return ByteArrayRef(nullptr, 0);
  }

  void SubgraphExecutionState::interpolate(const void *args, size_t arglen)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(const CompiledSubgraph::Interp &it : c.interps) {
      if((it.src_offset + it.bytes) > arglen)
        SUBGRAPH_FATAL(subgraph->me, "interpolation reads " << it.src_offset << "+" << it.bytes
                                                            << " bytes of the instantiation "
                                                               "arguments, but only "
                                                            << arglen << " were given");
      const CompiledSubgraph::Op &o = c.ops[it.op];
      char *dst = blocks[o.args_domain] + o.args_offset + it.dst_offset;
      const char *src = static_cast<const char *>(args) + it.src_offset;
      if(it.redop_id == 0) {
        memcpy(dst, src, it.bytes);
      } else {
        const ReductionOpUntyped *redop =
            get_runtime()->reduce_op_table.get(it.redop_id, nullptr);
        (redop->cpu_apply_excl_fn)(dst, 0, src, 0, 1 /*count*/, redop->userdata);
      }
    }
  }

  void SubgraphExecutionState::setup_profiling(
      const SubgraphInstantiationProfiling &iprof)
  {
    const SubgraphDefinition &d = *subgraph->defn;
    const CompiledSubgraph &c = subgraph->compiled;
    if(!iprof.copies.empty()) {
      // copies profile through Realm's transfer operations: hand each one
      // the definition's requests merged with this instantiation's
      copy_prs.resize(d.copies.size());
      for(size_t i = 0; i < d.copies.size(); i++)
        copy_prs[i].import_requests(d.copies[i].prs);
      for(const auto &kv : iprof.copies) {
        if(kv.first >= d.copies.size())
          SUBGRAPH_FATAL(subgraph->me, "profiling requested for copy "
                                           << kv.first << ", which does not exist");
        copy_prs[kv.first].import_requests(kv.second);
      }
    }
    if(!c.any_task_profiling && iprof.tasks.empty())
      return;
    prof_index.assign(c.ops.size(), -1);
    auto entry_for_task = [&](unsigned task) -> OpProfiling & {
      if(task >= d.tasks.size())
        SUBGRAPH_FATAL(subgraph->me, "profiling requested for task " << task
                                                                     << ", which does not exist");
      int32_t &idx = prof_index[c.task_ops[task]];
      if(idx < 0) {
        idx = int32_t(profiling.size());
        profiling.emplace_back(new OpProfiling);
        profiling.back()->requests.import_requests(d.tasks[task].prs);
      }
      return *profiling[idx];
    };
    for(size_t t = 0; t < d.tasks.size(); t++)
      if(!d.tasks[t].prs.empty())
        entry_for_task(unsigned(t));
    for(const auto &kv : iprof.tasks)
      entry_for_task(kv.first).requests.import_requests(kv.second);
    for(auto &p : profiling) {
      p->measurements.import_requests(p->requests);
      p->wants_timeline =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationTimeline>();
      p->wants_proc =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationProcessorUsage>();
      p->wants_status =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationStatus>();
      p->wants_fevent =
          p->measurements.wants_measurement<ProfilingMeasurements::OperationFinishEvent>();
      if(p->wants_timeline)
        p->timeline.record_create_time();
    }
  }

  bool SubgraphExecutionState::op_poisoned(uint32_t op) const
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const uint64_t *bits = c.op_inputs.data() + size_t(op) * c.input_words;
    for(size_t w = 0; w < c.input_words; w++)
      if(bits[w] & poisoned_inputs[w].load_acquire())
        return true;
    return false;
  }

  void SubgraphExecutionState::input_triggered(uint32_t input, bool is_poisoned)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const CompiledSubgraph::Input &in = c.inputs[input];
    if(is_poisoned) {
      // Published before any dependent operation can become ready.
      poisoned_inputs[input / 64].fetch_or_acqrel(uint64_t(1) << (input % 64));
      poisoned.store_release(true);
    }
    for(uint32_t p : in.procs)
      pending_inputs(p).fetch_sub_acqrel(1);
    for(uint32_t t : in.targets)
      if(counter(t).fetch_sub_acqrel(1) == 1)
        op_ready(t);
  }

  void SubgraphExecutionState::start(void)
  {
    for(uint32_t r : subgraph->compiled.roots)
      if(counter(r).fetch_sub_acqrel(1) == 1)
        op_ready(r);
  }

  void SubgraphExecutionState::op_ready(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const CompiledSubgraph::Op &o = c.ops[op];
    if(OpProfiling *pf = prof(op))
      if(pf->wants_timeline)
        pf->timeline.record_ready_time();
    if(o.proc >= 0) {
      const uint64_t slot = tail(o.proc).fetch_add_acqrel(1);
      queue(o.proc)[slot].store_release(int64_t(op));
      c.procs[o.proc].impl->notify_scheduler_of_new_work();
      return;
    }
    // Directly launched operation: performed by whoever made it ready.
    switch(o.kind) {
    case SubgraphDefinition::OPKIND_ARRIVAL:
    {
      if(!op_poisoned(op)) {
        const SubgraphDefinition::ArrivalDesc &ad = subgraph->defn->arrivals[o.index];
        Barrier b = ad.barrier;
        const void *value = ad.reduce_value.base();
        size_t value_size = ad.reduce_value.size();
        if(o.args_domain >= 0) {
          ByteArrayRef a = op_args(op);
          memcpy(&b, a.base(), sizeof(Barrier));
          value = static_cast<const char *>(a.base()) + sizeof(Barrier);
          value_size = a.size() - sizeof(Barrier);
        }
        b.arrive(ad.count, Event::NO_EVENT, value, value_size);
      }
      break;
    }
    case SubgraphDefinition::OPKIND_COPY:
      if(!op_poisoned(op)) {
        launch_copy(op); // completes asynchronously
        return;
      }
      break;
    default:
      SUBGRAPH_FATAL(subgraph->me, "internal error: operation " << op
                                                                << " of kind "
                                                                << op_kind_name(o.kind)
                                                                << " cannot be launched directly");
    }
    op_body_done(op);
    op_finished(op);
  }

  namespace {
    // A transfer on a compiled plan that reports back to the instantiation
    // when the data has moved. Realm's own bookkeeping (profiling responses,
    // the operation's finish event) runs first.
    class SubgraphTransferOperation : public TransferOperation {
    public:
      SubgraphTransferOperation(TransferDesc &desc, GenEventImpl *finish_event,
                                EventImpl::gen_t finish_gen, int priority,
                                const ProfilingRequestSet &prs,
                                SubgraphExecutionState *_state, uint32_t _op)
        : TransferOperation(desc, Event::NO_EVENT, finish_event, finish_gen, priority, prs)
        , state(_state)
        , op(_op)
      {}

    protected:
      virtual void mark_completed(void) override
      {
        SubgraphExecutionState *s = state;
        const uint32_t o = op;
        TransferOperation::mark_completed(); // may delete this operation
        s->op_body_done(o);
        s->op_finished(o);
      }

      SubgraphExecutionState *state;
      uint32_t op;
    };
  } // namespace

  void SubgraphExecutionState::launch_copy(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    const CompiledSubgraph::Op &o = c.ops[op];
    const SubgraphDefinition::CopyDesc &cd = subgraph->defn->copies[o.index];
    TransferDesc *plan = c.copy_plans[o.index];
    // every transfer operation needs a finish event of its own
    GenEventImpl *finish_event = GenEventImpl::create_genevent();
    Event ev = finish_event->current_event();
    const ProfilingRequestSet &prs = copy_prs.empty() ? cd.prs : copy_prs[o.index];
    SubgraphTransferOperation *top = new SubgraphTransferOperation(
        *plan, finish_event, ID(ev).event_generation(), priority + cd.priority, prs, this,
        op);
    // the plan is analyzed and the graph satisfied the preconditions: this
    // allocates intermediate buffers and creates the transfer descriptors
    top->start_or_defer();
  }

  void SubgraphExecutionState::op_body_done(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(uint64_t i = c.successors.offsets[op]; i < c.successors.offsets[op + 1]; i++) {
      const uint32_t sx = c.successors.data[i];
      if(counter(sx).fetch_sub_acqrel(1) == 1)
        op_ready(sx);
    }
  }

  void SubgraphExecutionState::op_finished(uint32_t op)
  {
    const CompiledSubgraph &c = subgraph->compiled;
    for(uint64_t i = c.postconds_of.offsets[op]; i < c.postconds_of.offsets[op + 1]; i++) {
      const uint32_t pc = c.postconds_of.data[i];
      if(postcond_counter(pc).fetch_sub_acqrel(1) != 1)
        continue;
      bool pc_poisoned = false;
      const uint64_t *bits = c.postcond_inputs.data() + size_t(pc) * c.input_words;
      for(size_t w = 0; w < c.input_words; w++)
        pc_poisoned = pc_poisoned || ((bits[w] & poisoned_inputs[w].load_acquire()) != 0);
      GenEventImpl::trigger(postconditions[pc], pc_poisoned);
    }
    for(uint64_t i = c.late_successors.offsets[op]; i < c.late_successors.offsets[op + 1];
        i++) {
      const uint32_t sx = c.late_successors.data[i];
      if(counter(sx).fetch_sub_acqrel(1) == 1)
        op_ready(sx);
    }
    // Last: a processor's share of the finish counter goes when its last
    // operation completes, whichever one that is (tasks may block or
    // complete asynchronously), and the state may be freed right after.
    const int32_t proc = c.ops[op].proc;
    if(proc < 0) {
      contributor_finished();
    } else if(remaining(uint32_t(proc)).fetch_sub_acqrel(1) == 1) {
      contributor_finished();
    }
  }

  void SubgraphExecutionState::complete_op_profiling(uint32_t op, OpProfiling *pf,
                                                     Event finish_event, bool skipped)
  {
    if(!pf)
      return;
    if(pf->wants_timeline) {
      if(!skipped)
        pf->timeline.record_complete_time();
      pf->measurements.add_measurement(pf->timeline);
    }
    if(pf->wants_proc) {
      ProfilingMeasurements::OperationProcessorUsage usage;
      usage.proc = subgraph->compiled.procs[subgraph->compiled.ops[op].proc].proc;
      pf->measurements.add_measurement(usage);
    }
    if(pf->wants_status) {
      ProfilingMeasurements::OperationStatus status;
      status.result = skipped ? ProfilingMeasurements::OperationStatus::CANCELLED
                              : ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY;
      status.error_code = 0;
      pf->measurements.add_measurement(status);
    }
    if(pf->wants_fevent) {
      ProfilingMeasurements::OperationFinishEvent fe;
      fe.finish_event = finish_event;
      pf->measurements.add_measurement(fe);
    }
    pf->measurements.send_responses(pf->requests);
  }

  void SubgraphExecutionState::AsyncOp::async_completed(void)
  {
    // The work this task launched has completed (called from the
    // processor's completion machinery, e.g. the GPU worker).
    if(finish_event.exists())
      GenEventImpl::trigger(finish_event, false /*!poisoned*/);
    state->complete_op_profiling(op, state->prof(op), finish_event, false /*!skipped*/);
    state->op_finished(op);
  }

  void SubgraphExecutionState::contributor_finished(void)
  {
    // Copy out what is needed first: once the counter reaches zero and the
    // event triggers, this state may be released at any moment.
    Event fe = finish_event;
    bool p = poisoned.load_acquire();
    if(finish_counter.fetch_sub_acqrel(1) == 1)
      GenEventImpl::trigger(fe, p);
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class ProcSubgraphExecutor
  //

  /*static*/ int ProcSubgraphExecutor::poll_budget_us = 100;

  ProcSubgraphExecutor::ProcSubgraphExecutor(Processor _proc)
    : proc(_proc)
    , numa_node_(-2)
    , pending_count(0)
    , scan_start(0)
    , peeked_cursor(0)
    , peeked_op(SUBGRAPH_EMPTY_QUEUE_ENTRY)
    , activity_epoch(0)
    , polled_epoch(0)
    , poll_deadline_ns(0)
  {}

  ProcSubgraphExecutor::~ProcSubgraphExecutor() {}

  void ProcSubgraphExecutor::note_worker_started(void)
  {
    if(numa_node_ != -2)
      return;
    int node = -1;
#ifdef __linux__
    int cpu = sched_getcpu();
    if((cpu >= 0) && numasysif_numa_available()) {
      const HardwareTopology &topo = get_runtime()->host_topology;
      for(int dm = 0; (dm < 1024) && (node < 0); dm++)
        if(topo.numa_domain_has_processors(dm) &&
           topo.get_processors_by_domain(dm).count(cpu))
          node = dm;
    }
#endif
    numa_node_ = node;
  }

  void ProcSubgraphExecutor::enqueue_subgraph(SubgraphExecutionState *state)
  {
    AutoLock<> al(pending_mutex);
    pending.push_back(state);
    pending_count.store_release(int64_t(pending.size()));
  }

  void ProcSubgraphExecutor::absorb_pending(void)
  {
    {
      AutoLock<> al(pending_mutex);
      pending.swap(pending_scratch);
      pending_count.store(0);
    }
    for(SubgraphExecutionState *state : pending_scratch) {
      const CompiledSubgraph &c = state->subgraph->compiled;
      auto it = c.proc_index.find(proc);
      assert(it != c.proc_index.end());
      Cursor cur;
      cur.state = state;
      cur.proc = it->second;
      cur.queue = state->queue(cur.proc);
      cur.front = 0;
      cur.end = c.procs[cur.proc].num_ops;
      cur.priority = state->get_priority();
      assert(cur.end > 0);
      active.push_back(cur);
    }
    pending_scratch.clear();
    activity_epoch++;
  }

  bool ProcSubgraphExecutor::peek(int &priority)
  {
    if(pending_count.load_acquire() > 0)
      absorb_pending();

    // Highest-priority ready instantiation wins; ties go round robin.
    // Instantiations below the floor wait, like normal tasks do, even if
    // the ones holding the processor have nothing ready right now.
    const int floor = active_floor(INT_MIN);
    const size_t n = active.size();
    bool found = false;
    for(size_t k = 0; k < n; k++) {
      size_t i = scan_start + k;
      if(i >= n)
        i -= n;
      const Cursor &c = active[i];
      if(c.priority < floor)
        continue;
      if(found && (c.priority <= priority))
        continue;
      int64_t op = c.queue[c.front].load_acquire();
      if(op != SUBGRAPH_EMPTY_QUEUE_ENTRY) {
        peeked_cursor = i;
        peeked_op = op;
        priority = c.priority;
        found = true;
      }
    }
    return found;
  }

  int ProcSubgraphExecutor::active_floor(int none) const
  {
    int floor = none;
    bool any = false;
    for(const Cursor &c : active) {
      if(c.state->pending_inputs(c.proc).load_acquire() != 0)
        continue; // may still be waiting on work this processor would block
      if(!any || (c.priority > floor))
        floor = c.priority;
      any = true;
    }
    return floor;
  }

  void ProcSubgraphExecutor::dequeue(ReadyEntry &entry)
  {
    assert((peeked_cursor < active.size()) && (peeked_op != SUBGRAPH_EMPTY_QUEUE_ENTRY));
    Cursor &c = active[peeked_cursor];
    entry.state = c.state;
    entry.op = uint32_t(peeked_op);
    entry.priority = c.priority;
    c.front++;
    if(c.front == c.end) {
      // Drop the cursor now: once this processor's operations have all
      // completed, its finish decrement may let the state be released at
      // any time.
      active[peeked_cursor] = active.back();
      active.pop_back();
      scan_start = peeked_cursor;
    } else {
      scan_start = peeked_cursor + 1;
    }
    if(scan_start >= active.size())
      scan_start = 0;
    peeked_op = SUBGRAPH_EMPTY_QUEUE_ENTRY;
    activity_epoch++;
  }

  void ProcSubgraphExecutor::execute(const ReadyEntry &entry)
  {
    SubgraphExecutionState *state = entry.state;
    SubgraphImpl *impl = state->subgraph;
    const CompiledSubgraph &c = impl->compiled;
    const CompiledSubgraph::Op &op = c.ops[entry.op];
    assert((op.kind == SubgraphDefinition::OPKIND_TASK) && (op.proc >= 0));

    LocalTaskProcessor *proc_impl = c.procs[op.proc].impl;
    SubgraphExecutionState::OpProfiling *pf = state->prof(entry.op);

    if(state->op_poisoned(entry.op)) {
      // A task depending on a poisoned input is skipped; its successors
      // still drain and the finish event ends up poisoned.
      state->op_body_done(entry.op);
      state->complete_op_profiling(entry.op, pf, Event::NO_EVENT, true /*skipped*/);
      state->op_finished(entry.op);
      return;
    }

    const SubgraphDefinition::TaskDesc &task_desc = impl->defn->tasks[op.index];

    // Tokens of the deferred-effects predecessors this task's work must follow.
    const uint64_t tw0 = c.token_waits.offsets[entry.op];
    const size_t num_tokens = size_t(c.token_waits.offsets[entry.op + 1] - tw0);
    const void *token_buf[16];
    std::vector<const void *> token_vec;
    const void **wait_tokens = token_buf;
    if(num_tokens > 16) {
      token_vec.resize(num_tokens);
      wait_tokens = token_vec.data();
    }
    for(size_t k = 0; k < num_tokens; k++)
      wait_tokens[k] = state->tokens[c.token_waits.data[tw0 + k]];

    // Run the task function on this thread. There is no Operation: the flag
    // lets Processor::get_current_finish_event create a finish event on
    // demand, triggered once the task is complete. Context managers are not
    // applied; the processor's subgraph hooks provide the equivalent.
    Thread *thread = Thread::self();
    thread->subgraph_finish_event() =
        (pf && pf->wants_fevent) ? UserEvent::create_user_event().id : 0;
    if(pf && pf->wants_timeline)
      pf->timeline.record_start_time();
    void *context = proc_impl->begin_subgraph_task(wait_tokens, num_tokens);
    ThreadLocal::current_processor = proc;
    thread->start_subgraph_task_execution();
    proc_impl->execute_task(task_desc.task_id, state->op_args(entry.op));
    thread->stop_subgraph_task_execution();
    ThreadLocal::current_processor = Processor::NO_PROC;
    void *token = proc_impl->end_subgraph_task(context, op.deferred);
    if(pf && pf->wants_timeline)
      pf->timeline.record_end_time();
    Event finish_event;
    finish_event.id = thread->subgraph_finish_event();
    thread->subgraph_finish_event() = 0;

    if(!op.async) {
      if(finish_event.exists())
        GenEventImpl::trigger(finish_event, false /*!poisoned*/);
      state->op_body_done(entry.op);
      state->complete_op_profiling(entry.op, pf, finish_event, false /*!skipped*/);
      state->op_finished(entry.op);
      return;
    }

    // The function returned but the work goes on. Successors that may start
    // now are released first: arming the completion may finish the
    // instantiation, and free the state, before it returns.
    SubgraphExecutionState::AsyncOp &ao = state->async_ops[op.async_index];
    ao.finish_event = finish_event;
    state->tokens[op.async_index] = token;
    state->op_body_done(entry.op);
    proc_impl->arm_subgraph_task_completion(context, token, &ao);
  }

  bool ProcSubgraphExecutor::keep_polling(void)
  {
    if((poll_budget_us <= 0) || (activity_epoch == 0))
      return false;
    // While an instantiation holds this processor, nothing else may run
    // here anyway: keep polling until it is done.
    if(!active.empty())
      return true;
    long long now = Clock::current_time_in_nanoseconds();
    if(activity_epoch != polled_epoch) {
      polled_epoch = activity_epoch;
      poll_deadline_ns = now + 1000LL * poll_budget_us;
    }
    return now < poll_deadline_ns;
  }

}; // namespace Realm
