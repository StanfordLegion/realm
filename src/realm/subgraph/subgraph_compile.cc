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

// Compiled subgraphs: validation of a SubgraphDefinition and its compilation
// into the per-processor queues, counters, plans and NUMA-placed block images
// described in subgraph_impl.h.

#include "realm/subgraph/subgraph_impl.h"
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

  namespace {
    size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }
  } // namespace

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
        case SubgraphDefinition::OPKIND_COPY:
          limit = d.copies.size();
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
    alive = true;
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


}; // namespace Realm
