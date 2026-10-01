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

#include <algorithm>
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
  // struct CompiledSubgraph
  //

  void CompiledSubgraph::clear()
  {
    ops.clear();
    procs.clear();
    domains.clear();
    proc_index.clear();
    successors.clear();
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

    // ---- validation: everything the compiled implementation cannot run yet
    if((d.concurrency_mode != SubgraphDefinition::ONE_SHOT) &&
       (d.concurrency_mode != SubgraphDefinition::INSTANTIATION_ORDER))
      SUBGRAPH_FATAL(me, "concurrency modes SERIALIZABLE and CONCURRENT are not "
                         "implemented; use ONE_SHOT or INSTANTIATION_ORDER");
    if(!d.copies.empty())
      SUBGRAPH_FATAL(me, "copies and fills are not implemented (" << d.copies.size()
                                                                 << " in definition)");
    if(!d.arrivals.empty())
      SUBGRAPH_FATAL(me, "barrier arrivals are not implemented (" << d.arrivals.size()
                                                                 << " in definition)");
    if(!d.instantiations.empty())
      SUBGRAPH_FATAL(me, "nested subgraph instantiations are not implemented ("
                             << d.instantiations.size() << " in definition)");
    if(!d.acquires.empty() || !d.releases.empty())
      SUBGRAPH_FATAL(me, "reservation acquires and releases are not implemented");
    if(!d.interpolations.empty())
      SUBGRAPH_FATAL(me, "interpolations are not implemented (" << d.interpolations.size()
                                                               << " in definition)");

    for(size_t i = 0; i < d.tasks.size(); i++) {
      const SubgraphDefinition::TaskDesc &t = d.tasks[i];
      if(!t.proc.exists())
        SUBGRAPH_FATAL(me, "task " << i << " has no processor");
      if(t.proc.kind() != Processor::LOC_PROC)
        SUBGRAPH_FATAL(me, "task " << i << " runs on " << t.proc
                                   << ", which is not a LOC_PROC; only CPU tasks are "
                                      "implemented");
      if(NodeID(t.proc.address_space()) != Network::my_node_id)
        SUBGRAPH_FATAL(me, "task " << i << " runs on " << t.proc
                                   << ", which belongs to another node; only local "
                                      "tasks are implemented");
      if(t.priority != 0)
        SUBGRAPH_FATAL(me, "task " << i << " has priority " << t.priority
                                   << "; per-task priorities are not implemented");
      if(!t.prs.empty())
        SUBGRAPH_FATAL(me, "task " << i
                                   << " has profiling requests; per-task profiling is "
                                      "not implemented");
    }

    for(size_t i = 0; i < d.dependencies.size(); i++) {
      const SubgraphDefinition::Dependency &dep = d.dependencies[i];
      for(OpKind k : {dep.src_op_kind, dep.tgt_op_kind}) {
        if((k == SubgraphDefinition::OPKIND_EXT_PRECOND) ||
           (k == SubgraphDefinition::OPKIND_EXT_POSTCOND))
          SUBGRAPH_FATAL(me, "dependency " << i
                                           << " uses an external pre/postcondition; "
                                              "external conditions are not implemented");
        if(k != SubgraphDefinition::OPKIND_TASK)
          SUBGRAPH_FATAL(me, "dependency " << i << " refers to a " << op_kind_name(k)
                                           << ", which is not implemented");
      }
      if((dep.src_op_index >= d.tasks.size()) || (dep.tgt_op_index >= d.tasks.size()))
        SUBGRAPH_FATAL(me, "dependency " << i << " refers to task "
                                         << std::max(dep.src_op_index, dep.tgt_op_index)
                                         << ", but the definition has " << d.tasks.size()
                                         << " tasks");
      if((dep.src_op_port != 0) || (dep.tgt_op_port != 0))
        SUBGRAPH_FATAL(me, "dependency " << i
                                         << " uses a nonzero port; ports are not "
                                            "implemented");
    }

    // ---- processors and NUMA domains
    // Processors in order of first appearance, then sorted by NUMA node so
    // that each domain's processors are contiguous.
    const size_t n = d.tasks.size();
    std::vector<Processor> procs_seen;
    std::unordered_map<Processor, LocalTaskProcessor *> proc_impls;
    std::unordered_map<Processor, int> proc_nodes;
    for(size_t i = 0; i < n; i++) {
      Processor p = d.tasks[i].proc;
      if(proc_impls.count(p))
        continue;
      LocalTaskProcessor *impl =
          dynamic_cast<LocalTaskProcessor *>(get_runtime()->get_processor_impl(p));
      if(!impl)
        SUBGRAPH_FATAL(me, "task " << i << " runs on " << p
                                   << ", which is not a task-running processor");
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
      cp.initial_ready = 0;
      cp.pending_inputs = 0; // no external inputs are implemented yet
      c.proc_index[cp.proc] = uint32_t(i);
    }

    // ---- operations: the tasks of each processor, contiguous per processor
    std::vector<uint32_t> task_to_op(n);
    c.ops.reserve(n);
    for(size_t pi = 0; pi < c.procs.size(); pi++) {
      CompiledSubgraph::Proc &cp = c.procs[pi];
      cp.first_op = uint32_t(c.ops.size());
      for(size_t i = 0; i < n; i++) {
        if(d.tasks[i].proc != cp.proc)
          continue;
        task_to_op[i] = uint32_t(c.ops.size());
        CompiledSubgraph::Op op;
        op.kind = SubgraphDefinition::OPKIND_TASK;
        op.index = unsigned(i);
        op.proc = int32_t(pi);
        op.counter_domain = cp.domain;
        op.counter_offset = 0;
        op.is_final = true;
        c.ops.push_back(op);
      }
      cp.num_ops = uint32_t(c.ops.size()) - cp.first_op;
    }

    // ---- edges (deduplicated), predecessor counts, cycle check
    std::vector<std::vector<uint32_t>> succ(n), pred(n);
    for(const SubgraphDefinition::Dependency &dep : d.dependencies) {
      uint32_t s = task_to_op[dep.src_op_index];
      uint32_t t = task_to_op[dep.tgt_op_index];
      if(s == t)
        SUBGRAPH_FATAL(me, "task " << dep.src_op_index << " depends on itself");
      succ[s].push_back(t);
      pred[t].push_back(s);
    }
    for(size_t i = 0; i < n; i++) {
      for(std::vector<uint32_t> *v : {&succ[i], &pred[i]}) {
        std::sort(v->begin(), v->end());
        v->erase(std::unique(v->begin(), v->end()), v->end());
      }
      c.ops[i].is_final = succ[i].empty();
    }
    {
      // Kahn's algorithm: every operation must become ready eventually.
      std::vector<uint32_t> indeg(n), ready;
      for(size_t i = 0; i < n; i++) {
        indeg[i] = uint32_t(pred[i].size());
        if(indeg[i] == 0)
          ready.push_back(uint32_t(i));
      }
      size_t done = 0;
      while(!ready.empty()) {
        uint32_t o = ready.back();
        ready.pop_back();
        done++;
        for(uint32_t s : succ[o])
          if(--indeg[s] == 0)
            ready.push_back(s);
      }
      if(done != n)
        SUBGRAPH_FATAL(me, "the dependencies contain a cycle (" << (n - done)
                                                                << " tasks can never run)");
    }
    c.successors = FlattenedSparseMatrix<uint32_t>(succ);

    // ---- counter placement: with the predecessors when they share a domain
    for(size_t i = 0; i < n; i++) {
      if(pred[i].empty())
        continue;
      int32_t dom = c.procs[c.ops[pred[i][0]].proc].domain;
      bool same = true;
      for(uint32_t p : pred[i])
        same = same && (c.procs[c.ops[p].proc].domain == dom);
      if(same)
        c.ops[i].counter_domain = dom;
    }

    // ---- block layout per domain and initial images
    std::vector<size_t> off(c.domains.size(), 0);
    for(size_t i = 0; i < n; i++) {
      CompiledSubgraph::Op &op = c.ops[i];
      op.counter_offset = uint32_t(off[op.counter_domain]);
      off[op.counter_domain] += sizeof(int64_t);
    }
    for(size_t dm = 0; dm < c.domains.size(); dm++)
      off[dm] = round_up(off[dm], SUBGRAPH_CACHE_LINE_BYTES);
    for(CompiledSubgraph::Proc &cp : c.procs) {
      cp.queue_offset = uint32_t(off[cp.domain]);
      off[cp.domain] += round_up(size_t(cp.num_ops) * sizeof(int64_t),
                                 SUBGRAPH_CACHE_LINE_BYTES);
      // tail and pending-inputs counter share one rarely written line
      cp.tail_offset = uint32_t(off[cp.domain]);
      cp.inputs_offset = cp.tail_offset + uint32_t(sizeof(uint64_t));
      off[cp.domain] += SUBGRAPH_CACHE_LINE_BYTES;
    }
    for(size_t dm = 0; dm < c.domains.size(); dm++) {
      CompiledSubgraph::Domain &dom = c.domains[dm];
      dom.bytes = std::max(off[dm], SUBGRAPH_CACHE_LINE_BYTES);
      dom.image.assign(dom.bytes, 0);
    }
    for(size_t i = 0; i < n; i++) {
      const CompiledSubgraph::Op &op = c.ops[i];
      int64_t count = int64_t(pred[i].size());
      memcpy(c.domains[op.counter_domain].image.data() + op.counter_offset, &count,
             sizeof(count));
    }
    for(CompiledSubgraph::Proc &cp : c.procs) {
      std::vector<char> &img = c.domains[cp.domain].image;
      int64_t *slots = reinterpret_cast<int64_t *>(img.data() + cp.queue_offset);
      for(uint32_t k = 0; k < cp.num_ops; k++)
        slots[k] = SUBGRAPH_EMPTY_QUEUE_ENTRY;
      uint32_t ready = 0;
      for(uint32_t o = cp.first_op; o < cp.first_op + cp.num_ops; o++)
        if(pred[o].empty())
          slots[ready++] = int64_t(o);
      cp.initial_ready = ready;
      uint64_t tail = ready;
      memcpy(img.data() + cp.tail_offset, &tail, sizeof(tail));
      int64_t inputs = cp.pending_inputs;
      memcpy(img.data() + cp.inputs_offset, &inputs, sizeof(inputs));
    }

    for(size_t dm = 0; dm < c.domains.size(); dm++)
      log_subgraph.info() << "subgraph " << me << ": domain " << dm << " numa_node="
                          << c.domains[dm].numa_node << " bytes=" << c.domains[dm].bytes;
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
                                 const ProfilingRequestSet &prs,
                                 span<const Event> preconditions,
                                 span<const Event> postconditions, Event start_event,
                                 Event finish_event, int priority_adjust)
  {
    if(!prs.empty())
      SUBGRAPH_FATAL(me, "profiling requests on instantiate are not implemented");
    if(!preconditions.empty() || !postconditions.empty())
      SUBGRAPH_FATAL(me, "external preconditions and postconditions are not implemented");

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
        new SubgraphExecutionState(this, args, arglen, finish_event, priority_adjust);
    // Release the execution state once the instantiation has finished.
    EventImpl::add_waiter(finish_event, new SubgraphInstantiationCleanup(state));
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
    if(poisoned) {
      // Nothing runs; the finish event is poisoned and cleanup proceeds as usual.
      log_subgraph.info() << "poisoned precondition: subgraph=" << state->subgraph->me;
      GenEventImpl::trigger(state->finish_event, true /*poisoned*/);
      return;
    }
    const std::vector<CompiledSubgraph::Proc> &procs = state->subgraph->compiled.procs;
    if(procs.empty()) {
      // A subgraph without operations has nothing to wait for.
      GenEventImpl::trigger(state->finish_event, false /*!poisoned*/);
      return;
    }
    for(const CompiledSubgraph::Proc &p : procs)
      p.impl->enqueue_subgraph(state);
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
                                                 const void *_args, size_t _arglen,
                                                 Event _finish_event, int _priority)
    : subgraph(_subgraph)
    , args(nullptr)
    , arglen(_arglen)
    , finish_counter(int64_t(_subgraph->compiled.procs.size()))
    , finish_event(_finish_event)
    , priority(_priority)
  {
    if((_args != nullptr) && (arglen > 0)) {
      args = malloc(arglen);
      memcpy(args, _args, arglen);
    }
    subgraph->acquire_blocks(blocks);
    const CompiledSubgraph &c = subgraph->compiled;
    for(size_t dm = 0; dm < c.domains.size(); dm++)
      memcpy(blocks[dm], c.domains[dm].image.data(), c.domains[dm].bytes);
  }

  SubgraphExecutionState::~SubgraphExecutionState()
  {
    free(args);
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
    const size_t n = active.size();
    bool found = false;
    for(size_t k = 0; k < n; k++) {
      size_t i = scan_start + k;
      if(i >= n)
        i -= n;
      const Cursor &c = active[i];
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
    entry.last_for_processor = (c.front == c.end);
    if(entry.last_for_processor) {
      // Drop the cursor now: once the operation completes, this processor's
      // finish decrement may let the state be released at any time.
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

  namespace {
    [[noreturn]] void launch_direct(SubgraphExecutionState *state, uint32_t op)
    {
      SUBGRAPH_FATAL(state->get_subgraph()->me,
                     "internal error: operation " << op
                                                  << " is not bound to a processor");
    }
  } // namespace

  void ProcSubgraphExecutor::execute(const ReadyEntry &entry)
  {
    SubgraphExecutionState *state = entry.state;
    SubgraphImpl *impl = state->subgraph;
    const CompiledSubgraph &c = impl->compiled;
    const CompiledSubgraph::Op &op = c.ops[entry.op];
    assert((op.kind == SubgraphDefinition::OPKIND_TASK) && (op.proc >= 0));
    const SubgraphDefinition::TaskDesc &task_desc = impl->defn->tasks[op.index];
    LocalTaskProcessor *proc_impl = c.procs[op.proc].impl;

    // Run the task on this thread, flagged so that operations a subgraph task
    // may not perform (waiting, querying its finish event) are rejected.
    // TODO: task context managers are not applied to subgraph tasks.
    Thread *thread = Thread::self();
    ThreadLocal::current_processor = proc;
    thread->start_subgraph_task_execution();
    proc_impl->execute_task(task_desc.task_id, task_desc.args);
    thread->stop_subgraph_task_execution();
    ThreadLocal::current_processor = Processor::NO_PROC;

    // Satisfy outgoing edges. A successor whose last predecessor this was
    // becomes ready on its own processor's queue.
    for(uint64_t i = c.successors.offsets[entry.op]; i < c.successors.offsets[entry.op + 1];
        i++) {
      const uint32_t s = c.successors.data[i];
      if(state->counter(s).fetch_sub_acqrel(1) != 1)
        continue;
      const CompiledSubgraph::Op &so = c.ops[s];
      if(so.proc < 0)
        launch_direct(state, s);
      const uint64_t slot = state->tail(so.proc).fetch_add_acqrel(1);
      state->queue(so.proc)[slot].store_release(int64_t(s));
      c.procs[so.proc].impl->notify_scheduler_of_new_work();
    }

    if(entry.last_for_processor) {
      // Copy out what is needed first: once the counter reaches zero and the
      // event triggers, the state may be released at any moment.
      Event finish_event = state->finish_event;
      if(state->finish_counter.fetch_sub_acqrel(1) == 1)
        GenEventImpl::trigger(finish_event, false /*!poisoned*/);
    }
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
