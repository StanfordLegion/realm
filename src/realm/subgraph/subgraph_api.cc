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

// Compiled subgraphs: the public Subgraph API, the SubgraphImpl lifecycle
// (creation, instantiation requests, destruction, block pool), the network
// messages and the event waiters and background items that glue them together.
// Compilation is in subgraph_compile.cc, execution in subgraph_exec.cc.

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

  Logger log_subgraph("subgraph");

  void subgraph_fatal(ID me, const std::string &what)
  {
    log_subgraph.fatal() << "subgraph " << me << ": " << what;
    abort();
  }

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


}; // namespace Realm
