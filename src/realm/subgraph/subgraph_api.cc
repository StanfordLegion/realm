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
#include <cstdio>
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
    // the default log stream is stdout, fully buffered when it is not a
    // terminal: flush, or the message dies with the process
    fflush(stdout);
    fflush(stderr);
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
      SUBGRAPH_FATAL(impl->me,
                     "profiling requests on create_subgraph are not implemented");

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
      // The message goes out once every instantiation this node sent has
      // finished (see RemoteSubgraphInstantiations); wait_on is passed to
      // the owner as given.
      UserEvent done = UserEvent::create_user_event();
      Event unfinished = get_runtime()->remote_subgraph_instantiations.take(*this);
      DeferredRemoteDestroy::send(owner, *this, wait_on, done, unfinished);
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
      get_runtime()->remote_subgraph_instantiations.note(*this, finish_event, amsg->seq,
                                                         amsg->base);
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
      get_runtime()->remote_subgraph_instantiations.note(*this, finish_event, amsg->seq,
                                                         amsg->base);
      amsg.commit();
    }
    return finish_event;
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class RemoteSubgraphInstantiations
  //

  void RemoteSubgraphInstantiations::note(Subgraph subgraph, Event finish_event,
                                          uint64_t &seq, uint64_t &base)
  {
    AutoLock<> al(mutex);
    Entry &e = entries[ID(subgraph).id];
    seq = e.next_seq++;
    // The base must be exact: a finished entry left in front would lower
    // it, and when the owner has since destroyed the subgraph and reused
    // its id, ask the new subgraph for a sequence number it never sees.
    while(!e.sent.empty()) {
      bool poisoned = false;
      if(!e.sent.front().finish.has_triggered_faultaware(poisoned))
        break;
      e.sent.pop_front();
    }
    base = e.sent.empty() ? seq : e.sent.front().seq;
    e.sent.push_back(Sent{seq, finish_event});
  }

  Event RemoteSubgraphInstantiations::take(Subgraph subgraph)
  {
    std::vector<Event> evs;
    {
      AutoLock<> al(mutex);
      std::unordered_map<ID::IDType, Entry>::iterator it =
          entries.find(ID(subgraph).id);
      if(it == entries.end())
        return Event::NO_EVENT;
      for(const Sent &sent : it->second.sent) {
        bool poisoned = false;
        if(!sent.finish.has_triggered_faultaware(poisoned))
          evs.push_back(sent.finish);
      }
      entries.erase(it);
    }
    if(evs.empty())
      return Event::NO_EVENT;
    if(evs.size() == 1)
      return evs[0];
    return Event::merge_events_ignorefaults(evs.data(), evs.size());
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class DeferredRemoteDestroy
  //

  DeferredRemoteDestroy::DeferredRemoteDestroy(NodeID _owner, Subgraph _subgraph,
                                               Event _wait_on, UserEvent _to_trigger)
    : owner(_owner)
    , subgraph(_subgraph)
    , wait_on(_wait_on)
    , to_trigger(_to_trigger)
  {}

  /*static*/ void DeferredRemoteDestroy::send(NodeID owner, Subgraph subgraph,
                                              Event wait_on, UserEvent to_trigger,
                                              Event after)
  {
    bool poisoned = false; // irrelevant: the event only orders the send
    if(!after.exists() || after.has_triggered_faultaware(poisoned)) {
      send_now(owner, subgraph, wait_on, to_trigger);
      return;
    }
    DeferredRemoteDestroy *w =
        new DeferredRemoteDestroy(owner, subgraph, wait_on, to_trigger);
    EventImpl::add_waiter(after, w);
  }

  /*static*/ void DeferredRemoteDestroy::send_now(NodeID owner, Subgraph subgraph,
                                                  Event wait_on, UserEvent to_trigger)
  {
    ActiveMessage<SubgraphDestroyMessage> amsg(owner);
    amsg->subgraph = subgraph;
    amsg->wait_on = wait_on;
    amsg->to_trigger = to_trigger;
    amsg.commit();
  }

  void DeferredRemoteDestroy::event_triggered(bool poisoned, TimeLimit work_until)
  {
    send_now(owner, subgraph, wait_on, to_trigger);
    delete this;
  }

  void DeferredRemoteDestroy::print(std::ostream &os) const
  {
    os << "deferred remote subgraph destroy: subgraph=" << subgraph << " owner=" << owner;
  }

  Event DeferredRemoteDestroy::get_finish_event(void) const { return to_trigger; }

  SubgraphImpl::SubgraphImpl()
    : me(Subgraph::NO_SUBGRAPH)
    , next_free(nullptr)
    , defn(nullptr)
  {}

  SubgraphImpl::~SubgraphImpl()
  {
    // A subgraph that was still alive at runtime teardown was never
    // destroyed by the application: free what destroy() would have freed.
    // Nothing else references this object by now (the tables are being
    // deleted, the background workers have stopped).
    if(alive.load_acquire())
      release_resources();
  }

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
    instantiate(args, arglen, prs, profiling, preconditions, postconditions, start_event,
                finish_event, priority_adjust, nullptr /*remote*/);
  }

  void SubgraphImpl::instantiate(const void *args, size_t arglen,
                                 const ProfilingRequestSet &prs,
                                 const SubgraphInstantiationProfiling &profiling,
                                 span<const Event> preconditions,
                                 span<const Event> postconditions, Event start_event,
                                 Event finish_event, int priority_adjust,
                                 const RemoteMessage *remote)
  {
    if(!alive.load_acquire())
      SUBGRAPH_FATAL(me,
                     "instantiate of a subgraph that was destroyed (or never created)");
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

    Event order_event = Event::NO_EVENT;
    {
      AutoLock<> al(lifecycle_lock);
      if(destroy_requested)
        SUBGRAPH_FATAL(me, "instantiated after destroy (the destroy's precondition had "
                           "triggered)");
      if((defn->concurrency_mode == SubgraphDefinition::ONE_SHOT) && instantiated)
        SUBGRAPH_FATAL(me, "ONE_SHOT subgraph instantiated a second time");
      if(defn->concurrency_mode == SubgraphDefinition::INSTANTIATION_ORDER) {
        if(remote && !remote_order_ready_locked(remote->order, true /*record*/)) {
          // An earlier instantiate from the same node has not arrived yet:
          // hold this one until it has (handle_remote_instantiate).
          HeldInstantiate *held = new HeldInstantiate;
          held->msg = *remote;
          held->payload.assign(static_cast<const char *>(remote->data),
                               static_cast<const char *>(remote->data) + remote->datalen);
          held->msg.data = held->payload.data();
          sender_order[remote->order.sender].held.push_back(held);
          log_subgraph.info() << "instantiate held for order: subgraph=" << me
                              << " sender=" << remote->order.sender
                              << " seq=" << remote->order.seq
                              << " base=" << remote->order.base;
          return;
        }
        // Instantiations run in order: this one starts once the previous one
        // has finished, and the next one will wait for this one. (CONCURRENT
        // and ONE_SHOT instantiations are independent: the engine runs any
        // number of them at once.)
        order_event = previous_instantiation_completion;
        previous_instantiation_completion = finish_event;
      }
      instantiated = true;
      outstanding_instantiations++;
    }

    // priority_adjust is the priority of this instantiation as a whole
    SubgraphExecutionState *state = acquire_state();
    state->reset(finish_event, priority_adjust, preconditions, postconditions);
    state->interpolate(args, arglen);
    state->setup_profiling(profiling);
    get_runtime()->subgraph_resource_reaper.instantiation_created();
    // Start once the start event (and the previous instantiation, in order)
    // has triggered; the external preconditions are delivered then. Nothing
    // may touch the state after this: it may already be finished and recycled.
    state->launcher.arm(start_event, order_event);
  }

  void SubgraphImpl::release_resources(void)
  {
    alive.store_release(false);
    delete defn;
    defn = nullptr;
    {
      // No instantiation is outstanding: every state is in the pool. Their
      // destructors return the blocks (taking the pool lock), then the
      // slabs go, before compiled.clear(): the slabs' domains live there.
      std::vector<SubgraphExecutionState *> states;
      {
        AutoLock<> al(state_pool_lock);
        states.swap(state_pool);
      }
      for(SubgraphExecutionState *st : states)
        delete st;
      AutoLock<> al(block_pool_lock);
      free_slabs();
    }
    compiled.clear();
  }

  void SubgraphImpl::destroy(void)
  {
    release_resources();

    previous_instantiation_completion = Event::NO_EVENT;
    outstanding_instantiations = 0;
    instantiated = false;
    destroy_pending = false;
    destroy_requested = false;
    destroy_done = UserEvent::NO_USER_EVENT;
    for(std::unordered_map<NodeID, SenderOrder>::iterator it = sender_order.begin();
        it != sender_order.end(); ++it) {
      // a held instantiate here is one issued after the destroy took effect
      // (its predecessor arrives later and aborts the same way)
      for(HeldInstantiate *h : it->second.held)
        delete h;
    }
    sender_order.clear();

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
    // The destroy takes effect when its precondition triggers. Until then,
    // instantiations are still accepted and counted: an application orders
    // them before the destroy by ordering their finish events before
    // wait_on (and across nodes that is the only way it can). A poisoned
    // precondition does not keep a subgraph from being destroyed.
    bool poisoned = false;
    bool triggered = wait_on.has_triggered_faultaware(poisoned);
    UserEvent done;
    {
      AutoLock<> al(lifecycle_lock);
      if(!alive.load_acquire())
        SUBGRAPH_FATAL(me, "destroy of a subgraph that was already destroyed (or never "
                           "created)");
      if(destroy_pending)
        SUBGRAPH_FATAL(me, "destroyed twice (a destroy was already pending)");
      destroy_pending = true;
      if(!triggered) {
        destroy_done = UserEvent::create_user_event();
        done = destroy_done;
      } else {
        destroy_requested = true;
        if(outstanding_instantiations > 0) {
          // The last instantiation to release its state completes the destroy.
          destroy_done = UserEvent::create_user_event();
          return destroy_done;
        }
      }
    }
    if(!triggered) {
      // Registered without the lock: the event may trigger meanwhile and
      // call destroy_precondition_triggered right here.
      get_runtime()->subgraph_resource_reaper.destroy_deferred();
      deferred_destroy.defer(this, wait_on);
      return done;
    }
    if(poisoned)
      log_subgraph.warning() << "destroy precondition was poisoned: subgraph=" << me;
    destroy();
    return Event::NO_EVENT;
  }

  void SubgraphImpl::destroy_precondition_triggered(bool poisoned)
  {
    if(poisoned)
      log_subgraph.warning() << "destroy precondition was poisoned: subgraph=" << me;
    get_runtime()->subgraph_resource_reaper.destroy_precondition_triggered();
    UserEvent done;
    {
      AutoLock<> al(lifecycle_lock);
      assert(destroy_pending && !destroy_requested);
      destroy_requested = true;
      if(outstanding_instantiations > 0)
        return; // the last release completes the destroy
      done = destroy_done;
    }
    // destroy() returns this object to the free list, so nothing of it is
    // read afterwards
    destroy();
    done.trigger();
  }

  void SubgraphImpl::instantiation_released(void)
  {
    UserEvent done;
    {
      AutoLock<> al(lifecycle_lock);
      assert(outstanding_instantiations > 0);
      outstanding_instantiations--;
      if((outstanding_instantiations > 0) || !destroy_requested)
        return;
      done = destroy_done;
    }
    destroy();
    done.trigger();
  }

  bool SubgraphImpl::remote_order_ready_locked(const RemoteOrder &order, bool record)
  {
    SenderOrder &so = sender_order[order.sender];
    if(order.base > so.next) {
      // everything below the base had finished, so it was handled
      so.next = order.base;
      so.handled.erase(so.handled.begin(), so.handled.lower_bound(so.next));
      while(!so.handled.empty() && (*so.handled.begin() == so.next)) {
        so.handled.erase(so.handled.begin());
        so.next++;
      }
    }
    // in order (the common case): nothing to look up
    for(uint64_t q = std::max(order.base, so.next); q < order.seq; q++)
      if(so.handled.count(q) == 0)
        return false;
    if(record) {
      if(order.seq == so.next) {
        so.next++;
        while(!so.handled.empty() && (*so.handled.begin() == so.next)) {
          so.handled.erase(so.handled.begin());
          so.next++;
        }
      } else if(order.seq > so.next) {
        so.handled.insert(order.seq);
      }
    }
    return true;
  }

  SubgraphImpl::HeldInstantiate *SubgraphImpl::pop_ready_held(NodeID sender)
  {
    AutoLock<> al(lifecycle_lock);
    std::unordered_map<NodeID, SenderOrder>::iterator it = sender_order.find(sender);
    if(it == sender_order.end())
      return nullptr;
    std::vector<HeldInstantiate *> &held = it->second.held;
    for(size_t i = 0; i < held.size(); i++) {
      if(remote_order_ready_locked(held[i]->msg.order, false /*record*/)) {
        HeldInstantiate *h = held[i];
        held[i] = held.back();
        held.pop_back();
        return h;
      }
    }
    return nullptr;
  }

  void SubgraphImpl::handle_remote_instantiate(const RemoteMessage &first)
  {
    const RemoteMessage *msg = &first;
    HeldInstantiate *held = nullptr;
    while(true) {
      span<const Event> preconditions, postconditions;
      ProfilingRequestSet prs;
      SubgraphInstantiationProfiling profiling;
      Serialization::FixedBufferDeserializer fbd(msg->data, msg->datalen);
      // skip over the instantiation args - they are accessed directly
      fbd.extract_bytes(0, msg->arglen);
      bool ok = ((fbd >> preconditions) && (fbd >> postconditions));
      if(ok && (fbd.bytes_left() > 0))
        ok = (fbd >> prs);
      if(ok && (fbd.bytes_left() > 0))
        ok = (fbd >> profiling);
      assert(ok);

      instantiate(msg->data, msg->arglen, prs, profiling, preconditions, postconditions,
                  msg->start_event, msg->finish_event, msg->priority_adjust, msg);
      delete held;

      // this may have been the predecessor an earlier arrival waits for
      held = pop_ready_held(first.order.sender);
      if(!held)
        return;
      msg = &held->msg;
    }
  }

  // Blocks are placed on the processors' NUMA nodes, which is page-granular
  // and costs system calls, so they are allocated once and recycled.
  void SubgraphImpl::acquire_blocks(std::vector<char *> &blocks)
  {
    AutoLock<> al(block_pool_lock);
    if(block_pool.empty())
      grow_pool_locked();
    blocks.swap(block_pool.back());
    block_pool.pop_back();
  }

  SubgraphExecutionState *SubgraphImpl::acquire_state(void)
  {
    {
      AutoLock<> al(state_pool_lock);
      if(!state_pool.empty()) {
        SubgraphExecutionState *st = state_pool.back();
        state_pool.pop_back();
        return st;
      }
    }
    return new SubgraphExecutionState(this);
  }

  void SubgraphImpl::recycle_state(SubgraphExecutionState *state)
  {
    state->retire();
    AutoLock<> al(state_pool_lock);
    state_pool.push_back(state);
  }

  void SubgraphImpl::release_blocks(std::vector<char *> &blocks)
  {
    AutoLock<> al(block_pool_lock);
    block_pool.emplace_back();
    block_pool.back().swap(blocks);
  }

  // Slab size: enough blocks to amortize the allocation, at least 8
  // instantiations' worth and at least 64 KiB for the largest domain's
  // slab (the same count applies to every domain, so the largest block
  // decides, or a tiny domain would make a huge one allocate megabytes).
  void SubgraphImpl::grow_pool_locked(void)
  {
    static const bool numa_available = numasysif_numa_available();
    const size_t ndom = compiled.domains.size();
    size_t largest = 1;
    for(size_t dm = 0; dm < ndom; dm++)
      largest = std::max(largest, compiled.domains[dm].bytes);
    const size_t count = std::max<size_t>(8, (size_t(64) << 10) / largest);
    std::vector<char *> bases(ndom, nullptr);
    for(size_t dm = 0; dm < ndom; dm++) {
      const CompiledSubgraph::Domain &dom = compiled.domains[dm];
      Slab slab;
      slab.domain = int32_t(dm);
      slab.bytes = count * dom.bytes;
      slab.numa = false;
      void *p = nullptr;
      if(numa_available && (dom.numa_node >= 0)) {
        p = numasysif_alloc_mem(dom.numa_node, slab.bytes, false /*!pin*/);
        slab.numa = (p != nullptr);
      }
      if(!p) {
#ifdef _MSC_VER
        p = _aligned_malloc(slab.bytes, SUBGRAPH_CACHE_LINE_BYTES);
#else
        if(posix_memalign(&p, SUBGRAPH_CACHE_LINE_BYTES, slab.bytes) != 0)
          p = nullptr;
#endif
      }
      if(!p)
        SUBGRAPH_FATAL(me, "failed to allocate " << slab.bytes
                                                 << " bytes of instantiation state");
      slab.base = static_cast<char *>(p);
      slabs.push_back(slab);
      bases[dm] = slab.base;
    }
    for(size_t i = 0; i < count; i++) {
      block_pool.emplace_back(ndom, nullptr);
      for(size_t dm = 0; dm < ndom; dm++)
        block_pool.back()[dm] = bases[dm] + i * compiled.domains[dm].bytes;
    }
  }

  void SubgraphImpl::free_slabs(void)
  {
    for(const Slab &slab : slabs) {
      bool freed = false;
      if(slab.numa)
        freed = numasysif_free_mem(compiled.domains[slab.domain].numa_node, slab.base,
                                   slab.bytes);
      if(!freed) {
#ifdef _MSC_VER
        _aligned_free(slab.base);
#else
        free(slab.base);
#endif
      }
    }
    slabs.clear();
    block_pool.clear();
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
    // may return the SubgraphImpl, and with it this waiter, to the free list
    subgraph->destroy_precondition_triggered(poisoned);
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
    SubgraphImpl::RemoteMessage rm;
    rm.order.sender = sender;
    rm.order.seq = msg.seq;
    rm.order.base = msg.base;
    rm.start_event = msg.wait_on;
    rm.finish_event = msg.finish_event;
    rm.priority_adjust = msg.priority_adjust;
    rm.arglen = msg.arglen;
    rm.data = data;
    rm.datalen = datalen;
    subgraph->handle_remote_instantiate(rm);
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
  {
    for(Input &in : inputs)
      in.owner = this;
    inputs[1].carries_poison = false; // the order event
  }

  void SubgraphWorkLauncher::arm(Event start_event, Event order_event)
  {
    // The count starts at one so that inputs already triggered (add_waiter
    // calls back right away) cannot launch before both have been examined.
    pending.store(1);
    poisoned.store(false);
    // Chained replays pass the previous finish event as the start event, which
    // is also the order event: one waiter is enough then.
    const Event evs[2] = {start_event,
                          (order_event == start_event) ? Event::NO_EVENT : order_event};
    for(int k = 0; k < 2; k++) {
      if(!evs[k].exists())
        continue;
      bool p = false;
      if(evs[k].has_triggered_faultaware(p)) {
        if(p && inputs[k].carries_poison)
          poisoned.store(true);
        continue;
      }
      pending.fetch_add(1);
      EventImpl::add_waiter(evs[k], &inputs[k]);
    }
    input_triggered(false);
  }

  void SubgraphWorkLauncher::input_triggered(bool p)
  {
    if(p)
      poisoned.store(true);
    if(pending.fetch_sub_acqrel(1) == 1)
      launch(state, poisoned.load());
  }

  /*static*/ void SubgraphWorkLauncher::launch(SubgraphExecutionState *state,
                                               bool poisoned)
  {
    const CompiledSubgraph &c = state->subgraph->compiled;
    // No input waiter is registered yet and no processor knows the state,
    // so the two early exits below may finish directly.
    if(poisoned) {
      // Nothing runs; the finish event and every postcondition are
      // poisoned and cleanup proceeds as usual.
      log_subgraph.info() << "poisoned precondition: subgraph=" << state->subgraph->me;
      for(Event pc : state->postconditions)
        GenEventImpl::trigger(pc, true /*poisoned*/);
      state->finish(true /*poisoned*/);
      return;
    }
    if(c.ops.empty()) {
      // Nothing to wait for.
      for(Event pc : state->postconditions)
        GenEventImpl::trigger(pc, false /*!poisoned*/);
      state->finish(false /*!poisoned*/);
      return;
    }
    for(const CompiledSubgraph::Proc &p : c.procs)
      p.impl->enqueue_subgraph(state);
    // The launcher's share of the finish counter (taken by reset) keeps the
    // state alive while inputs are delivered and the roots released, even
    // though processors may start, and finish, operations meanwhile.
    state->deliver_inputs();
    state->start();
    state->contributor_finished(); // may finish and recycle the state
  }

  void SubgraphWorkLauncher::Input::event_triggered(bool poisoned, TimeLimit work_until)
  {
    owner->input_triggered(poisoned && carries_poison);
  }

  void SubgraphWorkLauncher::Input::print(std::ostream &os) const
  {
    os << "SubgraphWorkLauncher: subgraph=" << owner->state->get_subgraph()->me;
  }

  Event SubgraphWorkLauncher::Input::get_finish_event(void) const
  {
    return Event::NO_EVENT;
  }

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
    // Nothing may touch the state (and so *this) after recycling: the next
    // instantiation can take it from the pool at once, and completing a
    // pending destroy deletes the pooled states. The reaper's count goes
    // last: the runtime's shutdown waits on it, and the destroy that the
    // release may complete still looks events up.
    SubgraphImpl *subgraph = state->get_subgraph();
    subgraph->recycle_state(state);
    subgraph->instantiation_released();
    get_runtime()->subgraph_resource_reaper.instantiation_released();
  }

  ////////////////////////////////////////////////////////////////////////
  //
  // class SubgraphInputWaiter
  //

  SubgraphInputWaiter::SubgraphInputWaiter(SubgraphExecutionState *_state,
                                           uint32_t _input)
    : state(_state)
    , input(_input)
  {}

  void SubgraphInputWaiter::event_triggered(bool poisoned, TimeLimit work_until)
  {
    // This waiter's share keeps the state alive until the delivery has
    // returned; releasing it may finish and recycle the state (and *this).
    SubgraphExecutionState *s = state;
    s->input_triggered(input, poisoned);
    s->contributor_finished();
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

    item->cleanup(); // decrements outstanding
    return left > 0;
  }

  void SubgraphResourceReaper::wait_for_idle(void)
  {
    // Called from the shutdown path, after the check that every
    // instantiation has finished, with the background workers still
    // running: only the release of finished states is left, which depends
    // on no application event. Not bounded: a bound would be a guess that
    // eventually misfires on a slow but correct teardown. A periodic
    // warning makes a stuck wait (a runtime bug) diagnosable.
    long long warned_at = Clock::current_time_in_nanoseconds();
    while(!idle()) {
      Thread::yield();
      long long now = Clock::current_time_in_nanoseconds();
      if(now - warned_at > 5000000000LL) {
        log_subgraph.warning() << "shutdown waiting for " << outstanding.load()
                               << " finished subgraph instantiation(s) to be released";
        warned_at = now;
      }
    }
  }

}; // namespace Realm
