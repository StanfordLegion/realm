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

// Event/UserEvent implementations for Realm

#ifndef REALM_EVENT_IMPL_H
#define REALM_EVENT_IMPL_H

#include "realm/event.h"
#include "realm/id.h"
#include "realm/nodeset.h"
#include "realm/faults.h"

#include "realm/network.h"
#include <realm/activemsg.h>

#include "realm/lists.h"
#include "realm/threads.h"
#include "realm/logging.h"
#include "realm/redop.h"
#include "realm/bgwork.h"
#include "realm/dynamic_table.h"

#include <vector>
#include <map>
#include <memory>

namespace Realm {

  class GenEventImpl;

  extern Logger log_poison; // defined in event_impl.cc
  class ProcessorImpl;      // defined in proc_impl.h

  class EventWaiter {
  public:
    virtual ~EventWaiter(void) {}
    virtual void event_triggered(bool poisoned, TimeLimit work_until) = 0;
    virtual void print(std::ostream &os) const = 0;
    virtual Event get_finish_event(void) const = 0;

    IntrusiveListLink<EventWaiter> ew_list_link;
    REALM_PMTA_DEFN(EventWaiter, IntrusiveListLink<EventWaiter>, ew_list_link);
    typedef IntrusiveList<EventWaiter, REALM_PMTA_USE(EventWaiter, ew_list_link),
                          DummyLock>
        EventWaiterList;
  };

  // triggering events can often result in recursive expansion of work -
  //  this widget flattens the call stack and defers excessive triggers
  //  to avoid stalling the initial triggerer longer than they want
  class EventTriggerNotifier : public BackgroundWorkItem {
  public:
    EventTriggerNotifier();

    void trigger_event_waiters(EventWaiter::EventWaiterList &to_trigger, bool poisoned,
                               TimeLimit trigger_until);

    virtual bool do_work(TimeLimit work_until);

  protected:
    Mutex mutex;
    EventWaiter::EventWaiterList delayed_normal;
    EventWaiter::EventWaiterList delayed_poisoned;

    static thread_local EventWaiter::EventWaiterList *nested_normal;
    static thread_local EventWaiter::EventWaiterList *nested_poisoned;
  };

  // parent class of GenEventImpl and BarrierImpl
  class EventImpl {
  public:
    typedef unsigned gen_t;

    EventImpl(void);
    virtual ~EventImpl(void);

    // test whether an event has triggered without waiting
    virtual bool has_triggered(gen_t needed_gen, bool &poisoned) = 0;

    virtual void subscribe(gen_t subscribe_gen) = 0;

    // causes calling thread to block until event has occurred
    // void wait(Event::gen_t needed_gen);

    virtual void external_wait(gen_t needed_gen, bool &poisoned) = 0;
    virtual bool external_timedwait(gen_t needed_gen, bool &poisoned,
                                    long long max_ns) = 0;

    // helper to create the Event for an arbitrary generation
    Event make_event(gen_t gen) const;

    virtual bool add_waiter(gen_t needed_gen,
                            EventWaiter *waiter /*, bool pre_subscribed = false*/) = 0;

    static bool add_waiter(Event needed, EventWaiter *waiter);

    // use this sparingly - it has to hunt through waiter lists while
    //  holding locks
    virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter) = 0;

    static bool detect_event_chain(Event search_from, Event target, int max_depth,
                                   bool print_chain);

  public:
    ID me;
    ProcessorImpl *owning_processor;
    NodeID owner;
  };

  class GenEventImpl;

  class EventMerger {
  public:
    EventMerger(GenEventImpl *_event_impl);
    ~EventMerger(void);

    bool is_active(void) const;

    void prepare_merger(Event _finish_event, bool _ignore_faults,
                        std::optional<size_t> expected_events = std::optional<size_t>());

    void add_precondition(Event wait_for);

    void arm_merger(void);

    class MergeEventPrecondition : public EventWaiter {
    public:
      MergeEventPrecondition(void) = default;
      MergeEventPrecondition(const MergeEventPrecondition &) = delete;
      MergeEventPrecondition(MergeEventPrecondition &&) = delete;
      virtual ~MergeEventPrecondition(void) = default;

    public:
      EventMerger *merger;

      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;
    };

    // as an alternative to add_precondition, get_next_precondition can
    //  be used to get a precondition that can manually be added to a waiter
    //  list
    MergeEventPrecondition *get_next_precondition(void);

  protected:
    void precondition_triggered(bool poisoned, TimeLimit work_until,
                                MergeEventPrecondition *precondition = nullptr);

    friend class MergeEventPrecondition;

    GenEventImpl *event_impl;
    EventImpl::gen_t finish_gen;
    unsigned precondition_offset;
    bool ignore_faults;
    bool recycle_preconditions;
    atomic<int> count_needed;
    atomic<int> faults_observed;

    static constexpr size_t MAX_INLINE_PRECONDITIONS = 6;
    MergeEventPrecondition inline_preconditions[MAX_INLINE_PRECONDITIONS];
    // std::deque does not invalidate references on resize
    std::deque<MergeEventPrecondition> overflow_preconditions;
    EventWaiter::EventWaiterList free_preconditions;
  };

  class EventCommunicator {
  public:
    virtual ~EventCommunicator() = default;

    virtual void trigger(Event event, NodeID owner, bool poisoned);

    virtual void update(Event event, NodeSet to_update,
                        span<EventImpl::gen_t> poisoned_generations);

    // the trailing taint fields piggyback deferred-allocation taint views on
    //  the update message (subscription acks and narrowing pushes) -
    //  taint_gen == 0 means "no taint information carried"
    virtual void update(Event event, NodeID to_update,
                        span<EventImpl::gen_t> poisoned_generations,
                        EventImpl::gen_t taint_gen = 0, uint8_t taint_kind = 0,
                        realm_id_t taint_inst = 0);

    virtual void subscribe(Event event, NodeID owner,
                           EventImpl::gen_t previous_subscribe_gen);
  };

  class GenEventImpl : public EventImpl {
  public:
    static const ID::ID_Types ID_TYPE = ID::ID_EVENT;

    GenEventImpl(void);
    GenEventImpl(EventTriggerNotifier *_event_triggerer, EventCommunicator *_event_comm);
    ~GenEventImpl(void);

    void init(ID _me, unsigned _init_owner);

    // --- taint tracking for deferred-allocation funding safety (BUG-8) ---
    // an event's "taint" is the set of PENDING deferred instance creations
    //  its trigger may (transitively) depend on - see
    //  tla/allocation/DIST-DESIGN.md section 5c and
    //  tla/allocation/bugs/BUG-8.md.  The set is kept in inline-one-id form:
    //  NONE, exactly one instance, or TOP (may depend on anything - unbound
    //  user events, unfired barrier generations, and the >=2-id overflow).
    //  Taint only ever NARROWS (TOP -> one-id -> NONE), so a stale view is
    //  always conservative; it dies at trigger (a fired event cannot depend
    //  on any still-pending creation).
    //
    // NOTE TO MINT-SITE AUTHORS (safe-by-default): a fresh, untriggered
    //  generation with NO recorded taint reads as TOP on its owner - suspect
    //  by construction.  A mint site that stores nothing is therefore always
    //  SOUND; the cost is only lost funding opportunities (a deletion gated
    //  on such an event cannot fund remote-origin creations until it fires).
    //  set_taint / set_taint_from_inputs are NARROWING optimizations, and
    //  the single obligation on any stored taint is that it DOMINATES the
    //  event's true transitive dependence on pending instance creations.
    //  The load-bearing narrowing sites that carry the performance:
    //   - task spawn (finish event = union of launch preconditions, frozen)
    //   - event merges (union of inputs, PENDING when unresolved)
    //   - copy/fill completion (union of the launch precondition)
    //   - instance ready events (ROOT_UNION: INST(self) u precondition)
    //   - user-event bind (owner-side narrowing push to subscribers)
    //
    // PERFORMANCE INVARIANT (DIST-DESIGN.md section 5e): trigger paths do
    //  NO taint work.  Death-at-trigger is implemented READ-side (the
    //  has_triggered check in read_taint plus the taint_gen tag) - nothing
    //  stores, clears, or sends taint when an event fires, and pure trigger
    //  propagation carries taint_gen == 0.  Taint cost lives only at event
    //  CREATION (O(#inputs), TAINT_MAX_INPUTS-bounded storage) and inside
    //  ALLOCATOR decisions (lazy PENDING resolution, funding filter).  Do
    //  not add taint reads/writes/messages to trigger or poison paths.
    enum TaintKind : uint8_t
    {
      TAINT_UNKNOWN = 0, // (non-owner view) no taint info delivered yet -
                         //  MUST be treated as suspect, never fundable
      TAINT_NONE = 1,    // provably independent of every pending creation
      TAINT_INST = 2,    // may depend on exactly one pending creation
      TAINT_TOP = 3,     // may depend on anything
      TAINT_PENDING = 4, // (owner side) derived event whose inputs' taints
                         //  were not all resolved at creation - lazily
                         //  re-resolved via taint_inputs (late reads only
                         //  ever observe NARROWER input taints, so they
                         //  remain sound over-approximations)
    };
    static const int TAINT_MAX_INPUTS = 2;

    // set the live generation's taint (event-creation sites)
    void set_taint(uint8_t kind, realm_id_t inst = 0);
    // derived-event rule: the new event's taint must dominate the union of
    //  ALL input taints (multi-input obligation, DIST-DESIGN.md 5c) - inputs
    //  whose taint cannot be resolved yet leave the event TAINT_PENDING
    void set_taint_from_inputs(span<const Event> inputs);
    // read the taint of 'e' as visible on this node; triggered events are
    //  TAINT_NONE by definition; remote events with no delivered view return
    //  TAINT_UNKNOWN (callers must already hold/register a subscription so
    //  the view eventually arrives)
    static uint8_t read_taint(Event e, realm_id_t &inst_out, int depth = 0);
    // non-owner side: store a taint view delivered by the owner
    //  (generation-checked; stale-generation updates are dropped)
    void apply_taint_update(gen_t gen, uint8_t kind, realm_id_t inst);
    // owner side: recompute/narrow a user event's taint at deferred-trigger
    //  (bind) time and push the narrowing to current subscribers
    void narrow_user_event_taint(Event wait_on);

    static GenEventImpl *
    create_genevent(void); // TODO: remove this once we get rid of get_runtime()

    static GenEventImpl *create_genevent(RuntimeImpl *runtime_impl);

    static ID make_id(const GenEventImpl &dummy, int owner, ID::IDType index)
    {
      return ID::make_event(owner, index, 0);
    }

    // get the Event (id+generation) for the current (i.e. untriggered) generation
    Event current_event(void) const;

    // test whether an event has triggered without waiting
    virtual bool has_triggered(gen_t needed_gen, bool &poisoned);

    virtual void subscribe(gen_t subscribe_gen);
    void handle_remote_subscription(NodeID sender, gen_t subscribe_gen,
                                    gen_t previous_subscribe_gen);

    virtual void external_wait(gen_t needed_gen, bool &poisoned);
    virtual bool external_timedwait(gen_t needed_gen, bool &poisoned, long long max_ns);

    virtual bool add_waiter(gen_t needed_gen, EventWaiter *waiter);

    // use this sparingly - it has to hunt through waiter lists while
    //  holding locks
    virtual bool remove_waiter(gen_t needed_gen, EventWaiter *waiter);

    // creates an event that won't trigger until all input events have
    static Event merge_events(span<const Event> wait_for, bool ignore_faults);
    static Event merge_events(Event ev1, Event ev2, Event ev3 = Event::NO_EVENT,
                              Event ev4 = Event::NO_EVENT, Event ev5 = Event::NO_EVENT,
                              Event ev6 = Event::NO_EVENT);
    static Event ignorefaults(Event wait_for);

    // record that the event has triggered and notify anybody who cares
    bool trigger(gen_t gen_triggered, int trigger_node, bool poisoned,
                 TimeLimit work_until);

    // helper for triggering with an Event (which must be backed by a GenEventImpl)
    static void trigger(Event e, bool poisoned);
    static void trigger(Event e, bool poisoned, TimeLimit work_until);

    // process an update message from the owner
    void process_update(gen_t current_gen, const gen_t *new_poisoned_generations,
                        int new_poisoned_count, TimeLimit work_until);

    // Set the operation that will trigger this event's generation.
    void set_trigger_op(gen_t gen, Operation *op);
    // Get the operation that will trigger this event's generation.
    // The returned operation's reference is incremented and must be removed by the
    // caller.
    Operation *get_trigger_op(gen_t gen);

    struct GenEventImplAllocator {
      EventTriggerNotifier *triggerer{nullptr};

      GenEventImplAllocator(void) = default;

      GenEventImplAllocator(EventTriggerNotifier *t)
        : triggerer(t)
      {}

      void construct(GenEventImpl *storage, ID id, unsigned owner) const
      {
        storage->~GenEventImpl();
        new(storage) GenEventImpl(triggerer, new EventCommunicator());
        storage->init(id, owner);
      }
    };

  public: // protected:
    // these state variables are monotonic, so can be checked without a lock for
    //  early-out conditions
    atomic<gen_t> generation = atomic<gen_t>(0);
    atomic<gen_t> gen_subscribed = atomic<gen_t>(0);
    atomic<int> num_poisoned_generations = atomic<int>(0);
    bool has_local_triggers = false;

    bool is_generation_poisoned(gen_t gen) const; // helper function - linear search

    // this is only manipulated when the event is "idle"
    GenEventImpl *next_free{nullptr};

    // used for merge_events and delayed UserEvent triggers
    EventMerger merger;

    EventTriggerNotifier *event_triggerer{nullptr};
    std::unique_ptr<EventCommunicator> event_comm{nullptr};

    // everything below here protected by this mutex
    Mutex mutex;

    // The operation that will trigger this generation
    Operation *current_trigger_op = nullptr;

    // local waiters are tracked by generation - an easily-accessed list is used
    //  for the "current" generation, whereas a map-by-generation-id is used for
    //  "future" generations (i.e. ones ahead of what we've heard about if we're
    //  not the owner)
    EventWaiter::EventWaiterList current_local_waiters;
    std::map<gen_t, EventWaiter::EventWaiterList> future_local_waiters;

    // external waiters on this node are notifies via a condition variable
    bool has_external_waiters = false;
    // use kernel mutex for timedwait functionality
    KernelMutex external_waiter_mutex;
    KernelMutex::CondVar external_waiter_condvar;

    // remote waiters are kept in a bitmask for the current generation - this is
    //  only maintained on the owner, who never has to worry about more than one
    //  generation
    NodeSet remote_waiters;

    // we'll set an upper bound on how many times any given event can be poisoned - this
    // keeps update messages from growing without bound
    static const int POISONED_GENERATION_LIMIT = 16;

    // note - we don't bother sorting the list below - the overhead of a binary search
    //  dominates for short lists
    // we also can't use an STL vector because reallocation prevents us from reading the
    //  list without the lock - instead we'll allocate the max size if/when we need
    //  any space
    gen_t *poisoned_generations = 0;

    // local triggerings - if we're not the owner, but we've triggered/poisoned events,
    //  we need to give consistent answers for those generations, so remember what we've
    //  done until our view of the distributed event catches up
    // value stored in map is whether generation was poisoned
    std::map<gen_t, bool> local_triggers;

    // these resolve a race condition between the early trigger of a
    //  poisoned merge and the last precondition
    bool free_list_insertion_delayed = false;
    friend class EventMerger;

    // --- taint state (guarded by 'mutex') ---
    // valid only when taint_gen matches the generation being asked about -
    //  stale-generation state is never trusted (hygiene rule for impl
    //  recycling: a fresh generation starts TAINT_NONE on the owner and
    //  TAINT_UNKNOWN on mirrors)
    uint8_t taint_kind = TAINT_NONE;
    gen_t taint_gen = 0;
    realm_id_t taint_inst = 0;
    Event taint_inputs[TAINT_MAX_INPUTS];
  };
}; // namespace Realm

#include "realm/event_impl.inl"

#endif // ifndef REALM_EVENT_IMPL_H
