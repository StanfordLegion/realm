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

#ifndef REALM_SUBGRAPH_IMPL_H
#define REALM_SUBGRAPH_IMPL_H

#include "realm/atomics.h"
#include "realm/subgraph.h"
#include "realm/subgraph/subgraph_order.h"
#include "realm/id.h"
#include "realm/event_impl.h"
#include "realm/operation.h"
#include "realm/bgwork.h"
#include "realm/mutex.h"
#include "realm/logging.h"

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Realm {

  extern Logger log_subgraph;

  // Unsupported features and misuse are hard errors: a subgraph that ran
  // with different semantics than requested would be worse than one that
  // refuses to run.
  [[noreturn]] void subgraph_fatal(ID me, const std::string &what);
#define SUBGRAPH_FATAL(me, expr)                                                         \
  do {                                                                                   \
    std::ostringstream _ss;                                                              \
    _ss << expr;                                                                         \
    ::Realm::subgraph_fatal((me), _ss.str());                                            \
  } while(0)

  const char *op_kind_name(SubgraphDefinition::OpKind kind);

  class LocalTaskProcessor;
  class ProcSubgraphExecutor;
  class SubgraphExecutionState;
  class SubgraphWorkLauncher;
  class SubgraphInstantiationCleanup;
  class TransferDesc;

  // Sentinel value for empty slots in the per-processor ready queues.
  constexpr int64_t SUBGRAPH_EMPTY_QUEUE_ENTRY = -1;
  constexpr size_t SUBGRAPH_CACHE_LINE_BYTES = 64;

  // CSR representation of a list of lists, for cache-friendly iteration.
  template <typename T>
  struct FlattenedSparseMatrix {
    FlattenedSparseMatrix() {}
    FlattenedSparseMatrix(const std::vector<std::vector<T>> &input)
    {
      uint64_t count = 0;
      for(size_t i = 0; i < input.size(); i++) {
        offsets.push_back(count);
        for(const T &it : input[i]) {
          data.push_back(it);
          count++;
        }
      }
      offsets.push_back(count);
    }
    void clear()
    {
      offsets.clear();
      data.clear();
    }
    std::vector<uint64_t> offsets;
    std::vector<T> data;
  };

  // CompiledSubgraph is the executable form of a SubgraphDefinition.
  //
  // Operations are tasks, which run on a processor's ready queue, and
  // directly launched operations (barrier arrivals), which whoever satisfies
  // their last precondition performs inline. Operations are numbered so that
  // the tasks of one processor are contiguous and processors of one NUMA
  // domain are contiguous; directly launched operations come last.
  //
  // Graph inputs are the start event, an implicit predecessor of every
  // operation without in-graph predecessors, and the external preconditions
  // of the definition. Each operation records the set of external inputs it
  // transitively depends on, so a poisoned input skips exactly the operations
  // that depend on it and poisons exactly the postconditions and finish event
  // downstream of it.
  //
  // Each instantiation owns one memory block per NUMA domain holding the
  // mutable state touched at run time:
  //  - one precondition counter per operation, placed in the domain of the
  //    processors that decrement it (the operation's predecessors) when they
  //    all share one, otherwise in the consuming processor's domain;
  //  - one ready queue per processor (one slot per operation it runs), the
  //    queue's tail index and the processor's pending-inputs counter, placed
  //    in that processor's domain, each padded to a cache line;
  //  - one counter per external postcondition;
  //  - the arguments of interpolated operations, patched at instantiation.
  // Blocks are allocated once per subgraph on the right NUMA node and
  // recycled across instantiations; `image` is their initial contents.
  struct CompiledSubgraph {
    struct Op {
      SubgraphDefinition::OpKind kind;
      unsigned index;          // into the definition's list for `kind`
      int32_t proc;            // index into procs; -1 for directly launched operations
      int32_t counter_domain;  // index into domains
      uint32_t counter_offset; // byte offset of the precondition counter in that block
      // Interpolated arguments, if any: a patched copy of the operation's
      // argument bytes lives in domain args_domain at args_offset.
      int32_t args_domain;
      uint32_t args_offset;
      uint32_t args_size;
      // Tasks on processors whose work outlives the task function (GPUs)
      // are asynchronous: they complete through the processor's callback.
      // A deferred-effects task additionally promised that all its work is
      // on the stream it was given, so tasks after it on the same processor
      // may start as soon as its function returns, ordering their own work
      // after it with the token it left behind.
      bool async;
      bool deferred;
      int32_t async_index; // per-instantiation token/completion slot, -1 if not async
      // Tasks: the processor's table entry, resolved at compile (entries are
      // never removed), so a graph task runs without the table's lock.
      const void *task_entry;
    };
    struct Input {                   // one external precondition
      std::vector<uint32_t> targets; // operations it directly gates
      std::vector<uint32_t> procs;   // processors with operations depending on it
    };
    struct Postcond { // one external postcondition
      uint32_t num_sources;
      int32_t counter_domain;
      uint32_t counter_offset;
    };
    struct Interp { // one interpolation into an operation's argument copy
      uint32_t op;
      size_t src_offset, bytes, dst_offset;
      ReductionOpID redop_id;
    };
    struct Proc {
      Processor proc;
      LocalTaskProcessor *impl;
      int32_t domain;    // index into domains
      uint32_t first_op; // ops[first_op .. first_op + num_ops) run here
      uint32_t num_ops;
      uint32_t queue_offset; // byte offset of the ready queue in the domain block
      uint32_t tail_offset;  // byte offset of the queue tail (atomic<uint64_t>)
      uint32_t
          inputs_offset; // byte offset of the pending-inputs counter (atomic<int64_t>)
      // byte offset of the count of operations still to complete here
      // (atomic<int64_t>); the processor's share of the finish counter is
      // released when it reaches zero, so tasks that block and finish out of
      // order are accounted for
      uint32_t remaining_offset;
      uint32_t initial_ready; // operations ready when the instantiation starts
      // Number of graph inputs (external preconditions) that operations on
      // this processor transitively depend on. Until an instantiation has
      // seen them all trigger, the processor runs ready work of the graph
      // but does not hold off other work, since it might be waiting on it.
      uint32_t pending_inputs;
      // Asynchronous operations here occupy async slots [first_async,
      // first_async + num_async) of the instantiation.
      bool async;
      uint32_t first_async, num_async;
    };
    struct Domain {
      int numa_node;           // OS NUMA node, or -1 if unknown
      size_t bytes;            // block size, a multiple of the cache line
      std::vector<char> image; // initial block contents
    };

    std::vector<Op> ops;
    uint32_t num_direct_ops;        // directly launched operations, numbered last
    std::vector<uint32_t> task_ops; // task index -> op
    std::vector<uint32_t> copy_ops; // copy index -> op
    // One analyzed transfer plan per copy, shared by all instantiations
    // (a reference is held; released by clear()).
    std::vector<TransferDesc *> copy_plans;
    bool any_task_profiling;     // some task carries profiling requests
    std::vector<uint32_t> roots; // operations without in-graph predecessors
    std::vector<Proc> procs;
    std::vector<Domain> domains;
    std::unordered_map<Processor, uint32_t> proc_index;
    // op -> successors released when its function returns (everything for
    // synchronous operations; same-processor tasks after a deferred one)
    FlattenedSparseMatrix<uint32_t> successors;
    // op -> successors released only once its work has completed
    FlattenedSparseMatrix<uint32_t> late_successors;
    // op -> async slots of the deferred predecessors whose tokens it waits on
    FlattenedSparseMatrix<uint32_t> token_waits;
    uint32_t num_async_ops;
    FlattenedSparseMatrix<uint32_t> postconds_of; // op -> postconditions it feeds
    std::vector<Input> inputs;
    std::vector<Postcond> postconds;
    std::vector<Interp> interps;
    // Bitsets over inputs, input_words 64-bit words each: per operation and
    // per postcondition, the external inputs transitively depended on.
    size_t input_words;
    std::vector<uint64_t> op_inputs;
    std::vector<uint64_t> postcond_inputs;

    void clear();
  };

  class SubgraphImpl {
  public:
    SubgraphImpl();
    ~SubgraphImpl();

    void init(ID _me, int _owner);

    // used by the dynamic table that allocates SubgraphImpls
    static ID make_id(const SubgraphImpl &dummy, int owner, ID::IDType index)
    {
      return ID::make_subgraph(owner, 0, index);
    }

    // Compiles `defn`, aborting with a message naming the operation and
    // feature for anything unsupported.
    void compile(void);
    // Set by compile, cleared by destroy: catches use after destroy for as
    // long as this slot is not reused (subgraph ids carry no generation).
    // Read without the lifecycle lock, so atomic.
    atomic<bool> alive{false};

    void instantiate(const void *args, size_t arglen, const ProfilingRequestSet &prs,
                     const SubgraphInstantiationProfiling &profiling,
                     span<const Event> preconditions, span<const Event> postconditions,
                     Event start_event, Event finish_event, int priority_adjust);

    void destroy(void);

    // Waits for a destroy's precondition (see request_destroy).
    class DeferredDestroy : public EventWaiter {
    public:
      void defer(SubgraphImpl *_subgraph, Event wait_on);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

    protected:
      SubgraphImpl *subgraph;
    };

    // Requests destruction. The destroy takes effect when wait_on triggers:
    // instantiations handled before that point (whether or not the destroy
    // was requested first) are covered by it, and the subgraph is destroyed
    // once the last of them has released its resources. Instantiating after
    // wait_on has triggered is an error. The returned event triggers when
    // the destroy is complete, NO_EVENT if it was completed on the spot.
    Event request_destroy(Event wait_on);
    // Called by DeferredDestroy when the destroy's precondition triggers.
    void destroy_precondition_triggered(bool poisoned);
    // Called when an instantiation's execution state has been released.
    void instantiation_released(void);

    // A remote instantiate: who sent it and where it stands in that node's
    // sequence of instantiates of this subgraph (see
    // RemoteSubgraphInstantiations), plus the message payload, which is
    // copied if the instantiate has to be held back.
    struct RemoteOrder {
      NodeID sender;
      uint64_t seq, base;
    };
    struct RemoteMessage {
      RemoteOrder order;
      Event start_event, finish_event;
      int priority_adjust;
      const void *data; // the payload (see SubgraphInstantiateMessage)
      size_t datalen;
    };
    // Handles a remote instantiate, holding it back if an earlier one from
    // the same node has not arrived yet and the mode is INSTANTIATION_ORDER,
    // then any held ones from that node that are ready now.
    void handle_remote_instantiate(const RemoteMessage &msg);

    // Execution states are recycled: an instantiation takes one from the
    // pool (or builds one, with its NUMA-placed blocks) and returns it once
    // the instantiation has been released.
    SubgraphExecutionState *acquire_state(void);
    void recycle_state(SubgraphExecutionState *state);

    // Per-instantiation memory blocks (one per NUMA domain), recycled.
    void acquire_blocks(std::vector<char *> &blocks);
    void release_blocks(std::vector<char *> &blocks);

  public:
    ID me;
    SubgraphImpl *next_free;
    SubgraphDefinition *defn;
    DeferredDestroy deferred_destroy;
    CompiledSubgraph compiled;

  protected:
    // A remote instantiate held until its predecessors from the same node
    // have been handled (INSTANTIATION_ORDER): a copy of the message.
    struct HeldInstantiate {
      RemoteMessage msg; // msg.data points into payload
      std::vector<char> payload;
    };
    // The core of instantiate. For a remote instantiate of an
    // INSTANTIATION_ORDER subgraph it may hold the message instead (returns
    // nullptr then too). Returns a held message from the same node that is
    // ready now, if any, popped in the same critical section that recorded
    // this one: after the instantiation is published nothing may touch this
    // object again without such a message in hand (a destroy covering the
    // published instantiation may complete and return the object to the
    // free list at any moment), and a held message is an unhandled
    // instantiate, which a conforming program cannot have destroyed.
    HeldInstantiate *instantiate(const void *args, size_t arglen,
                                 const ProfilingRequestSet &prs,
                                 const SubgraphInstantiationProfiling &profiling,
                                 span<const Event> preconditions,
                                 span<const Event> postconditions, Event start_event,
                                 Event finish_event, int priority_adjust,
                                 const RemoteMessage *remote);

    // Lifecycle state, protected by lifecycle_lock (including `alive` and
    // the reset in destroy()): the finish event of the most recent
    // instantiation (INSTANTIATION_ORDER chains the next one after it), the
    // number of instantiations whose state is still alive, and the destroy
    // request. A destroy is `pending` from the request until its
    // precondition triggers (a second request is an error) and `requested`
    // from then on (an instantiate is an error); whoever sees it requested
    // with the count at zero completes it.
    Mutex lifecycle_lock;
    Event previous_instantiation_completion = Event::NO_EVENT;
    int64_t outstanding_instantiations = 0;
    bool instantiated = false; // ONE_SHOT: a second instantiate is an error
    bool destroy_pending = false;
    bool destroy_requested = false;
    UserEvent destroy_done = UserEvent::NO_USER_EVENT;
    // per sending node, INSTANTIATION_ORDER only (see subgraph_order.h)
    std::unordered_map<NodeID, RemoteOrderTracker> sender_order;

    // Blocks come from per-domain slabs placed on the domain's NUMA node:
    // one mmap per slab, not per instantiation (an application may have
    // hundreds of instantiations in flight, and a page-granular NUMA
    // allocation costs hundreds of microseconds).
    struct Slab {
      int32_t domain;
      char *base;
      size_t bytes;
      bool numa; // allocated with numasysif_alloc_mem
    };
    Mutex block_pool_lock;
    std::vector<std::vector<char *>> block_pool; // released block sets
    // Its own lock: a state is recycled on the critical path between two
    // instantiations in order, and must not wait behind slab allocation.
    Mutex state_pool_lock;
    std::vector<SubgraphExecutionState *> state_pool; // released states
    std::vector<Slab> slabs;
    void grow_pool_locked(void); // carves one more slab per domain into the pool
    void free_slabs(void);
    // Frees the definition, the pooled states and slabs, and the compiled
    // form. Used by destroy, and by the destructor for a subgraph the
    // application never destroyed (runtime teardown).
    void release_resources(void);
  };

  // Instantiations this node has sent to remote owners, by subgraph. Two
  // things are derived from them, because active messages are not ordered:
  // not across message types, and not within one type either (a payload
  // large enough for a rendezvous protocol reaches the owner's handler
  // after smaller messages sent later).
  //  - A remote destroy's message is sent only once every instantiation
  //    this node sent before it has finished (DeferredRemoteDestroy), so
  //    the owner cannot handle the destroy ahead of one of them. The finish
  //    events' poison is irrelevant to the destroy and is never merged
  //    into its precondition.
  //  - Each instantiate carries its sequence number among this node's
  //    instantiates of the subgraph, and the lowest sequence number still
  //    unfinished when it was sent (`base`): everything below `base` had
  //    finished, so the owner needs only [base, seq) to have been handled
  //    before this one to preserve the issue order (INSTANTIATION_ORDER).
  // Entries are keyed by the reusable subgraph id and removed by the destroy
  // from this node; one left behind (a subgraph this node instantiated but
  // did not destroy) is harmless: its events are long triggered and a new
  // subgraph with the same id starts with a fresh order on the owner, which
  // `base` accounts for.
  class RemoteSubgraphInstantiations {
  public:
    // Records an instantiate about to be sent: its sequence number and base.
    void note(Subgraph subgraph, Event finish_event, uint64_t &seq, uint64_t &base);
    // The finish events of this node's unfinished instantiations of the
    // subgraph, merged ignoring faults (NO_EVENT if none); removes the entry.
    Event take(Subgraph subgraph);

  private:
    Mutex mutex;
    std::unordered_map<ID::IDType, RemoteOrderSender<Event>> entries;
  };

  // Sends a destroy message to a subgraph's owner once the instantiations
  // this node sent before it have finished (see RemoteSubgraphInstantiations).
  class DeferredRemoteDestroy : public EventWaiter {
  public:
    // Sends at once if `after` has triggered (poisoned or not), otherwise
    // when it does.
    static void send(NodeID owner, Subgraph subgraph, Event wait_on, UserEvent to_trigger,
                     Event after);

    virtual void event_triggered(bool poisoned, TimeLimit work_until);
    virtual void print(std::ostream &os) const;
    virtual Event get_finish_event(void) const;

  protected:
    DeferredRemoteDestroy(NodeID _owner, Subgraph _subgraph, Event _wait_on,
                          UserEvent _to_trigger);
    static void send_now(NodeID owner, Subgraph subgraph, Event wait_on,
                         UserEvent to_trigger);

    NodeID owner;
    Subgraph subgraph;
    Event wait_on;
    UserEvent to_trigger;
  };

  // active messages

  // The header must fit every network's budget. GASNet-EX packs a medium
  // request's header into 15 four-byte arguments (60 bytes), and a payload
  // too large for one message is fragmented, which prepends a 16-byte
  // FragmentInfo: 44 bytes remain for this struct, which with 8-byte
  // alignment means 40. (The RelWithDebInfo build compiles the check in
  // gasnetex_handlers.cc out; a header over the budget shows up as a
  // fortify "buffer overflow detected" on the sender.) The payload is:
  // uint64_t arglen, the instantiation arguments, then the serialized
  // preconditions, postconditions, profiling request set and profiling.
  struct SubgraphInstantiateMessage {
    Subgraph subgraph;
    Event wait_on, finish_event;
    uint64_t seq;       // see RemoteSubgraphInstantiations
    uint32_t base_back; // seq - base: the unfinished window never reaches 2^32
                        // (a node cannot hold that many live finish events)
    int priority_adjust;

    static void handle_message(NodeID sender, const SubgraphInstantiateMessage &msg,
                               const void *data, size_t datalen);
  };
  static_assert(sizeof(SubgraphInstantiateMessage) <= 44,
                "SubgraphInstantiateMessage must fit GASNet-EX's 60-byte medium header "
                "budget minus the 16-byte FragmentInfo of a fragmented message");

  struct SubgraphDestroyMessage {
    Subgraph subgraph;
    Event wait_on;
    UserEvent to_trigger;

    static void handle_message(NodeID sender, const SubgraphDestroyMessage &msg,
                               const void *data, size_t datalen);
  };

  // SubgraphWorkLauncher installs an instantiation onto its processors once
  // the instantiation's precondition has triggered.
  // The waiters below are members of the execution state, so an
  // instantiation allocates nothing beyond its finish event: they are armed
  // once per instantiation and are free again once the state is recycled.

  // Starts the instantiation once its start event and, for
  // INSTANTIATION_ORDER, the previous instantiation's finish event have
  // triggered. Two inputs instead of a merged event: no event allocation
  // and one trigger hop less on the critical path between instantiations.
  class SubgraphWorkLauncher {
  public:
    explicit SubgraphWorkLauncher(SubgraphExecutionState *state);
    void arm(Event start_event, Event order_event);
    static void launch(SubgraphExecutionState *state, bool poisoned);

  private:
    struct Input : public EventWaiter {
      SubgraphWorkLauncher *owner = nullptr;
      // Only the start event's poison poisons the instantiation. The order
      // event is the previous instantiation's finish event: its poison was
      // that instantiation's, and must not cascade down the chain.
      bool carries_poison = true;
      virtual void event_triggered(bool poisoned, TimeLimit work_until) override;
      virtual void print(std::ostream &os) const override;
      virtual Event get_finish_event(void) const override;
    };
    void input_triggered(bool poisoned);

    SubgraphExecutionState *state;
    Input inputs[2];
    atomic<int> pending{0};
    atomic<bool> poisoned{false};
  };

  // SubgraphInputWaiter delivers an external precondition to an instantiation.
  // Registered by the launcher once the instantiation has started (nothing
  // can run before then anyway, and an instantiation whose start was
  // poisoned never registers them). Each registered waiter holds a share of
  // the finish counter until its callback has returned, so the state is
  // never recycled while a waiter is still registered or running.
  class SubgraphInputWaiter : public EventWaiter {
  public:
    SubgraphInputWaiter(SubgraphExecutionState *state, uint32_t input);
    virtual void event_triggered(bool poisoned, TimeLimit work_until) override;
    virtual void print(std::ostream &os) const override;
    virtual Event get_finish_event(void) const override;

  private:
    SubgraphExecutionState *state;
    uint32_t input;
  };

  // SubgraphInstantiationCleanup waits for an instantiation's finish event
  // and then releases its execution state on a background worker.
  // Recycles the state after the finish event has been triggered. Not an
  // event waiter: the trigger's waiters include the launcher of the next
  // instantiation in order, and the trigger context's time budget is small,
  // so nothing may run ahead of them. The state's finishing path calls
  // cleanup() itself once the trigger has returned, inline when that is
  // cheap (no processor tokens to return), otherwise through the reaper.
  class SubgraphInstantiationCleanup {
  public:
    explicit SubgraphInstantiationCleanup(SubgraphExecutionState *state);
    void cleanup();

  private:
    SubgraphExecutionState *state;
  };

  // SubgraphResourceReaper is a background work item that processes
  // SubgraphInstantiationCleanup items asynchronously.
  class SubgraphResourceReaper : public BackgroundWorkItem {
  public:
    SubgraphResourceReaper();

    void enqueue_cleanup(SubgraphInstantiationCleanup *item);

    virtual bool do_work(TimeLimit work_until) override;

    // Two counts of the instantiations owned by this node, for shutdown:
    //  - unfinished: instantiated, finish event not yet triggered. This is
    //    pending application work, exactly like an Operation whose finish
    //    event has not triggered: once the shutdown precondition and the
    //    processors' flush tasks are done, a nonzero count is an application
    //    bug and the runtime aborts at once, as it does for operations. No
    //    waiting (and no timer) is involved.
    //  - outstanding: instantiated, state not yet released. After the finish
    //    event has triggered, releasing the state is bounded internal work
    //    on the background workers (recycle it, instantiation_released, and
    //    possibly a deferred destroy, which looks events up), so the runtime
    //    waits for it before tearing down. An application is allowed to see
    //    an instantiation's finish event, destroy the subgraph without
    //    waiting for the returned event and shut down. The wait depends on
    //    no application event and so always ends.
    //  - pending_destroys: destroys requested on this node whose
    //    precondition has not triggered. The precondition is an application
    //    event, so this too is pending application work at shutdown (and a
    //    destroy that never takes effect would leave its waiter registered
    //    on an event through teardown).
    void instantiation_created(void)
    {
      unfinished.fetch_add(1);
      outstanding.fetch_add(1);
    }
    void instantiation_finished(void) { unfinished.fetch_sub_acqrel(1); }
    void instantiation_released(void) { outstanding.fetch_sub_acqrel(1); }
    void destroy_deferred(void) { pending_destroys.fetch_add(1); }
    void destroy_precondition_triggered(void) { pending_destroys.fetch_sub_acqrel(1); }
    int64_t num_unfinished(void) const { return unfinished.load_acquire(); }
    int64_t num_pending_destroys(void) const { return pending_destroys.load_acquire(); }
    bool idle(void) const { return outstanding.load_acquire() == 0; }
    void wait_for_idle(void);

  private:
    Mutex mutex;
    std::queue<SubgraphInstantiationCleanup *> pending_cleanups;
    atomic<int64_t> unfinished{0};
    atomic<int64_t> outstanding{0};
    atomic<int64_t> pending_destroys{0};
  };

  // Completion callback for subgraph tasks on processors whose tasks keep
  // working after their function returns (see the LocalTaskProcessor hooks
  // in proc_impl.h).
  class SubgraphAsyncCompletion {
  public:
    virtual ~SubgraphAsyncCompletion(void) {}
    virtual void async_completed(void) = 0;
  };

  // SubgraphExecutionState is the per-instantiation state of a subgraph:
  // the NUMA-placed blocks described by CompiledSubgraph plus finish tracking.
  class SubgraphExecutionState {
  public:
    explicit SubgraphExecutionState(SubgraphImpl *subgraph);
    ~SubgraphExecutionState();
    // Prepares the (new or recycled) state for one instantiation.
    void reset(Event finish_event, int priority, span<const Event> preconditions,
               span<const Event> postconditions);
    // Returns processor resources after the instantiation has finished.
    void retire(void);
    // Triggers the finish event (launching whatever waits on it, including
    // the next instantiation in order) and then recycles this state.
    // Nothing may touch the state after calling this.
    void finish(bool poisoned);
    // True when retire() has nothing to hand back to a processor, so the
    // state can be recycled in the finish event's trigger context.
    bool retires_inline(void) const;

    SubgraphWorkLauncher launcher;
    SubgraphInstantiationCleanup cleanup;
    std::vector<SubgraphInputWaiter> input_waiters; // one per external input
    SubgraphImpl *get_subgraph() const { return subgraph; }
    int get_priority() const { return priority; }
    Event get_finish_event() const { return finish_event; }

    atomic<int64_t> &counter(uint32_t op) const;
    atomic<int64_t> *queue(uint32_t proc) const;
    atomic<uint64_t> &tail(uint32_t proc) const;
    atomic<int64_t> &pending_inputs(uint32_t proc) const;
    atomic<int64_t> &remaining(uint32_t proc) const;
    atomic<int64_t> &postcond_counter(uint32_t pc) const;
    // Argument bytes for an operation: its interpolated copy if it has one,
    // otherwise the definition's.
    ByteArrayRef op_args(uint32_t op) const;

    // Applies the instantiation arguments to the interpolated operations.
    void interpolate(const void *args, size_t arglen);
    // Sets up profiling for operations with definition-time or
    // instantiation-time requests.
    void setup_profiling(const SubgraphInstantiationProfiling &profiling);
    struct OpProfiling {
      ProfilingRequestSet requests;
      ProfilingMeasurementCollection measurements;
      bool wants_timeline = false, wants_proc = false, wants_status = false,
           wants_fevent = false;
      ProfilingMeasurements::OperationTimeline timeline;
    };
    OpProfiling *prof(uint32_t op) const
    {
      if(prof_index.empty() || (prof_index[op] < 0))
        return nullptr;
      return profiling[prof_index[op]].get();
    }
    // Records that external input `input` has triggered (possibly poisoned)
    // and makes dependent operations ready.
    void input_triggered(uint32_t input, bool poisoned);
    // Delivers the external preconditions: those already triggered at once,
    // the rest through input_waiters, each holding a finish share.
    void deliver_inputs(void);
    // Satisfies the implicit start input of every root operation.
    void start(void);
    // An operation's last precondition has been satisfied.
    void op_ready(uint32_t op);
    // Starts a copy operation on its compiled plan; it completes through
    // op_body_done/op_finished when the transfer finishes.
    void launch_copy(uint32_t op);
    // An operation's function has returned (or it was skipped because an
    // input it depends on was poisoned): releases the successors that may
    // start now.
    void op_body_done(uint32_t op);
    // called when active_work reaches zero: wakes the processors that may
    // have been holding lower-priority work back for this instantiation
    void stalled(void);
    // An operation's work is complete: releases the remaining successors
    // and postconditions, then its share of the finish counter. The state
    // may be freed once this returns.
    void op_finished(uint32_t op);
    // Adds and sends the measurements of a completed operation, if any.
    void complete_op_profiling(uint32_t op, OpProfiling *pf, Event finish_event,
                               bool skipped);
    // True if the operation depends on a poisoned input and must be skipped.
    bool op_poisoned(uint32_t op) const;
    // Releases one share of the finish counter (see finish_counter).
    void contributor_finished(void);

    // Asynchronous operations: the processor notifies the completion once the
    // work is done; the token orders dependents' work after it.
    struct AsyncOp : public SubgraphAsyncCompletion {
      SubgraphExecutionState *state = nullptr;
      uint32_t op = 0;
      Event finish_event = Event::NO_EVENT; // created on demand by the task
      virtual void async_completed(void) override;
    };
    std::vector<AsyncOp> async_ops; // by async slot
    std::vector<void *> tokens;     // by async slot, returned at destruction

  private:
    friend class ProcSubgraphExecutor;
    friend class SubgraphWorkLauncher;

    SubgraphImpl *subgraph;

    // Lifetime of this state during one instantiation. Every party that may
    // read the state holds one share: each processor (released when its
    // last operation completes), each directly launched operation, the
    // launcher (while it delivers inputs and starts the roots) and each
    // registered input waiter (until its callback returns). Making an
    // operation ready publishes it to a processor that may run it, complete
    // the instantiation and recycle the state at once, so whoever publishes
    // must hold a share for as long as it goes on reading the state.
    // Whoever brings the count to zero triggers finish_event and recycles.
    atomic<int64_t> finish_counter;
    Event finish_event;
    // Events the caller gave for the external preconditions and postconditions.
    std::vector<Event> preconditions;
    std::vector<Event> postconditions;
    // Bitset of external inputs that triggered poisoned.
    std::vector<atomic<uint64_t>> poisoned_inputs;
    atomic<bool> poisoned;
    // the start event was poisoned: every operation is skipped and every
    // postcondition poisoned, after the inputs have been waited for
    atomic<bool> start_poisoned;

    // Scheduling priority of this instantiation. While it is active on a
    // processor whose inputs are all satisfied, that processor runs only
    // work of equal or higher priority (plus tasks that already started).
    int priority;
    // Operations ready, in flight or running (not blocked) anywhere: while
    // nonzero the instantiation is making progress and holds its
    // processors (see ProcSubgraphExecutor::active_floor). Incremented when
    // an operation becomes ready, decremented when its function returns
    // (or the copy completes), and around a graph task blocking.
    atomic<int64_t> active_work;

    std::vector<char *> blocks; // one per CompiledSubgraph::Domain

    // Profiling, only populated when some operation requested it.
    std::vector<int32_t> prof_index; // per op, -1 if none
    std::vector<std::unique_ptr<OpProfiling>> profiling;
    // Per copy, the definition's requests merged with this instantiation's;
    // empty when the instantiation added none (the plan's requests apply).
    std::vector<ProfilingRequestSet> copy_prs;
  };

  // ProcSubgraphExecutor is the per-scheduler component that feeds subgraph
  // tasks to a processor's scheduler loop.
  //
  // Threading: enqueue_subgraph may be called from any thread. peek and
  // dequeue must be called with the owning scheduler's lock held; execute
  // must be called without it. The executor never touches the scheduler
  // lock itself; the scheduler loop treats a dequeued entry exactly like a
  // ready task (worker accounting, unlock, run, relock).
  //
  // Any number of instantiations may be active on one processor at a time.
  // The executor serves whichever has a ready operation, round robin among
  // the active ones, so progress never depends on the order in which
  // processors, or nodes, learned about the instantiations.
  class ProcSubgraphExecutor {
  public:
    ProcSubgraphExecutor(Processor proc);
    ~ProcSubgraphExecutor();

    struct ReadyEntry {
      SubgraphExecutionState *state;
      uint32_t op;  // into CompiledSubgraph::ops
      int priority; // the instantiation's priority
    };

    // Thread-safe. The caller is responsible for waking the scheduler.
    void enqueue_subgraph(SubgraphExecutionState *state);

    // Scheduler lock held. Returns true if an operation is ready to run and
    // reports its priority; among several ready instantiations the highest
    // priority wins, ties round robin. Work of instantiations below the
    // active floor (see below) is not offered.
    bool peek(int &priority);
    // Scheduler lock held. The priority floor this processor is held at,
    // or `none` if it is not held: the highest priority among the graph
    // tasks running here that have not blocked, and the instantiations with
    // operations still to run here whose inputs here have all triggered and
    // that are making progress somewhere (a task of theirs is running, or
    // an operation is ready or in flight: SubgraphExecutionState::
    // active_work). Work below the floor, normal tasks and other
    // instantiations alike, must wait. An instantiation that is stalled
    // everywhere (every started task of it is blocked and nothing is ready)
    // does not hold its processors: holding them would only idle them and,
    // if the graph is waiting on lower-priority work that needs one of
    // them, deadlock. Started tasks always resume, so the holds return as
    // soon as a task of it resumes. A held floor is thus always backed by
    // work that is executing or about to, and nothing can wait forever on
    // it (the highest-priority ready operation on a processor is never
    // excluded).
    int active_floor(int none) const;
    // Scheduler lock held. Running graph tasks on this processor, for the
    // floor: started when dequeued, finished when execute returns, and
    // blocked/resumed around a wait inside the task (the latter two also
    // maintain the instantiation's active_work).
    void task_started(int priority);
    void task_finished(int priority);
    void task_blocked(int priority, SubgraphExecutionState *state);
    void task_resumed(int priority, SubgraphExecutionState *state);
    // Scheduler lock held. Removes the operation found by the last
    // successful peek.
    void dequeue(ReadyEntry &entry);

    // No lock held. Runs the operation on the calling thread and propagates
    // its completion: successors become ready on their processors and, if
    // this was the processor's last operation, the finish counter drops.
    void execute(const ReadyEntry &entry);

    // Scheduler lock held; called when the scheduler has nothing else to do.
    // Returns true if it should keep polling instead of sleeping: subgraph
    // work ran or arrived on this processor within the last poll_budget_us,
    // so dependent operations from other processors are likely imminent and
    // should not pay a wake-up. The budget applies even while instantiations
    // are active here: one that is stalled (on another processor's work or
    // on an external input) must not keep this processor spinning for the
    // whole wait. (-ll:subgraph_poll, 0 disables)
    bool keep_polling(void);
    static int poll_budget_us;

    // Called by each worker thread of the owning scheduler when it starts;
    // records the NUMA node the processor's workers run on.
    void note_worker_started(void);
    // OS NUMA node of this processor's workers, or -1 if unknown.
    // Written once by the first worker to start, read by compile() on other
    // threads: -2 until known, -1 if unknown.
    int numa_node(void) const { return numa_node_.load_acquire(); }

  private:
    struct Cursor {
      SubgraphExecutionState *state;
      atomic<int64_t> *queue; // this processor's ready queue in the state
      uint32_t front;         // next slot to read
      uint32_t end;           // number of slots (== operations for this processor)
      uint32_t proc;          // this processor's index within the subgraph
      int priority;
    };
    void absorb_pending(void);

    Processor proc;
    atomic<int> numa_node_;

    // Instantiations handed to this processor but not yet picked up by the
    // scheduler loop. Written by launchers, drained under the scheduler lock.
    Mutex pending_mutex;
    std::vector<SubgraphExecutionState *> pending;
    std::vector<SubgraphExecutionState *> pending_scratch;
    atomic<int64_t> pending_count;

    // Instantiations this processor is working on (scheduler lock).
    std::vector<Cursor> active;
    size_t scan_start;
    // Graph tasks running (started and not blocked) here, as (priority,
    // count) pairs: a handful of entries at most, searched linearly, so a
    // task costs no allocation (scheduler lock).
    std::vector<std::pair<int, int>> running;
    // Result of the last successful peek, consumed by dequeue.
    size_t peeked_cursor;
    int64_t peeked_op;

    // Polling state: activity_epoch is bumped whenever work is dequeued or
    // an instantiation arrives; keep_polling extends the deadline when it
    // sees a new epoch.
    uint64_t activity_epoch;
    uint64_t polled_epoch;
    long long poll_deadline_ns;
  };

}; // namespace Realm

#endif
