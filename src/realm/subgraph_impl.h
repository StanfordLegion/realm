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
#include "realm/id.h"
#include "realm/event_impl.h"
#include "realm/operation.h"
#include "realm/proc_impl.h"
#include "realm/bgwork.h"
#include "realm/mutex.h"

#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace Realm {

  class LocalTaskProcessor;
  class ProcSubgraphExecutor;
  class SubgraphExecutionState;
  class SubgraphWorkLauncher;
  class SubgraphInstantiationCleanup;

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
    };
    struct Input { // one external precondition
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
      int32_t domain;         // index into domains
      uint32_t first_op;      // ops[first_op .. first_op + num_ops) run here
      uint32_t num_ops;
      uint32_t queue_offset;  // byte offset of the ready queue in the domain block
      uint32_t tail_offset;   // byte offset of the queue tail (atomic<uint64_t>)
      uint32_t inputs_offset; // byte offset of the pending-inputs counter (atomic<int64_t>)
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
    uint32_t num_direct_ops;          // directly launched operations, numbered last
    std::vector<uint32_t> task_ops;   // task index -> op
    bool any_task_profiling;          // some task carries profiling requests
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

    void instantiate(const void *args, size_t arglen, const ProfilingRequestSet &prs,
                     const SubgraphInstantiationProfiling &profiling,
                     span<const Event> preconditions, span<const Event> postconditions,
                     Event start_event, Event finish_event, int priority_adjust);

    void destroy(void);

    class DeferredDestroy : public EventWaiter {
    public:
      void defer(SubgraphImpl *_subgraph, Event wait_on, UserEvent to_trigger);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

    protected:
      SubgraphImpl *subgraph;
      UserEvent to_trigger;
    };

    // Requests destruction. The returned event triggers once every
    // outstanding instantiation has released its resources and wait_on has
    // triggered. Instantiating after this is an error.
    Event request_destroy(Event wait_on);
    // Called when an instantiation's execution state has been released.
    void instantiation_released(void);

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
    Event complete_destroy(Event wait_on, UserEvent done);

    // Lifecycle state, protected by lifecycle_lock: the finish event of the
    // most recent instantiation (INSTANTIATION_ORDER chains the next one
    // after it), the number of instantiations whose state is still alive,
    // and a pending destroy request carried out by whoever observes the
    // count reach zero.
    Mutex lifecycle_lock;
    Event previous_instantiation_completion = Event::NO_EVENT;
    int64_t outstanding_instantiations = 0;
    bool destroy_requested = false;
    Event destroy_wait_on = Event::NO_EVENT;
    UserEvent destroy_done = UserEvent::NO_USER_EVENT;

    Mutex block_pool_lock;
    std::vector<std::vector<char *>> block_pool;
    void free_blocks(std::vector<char *> &blocks);
  };

  // active messages

  struct SubgraphInstantiateMessage {
    Subgraph subgraph;
    Event wait_on, finish_event;
    size_t arglen;
    int priority_adjust;

    static void handle_message(NodeID sender, const SubgraphInstantiateMessage &msg,
                               const void *data, size_t datalen);
  };

  struct SubgraphDestroyMessage {
    Subgraph subgraph;
    Event wait_on;
    UserEvent to_trigger;

    static void handle_message(NodeID sender, const SubgraphDestroyMessage &msg,
                               const void *data, size_t datalen);
  };

  // SubgraphWorkLauncher installs an instantiation onto its processors once
  // the instantiation's precondition has triggered.
  class SubgraphWorkLauncher : public EventWaiter {
  public:
    SubgraphWorkLauncher(SubgraphExecutionState *state);
    static void launch_or_defer(SubgraphExecutionState *state, Event wait_on);
    // Hands the instantiation to every processor it uses. A poisoned
    // precondition poisons the instantiation instead of running it.
    static void launch(SubgraphExecutionState *state, bool poisoned);

    virtual void event_triggered(bool poisoned, TimeLimit work_until) override;
    virtual void print(std::ostream &os) const override;
    virtual Event get_finish_event(void) const override;

  private:
    SubgraphExecutionState *state;
  };

  // SubgraphInputWaiter delivers an external precondition to an instantiation.
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
  class SubgraphInstantiationCleanup : public EventWaiter {
  public:
    SubgraphInstantiationCleanup(SubgraphExecutionState *state);
    void cleanup();

    virtual void event_triggered(bool poisoned, TimeLimit work_until) override;
    virtual void print(std::ostream &os) const override;
    virtual Event get_finish_event(void) const override;

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

  private:
    Mutex mutex;
    std::queue<SubgraphInstantiationCleanup *> pending_cleanups;
  };

  // SubgraphExecutionState is the per-instantiation state of a subgraph:
  // the NUMA-placed blocks described by CompiledSubgraph plus finish tracking.
  class SubgraphExecutionState {
  public:
    SubgraphExecutionState(SubgraphImpl *subgraph, Event finish_event, int priority,
                           span<const Event> postconditions);
    ~SubgraphExecutionState();
    SubgraphImpl *get_subgraph() const { return subgraph; }
    int get_priority() const { return priority; }

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
    // Satisfies the implicit start input of every root operation.
    void start(void);
    // An operation's last precondition has been satisfied.
    void op_ready(uint32_t op);
    // An operation's function has returned (or it was skipped because an
    // input it depends on was poisoned): releases the successors that may
    // start now.
    void op_body_done(uint32_t op);
    // An operation's work is complete: releases the remaining successors
    // and postconditions, then its share of the finish counter. The state
    // may be freed once this returns.
    void op_finished(uint32_t op);
    // Adds and sends the measurements of a completed operation, if any.
    void complete_op_profiling(uint32_t op, OpProfiling *pf, Event finish_event,
                               bool skipped);
    // True if the operation depends on a poisoned input and must be skipped.
    bool op_poisoned(uint32_t op) const;
    // Called by each processor after its last operation and by each directly
    // launched operation after completing.
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

    // Number of processors and directly launched operations still working
    // on this instantiation. Whoever brings it to zero triggers finish_event.
    atomic<int64_t> finish_counter;
    Event finish_event;
    // Events the caller gave for the external postconditions.
    std::vector<Event> postconditions;
    // Bitset of external inputs that triggered poisoned.
    std::vector<atomic<uint64_t>> poisoned_inputs;
    atomic<bool> poisoned;

    // Scheduling priority of this instantiation. While it is active on a
    // processor whose inputs are all satisfied, that processor runs only
    // work of equal or higher priority (plus tasks that already started).
    int priority;

    std::vector<char *> blocks; // one per CompiledSubgraph::Domain

    // Profiling, only populated when some operation requested it.
    std::vector<int32_t> prof_index; // per op, -1 if none
    std::vector<std::unique_ptr<OpProfiling>> profiling;
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
      uint32_t op;             // into CompiledSubgraph::ops
      int priority;            // the instantiation's priority
    };

    // Thread-safe. The caller is responsible for waking the scheduler.
    void enqueue_subgraph(SubgraphExecutionState *state);

    // Scheduler lock held. Returns true if an operation is ready to run and
    // reports its priority; among several ready instantiations the highest
    // priority wins, ties round robin. Work of instantiations below the
    // active floor (see below) is not offered.
    bool peek(int &priority);
    // Scheduler lock held. Highest priority among active instantiations
    // whose inputs are all satisfied, or `none` if there is none. Work below
    // this priority, normal tasks and other instantiations alike, must wait
    // on this processor.
    int active_floor(int none) const;
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
    // should not pay a wake-up. (-ll:subgraph_poll, 0 disables)
    bool keep_polling(void);
    static int poll_budget_us;

    // Called by each worker thread of the owning scheduler when it starts;
    // records the NUMA node the processor's workers run on.
    void note_worker_started(void);
    // OS NUMA node of this processor's workers, or -1 if unknown.
    int numa_node(void) const { return numa_node_; }

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
    int numa_node_;

    // Instantiations handed to this processor but not yet picked up by the
    // scheduler loop. Written by launchers, drained under the scheduler lock.
    Mutex pending_mutex;
    std::vector<SubgraphExecutionState *> pending;
    std::vector<SubgraphExecutionState *> pending_scratch;
    atomic<int64_t> pending_count;

    // Instantiations this processor is working on (scheduler lock).
    std::vector<Cursor> active;
    size_t scan_start;
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
