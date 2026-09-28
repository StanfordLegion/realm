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
#include "realm/bgwork.h"

#include <queue>

namespace Realm {

  class LocalTaskProcessor;
  class ProcSubgraphExecutor;
  class ThreadedTaskScheduler;
  class SubgraphExecutionState;
  class SubgraphWorkLauncher;

  struct SubgraphScheduleEntry {
    SubgraphDefinition::OpKind op_kind;
    unsigned op_index;
    std::vector<std::pair<unsigned, int>> preconditions;
    unsigned first_interp, num_interps;
    unsigned intermediate_event_base, intermediate_event_count;
    bool is_final_event;
  };

  // Sentinel value for empty entries in the processor-local queues.
  constexpr int64_t SUBGRAPH_EMPTY_QUEUE_ENTRY = -1;

  // FlattenedSparseMatrix is a helper class that represents a sparse
  // matrix in a flattened format for better cache locality. It is
  // basically a CSR representation of a sparse matrix.
  template <typename T>
  struct FlattenedSparseMatrix {
    FlattenedSparseMatrix() {}
    FlattenedSparseMatrix(const std::vector<std::vector<T>> &input)
    {
      uint64_t count = 0;
      for(size_t i = 0; i < input.size(); i++) {
        offsets.push_back(count);
        for(auto &it : input[i]) {
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

  class SubgraphImpl {
  public:
    SubgraphImpl();

    ~SubgraphImpl();

    void init(ID _me, int _owner);

    static ID make_id(const SubgraphImpl &dummy, int owner, ID::IDType index)
    {
      return ID::make_subgraph(owner, 0, index);
    }

    // compile/analyze the subgraph
    bool compile(void);

    void instantiate(const void *args, size_t arglen, const ProfilingRequestSet &prs,
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

  protected:
    // Fields populated by the compilation step in the compiled
    // execution mode of subgraphs.

    // Maintain the processors and a mapping from each processor its index.
    // We extract the LocalTaskProcessor* implementations from each processor
    // so that we don't have to query the runtime for these during the execution
    // of the subgraph.
    std::vector<Processor> subgraph_processors;
    std::vector<LocalTaskProcessor *> subgraph_processor_impls;
    std::unordered_map<Processor, int32_t> processor_to_index;

    struct SubgraphOperationDesc {
      SubgraphOperationDesc(SubgraphDefinition::OpKind _op_kind, unsigned _op_index,
                            bool _is_final_event, bool _is_async)
        : op_kind(_op_kind)
        , op_index(_op_index)
        , proc_index(-1)
        , is_final_event(_is_final_event)
        , is_async(_is_async)
      {}

      SubgraphDefinition::OpKind op_kind;
      unsigned op_index;
      // Index into subgraph_processors of the processor running this
      // operation, so edge propagation needs no lookups.
      int32_t proc_index;
      bool is_final_event;
      bool is_async;
    };
    // Holds all operations in the compiled subgraph.
    std::vector<SubgraphOperationDesc> compiled_subgraph_operations;

    // EdgeInfo contains the necessary metadata about an edge
    // in the compiled subgraph to trigger dependencies.
    struct EdgeInfo {
      EdgeInfo(uint64_t _index)
        : index(_index)
      {}
      uint64_t index;
    };
    // operation_{incoming,outgoing}_edges contains the edges that
    // every operation in compiled_subgraph_operations needs to
    // {wait for, notify} for when the operation {begins, finishes}.
    FlattenedSparseMatrix<EdgeInfo> operation_incoming_edges;
    FlattenedSparseMatrix<EdgeInfo> operation_outgoing_edges;
    // operation_precondition_counters contains for each entry of
    // compiled_subgraph_operations the number of predecessor operations
    // that must complete before the subgraph operation can begin.
    // This data will not be modified,
    std::vector<int64_t> operation_precondition_counters;

    // initial_processor_queues contains the initial queue entries
    // for each processor.
    FlattenedSparseMatrix<int64_t> initial_processor_queues;
    // initial_queue_entry_counts contains the number of initial
    // queue entries for each processor.
    std::vector<int64_t> initial_queue_entry_counts;

    friend class ProcSubgraphExecutor;
    friend class SubgraphExecutionState;
    friend class SubgraphWorkLauncher;
    friend class Subgraph;

    // Lifecycle state for compiled subgraphs, all protected by lifecycle_lock:
    //  - for INSTANTIATION_ORDER, the finish event of the most recent
    //    instantiation, which the next instantiation must wait for;
    //  - the number of instantiations whose execution state has not been
    //    released yet;
    //  - a pending destroy request, carried out by whoever observes the
    //    outstanding count reach zero.
    Mutex lifecycle_lock;
    Event previous_instantiation_completion = Event::NO_EVENT;
    int64_t outstanding_instantiations = 0;
    bool destroy_requested = false;
    Event destroy_wait_on = Event::NO_EVENT;
    UserEvent destroy_done = UserEvent::NO_USER_EVENT;

    // Performs (or defers until wait_on) the destruction once no
    // instantiations are outstanding, returning the event to hand back to
    // the caller of Subgraph::destroy.
    Event complete_destroy(Event wait_on, UserEvent done);

  public:
    // Requests destruction. The returned event triggers once every
    // outstanding instantiation has released its resources and wait_on has
    // triggered. Instantiating after this is an error.
    Event request_destroy(Event wait_on);
    // Called when an instantiation's execution state has been released.
    void instantiation_released(void);

  public:
    ID me;
    SubgraphImpl *next_free;
    SubgraphDefinition *defn;
    std::vector<SubgraphScheduleEntry> interpreted_schedule;
    size_t num_intermediate_events, num_final_events, max_preconditions;

    DeferredDestroy deferred_destroy;
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

  // SubgraphExecutionState is the per-instantiation state of a compiled
  // subgraph: fresh copies of the precondition counters and per-processor
  // ready queues, plus the finish tracking.
  class SubgraphExecutionState {
  public:
    SubgraphExecutionState(SubgraphImpl *subgraph, const void *args, size_t arglen,
                           UserEvent finish_event);
    ~SubgraphExecutionState();
    SubgraphImpl *get_subgraph() const { return subgraph; }

  private:
    friend class ProcSubgraphExecutor;
    friend class SubgraphWorkLauncher;

    SubgraphImpl *subgraph;

    // Local copy of the instantiation arguments (input to interpolation,
    // which compiled subgraphs do not support yet).
    void *args;
    size_t arglen;

    // Number of processors (and, in the future, asynchronous work items)
    // still working on this instantiation. Whoever brings it to zero
    // triggers finish_event.
    atomic<int64_t> finish_counter;
    UserEvent finish_event;

    // Remaining predecessor count for each entry of
    // SubgraphImpl::compiled_subgraph_operations.
    atomic<int64_t> *preconditions;

    // All per-processor ready queues, laid out contiguously using the
    // offsets in SubgraphImpl::initial_processor_queues. A slot holds an
    // index into compiled_subgraph_operations or SUBGRAPH_EMPTY_QUEUE_ENTRY.
    atomic<int64_t> *processor_queues;

    // Per-processor producer state, one cache line each.
    struct alignas(64) ProcessorLocalState {
      // Next free slot, relative to the processor's queue region.
      atomic<uint64_t> queue_back;
    };
    std::vector<ProcessorLocalState> processor_state;
  };

  // ProcSubgraphExecutor is the per-scheduler component that feeds compiled
  // subgraph tasks to a processor's scheduler loop.
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

    // A unit of work handed to the scheduler loop.
    struct ReadyEntry {
      SubgraphExecutionState *state;
      uint64_t op_index;       // into SubgraphImpl::compiled_subgraph_operations
      int32_t proc_index;      // this processor's index within the subgraph
      bool last_for_processor; // nothing more for this processor in this instantiation
    };

    // Thread-safe. The caller is responsible for waking the scheduler.
    void enqueue_subgraph(SubgraphExecutionState *state);

    // Scheduler lock held. Returns true if an operation is ready to run and
    // reports the priority it should be scheduled at.
    bool peek(int &priority);
    // Scheduler lock held. Removes the operation found by the last
    // successful peek.
    void dequeue(ReadyEntry &entry);

    // No lock held. Runs the operation on the calling thread and propagates
    // its completion: successors become ready on their processors and, if
    // this was the processor's last operation, the finish counter drops.
    void execute(const ReadyEntry &entry);

  private:
    struct Cursor {
      SubgraphExecutionState *state;
      uint64_t base;  // global index of this processor's first queue slot
      uint64_t front; // next slot to read, relative to base
      uint64_t end;   // number of slots (== operations for this processor)
      int32_t proc_index;
    };
    void absorb_pending(void);

    Processor proc;

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
  };

}; // namespace Realm

#endif
