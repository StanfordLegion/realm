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

namespace Realm {

  class LocalTaskProcessor;
  class ProcSubgraphExecutor;
  class ThreadedTaskScheduler;
  struct SubgraphExecutionState;
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
    // TODO (rohany): Should make many of these maps we access unordered_maps
    //  or a faster "small-map" implementation.

    // Fields populated by the compilation step in the compiled
    // execution mode of subgraphs.

    // Maintain the processors and a mapping from each processor its index.
    // We extract the LocalTaskProcessor* implementations from each processor
    // so that we don't have to query the runtime for these during the execution
    // of the subgraph.
    std::vector<Processor> subgraph_processors;
    std::vector<LocalTaskProcessor *> subgraph_processor_impls;
    std::map<Processor, int32_t> processor_to_index;

    struct SubgraphOperationDesc {
      SubgraphOperationDesc(SubgraphDefinition::OpKind _op_kind, unsigned _op_index,
                            bool _is_final_event, bool _is_async)
        : op_kind(_op_kind)
        , op_index(_op_index)
        , is_final_event(_is_final_event)
        , is_async(_is_async)
      {}

      SubgraphDefinition::OpKind op_kind;
      unsigned op_index;
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

    // When concurrency_mode == INSTANTIATION_ORDER, the subgraph will
    // implicitly order instantiations of the subgraph by tracking
    // the completion event of the last instantiation.
    mutable Mutex instantiation_lock;
    mutable Event previous_instantiation_completion = Event::NO_EVENT;
    // previous_cleanup_completion tracks cleanup work launched
    // by subgraph instantiations.
    mutable Event previous_cleanup_completion = Event::NO_EVENT;

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

    static void handle_message(NodeID sender, const SubgraphDestroyMessage &msg,
                               const void *data, size_t datalen);
  };

  // SubgraphWorkLauncher is a helper class that manages the logic of installing
  // a subgraph when the preconditions for execution are met.
  class SubgraphWorkLauncher : public EventWaiter {
  public:
    SubgraphWorkLauncher(SubgraphExecutionState *subgraph);
    static void launch_or_defer(SubgraphExecutionState *subgraph, Event wait_on);
    void launch();

    virtual void event_triggered(bool poisoned, TimeLimit work_until) override;
    virtual void print(std::ostream &os) const override;
    virtual Event get_finish_event(void) const override;

  private:
    SubgraphExecutionState *subgraph;
  };

  // SubgraphInstantiationCleanup waits for a subgraph's finish event
  // and then cleans up the SubgraphExecutionState resources.
  class SubgraphInstantiationCleanup : public EventWaiter {
  public:
    SubgraphInstantiationCleanup(SubgraphExecutionState *subgraph, UserEvent to_trigger);
    void cleanup();

    virtual void event_triggered(bool poisoned, TimeLimit work_until) override;
    virtual void print(std::ostream &os) const override;
    virtual Event get_finish_event(void) const override;

  private:
    SubgraphExecutionState *subgraph;
    UserEvent to_trigger;
  };

  // SubgraphResourceReaper is a background work item that processes
  // SubgraphInstantiationCleanup items asynchronously, freeing resources
  // allocated for subgraph instantiation after the subgraph has finished.
  class SubgraphResourceReaper : public BackgroundWorkItem {
  public:
    SubgraphResourceReaper();

    void enqueue_cleanup(SubgraphInstantiationCleanup *item);

    virtual bool do_work(TimeLimit work_until) override;

  private:
    Mutex mutex;
    std::queue<SubgraphInstantiationCleanup *> pending_cleanups;
  };

  // SubgraphExecutionState describes the state needed for a compiled
  // subgraph execution.
  struct SubgraphExecutionState {
    SubgraphExecutionState(SubgraphImpl *subgraph, const void *args, size_t arglen,
                           UserEvent finish_event);
    ~SubgraphExecutionState();

    // The subgraph being executed.
    SubgraphImpl *subgraph;

    // The arguments to the subgraph. This is a local copy
    // that can be modified by the interpolation process (which
    // will happen in future work).
    void *args;
    size_t arglen;

    // finish_counter tracks the amount of pending work launched by
    // this subgraph. Whoever decrements this counter to 0 (whether
    // a processor, background work item, CUDA stream notifier, etc.)
    // must trigger finish_event to wake up anyone who needs to know
    // when this subgraph is complete.
    atomic<int64_t> finish_counter;
    UserEvent finish_event;

    // The precondition array that contains the number of pending
    // preconditions for each operation in subgraph->compiled_subgraph_operations.
    atomic<int64_t> *preconditions;

    // A single array that contains the per-processor queues for subgraph execution.
    atomic<int64_t> *processor_queues;

    struct ProcessorLocalState {
      // Maintains the next available slot in the per-processor queue
      // to place ready operations. This slot is "zero-indexed", meaning
      // that it is local to the current processor only and is not a global
      // index into processor_queues.
      atomic<int64_t> queue_back;

      // Ensure processor-local state does not accidentally cause
      // false sharing between processor cache lines.
      char _cache_line_padding[64];
    };
    std::vector<ProcessorLocalState> processor_state;
  };

  // ProcSubgraphExecutor manages the logic of what a Processor should
  // actually do when executing components of a compiled subgraph.
  class ProcSubgraphExecutor {
  public:
    ProcSubgraphExecutor(Processor _proc, ThreadedTaskScheduler *_scheduler);
    ~ProcSubgraphExecutor();

    // This is the main point of interaction with the ThreadedTaskScheduler.
    // Executes a unit of subgraph work. Returns true if work was performed.
    bool execute_subgraph_work();

    // Enqueue work onto this subgraph executor. This method is thread-safe.
    void enqueue_subgraph(SubgraphExecutionState *subgraph);

  private:
    // Attempt to acquire a subgraph to execute. Returns true if
    // a new subgraph was acquired successfully.
    bool try_acquire_subgraph();
    // Reset resources from an acquired subgraph.
    void release_subgraph();

    // These context controllers should manage thread-local
    // state that should be set for the entirety of a subgraphs execution.
    void push_subgraph_execution_context();
    void pop_subgraph_execution_context();

    // Check if this executor has an active subgraph to execute.
    bool has_active_subgraph() const { return current_subgraph != nullptr; }

  private:
    RWLock pending_subgraphs_lock;
    std::queue<SubgraphExecutionState *> pending_subgraphs;
    SubgraphExecutionState *current_subgraph;
    Processor proc;
    int proc_index;
    int64_t queue_front;
    ThreadedTaskScheduler *scheduler;
  };

}; // namespace Realm

#endif
