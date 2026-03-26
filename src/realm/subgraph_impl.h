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

namespace Realm {

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
      void defer(SubgraphImpl *_subgraph, Event wait_on);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

    protected:
      SubgraphImpl *subgraph;
    };

  protected:
    // TODO (rohany): Make sure to handle clearing / initializing
    //  some of these fields when a SubgraphImpl gets reused.

    // Fields populated by the compilation step in the compiled
    // execution mode of subgraphs.

    std::vector<Processor> subgraph_processors;
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
      EdgeInfo(uint64_t _op_index)
        : op_index(_op_index)
      {}
      uint64_t op_index;
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

    // TODO (rohany): This is work for future PRs.
    // TODO (rohany): Asynchronous edges.
    // TODO (rohany): Extra stuff here like background work? Connections between
    //  the interpreted and compiled parts of the subgraph?
    // TODO (rohany): Instantiation lock.

    // TODO (rohany): This is work for the _current_ pull request.
    // TODO (rohany): The concrete replay state and the subgraph executor object.
    // TODO (rohany): The resource cleanup infrastructure.

  public:
    ID me;
    SubgraphImpl *next_free;
    SubgraphDefinition *defn;
    // TODO (rohany): Rename this later to be "interpreted_schedule" or something.
    std::vector<SubgraphScheduleEntry> schedule;
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

  // TODO (rohany): Comment ...
  // TODO (rohany): In the original design, there was a separate state object
  //  per-processor. Looking back, I don't think this is completely necessary
  //  and we can share a good amount of the state and have just a small piece
  //  of it be local to each processor.
  // TODO (rohany): Set up the initialization of these objects to happen in a constructor.
  struct SubgraphExecutionState {
    // TODO (rohany): Move this definition to the .cc file.
    SubgraphExecutionState(SubgraphImpl *subgraph, void *args, size_t arglen,
                           atomic<int64_t> *finish_counter, UserEvent finish_event,
                           atomic<int64_t> *preconditions,
                           atomic<int64_t> *processor_queues)
      : subgraph(subgraph)
      , args(args)
      , arglen(arglen)
      , finish_counter(finish_counter)
      , finish_event(finish_event)
      , preconditions(preconditions)
      , processor_queues(processor_queues)
    {}

    // TODO (rohany): Move this definition to the .cc file.
    ~SubgraphExecutionState()
    {
      // TODO (rohany): Implement this ...
      assert(false);
    }

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
    atomic<int64_t> *finish_counter;
    UserEvent finish_event;

    // The precondition array that contains the number of pending
    // preconditions for each operation in subgraph->compiled_subgraph_operations.
    atomic<int64_t> *preconditions;

    // A single array that contains the per-processor queues for subgraph execution.
    atomic<int64_t> *processor_queues;

    // TODO (rohany): This kind of setup will require us to put the "processor index"
    //  somewhere else when setting up the SubgraphExecutor logic.
    struct ProcessorLocalState {
      // Maintains the next available slot in the per-processor queue
      // to place ready operations.
      atomic<int64_t> next_queue_slot;

      // Ensure processor-local state does not accidentally cause
      // false sharing between processor cache lines.
      char _cache_line_padding[64];
    };
    ProcessorLocalState *processor_state;
  };

  // TODO (rohany): Not sure yet what this is going to look like.
  class SubgraphExecutor {
  public:
    SubgraphExecutor();
    ~SubgraphExecutor();

    bool try_acquire_subgraph(SubgraphImpl *subgraph);
    void release_subgraph(SubgraphImpl *subgraph);
  };

  // TODO (rohany): I need to look at the details of this Operation class to
  //  see if this is what I actually want or not. The alternative is to keep
  //  doing the EventWaiter thing that I have going on right now. I'm not entirely
  //  sure that this makes sense right now.
  // class ExecuteCompiledSubgraphOperation : public Operation {
  // public:
  //   ExecuteCompiledSubgraphOperation(SubgraphImpl *subgraph);
  //   ~ExecuteCompiledSubgraphOperation();

  //   virtual bool mark_ready(void);
  //   virtual bool mark_started(void);
  //   virtual void mark_finished(bool successful);
  //   virtual void mark_terminated(int error_code, const ByteArray &details);
  // };

}; // namespace Realm

#endif
