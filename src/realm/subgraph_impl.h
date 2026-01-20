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

#include "realm/subgraph.h"
#include "realm/id.h"
#include "realm/event_impl.h"
#include "realm/cuda/cuda_module.h"
#include "realm/mutex.h"

namespace Realm {

  // Forward declarations.
  class LocalTaskProcessor;
  class XferDes;

  struct SubgraphScheduleEntry {
    SubgraphDefinition::OpKind op_kind;
    unsigned op_index;
    std::vector<std::pair<unsigned, int>> preconditions;
    unsigned first_interp, num_interps;
    unsigned intermediate_event_base, intermediate_event_count;
    bool is_final_event;
  };

  constexpr int64_t SUBGRAPH_EMPTY_QUEUE_ENTRY = -1;

   void
   do_interpolation_inline(const std::vector<SubgraphDefinition::Interpolation> &interpolations,
                           unsigned first_interp, unsigned num_interps,
                           SubgraphDefinition::Interpolation::TargetKind target_kind,
                           unsigned target_index,
                           const void *srcdata, size_t srclen,
                           void *dstdata, size_t dstlen);

  // a typed version for interpolating small values
  template <typename T>
  static T
  do_interpolation(const std::vector<SubgraphDefinition::Interpolation> &interpolations,
                   unsigned first_interp, unsigned num_interps,
                   SubgraphDefinition::Interpolation::TargetKind target_kind,
                   unsigned target_index, const void *srcdata, size_t srclen, T dstdata)
  {
    T val = dstdata;

    for(unsigned i = 0; i < num_interps; i++) {
      const SubgraphDefinition::Interpolation &it = interpolations[first_interp + i];
      if((it.target_kind != target_kind) || (it.target_index != target_index))
        continue;

      assert((it.offset + it.bytes) <= srclen);
      if(it.redop_id == 0) {
        // overwrite
        assert((it.target_offset + it.bytes) <= sizeof(T));
        memcpy(reinterpret_cast<char *>(&val) + it.target_offset,
               reinterpret_cast<const char *>(srcdata) + it.offset, it.bytes);
      } else {
        // TODO (rohany): Can't implement this here, including runtime_impl.h
        //  leads to a circular dependence.
        assert(false);
        // const ReductionOpUntyped *redop =
        //     get_runtime()->reduce_op_table.get(it.redop_id, 0);
        // assert((it.target_offset + redop->sizeof_lhs) <= sizeof(T));
        // (redop->cpu_apply_excl_fn)(reinterpret_cast<char *>(&val) + it.target_offset, 0,
        //                            reinterpret_cast<const char *>(srcdata) + it.offset, 0,
        //                            1 /*count*/, redop->userdata);
      }
    }
    return val;
  }

   // FlattenedProcMap and FlattenedProcTaskMap are essentially sparse matrix
   // and sparse tensor representations of mappings of
   // * proc -> vec<T> (per task)
   // * proc -> task -> vec<T>
   template<typename T>
   struct FlattenedProcMap {
     FlattenedProcMap() {}
     FlattenedProcMap(const std::vector<Processor>& all_procs, const std::map<Processor, std::vector<T>>& input) {
       uint64_t proc_count = 0;
       for (size_t i = 0; i < all_procs.size(); i++) {
         auto proc = all_procs[i];
         proc_offsets.push_back(proc_count);
         for (auto& it : input.at(proc)) {
           data.push_back(it);
           proc_count++;
         }
       }
       proc_offsets.push_back(proc_count);
     }
     void clear() {
       proc_offsets.clear();
       data.clear();
     }
     std::vector<uint64_t> proc_offsets;
     std::vector<T> data;
   };

   template<typename T>
   struct FlattenedProcTaskMap {
     FlattenedProcTaskMap() {}
     FlattenedProcTaskMap(const std::vector<Processor>& all_procs, const std::map<Processor, std::vector<std::vector<T>>>& input) {
       uint64_t proc_count = 0;
       uint64_t task_count = 0;
       for (size_t i = 0; i < all_procs.size(); i++) {
         auto proc = all_procs[i];
         proc_offsets.push_back(proc_count);
         for (auto& infos : input.at(proc)) {
           task_offsets.push_back(task_count);
           proc_count++;
           for (auto& info : infos) {
             data.push_back(info);
             task_count++;
           }
         }
       }
       proc_offsets.push_back(proc_count);
       task_offsets.push_back(task_count);
     }
     void clear() {
       proc_offsets.clear();
       task_offsets.clear();
       data.clear();
     }
     std::vector<uint64_t> proc_offsets;
     std::vector<uint64_t> task_offsets;
     std::vector<T> data;
   };

  template <typename T>
  struct FlattenedSparseMatrix {
    FlattenedSparseMatrix() {}
    FlattenedSparseMatrix(const std::vector<std::vector<T>>& input) {
      uint64_t count = 0;
      for (size_t i = 0; i < input.size(); i++) {
        offsets.push_back(count);
        for (auto& it : input[i]) {
          data.push_back(it);
          count++;
        }
      }
      offsets.push_back(count);
    }
    void clear() {
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
      void analyze_copy(SubgraphDefinition::CopyDesc& copy, std::vector<XferDes*>& result);
      // Methods for understanding the asynchronous affects tasks may have.
      // Since subgraphs have a static view of the source program, they can
      // take more advantage of this asynchrony than Realm can today.
      bool task_has_async_effects(SubgraphDefinition::TaskDesc& task);
      bool copy_has_any_async_effects(unsigned op_idx);
      bool copy_has_all_async_effects(unsigned op_idx);
      bool operation_has_async_effects(SubgraphDefinition::OpKind op_kind, unsigned op_idx);
      bool task_respects_async_effects(SubgraphDefinition::TaskDesc& src, SubgraphDefinition::TaskDesc& dst);
      bool operation_respects_async_effects(SubgraphDefinition::OpKind src_op_kind, unsigned src_op_idx,
                                            SubgraphDefinition::OpKind dst_op_kind, unsigned dst_op_idx);

      // Returns the number of "operations" that each bgwork item will
      // launch. Bgwork items (like copies) may launch multiple asynchronous
      // pieces of work, and all operations must complete before the bgwork
      // item is considered to be done. bgwork_async_operation_count is the
      // same, but returns the number of work items that launch asynchronous
      // work themselves.
      int32_t bgwork_operation_count(SubgraphDefinition::OpKind op_kind, unsigned op_idx);
      int32_t bgwork_async_operation_count(SubgraphDefinition::OpKind op_kind, unsigned op_idx);

  public:
    ID me;
    SubgraphImpl *next_free;
    SubgraphDefinition *defn;
    std::vector<SubgraphScheduleEntry> schedule;
    size_t num_intermediate_events, num_final_events, max_preconditions;

    DeferredDestroy deferred_destroy;

  public:
    bool opt = false;
    // These objects consist of the "static schedule" information
    // for subgraph replays.
    std::vector<Processor> all_procs;
    std::vector<LocalTaskProcessor*> all_proc_impls;
    std::vector<int32_t> async_finish_events;

    FlattenedProcMap<std::pair<SubgraphDefinition::OpKind, unsigned>> operations;
    struct OpMeta {
        bool is_final_event = false;
        bool is_async = false;
      };
    FlattenedProcMap<OpMeta> operation_meta;

    struct CompletionInfo {
      enum EdgeKind {
        STATIC_TO_STATIC,
        STATIC_TO_DYNAMIC,
        STATIC_TO_BGWORK,
        DYNAMIC_TO_STATIC,
        DYNAMIC_TO_BGWORK,
        BGWORK_TO_STATIC,
        BGWORK_TO_DYNAMIC,
        BGWORK_TO_BGWORK,
      };
      CompletionInfo(int32_t _proc, uint64_t _index, EdgeKind _kind) : proc(_proc), index(_index), kind(_kind) {}
      int32_t proc = -1;
      uint64_t index = UINT64_MAX;
      EdgeKind kind = STATIC_TO_STATIC;
    };

    FlattenedProcTaskMap<CompletionInfo> completion_infos;
    FlattenedProcMap<int32_t> original_preconditions;
    // Metadata for operations with asynchronous side effects.
    FlattenedProcTaskMap<CompletionInfo> async_outgoing_infos;
    FlattenedProcTaskMap<CompletionInfo> async_incoming_infos;

    // Collapsed interpolation metadata.
    FlattenedProcTaskMap<SubgraphDefinition::Interpolation> interpolations;

    // Initial queue status, prefilled with operations that don't
    // have any preconditions.
    std::vector<int64_t> initial_queue_state;
    std::vector<uint64_t> initial_queue_entry_count;

    class SubgraphWorkLauncher : public EventWaiter {
      public:
        SubgraphWorkLauncher(ProcSubgraphReplayState* state, SubgraphImpl* subgraph);
        void launch();
        virtual void event_triggered(bool poisoned, TimeLimit work_until);
        virtual void print(std::ostream& os) const;
        virtual Event get_finish_event(void) const;
      private:
        ProcSubgraphReplayState* state;
        SubgraphImpl* subgraph;
    };

    struct ExternalPreconditionMeta {
      std::vector<CompletionInfo> to_trigger;
    };
    std::vector<ExternalPreconditionMeta> external_precondition_info;

    // Metadata for the interaction of the dynamic and
    // static components of the subgraph.
    std::vector<ExternalPreconditionMeta> dynamic_to_static_triggers;
    std::vector<int32_t> static_to_dynamic_counts;

    // The number of bgwork items that are final events of the subgrpah.
    int32_t bgwork_finish_events = 0;
    // The schedule entries designated for background workers.
    std::vector<SubgraphScheduleEntry> bgwork_items;
    // Precondition counts for all bgwork items.
    std::vector<int32_t> bgwork_preconditions;
    // A vector of indices into bgwork_items of all background
    // work operations that are ready to run as soon as the
    // subgraph can start.
    std::vector<int64_t> bgwork_items_without_preconditions;
    FlattenedSparseMatrix<CompletionInfo> bgwork_postconditions;
    FlattenedSparseMatrix<CompletionInfo> bgwork_async_postconditions;
    FlattenedSparseMatrix<CompletionInfo> bgwork_async_preconditions;
    // Maintain a mapping of indices into the bgwork_items vector
    // of async tokens that belong to each processor. This is used
    // to return async tokens (like CUDA events) back to the per-processor
    // pools they may have been allocated from.
    std::map<LocalTaskProcessor*, std::vector<unsigned>> bgwork_async_event_procs;

    // The number of async events created per bgwork item. This is
    // necessary because some bgwork items may launch multiple disjoint
    // pieces of async work.
    std::vector<int64_t> bgwork_async_event_counts;

    // A vector of XD's for all copies that were planned during
    // subgraph compilation.
    FlattenedSparseMatrix<XferDes*> planned_copy_xds;

    // When concurrency_mode == INSTANTIATION_ORDER, the subgraph will
    // implicitly order instantiations of the subgraph by tracking
    // the completion event of the last instantiation.
    mutable Mutex instantiation_lock;
    mutable Event previous_instantiation_completion = Event::NO_EVENT;

    class ExternalPreconditionTriggerer : public EventWaiter {
      public:
        ExternalPreconditionTriggerer() : EventWaiter() {}
        ExternalPreconditionTriggerer(
          SubgraphImpl* _subgraph,
          ExternalPreconditionMeta* meta,
          atomic<int32_t>* preconditions,
          ProcSubgraphReplayState* all_proc_states
        );
        void trigger();
        virtual void event_triggered(bool poisoned, TimeLimit work_until);
        virtual void print(std::ostream& os) const;
        virtual Event get_finish_event(void) const;
      private:
        SubgraphImpl* subgraph = nullptr;
        ExternalPreconditionMeta* meta = nullptr;
        atomic<int32_t>* preconditions = nullptr;
        ProcSubgraphReplayState* all_proc_states = nullptr;
    };

    class AsyncGPUWorkTriggerer : public Cuda::GPUCompletionNotification {
      public:
        AsyncGPUWorkTriggerer(
          ProcSubgraphReplayState* all_proc_states,
          span<CompletionInfo> _infos,
          atomic<int32_t>* _preconditions,
          atomic<int32_t>* _final_ev_counter
        );
        void request_completed() override;
      private:
        ProcSubgraphReplayState* all_proc_states = nullptr;
        span<CompletionInfo> infos = {};
        atomic<int32_t>* preconditions = nullptr;
        atomic<int32_t>* final_ev_counter = nullptr;
    };

    class InstantiationCleanup : public EventWaiter {
      public:
        InstantiationCleanup(
          size_t num_procs,
          ProcSubgraphReplayState* state,
          void* args, atomic<int32_t>* preconds, atomic<int64_t>* queue,
          ExternalPreconditionTriggerer* external_preconds,
          ExternalPreconditionTriggerer* dynamic_preconds,
          void** async_operation_events,
          AsyncGPUWorkTriggerer* async_operation_event_triggerers,
          void* bgwork_preconds,
          void** async_bgwork_events,
          atomic<int32_t>* dynamic_precond_counters,
          UserEvent* dynamic_events,
          atomic<int32_t>* finish_counter
        );
        virtual void event_triggered(bool poisoned, TimeLimit work_until);
        virtual void print(std::ostream& os) const;
        virtual Event get_finish_event(void) const;
      private:
        size_t num_procs;
        ProcSubgraphReplayState* state;
        void* args;
        atomic<int32_t>* preconds;
        atomic<int64_t>* queue;
        ExternalPreconditionTriggerer* external_preconds;
        ExternalPreconditionTriggerer* dynamic_preconds;
        void** async_operation_events;
        AsyncGPUWorkTriggerer* async_operation_event_triggerers;
        void* bgwork_preconds;
        void** async_bgwork_events;
        atomic<int32_t>* dynamic_precond_counters;
        UserEvent* dynamic_events;
        atomic<int32_t>* finish_counter;
    };
  };

  struct ProcSubgraphReplayState {
    // Maintain a pointer to all the processor states.
    ProcSubgraphReplayState* all_proc_states = nullptr;

    // TODO (rohany): This has to be multiple indexes when we consider
    //  multiple mailboxes.
    int64_t next_op_index = 0;
    int32_t proc_index = -1;
    SubgraphImpl* subgraph = nullptr;

    // Interpolation argument sources.
    void* args = nullptr;
    size_t arglen = 0;

    // Only the last processor to complete will trigger
    // the finish event.
    atomic<int32_t>* finish_counter = nullptr;
    UserEvent finish_event = UserEvent::NO_USER_EVENT;

    // Store the preconditions array local to this instantiation
    // of the subgraph.
    atomic<int32_t>* preconditions = nullptr;
    // Precondition information for operations in
    // the dynamic portion of the subgraph.
    atomic<int32_t>* dynamic_preconditions = nullptr;
    UserEvent* dynamic_events = nullptr;

    // Precondition array for bgwork items.
    atomic<int32_t>* bgwork_preconditions = nullptr;

    // A queue of operation indexes to execute.
    atomic<int64_t>* queue = nullptr;
    // The queue slot that enqueuers should bump to write into.
    atomic<uint64_t> next_queue_slot;

    // Data structures for management of asynchrony.
    void** async_operation_events = nullptr;
    SubgraphImpl::AsyncGPUWorkTriggerer* async_operation_effect_triggerers = nullptr;

    // Array to manage background work items that launch asynchronous work.
    void** async_bgwork_events = nullptr;

    // A counter to manage the number of pending async items
    // that must complete before the finish event of this
    // processor can be triggered.
    bool has_pending_async_work = false;
    atomic<int32_t> pending_async_count;

    // Avoid false sharing of counters.
    char _cache_line_padding[64];
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

  class SubgraphTriggerContext {
  public:
    virtual ~SubgraphTriggerContext() {};
    virtual void enter() {};
    virtual void exit() {};
  };

  // Helper method to consolidate the logic of triggering the completion
  // of an operation during subgraph replay. The SubgraphTriggerContext
  // is a context that the caller can provide if certain state needs
  // to be set up before triggering an event.
  void trigger_subgraph_operation_completion(
    ProcSubgraphReplayState* all_proc_states,
    const SubgraphImpl::CompletionInfo& info,
    bool incr_counter,
    SubgraphTriggerContext* ctx
  );

  void launch_async_bgwork_item(ProcSubgraphReplayState* all_proc_states, unsigned index);
  void maybe_trigger_subgraph_final_completion_event(ProcSubgraphReplayState& states);

}; // namespace Realm

#endif
