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

namespace Realm {

  // Forward declarations.
  class LocalTaskProcessor;

  struct SubgraphScheduleEntry {
    SubgraphDefinition::OpKind op_kind;
    unsigned op_index;
    std::vector<std::pair<unsigned, int>> preconditions;
    unsigned first_interp, num_interps;
    unsigned intermediate_event_base, intermediate_event_count;
    bool is_final_event;
  };

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

   // Methods for understanding the asynchronous affects tasks may have.
   // Since subgraphs have a static view of the source program, they can
   // take more advantage of this asynchrony than Realm can today.
   bool task_has_async_effects(SubgraphDefinition::TaskDesc& task);
   bool operation_has_async_effects(SubgraphDefinition* defn, SubgraphDefinition::OpKind op_kind, unsigned op_idx);

   bool task_respects_async_effects(SubgraphDefinition::TaskDesc& src, SubgraphDefinition::TaskDesc& dst);
   bool operation_respects_async_effects(SubgraphDefinition* defn,
                                         SubgraphDefinition::OpKind src_op_kind, unsigned src_op_idx,
                                         SubgraphDefinition::OpKind dst_op_kind, unsigned dst_op_idx);

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
     std::vector<uint64_t> proc_offsets;
     std::vector<uint64_t> task_offsets;
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

  public:
    ID me;
    SubgraphImpl *next_free;
    SubgraphDefinition *defn;
    std::vector<SubgraphScheduleEntry> schedule;
    size_t num_intermediate_events, num_final_events, max_preconditions;

    DeferredDestroy deferred_destroy;

  public:
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
      CompletionInfo(int32_t _proc, uint64_t _index) : proc(_proc), index(_index) {}
      int32_t proc = -1;
      uint64_t index = UINT64_MAX;
    };

    FlattenedProcTaskMap<CompletionInfo> completion_infos;
    FlattenedProcMap<int32_t> original_preconditions;
    // Metadata for operations with asynchronous side effects.
    FlattenedProcTaskMap<CompletionInfo> async_outgoing_infos;
    FlattenedProcTaskMap<CompletionInfo> async_incoming_infos;

    // Collapsed interpolation metadata.
    FlattenedProcTaskMap<SubgraphDefinition::Interpolation> interpolations;

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

    class ExternalPreconditionTriggerer : public EventWaiter {
      public:
        ExternalPreconditionTriggerer() : EventWaiter() {}
        ExternalPreconditionTriggerer(SubgraphImpl* _subgraph, ExternalPreconditionMeta* meta, atomic<int32_t>* preconditions);
        void trigger();
        virtual void event_triggered(bool poisoned, TimeLimit work_until);
        virtual void print(std::ostream& os) const;
        virtual Event get_finish_event(void) const;
      private:
        SubgraphImpl* subgraph = nullptr;
        ExternalPreconditionMeta* meta = nullptr;
        atomic<int32_t>* preconditions = nullptr;
    };

    class AsyncGPUWorkTriggerer : public Cuda::GPUCompletionNotification {
      public:
        AsyncGPUWorkTriggerer(
          SubgraphImpl* _subgraph,
          span<CompletionInfo> _infos,
          atomic<int32_t>* _preconditions,
          atomic<int32_t>* _final_ev_counter,
          UserEvent _final_event);
        void request_completed() override;
      private:
        SubgraphImpl* subgraph = nullptr;
        span<CompletionInfo> infos = {};
        atomic<int32_t>* preconditions = nullptr;
        atomic<int32_t>* final_ev_counter = nullptr;
        UserEvent final_event = UserEvent::NO_USER_EVENT;
    };

    class InstantiationCleanup : public EventWaiter {
      public:
        InstantiationCleanup(
          size_t num_procs,
          ProcSubgraphReplayState* state,
          void* args, atomic<int32_t>* preconds,
          ExternalPreconditionTriggerer* external_preconds,
          void** async_operation_events,
          AsyncGPUWorkTriggerer* async_operation_event_triggerers
        );
        virtual void event_triggered(bool poisoned, TimeLimit work_until);
        virtual void print(std::ostream& os) const;
        virtual Event get_finish_event(void) const;
      private:
        size_t num_procs;
        ProcSubgraphReplayState* state;
        void* args;
        atomic<int32_t>* preconds;
        ExternalPreconditionTriggerer* external_preconds;
        void** async_operation_events;
        AsyncGPUWorkTriggerer* async_operation_event_triggerers;
    };
  };

  struct ProcSubgraphReplayState {
    // TODO (rohany): This has to be multiple indexes when we consider
    //  multiple mailboxes.
    int64_t next_op_index = 0;
    int32_t proc_index = -1;
    SubgraphImpl* subgraph = nullptr;

    // Interpolation argument sources.
    void* args = nullptr;
    size_t arglen = 0;

    // Each processor is responsible for triggering a final
    // event to let the runtime know that its contribution
    // to the subgraph execution is done.
    UserEvent finish_event = UserEvent::NO_USER_EVENT;

    // Store the preconditions array local to this instantiation
    // of the subgraph.
    atomic<int32_t>* preconditions = nullptr;

    // Data structures for management of asynchrony.
    void** async_operation_events = nullptr;
    SubgraphImpl::AsyncGPUWorkTriggerer* async_operation_effect_triggerers = nullptr;

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

}; // namespace Realm

#endif
