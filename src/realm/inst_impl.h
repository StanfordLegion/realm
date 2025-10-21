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

// RegionInstance implementations for Realm

#ifndef REALM_INST_IMPL_H
#define REALM_INST_IMPL_H

#include "realm/instance.h"
#include "realm/id.h"
#include "realm/inst_layout.h"

#include "realm/mutex.h"

#include "realm/rsrv_impl.h"
#include "realm/metadata.h"
#include "realm/event_impl.h"
#include "realm/profiling.h"
#include "realm/mem_impl.h"
#include "realm/repl_heap.h"

namespace Realm {

  class CompiledInstanceLayout : public PieceLookup::CompiledProgram {
  public:
    CompiledInstanceLayout(ReplicatedHeap *_repl_heap);
    ~CompiledInstanceLayout();

    virtual void *allocate_memory(size_t bytes);

    virtual void commit_updates();

    void reset();

    void *program_base{nullptr};
    size_t program_size{0};
    ReplicatedHeap *repl_heap{nullptr};
  };

  class RegionInstanceImpl {
  public:
    // RegionInstanceImpl creation/deletion is handled by MemoryImpl
    RegionInstanceImpl(const RuntimeImpl *_runtime_impl, RegionInstance _me,
                       MemoryImpl *_memory);
    ~RegionInstanceImpl(void);

    class DeferredCreate : public EventWaiter {
    public:
      void defer(RegionInstanceImpl *_inst, MemoryImpl *_mem, bool _need_alloc_result,
                 Event wait_on);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

    protected:
      RegionInstanceImpl *inst;
      MemoryImpl *mem;
      bool need_alloc_result;
    };
    DeferredCreate deferred_create;

    class DeferredDestroy : public EventWaiter {
    public:
      void defer(RegionInstanceImpl *_inst, MemoryImpl *_mem, Event wait_on);
      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream &os) const;
      virtual Event get_finish_event(void) const;

    protected:
      RegionInstanceImpl *inst;
      MemoryImpl *mem;
    };
    DeferredDestroy deferred_destroy;

    // Some help for when a deferred redistrict is performed on
    // an instance that hasn't even been allocated yet
    std::vector<RegionInstanceImpl *> deferred_redistrict;

  public:
    // entry point for both create_instance and create_external_instance
    static Event create_instance(RegionInstance &inst, MemoryImpl *memory,
                                 InstanceLayoutGeneric *ilg,
                                 const ExternalInstanceResource *res,
                                 const ProfilingRequestSet &prs, Event wait_on);

    /// @brief Release this instance and the resources it holds after \p wait_on has been
    /// triggered
    /// @param wait_on precondition event that will defer the actual release until
    /// triggered
    void release(Event wait_on);

    Event redistrict(RegionInstance *instances, const InstanceLayoutGeneric **layouts,
                     size_t num_layouts, const ProfilingRequestSet *prs,
                     Event wait_on = Event::NO_EVENT);

    // the life cycle of an instance is defined in part by when the
    //  allocation and deallocation of storage occurs, but that is managed
    //  by the memory, which uses these callbacks to notify us
    void notify_allocation(MemoryImpl::AllocationResult result, size_t offset,
                           TimeLimit work_until);
    void notify_deallocation(void);

    bool get_strided_parameters(void *&base, size_t &stride, off_t field_offset) const;

    Event fetch_metadata(Processor target);

    Event request_metadata(void)
    {
      return metadata.request_data(int(ID(me).instance_creator_node()), me.id);
    }

    // ensures metadata is available on the specified node
    Event prefetch_metadata(NodeID target);

    // called once storage has been released and all remote metadata is invalidated
    void recycle_instance(void);

    template <int N, typename T>
    const PieceLookup::Instruction *
    get_lookup_program(FieldID field_id, unsigned allowed_mask, uintptr_t &field_offset);

    template <int N, typename T>
    const PieceLookup::Instruction *
    get_lookup_program(FieldID field_id, const Rect<N, T> &subrect, unsigned allowed_mask,
                       size_t &field_offset);

    void read_untyped(size_t offset, void *data, size_t datalen) const;

    void write_untyped(size_t offset, const void *data, size_t datalen);

    void reduce_apply_untyped(size_t offset, ReductionOpID redop_id, const void *data,
                              size_t datalen, bool exclusive /*= false*/);

    void reduce_fold_untyped(size_t offset, ReductionOpID redop_id, const void *data,
                             size_t datalen, bool exclusive /*= false*/);

    void *pointer_untyped(size_t offset, size_t datalen);

  public: // protected:
    void send_metadata(const NodeSet &early_reqs);

    friend class RegionInstance;

    RegionInstance me;
    // not part of metadata because it's determined from ID alone
    MemoryImpl *mem_impl = nullptr;
    // Profiling info only needed on creation node
    ProfilingRequestSet requests;
    ProfilingMeasurementCollection measurements;
    ProfilingMeasurements::InstanceTimeline timeline;

    // several special values exist for the 'inst_offset' field below
    static constexpr size_t INSTOFFSET_UNALLOCATED = size_t(-1);
    static constexpr size_t INSTOFFSET_FAILED = size_t(-2);
    static constexpr size_t INSTOFFSET_DELAYEDALLOC = size_t(-3);
    static constexpr size_t INSTOFFSET_DELAYEDDESTROY = size_t(-4);
    static constexpr size_t INSTOFFSET_DELAYEDREDISTRICT = size_t(-5);
    static constexpr size_t INSTOFFSET_MAXVALID = size_t(-6);
    class Metadata : public MetadataBase {
    public:
      Metadata(ReplicatedHeap *repl_heap);

      void *serialize(size_t &out_size) const;

      template <typename T>
      void serialize_msg(T &s) const
      {
        bool ok = ((s << inst_offset) && (s << *layout));
        assert(ok);
      }

      void deserialize(const void *in_data, size_t in_size);

    protected:
      virtual void do_invalidate(void);

    public:
      size_t inst_offset;
      Event ready_event;
      bool need_alloc_result, need_notify_dealloc;
      InstanceLayoutGeneric *layout;
      ExternalInstanceResource *ext_resource;
      MemSpecificInfo *mem_specific; // a place for memory's to hang info
      CompiledInstanceLayout lookup_program;

      template <typename T>
      T *find_mem_specific()
      {
        MemSpecificInfo *info = mem_specific;
        while(info) {
          T *downcast = dynamic_cast<T *>(info);
          if(downcast)
            return downcast;
          info = info->next;
        }
        return 0;
      }

      template <typename T>
      const T *find_mem_specific() const
      {
        const MemSpecificInfo *info = mem_specific;
        while(info) {
          const T *downcast = dynamic_cast<const T *>(info);
          if(downcast)
            return downcast;
          info = info->next;
        }
        return 0;
      }

      void add_mem_specific(MemSpecificInfo *info);
    };

    // used for atomic access to metadata
    Mutex mutex;
    Metadata metadata;
    // cache of prefetch events for other nodes
    std::map<NodeID, Event> prefetch_events;

    // used for serialized application access to contents of instance
    ReservationImpl lock;

  private:
    const RuntimeImpl *runtime_impl{nullptr};
  };

  // active messages

  struct InstanceMetadataPrefetchRequest {
    RegionInstance inst;
    Event valid_event;

    static void handle_message(NodeID sender, const InstanceMetadataPrefetchRequest &msg,
                               const void *data, size_t datalen, TimeLimit work_until);
  };

}; // namespace Realm

#endif // ifndef REALM_INST_IMPL_H
