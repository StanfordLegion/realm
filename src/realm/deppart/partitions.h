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

// index space partitioning for Realm

#ifndef REALM_PARTITIONS_H
#define REALM_PARTITIONS_H

#include "realm/indexspace.h"
#include "realm/sparsity.h"
#include "realm/activemsg.h"
#include "realm/id.h"
#include "realm/operation.h"
#include "realm/threads.h"
#include "realm/cmdline.h"
#include "realm/pri_queue.h"
#include "realm/nodeset.h"
#include "realm/interval_tree.h"
#include "realm/dynamic_templates.h"
#include "realm/deppart/sparsity_impl.h"
#include "realm/deppart/inst_helper.h"
#include "realm/bgwork.h"

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace Realm {

  class PartitioningMicroOp;
  class PartitioningOperation;

  template <typename T>
  constexpr std::string_view type_name() {
  #if defined(__clang__)
      std::string_view p = __PRETTY_FUNCTION__;
      return {p.data() + 34, p.size() - 34 - 1};
  #elif defined(__GNUC__)
      std::string_view p = __PRETTY_FUNCTION__;
      return {p.data() + 49, p.size() - 49 - 1};
  #elif defined(_MSC_VER)
      std::string_view p = __FUNCSIG__;
      return {p.data() + 84, p.size() - 84 - 7};
  #else
      return "unknown";
  #endif
  }

  template<typename T>
  struct HiFlag {
    T hi;
    uint8_t head;
  };

  struct DeltaFlag {
    int32_t delta;
    uint8_t head;
  };

  // Data representations for GPU micro-ops
  // src idx tracks which subspace each rect/point
  // belongs to and allows multiple subspaces to be
  // computed together in a micro-op
  template<int N, typename T>
  struct RectDesc {
    Rect<N,T> rect;
    size_t src_idx;
  };

  template<int N, typename T>
  struct PointDesc {
    Point<N,T> point;
    size_t src_idx;
  };

  // Combines one or multiple index spaces into a single struct
  // If multiple, offsets tracks transitions between spaces
  template<int N, typename T>
  struct collapsed_space {
    SparsityMapEntry<N, T>* entries_buffer;
    size_t num_entries;
    size_t* offsets;
    size_t num_children;
    Rect<N, T> bounds;
  };

  // Stores everything necessary to query a BVH
  // Used with GPUMicroOp<N, T>::build_bvh
  template<int N, typename T>
  struct BVH {
    int root;
    size_t num_leaves;
    Rect<N,T>* boxes;
    uint64_t* indices;
    size_t* labels;
    int* childLeft;
    int* childRight;
  };

  struct arena_oom : std::bad_alloc {
    const char* what() const noexcept override { return "arena_oom"; }
  };

  class Arena {
  public:
    using byte = std::byte;

    Arena() noexcept : base_(nullptr), cap_(0), parity_(false), left_(0), right_(0), base_left_(0), base_right_(0) {}
    Arena(void* buffer, size_t bytes) noexcept
      : base_(reinterpret_cast<byte*>(buffer)), cap_(bytes), parity_(false), left_(0), right_(0), base_left_(0), base_right_(0) {}

    size_t capacity() const noexcept { return cap_; }
    size_t used() const noexcept { return left_ + right_; }

    size_t mark() const noexcept {
      return parity_ ? right_ : left_;
    }

    void rollback(size_t mark) noexcept {
      if (parity_) {
        right_ = mark;
      } else {
        left_ = mark;
      }
    }

    template <typename T>
    T* alloc(size_t count = 1) {
      try {
        if (parity_) {
          return alloc_right<T>(count);
        } else {
          return alloc_left<T>(count);
        }
      } catch (arena_oom&) {
        std::cout << "Arena OOM: requested " << count << " of " << type_name<T>()
                  << " capacity " << cap_ << " bytes, "
                  << " used " << used() << " bytes, "
                  << " left " << (cap_ - left_ - right_) << " bytes.\n";
        throw arena_oom{};
      }
    }

    void flip_parity(void) noexcept {
      if (parity_) {
        // switching from right to left
        left_ = base_left_;
      } else {
        // switching from left to right
        right_ = base_right_;
      }
      parity_ = !parity_;
    }

    void commit(bool parity) noexcept {
      if (parity) {
        base_right_ = right_;
      } else {
        base_left_ = left_;
      }
    }

    void reset(bool parity) noexcept {
      if (parity) {
        base_right_ = 0;
        right_ = 0;
      } else {
        base_left_ = 0;
        left_ = 0;
      }
    }

    bool get_parity(void) const noexcept {
      return parity_;
    }

    void start(void) noexcept {
      left_ = base_left_;
      right_ = base_right_;
      parity_ = false;
    }

  private:

    void* alloc_left_bytes(size_t bytes, size_t align = alignof(std::max_align_t)) {
      const size_t aligned = align_up(left_, align);
      if (aligned + bytes + right_ > cap_) throw arena_oom{};
      void* p = base_ + aligned;
      left_ = aligned + bytes;
      return p;
    }

    void* alloc_right_bytes(size_t bytes, size_t align = alignof(std::max_align_t)) {
      if (bytes + right_ > cap_) throw arena_oom{};
      const size_t aligned = align_down(cap_ - right_ - bytes, align);
      if (aligned < left_) throw arena_oom{};
      void *p = base_ + aligned;
      right_ = cap_ - aligned;
      return p;
    }

    template <typename T>
    T* alloc_left(size_t count = 1) {
      static_assert(!std::is_void_v<T>, "alloc<void> is invalid");
      return reinterpret_cast<T*>(alloc_left_bytes(sizeof(T) * count, alignof(T)));
    }

    template <typename T>
    T* alloc_right(size_t count = 1) {
      static_assert(!std::is_void_v<T>, "alloc<void> is invalid");
      return reinterpret_cast<T*>(alloc_right_bytes(sizeof(T) * count, alignof(T)));
    }

    static size_t align_up(size_t x, size_t a) noexcept {
      return (x + (a - 1)) & ~(a - 1);
    }

    static size_t align_down(size_t x, size_t a) noexcept {
      return x & ~(a - 1);
    }

    byte* base_;
    size_t cap_;
    bool parity_;
    size_t left_;
    size_t right_;
    size_t base_left_;
    size_t base_right_;
  };

  template <int N, typename T>
  class OverlapTester {
  public:
    OverlapTester(void);
    ~OverlapTester(void);

    void add_index_space(int label, const IndexSpace<N,T>& space, bool use_approx = true);

    void construct(void);

    void test_overlap(const Rect<N,T>* rects, size_t count, std::set<int>& overlaps);
    void test_overlap(const IndexSpace<N,T>& space, std::set<int>& overlaps, bool approx);
    void test_overlap(const SparsityMapImpl<N,T> *sparsity, std::set<int>& overlaps, bool approx);

  protected:
    std::vector<int> labels;
    std::vector<IndexSpace<N,T> > spaces;
    std::vector<bool> approxs;
  };

  template <typename T>
  class OverlapTester<1,T> {
  public:
    OverlapTester(void);
    ~OverlapTester(void);

    void add_index_space(int label, const IndexSpace<1,T>& space, bool use_approx = true);

    void construct(void);

    void test_overlap(const Rect<1,T>* rects, size_t count, std::set<int>& overlaps);
    void test_overlap(const IndexSpace<1,T>& space, std::set<int>& overlaps, bool approx);
    void test_overlap(const SparsityMapImpl<1,T> *sparsity, std::set<int>& overlaps, bool approx);

  protected:
    IntervalTree<T,int> interval_tree;
  };


  /////////////////////////////////////////////////////////////////////////

  class AsyncMicroOp : public Operation::AsyncWorkItem {
  public:
    AsyncMicroOp(Operation *_op, PartitioningMicroOp *_uop);
    ~AsyncMicroOp();
    
    virtual void request_cancellation(void);
      
    virtual void print(std::ostream& os) const;

  protected:
    PartitioningMicroOp *uop;
  };

  class PartitioningMicroOp {
  public:
    PartitioningMicroOp(void);
    virtual ~PartitioningMicroOp(void);

    virtual void execute(void) = 0;

    void mark_started(void);
    void mark_finished(void);

    template <int N, typename T>
    void sparsity_map_ready(SparsityMapImpl<N,T> *sparsity, bool precise);

    static RegionInstance realm_malloc(size_t size, Memory location = Memory::NO_MEMORY);

    IntrusiveListLink<PartitioningMicroOp> uop_link;
    REALM_PMTA_DEFN(PartitioningMicroOp,IntrusiveListLink<PartitioningMicroOp>,uop_link);
    typedef IntrusiveList<PartitioningMicroOp, REALM_PMTA_USE(PartitioningMicroOp,uop_link), DummyLock> MicroOpList;

  protected:
    PartitioningMicroOp(NodeID _requestor, AsyncMicroOp *_async_microop);

    void finish_dispatch(PartitioningOperation *op, bool inline_ok);

    atomic<int> wait_count;  // how many sparsity maps are we still waiting for?
    NodeID requestor;
    AsyncMicroOp *async_microop;

    // helper code to ship a microop to another node
    template <typename T>
    static void forward_microop(NodeID target,
				PartitioningOperation *op, T *microop);
  };

  template <int N, typename T>
  class ComputeOverlapMicroOp : public PartitioningMicroOp {
  public:
    // tied to the ImageOperation * - cannot be moved around the system
    ComputeOverlapMicroOp(PartitioningOperation *_op);
    virtual ~ComputeOverlapMicroOp(void);

    void add_input_space(const IndexSpace<N,T>& input_space);
    void add_extra_dependency(const IndexSpace<N,T>& dep_space);

    virtual void execute(void);

    void dispatch(PartitioningOperation *op, bool inline_ok);

  protected:
    PartitioningOperation *op;
    std::vector<IndexSpace<N,T> > input_spaces;
    std::vector<SparsityMapImpl<N,T> *> extra_deps;
  };

  //The parent class for all GPU partitioning micro-ops. Provides output utility functions

  template<int N, typename T>
  class GPUMicroOp : public PartitioningMicroOp {
  public:
    GPUMicroOp(void) = default;
    virtual ~GPUMicroOp(void) = default;

    virtual void execute(void) = 0;

    template <typename space_t>
    static void collapse_multi_space(const std::vector<space_t>& field_data, collapsed_space<N, T> &out_space, Arena &my_arena, cudaStream_t stream);

    static void collapse_parent_space(const IndexSpace<N, T>& parent_space, collapsed_space<N, T> &out_space, Arena &my_arena, cudaStream_t stream);

    static void build_bvh(const collapsed_space<N, T> &space, BVH<N, T> &bvh, Arena &my_arena, cudaStream_t stream);

    template <typename out_t>
    static void construct_input_rectlist(const collapsed_space<N, T> &lhs, const collapsed_space<N, T> &rhs, out_t* &d_valid_rects, size_t& out_size, uint32_t* counters, uint32_t* out_offsets, Arena &my_arena, cudaStream_t stream);

    template <typename out_t>
    static void volume_prefix_sum(const out_t* d_rects, size_t total_rects, size_t* &d_prefix_rects, size_t& num_pts, Arena &my_arena, cudaStream_t stream);

    template<typename Container, typename IndexFn, typename MapFn>
    void complete_pipeline(PointDesc<N, T>* d_points, size_t total_pts, RectDesc<N, T>* &d_out_rects, size_t &out_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap);

    template<typename Container, typename IndexFn, typename MapFn>
    void complete_rect_pipeline(RectDesc<N, T>* d_rects, size_t total_rects, RectDesc<N, T>* &d_out_rects, size_t &out_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap);

    template<typename Container, typename IndexFn, typename MapFn>
    void complete1d_pipeline(RectDesc<N, T>* d_rects, size_t total_rects, RectDesc<N, T>* &d_out_rects, size_t &out_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap);

    template<typename Container, typename IndexFn, typename MapFn>
    void send_output(RectDesc<N, T>* d_rects, size_t total_rects, Arena &my_arena, const Container& ctr, IndexFn getIndex, MapFn getMap);

    bool exclusive = false;

  };

  ////////////////////////////////////////
  //
  
  class PartitioningOperation : public Operation {
  public:
    PartitioningOperation(const ProfilingRequestSet &reqs,
			  GenEventImpl *_finish_event,
			  EventImpl::gen_t _finish_gen);

    virtual void execute(void) = 0;

    // the type of 'tester' depends on which operation it is, so erase the type here...
    virtual void set_overlap_tester(void *tester);

    void launch(Event wait_for);

    // some partitioning operations are handled inline for simple cases
    // these cases must still supply all the requested profiling responses
    static void do_inline_profiling(const ProfilingRequestSet &reqs,
				    long long inline_start_time);

    IntrusiveListLink<PartitioningOperation> op_link;
    REALM_PMTA_DEFN(PartitioningOperation,IntrusiveListLink<PartitioningOperation>,op_link);
    typedef IntrusiveList<PartitioningOperation, REALM_PMTA_USE(PartitioningOperation,op_link), DummyLock> OpList;

    class DeferredLaunch : public EventWaiter {
    public:
      void defer(PartitioningOperation *_op, Event wait_on);

      virtual void event_triggered(bool poisoned, TimeLimit work_until);
      virtual void print(std::ostream& os) const;
      virtual Event get_finish_event(void) const;

    protected:
      PartitioningOperation *op;
    };
    DeferredLaunch deferred_launch;
  };


  ////////////////////////////////////////
  //

  class PartitioningOpQueue : public BackgroundWorkItem {
  public:
    PartitioningOpQueue(CoreReservation *_rsrv,
			BackgroundWorkManager *_bgwork);
    virtual ~PartitioningOpQueue(void);

    static void configure_from_cmdline(std::vector<std::string>& cmdline);
    static void start_worker_threads(CoreReservationSet& crs,
				     BackgroundWorkManager *_bgwork);
    static void stop_worker_threads(void);

    void enqueue_partitioning_operation(PartitioningOperation *op);
    void enqueue_partitioning_microop(PartitioningMicroOp *uop);

    void worker_thread_loop(void);

    // called by BackgroundWorkers
    bool do_work(TimeLimit work_until);

  protected:
    atomic<bool> shutdown_flag;
    CoreReservation *rsrv;
    PartitioningOperation::OpList op_list;
    PartitioningMicroOp::MicroOpList uop_list;
    Mutex mutex;
    Mutex::CondVar condvar;
    std::vector<Thread *> workers;
    bool work_advertised;
  };


  ////////////////////////////////////////
  //
  // active messages


  template <typename T>
  struct RemoteMicroOpMessage {
    PartitioningOperation *operation;
    AsyncMicroOp *async_microop;

    static void handle_message(NodeID sender,
			       const RemoteMicroOpMessage<T> &msg,
			       const void *data, size_t datalen)
    {
      Serialization::FixedBufferDeserializer fbd(data, datalen);
      T *uop = new T(sender, msg.async_microop, fbd);
      uop->dispatch(msg.operation, false /*not ok to run in this thread*/);
    }
  };


  struct RemoteMicroOpCompleteMessage {
    AsyncMicroOp *async_microop;

    static void handle_message(NodeID sender,
			       const RemoteMicroOpCompleteMessage &msg,
			       const void *data, size_t datalen);

    static ActiveMessageHandlerReg<RemoteMicroOpCompleteMessage> areg;
  };


};

#include "realm/deppart/partitions.inl"

#endif // REALM_PARTITIONS_H

