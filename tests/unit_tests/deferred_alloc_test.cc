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

// Regression tests for the LocalManagedMemory deferred-allocation / pending-
// release state machine.  These exercise the "bad-path" branches in
// release_storage_deferrable / release_storage_immediate / reuse_storage_*
// where a destroy can't be applied to current_allocator immediately because
// attempt_release_reordering can't (yet) satisfy the oldest pending alloc.
//
// Prior to the fix, those branches queued the destroy in pending_releases
// (with the tag still live in current_allocator) but immediately recycled the
// inst slot via notify_deallocation -> recycle_instance.  A subsequent
// new_instance() call could then hand the same inst ID back out while the
// allocator was still tracking the old one, producing double-tracking that
// later surfaced as assertion failures in BasicRangeAllocator::deallocate
// (`missing_ok`) or split_range (`allocated.find(new_tags[i]) == end()`).

#include "realm/mem_impl.h"
#include "realm/inst_impl.h"
#include "realm/inst_layout.h"
#include "realm/event_impl.h"
#include "test_mock.h"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace Realm {
  extern RuntimeImpl *runtime_singleton;
}

using namespace Realm;

class DeferredAllocBadPathTest : public ::testing::Test {
protected:
  static constexpr size_t kMemSize = 100;

  void SetUp() override
  {
    runtime_ = std::make_unique<MockRuntimeImplWithEventFreeList>();
    runtime_->init(1);
    Realm::runtime_singleton = runtime_.get();

    Memory mem_id = ID::make_memory(/*owner=*/0, /*mem_idx=*/0).convert<Memory>();
    buffer_.resize(kMemSize);
    mem_ = std::make_unique<LocalCPUMemory>(
        runtime_.get(), mem_id, kMemSize, /*numa_node=*/0, Memory::SYSTEM_MEM,
        buffer_.data(), /*segment=*/nullptr, /*enable_ipc=*/false);
  }

  void TearDown() override
  {
    mem_.reset();
    Realm::runtime_singleton = nullptr;
    runtime_->finalize();
    runtime_.reset();
  }

  // Allocate a fresh inst from this memory's slot table and attach an opaque
  // layout of the requested size.  The inst's metadata is left STATE_INVALID;
  // notify_allocation() will mark it valid when (or if) the allocation
  // actually completes.
  RegionInstanceImpl *make_inst(size_t bytes, size_t alignment = 1)
  {
    RegionInstanceImpl *impl = mem_->new_instance(ProfilingRequestSet());
    impl->metadata.layout = new InstanceLayoutOpaque(bytes, alignment);
    impl->metadata.ext_resource = nullptr;
    impl->metadata.need_alloc_result = false;
    impl->metadata.need_notify_dealloc = false;
    return impl;
  }

  bool slot_on_free_list(RegionInstanceImpl *inst)
  {
    int inst_idx = ID(inst->me).instance_inst_idx();
    RWLock::AutoReaderLock al(mem_->local_instances.mutex);
    for(size_t x : mem_->local_instances.free_list)
      if(static_cast<int>(x) == inst_idx)
        return true;
    return false;
  }

  std::unique_ptr<MockRuntimeImplWithEventFreeList> runtime_;
  std::vector<char> buffer_;
  std::unique_ptr<LocalCPUMemory> mem_;
};

// Layout for this test:
//
//   memory: [0...100)
//   alloc A=30 at [0,30), B=50 at [30,80), C=20 at [80,100)  -> full.
//
// Step 1: schedule B's destroy with a precondition that hasn't fired - B is
// queued in pending_releases (is_ready=false).
//
// Step 2: request D=50.  In the future state (B released) D fits at [30,80),
// so allocate_storage_deferrable returns ALLOC_DEFERRED and D goes onto
// pending_allocs.
//
// Step 3: destroy A with an already-triggered precondition.  triggered=true,
// pending_allocs is non-empty, attempt_release_reordering can't satisfy D
// against release_allocator (only [0,30) free), so we hit the bad-path branch.
// A's PendingRelease is queued with is_ready=true and current_allocator still
// holds A's tag.
//
// Invariants the fix preserves:
//   - A's entry now carries deferred_dealloc_notify=true.
//   - A's inst slot is NOT yet on local_instances.free_list.
//   - A fresh new_instance() does not hand A's slot back out.
//
// Step 4: fire B's precondition (we bypass the event/DeferredDestroy plumbing
// by calling release_storage_immediate directly - that's what the deferred-
// destroy callback does when the precondition triggers).  The oldest-path
// catch-up drains B, finishes D, and then drains A.  Once A's tag is finally
// gone from current_allocator the deferred-notify fires exactly once and the
// slot becomes available for reuse.
TEST_F(DeferredAllocBadPathTest, ReorderFailDefersInstanceRecycle)
{
  RegionInstanceImpl *A = make_inst(30);
  RegionInstanceImpl *B = make_inst(50);
  RegionInstanceImpl *C = make_inst(20);
  RegionInstanceImpl *D = make_inst(50);

  // Step 0: fill memory with A, B, C.
  ASSERT_EQ(
      MemoryImpl::ALLOC_INSTANT_SUCCESS,
      mem_->allocate_storage_deferrable(A, /*need_alloc_result=*/false, Event::NO_EVENT));
  ASSERT_EQ(MemoryImpl::ALLOC_INSTANT_SUCCESS,
            mem_->allocate_storage_deferrable(B, false, Event::NO_EVENT));
  ASSERT_EQ(MemoryImpl::ALLOC_INSTANT_SUCCESS,
            mem_->allocate_storage_deferrable(C, false, Event::NO_EVENT));

  // Step 1: queue a deferred destroy for B - precondition has not triggered,
  // so B lands in pending_releases with is_ready=false.
  Event b_precondition = GenEventImpl::create_genevent()->current_event();
  mem_->release_storage_deferrable(B, b_precondition);
  ASSERT_EQ(1u, mem_->pending_releases.size());
  ASSERT_EQ(B, mem_->pending_releases.front().inst);
  ASSERT_FALSE(mem_->pending_releases.front().is_ready);

  // Step 2: request D=50.  Future state with B's release applied has [30,80)
  // free, so this is ALLOC_DEFERRED rather than ALLOC_INSTANT_FAILURE.
  ASSERT_EQ(MemoryImpl::ALLOC_DEFERRED,
            mem_->allocate_storage_deferrable(D, false, Event::NO_EVENT));
  ASSERT_EQ(1u, mem_->pending_allocs.size());
  ASSERT_EQ(D, mem_->pending_allocs.front().inst);

  // Step 3: destroy A with a triggered (NO_EVENT) precondition.  This is the
  // bad path: triggered + pending_allocs non-empty + reorder-can't-fit-D.
  mem_->release_storage_deferrable(A, Event::NO_EVENT);

  // A should have been queued behind B with is_ready=true and the new
  // deferred-dealloc-notify flag set.
  ASSERT_EQ(2u, mem_->pending_releases.size());
  const LocalManagedMemory::PendingRelease &a_entry = mem_->pending_releases.back();
  EXPECT_EQ(A, a_entry.inst);
  EXPECT_TRUE(a_entry.is_ready);
  EXPECT_TRUE(a_entry.deferred_dealloc_notify);

  // The key invariant the fix preserves: A's slot must NOT have been
  // recycled, because current_allocator still tracks A's tag.  Without the
  // fix, A would be on the free list here and the next new_instance() could
  // hand the same inst ID right back out.
  EXPECT_FALSE(slot_on_free_list(A));

  // Stronger check: a fresh new_instance() must not return A's slot.
  RegionInstanceImpl *E = mem_->new_instance(ProfilingRequestSet());
  EXPECT_NE(ID(A->me).instance_inst_idx(), ID(E->me).instance_inst_idx());

  // Step 4: fire B's precondition.  release_storage_deferrable registered
  // B->deferred_destroy as a waiter on b_precondition; triggering the event
  // walks that waiter list and dispatches release_storage_immediate(B, ...)
  // via DeferredDestroy::event_triggered.  Calling release_storage_immediate
  // directly here would also drain the allocator state, but it would leave
  // the stale waiter pointing at B->deferred_destroy in b_precondition - and
  // when the runtime tore down b_precondition's GenEventImpl after the test,
  // ~GenEventImpl would walk the waiter list and hit a use-after-free on
  // the already-freed RegionInstanceImpl.
  GenEventImpl::trigger(b_precondition, /*poisoned=*/false);

  // B's release drains as the oldest, satisfies D, and the catch-up loop
  // then drains A as the next consecutive ready entry.  Both queues should
  // be empty.
  EXPECT_TRUE(mem_->pending_releases.empty());
  EXPECT_TRUE(mem_->pending_allocs.empty());

  // D should now be allocated where B used to live.
  EXPECT_EQ(static_cast<size_t>(30), D->metadata.inst_offset);

  // And A's slot should finally be on the free list - the deferred
  // notify_deallocation() fired during the drain because A's tag is now
  // gone from current_allocator.
  EXPECT_TRUE(slot_on_free_list(A));
}
