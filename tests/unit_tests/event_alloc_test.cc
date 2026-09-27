/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
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

// Verifies that constructing a GenEventImpl performs no heap allocations.
//
// Realm materializes GenEventImpls in bulk: 2^16 per leaf of a node's local event
// table and 2^7 per leaf of the lookaside tables that proxy remote nodes' events, and
// leaves are never freed while the runtime is up.  Any per-object allocation in that
// path is therefore multiplied by millions of event slots on large runs.  Two such
// allocations have been removed: a default-constructed std::deque in the EventMerger
// (~576 bytes of heap per event under libstdc++, more than half of node 0's
// event-table memory at 128 nodes) and a private EventCommunicator per event.
//
// The test replaces the global operator new/delete with counting versions, which is
// why it lives in its own executable rather than in realm_unit_tests.

#include "realm/event_impl.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

using namespace Realm;

namespace {

  std::atomic<bool> g_count_allocations{false};
  std::atomic<long> g_allocation_count{0};
  std::atomic<long> g_allocation_bytes{0};

  void *counted_allocate(std::size_t size)
  {
    if(g_count_allocations.load(std::memory_order_relaxed)) {
      g_allocation_count.fetch_add(1, std::memory_order_relaxed);
      g_allocation_bytes.fetch_add(static_cast<long>(size), std::memory_order_relaxed);
    }
    void *ptr = std::malloc(size ? size : 1);
    if(ptr == nullptr) {
      throw std::bad_alloc();
    }
    return ptr;
  }

  // Counts heap allocations made while it is in scope.
  class AllocationScope {
  public:
    AllocationScope(void)
    {
      g_allocation_count.store(0);
      g_allocation_bytes.store(0);
      g_count_allocations.store(true);
    }
    ~AllocationScope(void) { g_count_allocations.store(false); }
    AllocationScope(const AllocationScope &) = delete;
    AllocationScope &operator=(const AllocationScope &) = delete;

    long count(void) const { return g_allocation_count.load(); }
    long bytes(void) const { return g_allocation_bytes.load(); }
  };

  // Stands in for the network so a remote-owned event can be triggered without a
  // runtime.
  class CountingEventCommunicator : public EventCommunicator {
  public:
    virtual void trigger(Event event, NodeID owner, bool poisoned)
    {
      sent_trigger_count++;
      last_trigger_poisoned = poisoned;
    }
    virtual void update(Event event, NodeID to_update,
                        span<EventImpl::gen_t> poisoned_generations)
    {}
    virtual void subscribe(Event event, NodeID owner,
                           EventImpl::gen_t previous_subscribe_gen)
    {}

    int sent_trigger_count = 0;
    bool last_trigger_poisoned = false;
  };

  class GenEventImplAllocTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      event_notifier = new EventTriggerNotifier();
      event_comm = new CountingEventCommunicator();
    }

    void TearDown() override
    {
#ifdef DEBUG_REALM
      event_notifier->shutdown_work_item();
#endif
      delete event_notifier;
      delete event_comm;
    }

    // shared by every event in a test, just as the runtime shares its single
    // notifier and communicator across all of its events
    EventTriggerNotifier *event_notifier = nullptr;
    CountingEventCommunicator *event_comm = nullptr;
  };

} // namespace

void *operator new(std::size_t size) { return counted_allocate(size); }
void *operator new[](std::size_t size) { return counted_allocate(size); }
void operator delete(void *ptr) noexcept { std::free(ptr); }
void operator delete[](void *ptr) noexcept { std::free(ptr); }
void operator delete(void *ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void *ptr, std::size_t) noexcept { std::free(ptr); }

// Mirror a DynamicTable leaf exactly: the leaf's array default-constructs its elements,
// then GenEventImplAllocator::construct hands each one the runtime's shared notifier
// and communicator and initializes it.  Nothing in that path may touch the heap.
TEST_F(GenEventImplAllocTest, ConstructionDoesNotAllocate)
{
  // one remote lookaside leaf's worth of events
  constexpr size_t num_events = 1 << 7;
  using Storage = std::aligned_storage_t<sizeof(GenEventImpl), alignof(GenEventImpl)>;
  std::vector<Storage> storage(num_events);
  GenEventImpl *events = reinterpret_cast<GenEventImpl *>(storage.data());
  GenEventImpl::GenEventImplAllocator allocator(event_notifier, event_comm);

  long count = -1;
  long bytes = -1;
  {
    AllocationScope scope;
    for(size_t i = 0; i < num_events; i++) {
      new(&events[i]) GenEventImpl();
      allocator.construct(&events[i], ID::make_event(0, i, 0), 0);
    }
    count = scope.count();
    bytes = scope.bytes();
  }

  EXPECT_EQ(count, 0) << "constructing " << num_events << " GenEventImpls allocated "
                      << bytes << " bytes";
  // every event refers to the shared notifier and communicator rather than owning
  // its own
  for(size_t i = 0; i < num_events; i++) {
    EXPECT_EQ(events[i].event_triggerer, event_notifier);
    EXPECT_EQ(events[i].event_comm, static_cast<EventCommunicator *>(event_comm));
  }
  std::printf("[   INFO   ] sizeof(GenEventImpl)=%zu sizeof(EventMerger)=%zu "
              "sizeof(MergeEventPrecondition)=%zu\n",
              sizeof(GenEventImpl), sizeof(EventMerger),
              sizeof(EventMerger::MergeEventPrecondition));

  for(size_t i = 0; i < num_events; i++) {
    events[i].~GenEventImpl();
  }
}

namespace {
  // Arms a merge of 'num_preconditions' inputs on 'event' and returns the number of
  // heap allocations the arming path performed.  The preconditions are then triggered
  // and the merger armed outside the counting window (triggering a remote-owned event
  // records the generation in a std::map, which is not what is being measured).
  long allocations_to_arm_merge(GenEventImpl &event, GenEventImpl::gen_t gen,
                                size_t num_preconditions)
  {
    std::vector<EventMerger::MergeEventPrecondition *> preconditions;
    preconditions.reserve(num_preconditions);
    long count = -1;
    {
      AllocationScope scope;
      event.merger.prepare_merger(event.make_event(gen), false /*ignore faults*/,
                                  num_preconditions);
      for(size_t i = 0; i < num_preconditions; i++) {
        preconditions.push_back(event.merger.get_next_precondition());
      }
      count = scope.count();
    }
    for(EventMerger::MergeEventPrecondition *pre : preconditions) {
      pre->event_triggered(false /*!poisoned*/, TimeLimit::responsive());
    }
    event.merger.arm_merger();
    return count;
  }
} // namespace

// A merge that fits in the inline preconditions must not allocate, a wider one may
// allocate its overflow storage, and that storage must be gone again by the time the
// next merge on the same event is armed.
TEST_F(GenEventImplAllocTest, OnlyOverflowMergesAllocate)
{
  const NodeID owner = 1;
  bool poisoned = false;
  GenEventImpl event(event_notifier, event_comm);
  event.init(ID::make_event(0, 0, 0), owner);

  EXPECT_EQ(allocations_to_arm_merge(event, 1, EventMerger::MAX_INLINE_PRECONDITIONS), 0);
  EXPECT_TRUE(event.has_triggered(1, poisoned));

  const long overflow_allocations =
      allocations_to_arm_merge(event, 2, EventMerger::MAX_INLINE_PRECONDITIONS + 1);
  EXPECT_GT(overflow_allocations, 0);
  EXPECT_TRUE(event.has_triggered(2, poisoned));
  std::printf("[   INFO   ] first overflow precondition cost %ld allocation(s)\n",
              overflow_allocations);

  EXPECT_EQ(allocations_to_arm_merge(event, 3, EventMerger::MAX_INLINE_PRECONDITIONS), 0);
  EXPECT_TRUE(event.has_triggered(3, poisoned));
  EXPECT_FALSE(event.merger.is_active());
}
