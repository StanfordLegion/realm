/* Copyright 2026 Stanford University, NVIDIA Corporation
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

#include "realm/bgwork.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

using namespace Realm;

namespace {

  class TestWorkItem : public BackgroundWorkItem {
  public:
    TestWorkItem()
      : BackgroundWorkItem("bgwork_test")
    {}

    bool do_work(TimeLimit /*work_until*/) override
    {
      ran.store(true, std::memory_order_relaxed);
      return false;
    }

    void activate() { make_active(); }
    unsigned slot() const { return index; }

    void retire()
    {
#ifdef DEBUG_REALM
      shutdown_work_item();
#endif
    }

    std::atomic<bool> ran{false};
  };

  class TestWorkManager : public BackgroundWorkManager {
  public:
    // Simulate a worker scanning an active mask it loaded before slot release.
    void set_stale_active_bit(unsigned slot)
    {
      BitMask bit = BitMask(1) << (slot % BITMASK_BITS);
      active_work_item_mask[slot / BITMASK_BITS].fetch_or(bit);
    }
  };

} // namespace

TEST(BackgroundWorkTest, ConcurrentDiscoveryOfReleasedSlot)
{
  TestWorkManager manager;
  unsigned slot;
  {
    TestWorkItem item;
    item.add_to_manager(&manager);
    slot = item.slot();
    item.retire();
  }
  manager.set_stale_active_bit(slot);

  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for(int t = 0; t < 4; t++) {
    threads.emplace_back([&] {
      while(!go.load()) {
        std::this_thread::yield();
      }
      for(int i = 0; i < 2000; i++) {
        BackgroundWorkManager::Worker worker;
        worker.set_manager(&manager);
        worker.do_work(-1, nullptr);
      }
    });
  }
  go.store(true);
  for(std::thread &thread : threads) {
    thread.join();
  }
}

TEST(BackgroundWorkTest, DiscoveryReadsOrderedBeforeRelease)
{
  BackgroundWorkManager manager;
  auto item = std::make_unique<TestWorkItem>();
  item->add_to_manager(&manager, 0 /*numa domain*/);
  item->activate();

  // Keep the handoff relaxed so ThreadSanitizer checks the slot's own ordering.
  std::atomic<bool> discovered{false}, deleted{false};
  std::thread discoverer([&] {
    BackgroundWorkManager::Worker worker;
    worker.set_manager(&manager);
    worker.set_numa_domain(1); // Discover the item without running it.
    worker.do_work(-1, nullptr);
    discovered.store(true, std::memory_order_relaxed);
    while(!deleted.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
  });
  std::thread runner([&] {
    while(!discovered.load(std::memory_order_relaxed)) {
      std::this_thread::yield();
    }
    BackgroundWorkManager::Worker worker;
    worker.set_manager(&manager);
    while(!item->ran.load(std::memory_order_relaxed)) {
      worker.do_work(-1, nullptr);
    }
    item->retire();
    item.reset();
    deleted.store(true, std::memory_order_relaxed);
  });
  runner.join();
  discoverer.join();
}
