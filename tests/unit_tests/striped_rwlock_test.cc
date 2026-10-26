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

// Tests of StripedRWLock: readers and a writer exclude each other, readers
//  run concurrently, a writer is not starved by a steady stream of readers,
//  readers are not starved by back-to-back writers, and threads that share a
//  reader slot (more threads than slots) still get mutual exclusion.
// Sized for a small machine (a CI runner has four cpus and runs several tests
//  at once): few threads, short runs, and no bound on wall-clock time that a
//  slow or loaded machine could miss.

#include "realm/mutex.h"
#include "realm/timers.h"
#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

using namespace Realm;

namespace {
  // two counters that a writer updates one after the other; a reader that
  //  sees them differ has run concurrently with a writer
  struct Guarded {
    StripedRWLock lock;
    volatile long a = 0;
    volatile long b = 0;
  };

  void spin_for(long long ns)
  {
    long long t0 = Clock::current_time_in_nanoseconds();
    while(Clock::current_time_in_nanoseconds() - t0 < ns)
      ;
  }

  // threads wait here for the test to start without burning the cpus that
  //  the test's other threads still need to get created
  struct StartGate {
    std::atomic<bool> open{false};
    void wait() const
    {
      while(!open.load(std::memory_order_acquire))
        std::this_thread::yield();
    }
    void release() { open.store(true, std::memory_order_release); }
  };

  const int NUM_READERS = 4;
} // namespace

TEST(StripedRWLockTest, ReadersExcludeWriters)
{
  Guarded g;
  StartGate gate;
  std::atomic<bool> done(false);
  std::atomic<long> torn(0), reads(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < NUM_READERS; t++)
    readers.emplace_back([&] {
      gate.wait();
      while(!done.load()) {
        StripedRWLock::AutoReaderLock al(g.lock);
        long a = g.a;
        spin_for(100);
        long b = g.b;
        if(a != b)
          torn.fetch_add(1);
        reads.fetch_add(1);
      }
    });
  gate.release();
  // keep writing until the readers have overlapped with enough of it (on a
  //  busy machine they may not get a cpu for a while), but not forever
  long long t0 = Clock::current_time_in_nanoseconds();
  long writes = 0;
  while((writes < 1000) ||
        ((reads.load() < 1000) &&
         ((Clock::current_time_in_nanoseconds() - t0) < 10000000000LL))) {
    StripedRWLock::AutoWriterLock al(g.lock);
    g.a = g.a + 1;
    spin_for(200);
    g.b = g.b + 1;
    writes++;
  }
  done.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(torn.load(), 0);
  EXPECT_GE(reads.load(), 1000L);
  EXPECT_EQ(long(g.a), writes);
  EXPECT_EQ(long(g.b), writes);
}

TEST(StripedRWLockTest, WritersExcludeEachOther)
{
  Guarded g;
  std::vector<std::thread> writers;
  for(int t = 0; t < 4; t++)
    writers.emplace_back([&] {
      for(int i = 0; i < 1000; i++) {
        StripedRWLock::AutoWriterLock al(g.lock);
        long a = g.a;
        spin_for(50);
        g.a = a + 1;
        g.b = g.b + 1;
      }
    });
  for(std::thread &t : writers)
    t.join();
  EXPECT_EQ(long(g.a), 4000L);
  EXPECT_EQ(long(g.b), 4000L);
}

TEST(StripedRWLockTest, ReadersRunConcurrently)
{
  // every reader takes the lock and keeps it until all of them are inside at
  //  once (or gives up after a while): a lock that admitted one reader at a
  //  time would never get them all in
  StripedRWLock lock;
  std::atomic<int> inside(0);
  std::atomic<bool> all_in(false);
  std::vector<std::thread> readers;
  for(int t = 0; t < NUM_READERS; t++)
    readers.emplace_back([&] {
      StripedRWLock::AutoReaderLock al(lock);
      if((inside.fetch_add(1) + 1) == NUM_READERS)
        all_in.store(true);
      long long t0 = Clock::current_time_in_nanoseconds();
      while(!all_in.load() &&
            ((Clock::current_time_in_nanoseconds() - t0) < 5000000000LL))
        std::this_thread::yield();
    });
  for(std::thread &t : readers)
    t.join();
  EXPECT_TRUE(all_in.load());
}

TEST(StripedRWLockTest, WriterNotStarvedByReaders)
{
  // readers acquire and release back to back and never stop on their own;
  //  every writer must still get in (a reader-preferring lock could keep it
  //  out for as long as the readers keep coming)
  Guarded g;
  StartGate gate;
  std::atomic<bool> done(false);
  std::atomic<long> torn(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < NUM_READERS; t++)
    readers.emplace_back([&] {
      gate.wait();
      while(!done.load()) {
        StripedRWLock::AutoReaderLock al(g.lock);
        if(g.a != g.b)
          torn.fetch_add(1);
      }
    });
  gate.release();
  long long worst = 0;
  for(int i = 0; i < 200; i++) {
    long long t0 = Clock::current_time_in_nanoseconds();
    {
      StripedRWLock::AutoWriterLock al(g.lock);
      g.a = g.a + 1;
      g.b = g.b + 1;
    }
    long long dt = Clock::current_time_in_nanoseconds() - t0;
    if(dt > worst)
      worst = dt;
    spin_for(10000); // let the readers back in
  }
  done.store(true);
  for(std::thread &t : readers)
    t.join();
  // a writer waits for the readers already inside and for those that backed
  //  off for the previous writer: a few scheduler quanta even on a loaded
  //  machine, never seconds
  EXPECT_LT(worst, 5000000000LL) << "worst write acquire " << worst << " ns";
  EXPECT_EQ(torn.load(), 0);
  EXPECT_EQ(long(g.a), 200L);
}

TEST(StripedRWLockTest, ReadersNotStarvedByWriters)
{
  // the writer holds the lock long enough for a reader to give up spinning
  //  and sleep, then releases and re-acquires it at once, 100 times over;
  //  the reader must get in at the first release instead of having to win
  //  a race against the next acquire. Progress is counted in writer holds,
  //  not in time, so a slow machine can not fail this
  Guarded g;
  std::atomic<int> holds(0); // completed writer holds
  std::atomic<bool> held(false);
  std::thread writer([&] {
    for(int i = 0; i < 100; i++) {
      {
        StripedRWLock::AutoWriterLock al(g.lock);
        held.store(true);
        spin_for(5000000); // 5 ms
      }
      holds.fetch_add(1);
    }
  });
  while(!held.load())
    std::this_thread::yield();
  int holds_at_arrival = holds.load();
  int holds_at_entry;
  {
    StripedRWLock::AutoReaderLock al(g.lock);
    holds_at_entry = holds.load();
  }
  writer.join();
  // a reader that arrives during a hold is let in at the end of it; one that
  //  has to race the next acquire typically waits for most of the 100 holds
  EXPECT_LT(holds_at_entry - holds_at_arrival, 10)
      << "the reader arrived after " << holds_at_arrival << " holds and got in after "
      << holds_at_entry;
}

TEST(StripedRWLockTest, MoreThreadsThanSlots)
{
  // short-lived readers, twice as many as there are slots so that every slot
  //  is shared, against a writer that keeps updating
  Guarded g;
  std::atomic<long> torn(0);
  std::atomic<bool> done(false);
  std::thread writer([&] {
    while(!done.load()) {
      {
        StripedRWLock::AutoWriterLock al(g.lock);
        g.a = g.a + 1;
        spin_for(100);
        g.b = g.b + 1;
      }
      spin_for(2000); // leave the readers room between writes
    }
  });
  for(int round = 0; round < 2; round++) {
    std::vector<std::thread> readers;
    for(unsigned t = 0; t < 2 * StripedRWLock::NUM_SLOTS; t++)
      readers.emplace_back([&] {
        for(int i = 0; i < 20; i++) {
          StripedRWLock::AutoReaderLock al(g.lock);
          long a = g.a;
          spin_for(100);
          if(a != g.b)
            torn.fetch_add(1);
        }
      });
    for(std::thread &t : readers)
      t.join();
  }
  done.store(true);
  writer.join();
  EXPECT_EQ(torn.load(), 0);
  EXPECT_EQ(long(g.a), long(g.b));
}
