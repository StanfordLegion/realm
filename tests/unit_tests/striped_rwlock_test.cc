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
} // namespace

TEST(StripedRWLockTest, ReadersExcludeWriters)
{
  Guarded g;
  std::atomic<bool> go(false), done(false);
  std::atomic<long> torn(0), reads(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < 8; t++)
    readers.emplace_back([&] {
      while(!go.load())
        ;
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
  go.store(true);
  for(int i = 0; i < 2000; i++) {
    StripedRWLock::AutoWriterLock al(g.lock);
    g.a = g.a + 1;
    spin_for(200);
    g.b = g.b + 1;
  }
  done.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(torn.load(), 0);
  EXPECT_GT(reads.load(), 0);
  EXPECT_EQ(long(g.a), 2000L);
  EXPECT_EQ(long(g.b), 2000L);
}

TEST(StripedRWLockTest, WritersExcludeEachOther)
{
  Guarded g;
  std::vector<std::thread> writers;
  for(int t = 0; t < 8; t++)
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
  EXPECT_EQ(long(g.a), 8000L);
  EXPECT_EQ(long(g.b), 8000L);
}

TEST(StripedRWLockTest, ReadersRunConcurrently)
{
  // all readers must be inside the lock at once at some point: with 4
  //  threads each holding it for 1 ms over 200 ms this is certain
  StripedRWLock lock;
  std::atomic<int> inside(0), max_inside(0);
  std::atomic<bool> go(false);
  std::vector<std::thread> readers;
  for(int t = 0; t < 4; t++)
    readers.emplace_back([&] {
      while(!go.load())
        ;
      for(int i = 0; i < 200; i++) {
        StripedRWLock::AutoReaderLock al(lock);
        int n = inside.fetch_add(1) + 1;
        int m = max_inside.load();
        while((n > m) && !max_inside.compare_exchange_weak(m, n))
          ;
        spin_for(1000000);
        inside.fetch_sub(1);
      }
    });
  go.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(max_inside.load(), 4);
}

TEST(StripedRWLockTest, WriterNotStarvedByReaders)
{
  // readers acquire and release back to back; every writer must still get
  //  in promptly (a reader-preferring lock would let this take forever)
  Guarded g;
  std::atomic<bool> go(false), done(false);
  std::vector<std::thread> readers;
  for(int t = 0; t < 8; t++)
    readers.emplace_back([&] {
      while(!go.load())
        ;
      while(!done.load()) {
        StripedRWLock::AutoReaderLock al(g.lock);
        if(g.a != g.b)
          abort();
      }
    });
  go.store(true);
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
  // generous: a preempted reader can hold the lock for a scheduler quantum
  EXPECT_LT(worst, 1000000000LL);
  EXPECT_EQ(long(g.a), 200L);
}

TEST(StripedRWLockTest, ReadersNotStarvedByWriters)
{
  // the writer holds the lock long enough for a reader to give up spinning
  //  and sleep, then releases and re-acquires it at once, 100 times over;
  //  the reader must get in at the first release instead of having to win
  //  a race against the next acquire
  Guarded g;
  std::atomic<bool> held(false);
  std::thread writer([&] {
    for(int i = 0; i < 100; i++) {
      StripedRWLock::AutoWriterLock al(g.lock);
      held.store(true);
      spin_for(5000000); // 5 ms
    }
  });
  while(!held.load())
    ;
  long long t0 = Clock::current_time_in_nanoseconds();
  {
    StripedRWLock::AutoReaderLock al(g.lock);
  }
  long long waited = Clock::current_time_in_nanoseconds() - t0;
  writer.join();
  // one hold (5 ms) plus wake-up latency; a reader that has to race the next
  //  acquire typically waits for most of the 100 holds
  EXPECT_LT(waited, 50000000LL) << "the reader waited " << waited << " ns";
}

TEST(StripedRWLockTest, MoreThreadsThanSlots)
{
  // short-lived readers, several times more of them than there are slots,
  //  so that slots are shared, against a writer
  Guarded g;
  std::atomic<long> torn(0);
  std::atomic<bool> done(false);
  std::thread writer([&] {
    while(!done.load()) {
      StripedRWLock::AutoWriterLock al(g.lock);
      g.a = g.a + 1;
      spin_for(100);
      g.b = g.b + 1;
    }
  });
  for(int round = 0; round < 4; round++) {
    std::vector<std::thread> readers;
    for(unsigned t = 0; t < 2 * StripedRWLock::NUM_SLOTS; t++)
      readers.emplace_back([&] {
        for(int i = 0; i < 50; i++) {
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
