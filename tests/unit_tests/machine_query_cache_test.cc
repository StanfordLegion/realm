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

// Tests of the machine query result cache: that a query executed and then
//  refined gives the same answer as a fresh query with the same constraints,
//  that copies of a query share results safely, and that readers survive the
//  cache being invalidated and rebuilt underneath them. The two concurrent
//  tests only prove something under ThreadSanitizer (copies) and
//  AddressSanitizer (invalidation); in a plain build they are smoke tests.

#include "realm.h"
#include "realm/machine_impl.h"
#include "test_mock.h"
#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include <set>
#include <thread>
#include <vector>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
  extern RuntimeImpl *runtime_singleton;
}; // namespace Realm

class MachineQueryCacheTest : public ::testing::Test {
protected:
  // one node, 4 CPUs, 2 system memories:
  //   p0,p1 -> m0: bw 100 lat 10     p2,p3 -> m0: bw 50 lat 20
  //   p0,p1 -> m1: bw  80 lat 15     p2,p3 -> m1: bw 90 lat 12
  //   m0 <-> m1: bw 60 lat 25
  // so the expected answers below can be read off the tables
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl_ = std::make_unique<MockRuntimeImplMachineModel>();
    Realm::runtime_singleton = runtime_impl_.get();
    runtime_impl_->init(1);

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems;
    for(unsigned int i = 0; i < 4; i++)
      procs_mems.proc_infos.push_back({i, Processor::LOC_PROC, 0});
    procs_mems.mem_infos.push_back({0, Memory::SYSTEM_MEM, size_t(1024) * 1024, 0});
    procs_mems.mem_infos.push_back({1, Memory::SYSTEM_MEM, size_t(2048) * 1024, 0});
    procs_mems.proc_mem_affinities.push_back({0, 0, 100, 10});
    procs_mems.proc_mem_affinities.push_back({1, 0, 100, 10});
    procs_mems.proc_mem_affinities.push_back({2, 0, 50, 20});
    procs_mems.proc_mem_affinities.push_back({3, 0, 50, 20});
    procs_mems.proc_mem_affinities.push_back({0, 1, 80, 15});
    procs_mems.proc_mem_affinities.push_back({1, 1, 80, 15});
    procs_mems.proc_mem_affinities.push_back({2, 1, 90, 12});
    procs_mems.proc_mem_affinities.push_back({3, 1, 90, 12});
    procs_mems.mem_mem_affinities.push_back({0, 1, 60, 25});
    procs_mems.mem_mem_affinities.push_back({1, 0, 60, 25});
    runtime_impl_->setup_mock_proc_mems(procs_mems);

    for(unsigned int i = 0; i < 4; i++)
      procs[i] = ID::make_processor(0, i).convert<Processor>();
    for(unsigned int i = 0; i < 2; i++)
      mems[i] = ID::make_memory(0, i).convert<Memory>();
  }

  void TearDown() override
  {
    runtime_impl_->finalize();
    runtime_impl_.reset();
    Realm::runtime_singleton = nullptr;
  }

  Machine machine() { return Machine::get_machine(); }

  // every result accessor of a query, as a set, checking they agree
  template <typename Q, typename T>
  static std::set<T> results(const Q &q, T /*tag*/)
  {
    std::set<T> found;
    for(T t = q.first(); t.exists(); t = q.next(t))
      EXPECT_TRUE(found.insert(t).second) << "duplicate result";
    EXPECT_EQ(found.size(), q.count()) << "count disagrees with iteration";
    if(found.empty())
      EXPECT_FALSE(q.random().exists());
    else
      EXPECT_TRUE(found.count(q.random()) > 0) << "random() not among results";
    return found;
  }
  static std::set<Processor> results(const Machine::ProcessorQuery &q)
  {
    return results(q, Processor::NO_PROC);
  }
  static std::set<Memory> results(const Machine::MemoryQuery &q)
  {
    return results(q, Memory::NO_MEMORY);
  }

  // re-adds an existing affinity, which invalidates the query caches without
  //  changing any answer
  void invalidate_caches()
  {
    Machine::ProcessorMemoryAffinity pma;
    pma.p = procs[0];
    pma.m = mems[0];
    pma.bandwidth = 100;
    pma.latency = 10;
    runtime_impl_->machine->add_proc_mem_affinity(pma);
  }

  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl_;
  Processor procs[4];
  Memory mems[2];
};

// a query that has already been executed must give the same answers after
//  refinement as a fresh query with the same constraints

TEST_F(MachineQueryCacheTest, RefineProcessorQueryWithBestAffinity)
{
  Machine::ProcessorQuery warmed(machine());
  warmed.only_kind(Processor::LOC_PROC);
  EXPECT_EQ(results(warmed),
            (std::set<Processor>{procs[0], procs[1], procs[2], procs[3]}));

  warmed.best_affinity_to(mems[0]);
  Machine::ProcessorQuery fresh = Machine::ProcessorQuery(machine())
                                      .only_kind(Processor::LOC_PROC)
                                      .best_affinity_to(mems[0]);
  std::set<Processor> expected{procs[0], procs[1]};
  EXPECT_EQ(results(fresh), expected);
  EXPECT_EQ(results(warmed), expected);
  EXPECT_EQ(warmed.first(), fresh.first());
}

TEST_F(MachineQueryCacheTest, RefineMemoryQueryWithBestAffinityToProcessor)
{
  Machine::MemoryQuery warmed(machine());
  warmed.only_kind(Memory::SYSTEM_MEM);
  EXPECT_EQ(results(warmed), (std::set<Memory>{mems[0], mems[1]}));

  warmed.best_affinity_to(procs[0]);
  Machine::MemoryQuery fresh = Machine::MemoryQuery(machine())
                                   .only_kind(Memory::SYSTEM_MEM)
                                   .best_affinity_to(procs[0]);
  std::set<Memory> expected{mems[0]};
  EXPECT_EQ(results(fresh), expected);
  EXPECT_EQ(results(warmed), expected);
}

TEST_F(MachineQueryCacheTest, RefineMemoryQueryWithBestAffinityToMemory)
{
  Machine::MemoryQuery warmed(machine());
  warmed.only_kind(Memory::SYSTEM_MEM);
  EXPECT_EQ(warmed.count(), 2u);

  // only m1 has an affinity from m0, so m0 itself is not a candidate
  warmed.best_affinity_to(mems[0]);
  Machine::MemoryQuery fresh = Machine::MemoryQuery(machine())
                                   .only_kind(Memory::SYSTEM_MEM)
                                   .best_affinity_to(mems[0]);
  std::set<Memory> expected{mems[1]};
  EXPECT_EQ(results(fresh), expected);
  EXPECT_EQ(results(warmed), expected);
}

TEST_F(MachineQueryCacheTest, AddSecondBestAffinityCostAfterExecution)
{
  Machine::MemoryQuery q(machine());
  q.only_kind(Memory::SYSTEM_MEM);
  EXPECT_EQ(q.count(), 2u);
  q.best_affinity_to(procs[0]);
  EXPECT_EQ(results(q), (std::set<Memory>{mems[0]}));
  // adding the memory component: m0 has no affinity from m0 and drops out,
  //  m1 scores 80 + 60
  q.best_affinity_to(mems[0]);
  Machine::MemoryQuery fresh = Machine::MemoryQuery(machine())
                                   .only_kind(Memory::SYSTEM_MEM)
                                   .best_affinity_to(procs[0])
                                   .best_affinity_to(mems[0]);
  EXPECT_EQ(results(fresh), (std::set<Memory>{mems[1]}));
  EXPECT_EQ(results(q), (std::set<Memory>{mems[1]}));
}

TEST_F(MachineQueryCacheTest, ChangeBestAffinityTargetAndWeightsAfterExecution)
{
  Machine::ProcessorQuery q(machine());
  q.only_kind(Processor::LOC_PROC).best_affinity_to(mems[0]);
  EXPECT_EQ(results(q), (std::set<Processor>{procs[0], procs[1]}));
  // a new target (this logs a warning, which is the documented behavior)
  q.best_affinity_to(mems[1]);
  EXPECT_EQ(results(q), (std::set<Processor>{procs[2], procs[3]}));
  // same target, latency-only weight: 20 beats 10
  q.best_affinity_to(mems[1], 0 /*bandwidth*/, 1 /*latency*/);
  EXPECT_EQ(results(q), (std::set<Processor>{procs[0], procs[1]}));
  Machine::ProcessorQuery fresh = Machine::ProcessorQuery(machine())
                                      .only_kind(Processor::LOC_PROC)
                                      .best_affinity_to(mems[1], 0, 1);
  EXPECT_EQ(results(fresh), results(q));
}

TEST_F(MachineQueryCacheTest, OtherRestrictionsAfterExecution)
{
  Machine::ProcessorQuery pq(machine());
  pq.only_kind(Processor::LOC_PROC);
  EXPECT_EQ(pq.count(), 4u);
  pq.has_affinity_to(mems[0], 60 /*min bandwidth*/);
  EXPECT_EQ(results(pq), (std::set<Processor>{procs[0], procs[1]}));
  pq.same_address_space_as(mems[0]);
  EXPECT_EQ(results(pq), (std::set<Processor>{procs[0], procs[1]}));

  Machine::MemoryQuery mq(machine());
  mq.only_kind(Memory::SYSTEM_MEM);
  EXPECT_EQ(mq.count(), 2u);
  mq.has_capacity(size_t(1536) * 1024);
  EXPECT_EQ(results(mq), (std::set<Memory>{mems[1]}));
  mq.has_affinity_to(procs[0]);
  EXPECT_EQ(results(mq), (std::set<Memory>{mems[1]}));
}

// copies of a query share its results until one of them is refined, and then
//  only the refined one changes

TEST_F(MachineQueryCacheTest, CopiesRefineIndependently)
{
  Machine::ProcessorQuery pq(machine());
  pq.only_kind(Processor::LOC_PROC);
  EXPECT_EQ(pq.count(), 4u);

  Machine::ProcessorQuery copy(pq);
  Machine::ProcessorQuery assigned(machine());
  assigned = pq;
  EXPECT_EQ(copy.count(), 4u);
  EXPECT_EQ(assigned.count(), 4u);

  copy.best_affinity_to(mems[0]);
  EXPECT_EQ(results(copy), (std::set<Processor>{procs[0], procs[1]}));
  EXPECT_EQ(pq.count(), 4u);
  EXPECT_EQ(assigned.count(), 4u);

  assigned.has_affinity_to(mems[1], 85);
  EXPECT_EQ(results(assigned), (std::set<Processor>{procs[2], procs[3]}));
  EXPECT_EQ(pq.count(), 4u);
  EXPECT_EQ(copy.count(), 2u);

  // copying an already-refined query keeps its results
  Machine::ProcessorQuery copy2(copy);
  EXPECT_EQ(results(copy2), results(copy));
}

TEST_F(MachineQueryCacheTest, ConcurrentReadsThroughCopies)
{
  Machine::ProcessorQuery pq(machine());
  pq.only_kind(Processor::LOC_PROC);
  Machine::MemoryQuery mq(machine());
  mq.only_kind(Memory::SYSTEM_MEM);
  Machine::ProcessorQuery bq(machine());
  bq.only_kind(Processor::LOC_PROC).best_affinity_to(mems[0]);

  std::atomic<bool> go(false);
  std::atomic<int> failures(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < 4; t++) {
    // each thread holds its own copy, all sharing one implementation
    readers.emplace_back([&, pq, mq, bq] {
      while(!go.load())
        ;
      for(int i = 0; i < 2000; i++) {
        if(pq.count() != 4)
          failures.fetch_add(1);
        if(mq.count() != 2)
          failures.fetch_add(1);
        if(bq.count() != 2)
          failures.fetch_add(1);
        size_t n = 0;
        for(Processor p = pq.first(); p.exists(); p = pq.next(p))
          n++;
        if(n != 4)
          failures.fetch_add(1);
        if(!mq.first().exists() || !bq.random().exists())
          failures.fetch_add(1);
      }
    });
  }
  go.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(failures.load(), 0);
  // the originals were not disturbed
  EXPECT_EQ(pq.count(), 4u);
  EXPECT_EQ(mq.count(), 2u);
  EXPECT_EQ(bq.count(), 2u);
}

// invalidating and rebuilding the shared cache while other queries are
//  reading it must neither crash nor change any answer

TEST_F(MachineQueryCacheTest, InvalidationUnderReaders)
{
  std::atomic<bool> go(false), done(false);
  std::atomic<int> failures(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < 4; t++) {
    readers.emplace_back([&] {
      // independent queries exercising each shared cache plus a private
      //  best-affinity list
      Machine::ProcessorQuery by_kind(machine());
      by_kind.only_kind(Processor::LOC_PROC);
      Machine::ProcessorQuery by_affinity(machine());
      by_affinity.only_kind(Processor::LOC_PROC).has_affinity_to(mems[0]);
      Machine::MemoryQuery mems_by_kind(machine());
      mems_by_kind.only_kind(Memory::SYSTEM_MEM);
      Machine::ProcessorQuery best(machine());
      best.only_kind(Processor::LOC_PROC).best_affinity_to(mems[0]);
      while(!go.load())
        ;
      while(!done.load()) {
        if(by_kind.count() != 4)
          failures.fetch_add(1);
        if(by_affinity.count() != 4)
          failures.fetch_add(1);
        if(mems_by_kind.count() != 2)
          failures.fetch_add(1);
        if(best.count() != 2)
          failures.fetch_add(1);
        size_t n = 0;
        for(Processor p = by_kind.first(); p.exists(); p = by_kind.next(p))
          n++;
        if(n != 4)
          failures.fetch_add(1);
        for(Memory m = mems_by_kind.first(); m.exists(); m = mems_by_kind.next(m))
          if(m.kind() != Memory::SYSTEM_MEM)
            failures.fetch_add(1);
      }
    });
  }
  go.store(true);
  for(int i = 0; i < 2000; i++) {
    invalidate_caches();
    // rebuild the shared lists so the readers' old ones are released
    Machine::ProcessorQuery(machine()).only_kind(Processor::LOC_PROC).count();
    Machine::MemoryQuery(machine()).only_kind(Memory::SYSTEM_MEM).count();
  }
  done.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(failures.load(), 0);
}

// queries that walk the machine model (predicates, node restrictions, best
//  affinity, processor-anchored memory queries) must hold the machine's
//  reader lock, so an affinity update can not race with them - visible only
//  under ThreadSanitizer, a smoke test otherwise
TEST_F(MachineQueryCacheTest, MachineUpdatesUnderWalkingReaders)
{
  std::atomic<bool> go(false), done(false);
  std::atomic<int> failures(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < 4; t++) {
    readers.emplace_back([&] {
      Machine::ProcessorQuery fast(machine());
      fast.only_kind(Processor::LOC_PROC).has_affinity_to(mems[0], 60 /*min bw*/);
      Machine::ProcessorQuery local(machine());
      local.only_kind(Processor::LOC_PROC).local_address_space();
      Machine::ProcessorQuery best(machine());
      best.only_kind(Processor::LOC_PROC).best_affinity_to(mems[0]);
      Machine::MemoryQuery anchored(machine());
      anchored.only_kind(Memory::SYSTEM_MEM).has_affinity_to(procs[0]);
      Machine::MemoryQuery big(machine());
      big.only_kind(Memory::SYSTEM_MEM).has_capacity(size_t(1536) * 1024);
      Machine::MemoryQuery best_mem(machine());
      best_mem.only_kind(Memory::SYSTEM_MEM).best_affinity_to(procs[0]);
      while(!go.load())
        ;
      while(!done.load()) {
        if(fast.count() != 2)
          failures.fetch_add(1);
        if(local.count() != 4)
          failures.fetch_add(1);
        if(best.count() != 2)
          failures.fetch_add(1);
        if(anchored.count() != 2)
          failures.fetch_add(1);
        if(big.count() != 1)
          failures.fetch_add(1);
        if(best_mem.count() != 1)
          failures.fetch_add(1);
        size_t n = 0;
        for(Processor p = fast.first(); p.exists(); p = fast.next(p))
          n++;
        for(Memory m = anchored.first(); m.exists(); m = anchored.next(m))
          n++;
        if(n != 4)
          failures.fetch_add(1);
        if(!local.random().exists() || !big.random().exists())
          failures.fetch_add(1);
      }
    });
  }
  go.store(true);
  for(int i = 0; i < 2000; i++)
    invalidate_caches(); // an affinity update, i.e. a writer of the machine model
  done.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(failures.load(), 0);
}

TEST_F(MachineQueryCacheTest, IterationSurvivesInvalidation)
{
  Machine::ProcessorQuery pq(machine());
  pq.only_kind(Processor::LOC_PROC);
  std::set<Processor> seen;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    seen.insert(p);
    invalidate_caches();
  }
  EXPECT_EQ(seen, (std::set<Processor>{procs[0], procs[1], procs[2], procs[3]}));

  Machine::MemoryQuery mq(machine());
  mq.only_kind(Memory::SYSTEM_MEM);
  std::set<Memory> seen_mems;
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    seen_mems.insert(m);
    invalidate_caches();
  }
  EXPECT_EQ(seen_mems, (std::set<Memory>{mems[0], mems[1]}));
}
