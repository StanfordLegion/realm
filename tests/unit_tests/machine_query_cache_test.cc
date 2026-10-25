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
//  that copies of a query share results safely, that the cache and a walk of
//  the machine agree, and that readers survive the cache being invalidated
//  and rebuilt underneath them. The concurrent tests only prove something
//  under ThreadSanitizer (copies, machine updates) and AddressSanitizer
//  (invalidation); in a plain build they are smoke tests.

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
  // one node, 4 CPUs, 2 system memories and a framebuffer nothing can reach:
  //   p0,p1 -> m0: bw 100 lat 10     p2,p3 -> m0: bw 50 lat 20
  //   p0,p1 -> m1: bw  80 lat 15     p2,p3 -> m1: bw 90 lat 12
  //   m0 <-> m1: bw 60 lat 25        m2: no affinities
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
    procs_mems.mem_infos.push_back({2, Memory::GPU_FB_MEM, size_t(4096) * 1024, 0});
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
    for(unsigned int i = 0; i < 3; i++)
      mems[i] = ID::make_memory(0, i).convert<Memory>();
  }

  void TearDown() override
  {
    runtime_impl_->finalize();
    runtime_impl_.reset();
    Realm::runtime_singleton = nullptr;
  }

  Machine machine() { return Machine::get_machine(); }

public:
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

  // the results of a query with the shared cache enabled and with it
  //  disabled (every query then walks the machine), which must agree
  template <typename Q>
  static void expect_cache_agrees_with_walk(const Q &q)
  {
    const bool saved = Config::use_machine_query_cache;
    Config::use_machine_query_cache = true;
    const auto cached = results(q);
    Config::use_machine_query_cache = false;
    const auto walked = results(q);
    Config::use_machine_query_cache = saved;
    EXPECT_EQ(cached, walked) << "cached and walked results differ";
  }

protected:
  // an affinity update: a machine-model writer, which also invalidates the
  //  query caches
  void set_affinity(Processor p, Memory m, unsigned bandwidth, unsigned latency)
  {
    Machine::ProcessorMemoryAffinity pma;
    pma.p = p;
    pma.m = m;
    pma.bandwidth = bandwidth;
    pma.latency = latency;
    runtime_impl_->machine->add_proc_mem_affinity(pma);
  }

  // re-adds an existing affinity, which invalidates the query caches without
  //  changing any answer
  void invalidate_caches() { set_affinity(procs[0], mems[0], 100, 10); }

  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl_;
  Processor procs[4];
  Memory mems[3];
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
//  reader lock, so an affinity update can not race with them. The writer
//  flips p0 -> m0 between bw 100 and bw 30, which changes the answers, so
//  every answer must be the one for one of the two machine states (a torn
//  read would be caught by ThreadSanitizer; a missing lock also shows up
//  here as an answer for neither state)
TEST_F(MachineQueryCacheTest, MachineUpdatesUnderWalkingReaders)
{
  std::atomic<bool> go(false), done(false);
  std::atomic<int> failures(0);
  std::vector<std::thread> readers;
  for(int t = 0; t < 4; t++) {
    readers.emplace_back([&] {
      // p0 -> m0 at bw 100: {p0, p1}; at bw 30: {p1}
      Machine::ProcessorQuery fast(machine());
      fast.only_kind(Processor::LOC_PROC).has_affinity_to(mems[0], 60 /*min bw*/);
      Machine::ProcessorQuery local(machine());
      local.only_kind(Processor::LOC_PROC).local_address_space();
      // best for m0: {p0, p1} at bw 100, {p1} at bw 30
      Machine::ProcessorQuery best(machine());
      best.only_kind(Processor::LOC_PROC).best_affinity_to(mems[0]);
      Machine::MemoryQuery anchored(machine());
      anchored.only_kind(Memory::SYSTEM_MEM).has_affinity_to(procs[0]);
      Machine::MemoryQuery big(machine());
      big.only_kind(Memory::SYSTEM_MEM).has_capacity(size_t(1536) * 1024);
      // best for p0: m0 (100 vs 80) at bw 100, m1 (80 vs 30) at bw 30
      Machine::MemoryQuery best_mem(machine());
      best_mem.only_kind(Memory::SYSTEM_MEM).best_affinity_to(procs[0]);
      while(!go.load())
        ;
      while(!done.load()) {
        size_t c = fast.count();
        if((c != 2) && (c != 1))
          failures.fetch_add(1);
        if(local.count() != 4)
          failures.fetch_add(1);
        c = best.count();
        if((c != 2) && (c != 1))
          failures.fetch_add(1);
        if(anchored.count() != 2)
          failures.fetch_add(1);
        if(big.count() != 1)
          failures.fetch_add(1);
        Memory bm = best_mem.first();
        if((bm != mems[0]) && (bm != mems[1]))
          failures.fetch_add(1);
        size_t n = 0;
        for(Processor p = fast.first(); p.exists(); p = fast.next(p)) {
          if((p != procs[0]) && (p != procs[1]))
            failures.fetch_add(1);
          n++;
        }
        for(Memory m = anchored.first(); m.exists(); m = anchored.next(m))
          n++;
        if((n != 4) && (n != 3))
          failures.fetch_add(1);
        if(!local.random().exists() || !big.random().exists())
          failures.fetch_add(1);
        Processor bp = best.first();
        if((bp != procs[0]) && (bp != procs[1]))
          failures.fetch_add(1);
      }
    });
  }
  go.store(true);
  for(int i = 0; i < 2000; i++)
    set_affinity(procs[0], mems[0], (i & 1) ? 100 : 30, 10);
  set_affinity(procs[0], mems[0], 100, 10);
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

// the shared cache serves kind-and-memory queries whose answer is empty (a
//  kind with no processor having an affinity to the memory) as cheaply as
//  any other, and sees the affinity that later fills them in

TEST_F(MachineQueryCacheTest, KindAndMemoryWithoutAffinity)
{
  // the framebuffer is reachable from nothing
  Machine::ProcessorQuery none(machine());
  none.only_kind(Processor::LOC_PROC).has_affinity_to(mems[2]);
  EXPECT_EQ(none.count(), 0u);
  EXPECT_FALSE(none.first().exists());
  EXPECT_FALSE(none.random().exists());
  expect_cache_agrees_with_walk(none);
  // a kind with no processors at all
  Machine::ProcessorQuery no_kind(machine());
  no_kind.only_kind(Processor::TOC_PROC).has_affinity_to(mems[0]);
  EXPECT_EQ(no_kind.count(), 0u);
  EXPECT_FALSE(no_kind.first().exists());
  EXPECT_EQ(Machine::ProcessorQuery(machine()).only_kind(Processor::TOC_PROC).count(), 0u);
  EXPECT_FALSE(Machine::ProcessorQuery(machine()).only_kind(Processor::TOC_PROC).first().exists());
  EXPECT_EQ(Machine::MemoryQuery(machine()).only_kind(Memory::DISK_MEM).count(), 0u);

  // give p2 an affinity to the framebuffer: the same query object and a
  //  fresh one both see it
  set_affinity(procs[2], mems[2], 40, 50);
  EXPECT_EQ(results(none), (std::set<Processor>{procs[2]}));
  Machine::ProcessorQuery fresh(machine());
  fresh.only_kind(Processor::LOC_PROC).has_affinity_to(mems[2]);
  EXPECT_EQ(results(fresh), (std::set<Processor>{procs[2]}));
  expect_cache_agrees_with_walk(fresh);
  EXPECT_EQ(results(Machine::MemoryQuery(machine()).has_affinity_to(procs[2])),
            (std::set<Memory>{mems[0], mems[1], mems[2]}));
}

// the cache and a walk of the machine must agree for every shape the cache
//  serves, including a query refined after it was executed

TEST_F(MachineQueryCacheTest, CacheAgreesWithWalk)
{
  expect_cache_agrees_with_walk(
      Machine::ProcessorQuery(machine()).only_kind(Processor::LOC_PROC));
  expect_cache_agrees_with_walk(Machine::ProcessorQuery(machine())
                                    .only_kind(Processor::LOC_PROC)
                                    .has_affinity_to(mems[1]));
  expect_cache_agrees_with_walk(Machine::ProcessorQuery(machine())
                                    .only_kind(Processor::LOC_PROC)
                                    .has_affinity_to(mems[1], 85 /*min bw*/));
  expect_cache_agrees_with_walk(
      Machine::MemoryQuery(machine()).only_kind(Memory::SYSTEM_MEM));
  expect_cache_agrees_with_walk(Machine::MemoryQuery(machine()).has_affinity_to(procs[3]));
  Machine::ProcessorQuery refined(machine());
  refined.only_kind(Processor::LOC_PROC);
  EXPECT_EQ(refined.count(), 4u);
  refined.has_affinity_to(mems[0]);
  expect_cache_agrees_with_walk(refined);
  refined.same_address_space_as(mems[0]);
  expect_cache_agrees_with_walk(refined);
}

// copies of a query share its implementation by reference count; the last
//  copy to go away deletes it, and a copy that is refined while others read
//  through theirs makes a private one. Both must be ordered after every
//  other copy's reads (visible only under ThreadSanitizer)

TEST_F(MachineQueryCacheTest, CopiesDroppedAndRefinedConcurrently)
{
  for(int mode = 0; mode < 2; mode++) {
    for(int iter = 0; iter < 100; iter++) {
      std::vector<Machine::MemoryQuery> mcopies;
      std::vector<Machine::ProcessorQuery> pcopies;
      {
        Machine::MemoryQuery mq(machine());
        mq.only_kind(Memory::SYSTEM_MEM);
        EXPECT_EQ(mq.count(), 2u);
        Machine::ProcessorQuery pq(machine());
        pq.only_kind(Processor::LOC_PROC).best_affinity_to(mems[1]);
        EXPECT_EQ(pq.count(), 2u);
        for(int t = 0; t < 4; t++) {
          mcopies.push_back(mq);
          pcopies.push_back(pq);
        }
        // the originals go away here: the copies alone keep the shared
        //  implementations alive
      }
      std::atomic<bool> go(false);
      std::atomic<int> failures(0);
      std::vector<std::thread> threads;
      for(int t = 0; t < 4; t++) {
        // each thread owns one copy (taken here, before the vectors are
        //  cleared below)
        threads.emplace_back([&, mode, mq = mcopies[t], pq = pcopies[t]]() mutable {
          while(!go.load())
            ;
          if((mq.count() != 2) || !mq.first().exists())
            failures.fetch_add(1);
          if((pq.count() != 2) || (pq.first() != procs[2]))
            failures.fetch_add(1);
          if(mode == 1) {
            // refine: this copy gets a private implementation
            mq.has_capacity(size_t(1536) * 1024);
            pq.same_address_space_as(mems[0]);
            if((mq.count() != 1) || (pq.count() != 2))
              failures.fetch_add(1);
          }
          // drop: the last one out deletes the shared implementation
        });
      }
      mcopies.clear();
      pcopies.clear();
      go.store(true);
      for(std::thread &th : threads)
        th.join();
      EXPECT_EQ(failures.load(), 0);
    }
  }
}

// an affinity from a processor on one node to a memory on another must be
//  seen the same way by the shared cache and by a walk of the machine

class MachineQueryCacheMultiNodeTest : public ::testing::Test {
protected:
  // two nodes with 2 CPUs and a system memory each; p0 and p1 on node 0, p2
  //  and p3 on node 1; every CPU reaches its own node's memory, and p2 also
  //  reaches node 0's memory
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl_ = std::make_unique<MockRuntimeImplMachineModel>();
    Realm::runtime_singleton = runtime_impl_.get();
    runtime_impl_->init(2);

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems;
    procs_mems.proc_infos.push_back({0, Processor::LOC_PROC, 0});
    procs_mems.proc_infos.push_back({1, Processor::LOC_PROC, 0});
    procs_mems.proc_infos.push_back({0, Processor::LOC_PROC, 1});
    procs_mems.proc_infos.push_back({1, Processor::LOC_PROC, 1});
    procs_mems.mem_infos.push_back({0, Memory::SYSTEM_MEM, size_t(1024) * 1024, 0});
    procs_mems.mem_infos.push_back({0, Memory::SYSTEM_MEM, size_t(1024) * 1024, 1});
    procs_mems.proc_mem_affinities.push_back({0, 0, 100, 10});
    procs_mems.proc_mem_affinities.push_back({1, 0, 100, 10});
    procs_mems.proc_mem_affinities.push_back({2, 1, 100, 10});
    procs_mems.proc_mem_affinities.push_back({3, 1, 100, 10});
    procs_mems.proc_mem_affinities.push_back({2, 0, 10, 1000});
    runtime_impl_->setup_mock_proc_mems(procs_mems);

    procs[0] = ID::make_processor(0, 0).convert<Processor>();
    procs[1] = ID::make_processor(0, 1).convert<Processor>();
    procs[2] = ID::make_processor(1, 0).convert<Processor>();
    procs[3] = ID::make_processor(1, 1).convert<Processor>();
    mems[0] = ID::make_memory(0, 0).convert<Memory>();
    mems[1] = ID::make_memory(1, 0).convert<Memory>();
  }

  void TearDown() override
  {
    runtime_impl_->finalize();
    runtime_impl_.reset();
    Realm::runtime_singleton = nullptr;
  }

  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl_;
  Processor procs[4];
  Memory mems[2];
};

TEST_F(MachineQueryCacheMultiNodeTest, NonLocalAffinity)
{
  Machine machine = Machine::get_machine();
  Machine::ProcessorQuery sharing(machine);
  sharing.only_kind(Processor::LOC_PROC).has_affinity_to(mems[0]);
  std::set<Processor> expected{procs[0], procs[1], procs[2]};
  EXPECT_EQ(MachineQueryCacheTest::results(sharing), expected);
  MachineQueryCacheTest::expect_cache_agrees_with_walk(sharing);
  // and the ordinary by-kind list spans both nodes
  Machine::ProcessorQuery all(machine);
  all.only_kind(Processor::LOC_PROC);
  EXPECT_EQ(all.count(), 4u);
  MachineQueryCacheTest::expect_cache_agrees_with_walk(all);
  Machine::MemoryQuery visible(machine);
  visible.has_affinity_to(procs[2]);
  EXPECT_EQ(MachineQueryCacheTest::results(visible), (std::set<Memory>{mems[0], mems[1]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(visible);
}

// a query restricted to one node is served the run of the shared by-kind
//  (or by-kind-and-memory) list that node owns: every accessor must see
//  exactly that node's entries, before and after the machine changes
TEST_F(MachineQueryCacheMultiNodeTest, NodeRestrictedQueries)
{
  Machine machine = Machine::get_machine();
  typedef std::set<Processor> PSet;
  typedef std::set<Memory> MSet;

  Machine::ProcessorQuery node1(machine);
  node1.only_kind(Processor::LOC_PROC).same_address_space_as(procs[2]);
  EXPECT_EQ(MachineQueryCacheTest::results(node1), (PSet{procs[2], procs[3]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(node1);

  Machine::ProcessorQuery node0(machine);
  node0.only_kind(Processor::LOC_PROC).same_address_space_as(mems[0]);
  EXPECT_EQ(MachineQueryCacheTest::results(node0), (PSet{procs[0], procs[1]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(node0);

  // this process is node 0 in the mock machine
  Machine::ProcessorQuery local(machine);
  local.local_address_space().only_kind(Processor::LOC_PROC);
  EXPECT_EQ(MachineQueryCacheTest::results(local), (PSet{procs[0], procs[1]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(local);

  // the node's run of a by-kind-and-memory list: only p2 on node 1 reaches m0
  Machine::ProcessorQuery sharing1(machine);
  sharing1.only_kind(Processor::LOC_PROC).has_affinity_to(mems[0]).same_address_space_as(
      procs[3]);
  EXPECT_EQ(MachineQueryCacheTest::results(sharing1), (PSet{procs[2]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(sharing1);
  Machine::ProcessorQuery sharing0(machine);
  sharing0.same_address_space_as(procs[0]).only_kind(Processor::LOC_PROC).has_affinity_to(
      mems[0]);
  EXPECT_EQ(MachineQueryCacheTest::results(sharing0), (PSet{procs[0], procs[1]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(sharing0);

  // a kind the node does not have, and two different nodes, match nothing
  Machine::ProcessorQuery no_kind(machine);
  no_kind.only_kind(Processor::TOC_PROC).same_address_space_as(procs[0]);
  EXPECT_TRUE(MachineQueryCacheTest::results(no_kind).empty());
  MachineQueryCacheTest::expect_cache_agrees_with_walk(no_kind);
  Machine::ProcessorQuery two_nodes(machine);
  two_nodes.only_kind(Processor::LOC_PROC)
      .same_address_space_as(procs[0])
      .same_address_space_as(procs[2]);
  EXPECT_TRUE(MachineQueryCacheTest::results(two_nodes).empty());
  MachineQueryCacheTest::expect_cache_agrees_with_walk(two_nodes);

  // memories by kind, per node
  Machine::MemoryQuery mnode1(machine);
  mnode1.only_kind(Memory::SYSTEM_MEM).same_address_space_as(procs[2]);
  EXPECT_EQ(MachineQueryCacheTest::results(mnode1), (MSet{mems[1]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(mnode1);
  Machine::MemoryQuery mlocal(machine);
  mlocal.local_address_space().only_kind(Memory::SYSTEM_MEM);
  EXPECT_EQ(MachineQueryCacheTest::results(mlocal), (MSet{mems[0]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(mlocal);
  Machine::MemoryQuery mno_kind(machine);
  mno_kind.only_kind(Memory::GPU_FB_MEM).same_address_space_as(procs[0]);
  EXPECT_TRUE(MachineQueryCacheTest::results(mno_kind).empty());
  MachineQueryCacheTest::expect_cache_agrees_with_walk(mno_kind);
  Machine::MemoryQuery mtwo_nodes(machine);
  mtwo_nodes.only_kind(Memory::SYSTEM_MEM)
      .same_address_space_as(procs[0])
      .same_address_space_as(procs[2]);
  EXPECT_TRUE(MachineQueryCacheTest::results(mtwo_nodes).empty());
  MachineQueryCacheTest::expect_cache_agrees_with_walk(mtwo_nodes);

  // next() from an entry outside the node has no successor, as in a walk
  {
    const bool saved = Config::use_machine_query_cache;
    for(bool cache : {true, false}) {
      Config::use_machine_query_cache = cache;
      EXPECT_EQ(node1.next(procs[0]), Processor::NO_PROC);
      EXPECT_EQ(node1.next(procs[2]), procs[3]);
      EXPECT_EQ(node1.next(procs[3]), Processor::NO_PROC);
      EXPECT_EQ(node0.next(procs[1]), Processor::NO_PROC);
      EXPECT_EQ(node0.next(procs[2]), Processor::NO_PROC);
      EXPECT_EQ(sharing1.next(procs[0]), Processor::NO_PROC);
      EXPECT_EQ(mnode1.next(mems[0]), Memory::NO_MEMORY);
      EXPECT_EQ(mlocal.next(mems[0]), Memory::NO_MEMORY);
    }
    Config::use_machine_query_cache = saved;
  }

  // a machine update (p3 now reaches m0 too) is seen by the executed queries
  Machine::ProcessorMemoryAffinity pma;
  pma.p = procs[3];
  pma.m = mems[0];
  pma.bandwidth = 10;
  pma.latency = 1000;
  runtime_impl_->machine->add_proc_mem_affinity(pma);
  EXPECT_EQ(MachineQueryCacheTest::results(sharing1), (PSet{procs[2], procs[3]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(sharing1);
  EXPECT_EQ(MachineQueryCacheTest::results(sharing0), (PSet{procs[0], procs[1]}));
  EXPECT_EQ(MachineQueryCacheTest::results(node1), (PSet{procs[2], procs[3]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(node1);
}
