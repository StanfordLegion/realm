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
#include <optional>
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
    // rebuild the shared lists, so the readers keep moving to new ones while
    //  the old ones stay valid (nothing is freed before shutdown)
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
//  every answer must be the one for one of the two machine states. A missing
//  lock is a data race that ThreadSanitizer reports, and CI runs this test
//  under it; a plain build only checks the answers, and since the writer
//  rewrites one field in place and both values are acceptable, a plain build
//  cannot tell a missing lock from a present one
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

// a best-affinity query keeps the list it built until the machine changes:
//  every update advances the generation its list is stamped with, so the same
//  query object must see an update that changes its answer, whether or not
//  the shared cache is in use

TEST_F(MachineQueryCacheTest, ExecutedBestAffinityQuerySeesUpdates)
{
  typedef std::set<Processor> PSet;
  typedef std::set<Memory> MSet;
  const bool saved = Config::use_machine_query_cache;
  for(bool cache : {true, false}) {
    Config::use_machine_query_cache = cache;
    // best for m1: p2 and p3 (90 against 80); best for p0: m0 (100 against 80)
    Machine::ProcessorQuery best(machine());
    best.only_kind(Processor::LOC_PROC).best_affinity_to(mems[1]);
    Machine::ProcessorQuery best_any(machine());
    best_any.best_affinity_to(mems[1]);
    Machine::MemoryQuery best_mem(machine());
    best_mem.only_kind(Memory::SYSTEM_MEM).best_affinity_to(procs[0]);
    EXPECT_EQ(results(best), (PSet{procs[2], procs[3]}));
    EXPECT_EQ(results(best_any), (PSet{procs[2], procs[3]}));
    EXPECT_EQ(results(best_mem), (MSet{mems[0]}));

    // p1 -> m1 raised above the rest, p0 -> m0 lowered below p0 -> m1
    set_affinity(procs[1], mems[1], 120, 15);
    set_affinity(procs[0], mems[0], 70, 10);
    EXPECT_EQ(results(best), (PSet{procs[1]})) << "cache " << cache;
    EXPECT_EQ(best.first(), procs[1]);
    EXPECT_EQ(best.count(), 1u);
    EXPECT_EQ(best.next(procs[1]), Processor::NO_PROC);
    EXPECT_EQ(results(best_any), (PSet{procs[1]})) << "cache " << cache;
    EXPECT_EQ(results(best_mem), (MSet{mems[1]})) << "cache " << cache;
    EXPECT_EQ(best_mem.first(), mems[1]);
    EXPECT_EQ(best_mem.count(), 1u);

    // and back
    set_affinity(procs[1], mems[1], 80, 15);
    set_affinity(procs[0], mems[0], 100, 10);
    EXPECT_EQ(results(best), (PSet{procs[2], procs[3]})) << "cache " << cache;
    EXPECT_EQ(results(best_any), (PSet{procs[2], procs[3]}));
    EXPECT_EQ(results(best_mem), (MSet{mems[0]})) << "cache " << cache;
  }
  Config::use_machine_query_cache = saved;
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
  EXPECT_EQ(Machine::ProcessorQuery(machine()).only_kind(Processor::TOC_PROC).count(),
            0u);
  EXPECT_FALSE(
      Machine::ProcessorQuery(machine()).only_kind(Processor::TOC_PROC).first().exists());
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
  expect_cache_agrees_with_walk(
      Machine::MemoryQuery(machine()).has_affinity_to(procs[3]));
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

// next(after) remembers, per thread, which list the previous call stepped
//  through and where, and checks the hint against the live list before using
//  it; so it must be right whatever the previous call on this thread was:
//  a step through another query's list, another position, a query whose
//  list has since been freed, or the list from before a machine update
TEST_F(MachineQueryCacheTest, NextIsRightWhateverThePreviousCallWas)
{
  Machine::ProcessorQuery cpus(machine());
  cpus.only_kind(Processor::LOC_PROC); // shared list: p0..p3
  Machine::MemoryQuery sys(machine());
  sys.only_kind(Memory::SYSTEM_MEM); // shared list: m0, m1
  Machine::ProcessorQuery best(machine());
  best.only_kind(Processor::LOC_PROC).best_affinity_to(mems[1]); // own list: p2, p3

  // three iterations interleaved on one thread
  EXPECT_EQ(cpus.first(), procs[0]);
  EXPECT_EQ(sys.first(), mems[0]);
  EXPECT_EQ(best.first(), procs[2]);
  EXPECT_EQ(cpus.next(procs[0]), procs[1]);
  EXPECT_EQ(sys.next(mems[0]), mems[1]);
  EXPECT_EQ(best.next(procs[2]), procs[3]);
  EXPECT_EQ(cpus.next(procs[1]), procs[2]);
  EXPECT_EQ(sys.next(mems[1]), Memory::NO_MEMORY);
  EXPECT_EQ(best.next(procs[3]), Processor::NO_PROC);
  EXPECT_EQ(cpus.next(procs[2]), procs[3]);
  EXPECT_EQ(cpus.next(procs[3]), Processor::NO_PROC);

  // out of order, repeated, backwards, and from values that are not results
  EXPECT_EQ(cpus.next(procs[0]), procs[1]);
  EXPECT_EQ(cpus.next(procs[3]), Processor::NO_PROC);
  EXPECT_EQ(cpus.next(procs[2]), procs[3]);
  EXPECT_EQ(cpus.next(procs[2]), procs[3]);
  EXPECT_EQ(cpus.next(procs[1]), procs[2]);
  EXPECT_EQ(cpus.next(procs[0]), procs[1]);
  EXPECT_EQ(cpus.next(Processor::NO_PROC), Processor::NO_PROC);
  EXPECT_EQ(sys.next(Memory::NO_MEMORY), Memory::NO_MEMORY);
  EXPECT_EQ(best.next(Processor::NO_PROC), Processor::NO_PROC);
  EXPECT_EQ(cpus.next(ID::make_processor(0, 9).convert<Processor>()), Processor::NO_PROC);
  EXPECT_EQ(best.next(procs[0]), procs[2]);
  EXPECT_EQ(best.next(procs[1]), procs[2]);
  EXPECT_EQ(sys.next(mems[2]), Memory::NO_MEMORY);

  // a query's own list dies with the query, and the next query's list can be
  //  allocated at the same address with other entries at the same positions:
  //  a hint left by the old list must not be trusted
  for(int i = 0; i < 64; i++) {
    const bool to_m1 = (i % 2) == 1;
    Machine::ProcessorQuery q(machine());
    q.only_kind(Processor::LOC_PROC).best_affinity_to(to_m1 ? mems[1] : mems[0]);
    const Processor first = to_m1 ? procs[2] : procs[0];
    const Processor second = to_m1 ? procs[3] : procs[1];
    if((i % 4) < 2)
      EXPECT_EQ(q.first(), first);
    EXPECT_EQ(q.next(first), second) << i;
    EXPECT_EQ(q.next(second), Processor::NO_PROC) << i;
  }

  // across a machine update the shared list is rebuilt elsewhere, and the
  //  iteration goes on from the same place in the new one
  EXPECT_EQ(cpus.first(), procs[0]);
  EXPECT_EQ(cpus.next(procs[0]), procs[1]);
  invalidate_caches();
  EXPECT_EQ(cpus.next(procs[1]), procs[2]);
  EXPECT_EQ(cpus.next(procs[2]), procs[3]);
  EXPECT_EQ(sys.first(), mems[0]);
  invalidate_caches();
  EXPECT_EQ(sys.next(mems[0]), mems[1]);
  EXPECT_EQ(best.first(), procs[2]);
  invalidate_caches();
  EXPECT_EQ(best.next(procs[2]), procs[3]);
}

// after a machine update, the first thread whose query misses the shared
//  snapshot builds it and the others wait for that build instead of each
//  building their own (QueryCacheSlot::get_or_build). Generations only move
//  forward and a build is attempted only for the live generation, so each
//  snapshot kind is built at most once per update, however many threads
//  query. The readers are made to query after every update, so without
//  single-flight builds the count would approach one build per reader per
//  update
TEST_F(MachineQueryCacheTest, OneSnapshotBuildPerUpdate)
{
  const int num_readers = 8;
  const int num_updates = 100;
  auto query_all = [&]() -> int {
    int wrong = 0;
    if(Machine::ProcessorQuery(machine()).only_kind(Processor::LOC_PROC).count() != 4)
      wrong++;
    if(Machine::ProcessorQuery(machine())
           .only_kind(Processor::LOC_PROC)
           .has_affinity_to(mems[0])
           .count() != 4)
      wrong++;
    if(Machine::MemoryQuery(machine()).only_kind(Memory::SYSTEM_MEM).count() != 2)
      wrong++;
    return wrong;
  };
  // the snapshots for the current generation exist before counting starts
  ASSERT_EQ(query_all(), 0);
  const unsigned long kind_before = ProcessorQueryImpl::kind_cache.build_count();
  const unsigned long affinity_before = ProcessorQueryImpl::affinity_cache.build_count();
  const unsigned long mem_before = MemoryQueryImpl::kind_cache.build_count();

  std::atomic<bool> stop(false);
  std::atomic<int> failures(0);
  std::unique_ptr<std::atomic<unsigned long>[]> completed(
      new std::atomic<unsigned long>[num_readers]);
  for(int r = 0; r < num_readers; r++)
    completed[r].store(0);
  std::vector<std::thread> readers;
  for(int r = 0; r < num_readers; r++)
    readers.emplace_back([&, r] {
      while(!stop.load()) {
        failures.fetch_add(query_all());
        completed[r].fetch_add(1);
      }
    });
  for(int u = 0; u < num_updates; u++) {
    unsigned long seen[num_readers];
    for(int r = 0; r < num_readers; r++)
      seen[r] = completed[r].load();
    invalidate_caches();
    // every reader finishes a round that started after the update (the round
    //  in flight at the update may have used the old snapshot; the next one
    //  cannot), so every reader misses on every update
    for(int r = 0; r < num_readers; r++)
      while(completed[r].load() < seen[r] + 2)
        std::this_thread::yield();
  }
  stop.store(true);
  for(std::thread &t : readers)
    t.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_LE(ProcessorQueryImpl::kind_cache.build_count() - kind_before,
            (unsigned long)num_updates);
  EXPECT_LE(ProcessorQueryImpl::affinity_cache.build_count() - affinity_before,
            (unsigned long)num_updates);
  EXPECT_LE(MemoryQueryImpl::kind_cache.build_count() - mem_before,
            (unsigned long)num_updates);
  // and the readers did make every snapshot kind get rebuilt (so none of the
  //  bounds above holds because the cache stopped being used)
  EXPECT_GE(ProcessorQueryImpl::kind_cache.build_count() - kind_before,
            (unsigned long)num_updates / 2);
  EXPECT_GE(ProcessorQueryImpl::affinity_cache.build_count() - affinity_before,
            (unsigned long)num_updates / 2);
  EXPECT_GE(MemoryQueryImpl::kind_cache.build_count() - mem_before,
            (unsigned long)num_updates / 2);
}

// best-affinity queries are built from the target memory's own affinity map:
//  only a processor or memory with a same-node affinity to the target can
//  score, so that map holds every possible result. The answers must be
//  exactly those of a brute-force walk over every candidate through the
//  public API, on a machine with a second processor kind, cross-node
//  affinities (which never score), a memory nothing reaches and a second node

namespace {
  int affinity_score(unsigned bandwidth, unsigned latency, int bandwidth_weight,
                     int latency_weight)
  {
    return int(bandwidth) * bandwidth_weight + int(latency) * latency_weight;
  }

  // what ProcessorQuery::best_affinity_to(m) must return: of the processors
  //  that pass the other restrictions, those with the highest score over a
  //  same-node affinity to m (a processor without one is not a candidate)
  std::set<Processor> reference_best_procs(const Machine &machine,
                                           std::optional<Processor::Kind> kind,
                                           std::optional<AddressSpace> node,
                                           std::optional<Memory> must_reach, Memory m,
                                           int bandwidth_weight, int latency_weight)
  {
    std::set<Processor> all;
    machine.get_all_processors(all);
    std::optional<int> best;
    std::set<Processor> result;
    for(Processor p : all) {
      if(kind && (p.kind() != *kind))
        continue;
      if(node && (p.address_space() != *node))
        continue;
      if(must_reach && !machine.has_affinity(p, *must_reach))
        continue;
      std::vector<Machine::ProcessorMemoryAffinity> pmas;
      machine.get_proc_mem_affinity(pmas, p, m, true /*local_only*/);
      if(pmas.empty())
        continue;
      int score = affinity_score(pmas[0].bandwidth, pmas[0].latency, bandwidth_weight,
                                 latency_weight);
      if(!best || (score > *best)) {
        best = score;
        result.clear();
        result.insert(p);
      } else if(score == *best)
        result.insert(p);
    }
    return result;
  }

  // what MemoryQuery::best_affinity_to(m) (with no processor target) must
  //  return: of the memories that pass the other restrictions, those with the
  //  highest score over a same-node affinity from m
  std::set<Memory> reference_best_mems(const Machine &machine,
                                       std::optional<Memory::Kind> kind,
                                       std::optional<AddressSpace> node,
                                       size_t min_capacity,
                                       std::optional<Processor> must_reach, Memory m,
                                       int bandwidth_weight, int latency_weight)
  {
    std::set<Memory> all;
    machine.get_all_memories(all);
    std::optional<int> best;
    std::set<Memory> result;
    for(Memory c : all) {
      if(kind && (c.kind() != *kind))
        continue;
      if(node && (c.address_space() != *node))
        continue;
      if((min_capacity > 0) && (c.capacity() < min_capacity))
        continue;
      if(must_reach && !machine.has_affinity(*must_reach, c))
        continue;
      std::vector<Machine::MemoryMemoryAffinity> mmas;
      machine.get_mem_mem_affinity(mmas, m, c, true /*local_only*/);
      if(mmas.empty())
        continue;
      int score = affinity_score(mmas[0].bandwidth, mmas[0].latency, bandwidth_weight,
                                 latency_weight);
      if(!best || (score > *best)) {
        best = score;
        result.clear();
        result.insert(c);
      } else if(score == *best)
        result.insert(c);
    }
    return result;
  }
} // namespace

class BestAffinityAnchorTest : public ::testing::Test {
protected:
  // node 0: CPUs p0 and p1, GPU g0; memories m0 (system), f0 (framebuffer,
  //  256 KB) and o0 (zero-copy, nothing reaches it). node 1: CPU p2 and
  //  system memory m1.
  //   p0 -> m0 100/10    p1 -> m0 100/10    g0 -> m0 200/5
  //   p0 -> f0  50/20                       g0 -> f0 500/1
  //   p2 -> m1 100/10
  //   cross-node, so never scored: p2 -> m0 1000/1 and p1 -> m1 300/1
  //   m0 -> f0 50/20, f0 -> m0 40/30; cross-node: m0 -> m1 and m1 -> m0 900/1
  void SetUp() override
  {
    Realm::enable_unit_tests = true;
    runtime_impl_ = std::make_unique<MockRuntimeImplMachineModel>();
    Realm::runtime_singleton = runtime_impl_.get();
    runtime_impl_->init(2);

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded pm;
    pm.proc_infos.push_back({0, Processor::LOC_PROC, 0});                    // p0
    pm.proc_infos.push_back({1, Processor::LOC_PROC, 0});                    // p1
    pm.proc_infos.push_back({2, Processor::TOC_PROC, 0});                    // g0
    pm.proc_infos.push_back({0, Processor::LOC_PROC, 1});                    // p2
    pm.mem_infos.push_back({0, Memory::SYSTEM_MEM, size_t(1024) * 1024, 0}); // m0
    pm.mem_infos.push_back({1, Memory::GPU_FB_MEM, size_t(256) * 1024, 0});  // f0
    pm.mem_infos.push_back({2, Memory::Z_COPY_MEM, size_t(1024) * 1024, 0}); // o0
    pm.mem_infos.push_back({0, Memory::SYSTEM_MEM, size_t(1024) * 1024, 1}); // m1
    pm.proc_mem_affinities.push_back({0, 0, 100, 10});
    pm.proc_mem_affinities.push_back({1, 0, 100, 10});
    pm.proc_mem_affinities.push_back({2, 0, 200, 5});
    pm.proc_mem_affinities.push_back({0, 1, 50, 20});
    pm.proc_mem_affinities.push_back({2, 1, 500, 1});
    pm.proc_mem_affinities.push_back({3, 3, 100, 10});
    pm.proc_mem_affinities.push_back({3, 0, 1000, 1});
    pm.proc_mem_affinities.push_back({1, 3, 300, 1});
    pm.mem_mem_affinities.push_back({0, 1, 50, 20});
    pm.mem_mem_affinities.push_back({1, 0, 40, 30});
    pm.mem_mem_affinities.push_back({0, 3, 900, 1});
    pm.mem_mem_affinities.push_back({3, 0, 900, 1});
    runtime_impl_->setup_mock_proc_mems(pm);

    p0 = ID::make_processor(0, 0).convert<Processor>();
    p1 = ID::make_processor(0, 1).convert<Processor>();
    g0 = ID::make_processor(0, 2).convert<Processor>();
    p2 = ID::make_processor(1, 0).convert<Processor>();
    m0 = ID::make_memory(0, 0).convert<Memory>();
    f0 = ID::make_memory(0, 1).convert<Memory>();
    o0 = ID::make_memory(0, 2).convert<Memory>();
    m1 = ID::make_memory(1, 0).convert<Memory>();
    // o0 has no affinities, so nothing above put it into the machine model
    runtime_impl_->machine->nodeinfos[0]->add_memory(o0);
    runtime_impl_->machine->update_kind_maps();
  }

  void TearDown() override
  {
    runtime_impl_->finalize();
    runtime_impl_.reset();
    Realm::runtime_singleton = nullptr;
  }

  std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl_;
  Processor p0, p1, g0, p2;
  Memory m0, f0, o0, m1;
};

TEST_F(BestAffinityAnchorTest, ProcessorSpotChecks)
{
  Machine machine = Machine::get_machine();
  typedef std::set<Processor> PSet;
  // g0 scores higher but is not a CPU; p2 reaches m0 at 1000 but from node 1
  EXPECT_EQ(MachineQueryCacheTest::results(Machine::ProcessorQuery(machine)
                                               .only_kind(Processor::LOC_PROC)
                                               .best_affinity_to(m0)),
            (PSet{p0, p1}));
  EXPECT_EQ(MachineQueryCacheTest::results(
                Machine::ProcessorQuery(machine).best_affinity_to(m0)),
            (PSet{g0}));
  // restricted to the other node: nothing there has a same-node affinity to m0
  EXPECT_TRUE(MachineQueryCacheTest::results(Machine::ProcessorQuery(machine)
                                                 .only_kind(Processor::LOC_PROC)
                                                 .same_address_space_as(p2)
                                                 .best_affinity_to(m0))
                  .empty());
  EXPECT_EQ(MachineQueryCacheTest::results(Machine::ProcessorQuery(machine)
                                               .only_kind(Processor::LOC_PROC)
                                               .same_address_space_as(p0)
                                               .best_affinity_to(m0)),
            (PSet{p0, p1}));
  // a predicate on top: only p0 also reaches f0
  EXPECT_EQ(MachineQueryCacheTest::results(Machine::ProcessorQuery(machine)
                                               .only_kind(Processor::LOC_PROC)
                                               .has_affinity_to(f0)
                                               .best_affinity_to(m0)),
            (PSet{p0}));
  // nothing reaches o0; p1 -> m1 at 300 is cross-node, so p2 wins m1
  EXPECT_TRUE(MachineQueryCacheTest::results(
                  Machine::ProcessorQuery(machine).best_affinity_to(o0))
                  .empty());
  EXPECT_EQ(MachineQueryCacheTest::results(Machine::ProcessorQuery(machine)
                                               .only_kind(Processor::LOC_PROC)
                                               .best_affinity_to(m1)),
            (PSet{p2}));
  EXPECT_TRUE(MachineQueryCacheTest::results(Machine::ProcessorQuery(machine)
                                                 .only_kind(Processor::TOC_PROC)
                                                 .best_affinity_to(m1))
                  .empty());
  // weights: the highest weighted sum wins, so a positive latency weight
  //  prefers the larger latency (20 over 1) - the existing semantics
  EXPECT_EQ(MachineQueryCacheTest::results(
                Machine::ProcessorQuery(machine).best_affinity_to(f0, 0, 1)),
            (PSet{p0}));
  EXPECT_EQ(MachineQueryCacheTest::results(
                Machine::ProcessorQuery(machine).best_affinity_to(f0, 1, -1)),
            (PSet{g0}));
}

TEST_F(BestAffinityAnchorTest, ProcessorsMatchReference)
{
  Machine machine = Machine::get_machine();
  const std::optional<Processor::Kind> kinds[] = {
      std::nullopt, Processor::LOC_PROC, Processor::TOC_PROC, Processor::IO_PROC};
  const std::optional<AddressSpace> nodes[] = {std::nullopt, 0, 1};
  const std::optional<Memory> reach[] = {std::nullopt, f0, m0, m1};
  const Memory targets[] = {m0, f0, o0, m1};
  const int weights[][2] = {{1, 0}, {0, 1}, {1, -1}, {2, 3}};
  int checked = 0;
  for(const std::optional<Processor::Kind> &kind : kinds)
    for(const std::optional<AddressSpace> &node : nodes)
      for(const std::optional<Memory> &must_reach : reach)
        for(Memory target : targets)
          for(const int *w : weights) {
            Machine::ProcessorQuery q(machine);
            if(kind)
              q.only_kind(*kind);
            if(node)
              q.same_address_space_as((*node == 0) ? p0 : p2);
            if(must_reach)
              q.has_affinity_to(*must_reach);
            q.best_affinity_to(target, w[0], w[1]);
            std::set<Processor> expected =
                reference_best_procs(machine, kind, node, must_reach, target, w[0], w[1]);
            EXPECT_EQ(MachineQueryCacheTest::results(q), expected)
                << "kind " << (kind ? int(*kind) : -1) << " node "
                << (node ? int(*node) : -1) << " reach "
                << (must_reach ? must_reach->id : 0) << " target " << target.id
                << " weights " << w[0] << "," << w[1];
            checked++;
          }
  EXPECT_EQ(checked, 4 * 3 * 4 * 4 * 4);
}

TEST_F(BestAffinityAnchorTest, MemorySpotChecks)
{
  Machine machine = Machine::get_machine();
  typedef std::set<Memory> MSet;
  // m1 at 900 is cross-node, so f0 is m0's best
  EXPECT_EQ(
      MachineQueryCacheTest::results(Machine::MemoryQuery(machine).best_affinity_to(m0)),
      (MSet{f0}));
  EXPECT_TRUE(MachineQueryCacheTest::results(Machine::MemoryQuery(machine)
                                                 .only_kind(Memory::SYSTEM_MEM)
                                                 .best_affinity_to(m0))
                  .empty());
  EXPECT_TRUE(
      MachineQueryCacheTest::results(
          Machine::MemoryQuery(machine).same_address_space_as(p2).best_affinity_to(m0))
          .empty());
  EXPECT_EQ(
      MachineQueryCacheTest::results(
          Machine::MemoryQuery(machine).same_address_space_as(p0).best_affinity_to(m0)),
      (MSet{f0}));
  // f0 is 256 KB
  EXPECT_TRUE(MachineQueryCacheTest::results(Machine::MemoryQuery(machine)
                                                 .has_capacity(size_t(512) * 1024)
                                                 .best_affinity_to(m0))
                  .empty());
  // predicates on top: g0 reaches f0, p1 does not
  EXPECT_EQ(MachineQueryCacheTest::results(
                Machine::MemoryQuery(machine).has_affinity_to(g0).best_affinity_to(m0)),
            (MSet{f0}));
  EXPECT_TRUE(MachineQueryCacheTest::results(
                  Machine::MemoryQuery(machine).has_affinity_to(p1).best_affinity_to(m0))
                  .empty());
  EXPECT_EQ(
      MachineQueryCacheTest::results(Machine::MemoryQuery(machine).best_affinity_to(f0)),
      (MSet{m0}));
  EXPECT_TRUE(
      MachineQueryCacheTest::results(Machine::MemoryQuery(machine).best_affinity_to(o0))
          .empty());
  // m1's only outgoing affinity is cross-node
  EXPECT_TRUE(
      MachineQueryCacheTest::results(Machine::MemoryQuery(machine).best_affinity_to(m1))
          .empty());
  // with a processor target too (the processor-anchored path): p0 reaches m0
  //  and f0, and only f0 has an affinity from m0
  EXPECT_EQ(MachineQueryCacheTest::results(
                Machine::MemoryQuery(machine).best_affinity_to(p0).best_affinity_to(m0)),
            (MSet{f0}));
  EXPECT_EQ(
      MachineQueryCacheTest::results(Machine::MemoryQuery(machine).best_affinity_to(g0)),
      (MSet{f0}));
}

TEST_F(BestAffinityAnchorTest, MemoriesMatchReference)
{
  Machine machine = Machine::get_machine();
  const std::optional<Memory::Kind> kinds[] = {std::nullopt, Memory::SYSTEM_MEM,
                                               Memory::GPU_FB_MEM, Memory::Z_COPY_MEM};
  const std::optional<AddressSpace> nodes[] = {std::nullopt, 0, 1};
  const size_t capacities[] = {0, size_t(512) * 1024};
  const std::optional<Processor> reach[] = {std::nullopt, g0, p1, p2};
  const Memory targets[] = {m0, f0, o0, m1};
  const int weights[][2] = {{1, 0}, {0, 1}, {1, -1}, {2, 3}};
  int checked = 0;
  for(const std::optional<Memory::Kind> &kind : kinds)
    for(const std::optional<AddressSpace> &node : nodes)
      for(size_t capacity : capacities)
        for(const std::optional<Processor> &must_reach : reach)
          for(Memory target : targets)
            for(const int *w : weights) {
              Machine::MemoryQuery q(machine);
              if(kind)
                q.only_kind(*kind);
              if(node)
                q.same_address_space_as((*node == 0) ? p0 : p2);
              if(capacity)
                q.has_capacity(capacity);
              if(must_reach)
                q.has_affinity_to(*must_reach);
              q.best_affinity_to(target, w[0], w[1]);
              std::set<Memory> expected = reference_best_mems(
                  machine, kind, node, capacity, must_reach, target, w[0], w[1]);
              EXPECT_EQ(MachineQueryCacheTest::results(q), expected)
                  << "kind " << (kind ? int(*kind) : -1) << " node "
                  << (node ? int(*node) : -1) << " capacity " << capacity << " reach "
                  << (must_reach ? must_reach->id : 0) << " target " << target.id
                  << " weights " << w[0] << "," << w[1];
              checked++;
            }
  EXPECT_EQ(checked, 4 * 3 * 2 * 4 * 4 * 4);
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
  EXPECT_EQ(MachineQueryCacheTest::results(visible),
            (std::set<Memory>{mems[0], mems[1]}));
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
  sharing1.only_kind(Processor::LOC_PROC)
      .has_affinity_to(mems[0])
      .same_address_space_as(procs[3]);
  EXPECT_EQ(MachineQueryCacheTest::results(sharing1), (PSet{procs[2]}));
  MachineQueryCacheTest::expect_cache_agrees_with_walk(sharing1);
  Machine::ProcessorQuery sharing0(machine);
  sharing0.same_address_space_as(procs[0])
      .only_kind(Processor::LOC_PROC)
      .has_affinity_to(mems[0]);
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
      // and nothing follows "no processor" / "no memory"
      EXPECT_EQ(node1.next(Processor::NO_PROC), Processor::NO_PROC);
      EXPECT_EQ(local.next(Processor::NO_PROC), Processor::NO_PROC);
      EXPECT_EQ(mnode1.next(Memory::NO_MEMORY), Memory::NO_MEMORY);
    }
    Config::use_machine_query_cache = saved;
  }

  // the same for node-restricted best-affinity queries, whose lists are built
  //  per query: next() must answer the same before the query has run (the
  //  list is built for the call) and after it (the list is attached). Both
  //  CPUs of node 1 tie for m1, and only m1 is on p2's node
  {
    const bool saved = Config::use_machine_query_cache;
    for(bool cache : {true, false})
      for(int executed = 0; executed < 2; executed++) {
        Config::use_machine_query_cache = cache;
        Machine::ProcessorQuery best1(machine);
        best1.only_kind(Processor::LOC_PROC)
            .same_address_space_as(procs[2])
            .best_affinity_to(mems[1]);
        Machine::MemoryQuery mbest1(machine);
        mbest1.same_address_space_as(procs[2]).best_affinity_to(procs[2]);
        if(executed) {
          EXPECT_EQ(best1.first(), procs[2]);
          EXPECT_EQ(mbest1.first(), mems[1]);
        }
        EXPECT_EQ(best1.next(procs[0]), Processor::NO_PROC)
            << "cache " << cache << " executed " << executed;
        EXPECT_EQ(best1.next(procs[2]), procs[3]);
        EXPECT_EQ(best1.next(procs[3]), Processor::NO_PROC);
        EXPECT_EQ(mbest1.next(mems[0]), Memory::NO_MEMORY)
            << "cache " << cache << " executed " << executed;
        EXPECT_EQ(mbest1.next(mems[1]), Memory::NO_MEMORY);
        EXPECT_EQ(best1.next(Processor::NO_PROC), Processor::NO_PROC);
        EXPECT_EQ(mbest1.next(Memory::NO_MEMORY), Memory::NO_MEMORY);
        EXPECT_EQ(MachineQueryCacheTest::results(best1), (PSet{procs[2], procs[3]}));
        EXPECT_EQ(MachineQueryCacheTest::results(mbest1), (MSet{mems[1]}));
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
