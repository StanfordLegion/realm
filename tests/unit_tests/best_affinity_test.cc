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

#include "realm.h"
#include "test_mock.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <limits>

using namespace Realm;

namespace Realm {
  extern bool enable_unit_tests;
  extern RuntimeImpl *runtime_singleton;
}; // namespace Realm

class BestAffinityTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    Realm::enable_unit_tests = true;
    runtime_impl_ = std::make_unique<MockRuntimeImplMachineModel>();

    // Set runtime singleton so Machine::get_machine() works
    Realm::runtime_singleton = runtime_impl_.get();

    runtime_impl_->init(1); // Single node

    // Set up mock machine model with processors and memories
    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded procs_mems;

    // Create 4 CPU processors
    for(unsigned int i = 0; i < 4; i++) {
      procs_mems.proc_infos.push_back({i, Processor::LOC_PROC, 0});
    }

    // Create 2 system memories with different sizes
    procs_mems.mem_infos.push_back(
        {0, Memory::SYSTEM_MEM, static_cast<size_t>(1024) * 1024 * 1024, 0});
    procs_mems.mem_infos.push_back(
        {1, Memory::SYSTEM_MEM, static_cast<size_t>(2048) * 1024 * 1024, 0});

    // Set up processor-memory affinities with varying bandwidth/latency
    // Proc 0 and 1 have high bandwidth to mem 0
    procs_mems.proc_mem_affinities.push_back({0, 0, 100, 10});
    procs_mems.proc_mem_affinities.push_back({1, 0, 100, 10});
    // Proc 2 and 3 have lower bandwidth to mem 0
    procs_mems.proc_mem_affinities.push_back({2, 0, 50, 20});
    procs_mems.proc_mem_affinities.push_back({3, 0, 50, 20});

    // All procs have access to mem 1 with different characteristics
    procs_mems.proc_mem_affinities.push_back({0, 1, 80, 15});
    procs_mems.proc_mem_affinities.push_back({1, 1, 80, 15});
    procs_mems.proc_mem_affinities.push_back({2, 1, 90, 12});
    procs_mems.proc_mem_affinities.push_back({3, 1, 90, 12});

    // Set up memory-memory affinities
    procs_mems.mem_mem_affinities.push_back({0, 1, 60, 25});
    procs_mems.mem_mem_affinities.push_back({1, 0, 60, 25});

    runtime_impl_->setup_mock_proc_mems(procs_mems);
  }

  static void TearDownTestSuite()
  {
    if(runtime_impl_) {
      runtime_impl_->finalize();
      runtime_impl_.reset();

      // Clear runtime singleton
      Realm::runtime_singleton = nullptr;
    }
  }

  // Helper to get machine
  static Machine get_machine() { return Machine::get_machine(); }

  static std::unique_ptr<MockRuntimeImplMachineModel> runtime_impl_;
};

std::unique_ptr<MockRuntimeImplMachineModel> BestAffinityTest::runtime_impl_ = nullptr;

// Test basic processor best affinity with default weights (bandwidth only)
TEST_F(BestAffinityTest, ProcessorBestAffinityDefault)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists()) << "No system memory found";

  // Query for best affinity to first memory (should be proc 0 and 1 with bandwidth 100)
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  size_t count = pq.count();
  EXPECT_EQ(count, 2) << "Should find 2 processors with best bandwidth";

  // Verify we can iterate through results
  std::vector<Processor> best_procs;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    EXPECT_TRUE(p.kind() == Processor::LOC_PROC);
    best_procs.push_back(p);
  }
  EXPECT_EQ(best_procs.size(), count) << "Iteration count doesn't match query count";
}

// Test processor best affinity with custom latency weight
TEST_F(BestAffinityTest, ProcessorBestAffinityCustomWeights)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists()) << "No system memory found";

  // Query with latency weight only (bandwidth_weight=0, latency_weight=1)
  // Best latency to mem 0: proc 0 and 1 have latency 10
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem, 0, 1);

  size_t count = pq.count();
  EXPECT_EQ(count, 2) << "Should find 2 processors with best latency";

  // Verify custom weights work
  size_t iter_count = 0;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    EXPECT_TRUE(p.kind() == Processor::LOC_PROC);
    iter_count++;
  }
  EXPECT_EQ(iter_count, count) << "Iteration count doesn't match query count";
}

// Test memory best affinity to processor
TEST_F(BestAffinityTest, MemoryBestAffinityToProcessor)
{
  Machine machine = get_machine();
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  ASSERT_TRUE(cpu.exists()) << "No CPU processor found";

  // Find which memory has best affinity to cpu
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  mq.best_affinity_to(cpu);

  size_t count = mq.count();
  EXPECT_GT(count, 0) << "No memories with best affinity found";

  // Verify we can iterate through results
  size_t iter_count = 0;
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    EXPECT_TRUE(m.kind() == Memory::SYSTEM_MEM);
    iter_count++;
  }
  EXPECT_EQ(iter_count, count) << "Iteration count doesn't match query count";
}

// Test memory best affinity to both processor and memory
TEST_F(BestAffinityTest, MemoryBestAffinityToBoth)
{
  Machine machine = get_machine();
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  ASSERT_TRUE(cpu.exists()) << "No CPU processor found";

  Machine::MemoryQuery temp_mq(machine);
  temp_mq.only_kind(Memory::SYSTEM_MEM);
  Memory target_mem = temp_mq.first();
  ASSERT_TRUE(target_mem.exists()) << "No system memory found";

  // Query for memories with best combined affinity to both
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  mq.best_affinity_to(cpu);
  mq.best_affinity_to(target_mem);

  size_t count = mq.count();
  EXPECT_GT(count, 0) << "No memories with combined best affinity found";
}

// Test that ties return multiple results
TEST_F(BestAffinityTest, BestAffinityTies)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists()) << "No system memory found";

  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  std::vector<Processor> best_procs;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    best_procs.push_back(p);
  }

  // We set up proc 0 and 1 with identical affinity, so should get both
  ASSERT_EQ(best_procs.size(), 2);

  // Verify they all have the same bandwidth
  std::vector<Machine::ProcessorMemoryAffinity> first_affinities;
  machine.get_proc_mem_affinity(first_affinities, best_procs[0]);
  int first_bandwidth = -1;
  for(const auto &aff : first_affinities) {
    if(aff.m == mem) {
      first_bandwidth = aff.bandwidth;
      break;
    }
  }

  for(size_t i = 1; i < best_procs.size(); i++) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, best_procs[i]);
    int bandwidth = -1;
    for(const auto &aff : affinities) {
      if(aff.m == mem) {
        bandwidth = aff.bandwidth;
        break;
      }
    }
    EXPECT_EQ(bandwidth, first_bandwidth) << "Tied results have different bandwidths";
  }
}

// Test cache behavior - multiple calls should use cached results
TEST_F(BestAffinityTest, CacheBehavior)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists());

  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  // First access - builds cache
  Processor first1 = pq.first();
  size_t count1 = pq.count();

  // Second access - should use cache
  Processor first2 = pq.first();
  size_t count2 = pq.count();

  EXPECT_EQ(first1, first2) << "first() returned different results";
  EXPECT_EQ(count1, count2) << "count() returned different results";

  // Verify iteration is consistent
  std::vector<Processor> procs1, procs2;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    procs1.push_back(p);
  }
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    procs2.push_back(p);
  }

  ASSERT_EQ(procs1.size(), procs2.size()) << "Iteration returned different counts";
  for(size_t i = 0; i < procs1.size(); i++) {
    EXPECT_EQ(procs1[i], procs2[i])
        << "Iteration returned different processors at index " << i;
  }
}

// Test random() with best affinity
TEST_F(BestAffinityTest, RandomWithBestAffinity)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists());

  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  // Get all best affinity processors
  std::set<Processor> best_procs;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    best_procs.insert(p);
  }
  ASSERT_GT(best_procs.size(), 0);

  // Test random() returns one of the best affinity processors
  for(int i = 0; i < 10; i++) {
    Processor random_proc = pq.random();
    if(random_proc.exists()) {
      EXPECT_TRUE(best_procs.count(random_proc) > 0)
          << "random() returned processor not in best affinity set";
    }
  }
}

// Test that count(), first(), and next() are consistent
TEST_F(BestAffinityTest, ConsistencyBetweenMethods)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists());

  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  size_t count = pq.count();

  // Count via iteration
  size_t iter_count = 0;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    iter_count++;
  }

  EXPECT_EQ(count, iter_count) << "count() and iteration gave different results";
}

// Test query copy with best affinity
TEST_F(BestAffinityTest, QueryCopyWithBestAffinity)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists());

  Machine::ProcessorQuery pq1(machine);
  pq1.only_kind(Processor::LOC_PROC);
  pq1.best_affinity_to(mem);

  // Copy the query
  Machine::ProcessorQuery pq2 = pq1;

  // Both queries should return the same results
  size_t count1 = pq1.count();
  size_t count2 = pq2.count();
  EXPECT_EQ(count1, count2);

  Processor first1 = pq1.first();
  Processor first2 = pq2.first();
  EXPECT_EQ(first1, first2);
}

// Test combined weights
TEST_F(BestAffinityTest, CombinedWeights)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists());

  // Query with equal bandwidth and latency weights
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem, 1, 1);

  size_t count = pq.count();
  EXPECT_GT(count, 0);

  // Find the best combined score
  Machine::ProcessorQuery all_cpus(machine);
  all_cpus.only_kind(Processor::LOC_PROC);

  int best_score = std::numeric_limits<int>::min();
  for(Processor p = all_cpus.first(); p.exists(); p = all_cpus.next(p)) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, p, mem);
    if(!affinities.empty()) {
      int score = affinities[0].bandwidth + affinities[0].latency;
      if(score > best_score) {
        best_score = score;
      }
    }
  }

  // Verify all results have the best score
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, p, mem);
    ASSERT_FALSE(affinities.empty());
    int score = affinities[0].bandwidth + affinities[0].latency;
    EXPECT_EQ(score, best_score);
  }
}
