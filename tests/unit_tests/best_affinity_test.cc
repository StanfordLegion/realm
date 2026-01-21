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
#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <limits>

using namespace Realm;

class BestAffinityTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    std::vector<const char *> cmdline_argv;
    const char dummy_args[] = "test";
    const char cpu_cmd[] = "-ll:cpu";
    const char cpu[] = "4";
    const char util_cmd[] = "-ll:util";
    const char util[] = "2";
    cmdline_argv.push_back(dummy_args);
    cmdline_argv.push_back(cpu_cmd);
    cmdline_argv.push_back(cpu);
    cmdline_argv.push_back(util_cmd);
    cmdline_argv.push_back(util);

    int argc = cmdline_argv.size();
    char **argv = const_cast<char **>(cmdline_argv.data());

    runtime_ = new Runtime();
    runtime_->init(&argc, &argv);
  }

  void TearDown() override
  {
    runtime_->shutdown();
    runtime_->wait_for_shutdown();
    delete runtime_;
    runtime_ = nullptr;
  }

  // Helper to get machine
  Machine get_machine() { return Machine::get_machine(); }

  Runtime *runtime_;
};

// Test basic processor best affinity with default weights
TEST_F(BestAffinityTest, ProcessorBestAffinityDefault)
{
  Machine machine = get_machine();
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  ASSERT_TRUE(mem.exists()) << "No system memory found";

  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  size_t count = pq.count();
  EXPECT_GT(count, 0) << "No processors with best affinity found";

  // Verify we can iterate through results
  size_t iter_count = 0;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    EXPECT_TRUE(p.kind() == Processor::LOC_PROC);
    iter_count++;
  }
  EXPECT_EQ(iter_count, count) << "Iteration count doesn't match query count";
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
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem, 0, 1);

  size_t count = pq.count();
  EXPECT_GT(count, 0) << "No processors with best latency found";

  // Verify custom weights work - just check we get some results
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

  // Find the best combined score
  Machine::MemoryQuery all_mems(machine);
  all_mems.only_kind(Memory::SYSTEM_MEM);

  int best_combined_score = std::numeric_limits<int>::min();
  for(Memory m = all_mems.first(); m.exists(); m = all_mems.next(m)) {
    int proc_score = std::numeric_limits<int>::min();
    int mem_score = std::numeric_limits<int>::min();

    std::vector<Machine::ProcessorMemoryAffinity> proc_affinities;
    machine.get_proc_mem_affinity(proc_affinities, cpu, m);
    if(!proc_affinities.empty()) {
      proc_score = proc_affinities[0].bandwidth;
    }

    std::vector<Machine::MemoryMemoryAffinity> mem_affinities;
    machine.get_mem_mem_affinity(mem_affinities, m, target_mem);
    if(!mem_affinities.empty()) {
      mem_score = mem_affinities[0].bandwidth;
    }

    if(proc_score != std::numeric_limits<int>::min() &&
       mem_score != std::numeric_limits<int>::min()) {
      int combined = proc_score + mem_score;
      if(combined > best_combined_score) {
        best_combined_score = combined;
      }
    }
  }

  // Verify all results have the best combined score
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    std::vector<Machine::ProcessorMemoryAffinity> proc_affinities;
    machine.get_proc_mem_affinity(proc_affinities, cpu, m);
    ASSERT_FALSE(proc_affinities.empty());

    std::vector<Machine::MemoryMemoryAffinity> mem_affinities;
    machine.get_mem_mem_affinity(mem_affinities, m, target_mem);
    ASSERT_FALSE(mem_affinities.empty());

    int combined = proc_affinities[0].bandwidth + mem_affinities[0].bandwidth;
    EXPECT_EQ(combined, best_combined_score)
        << "Memory " << m << " does not have best combined score";
  }
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

  ASSERT_GT(best_procs.size(), 0);

  // If there are multiple results, verify they all have the same bandwidth
  if(best_procs.size() > 1) {
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
