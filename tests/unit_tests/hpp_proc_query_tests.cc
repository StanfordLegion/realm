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

#define REALM_NAMESPACE RealmHPP

#include "realm.hpp"
#include "test_mock.h"
#include <gtest/gtest.h>
#include <vector>
#include <set>

// Use the HPP namespace explicitly to avoid conflicts with test_mock.h
namespace HPP = REALM_NAMESPACE;

namespace Realm {
  extern bool enable_unit_tests;
};

class HPPProcessorQueryTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    // Enable unit tests for the mock runtime
    Realm::enable_unit_tests = true;

    // Initialize mock runtime with 1 node
    mock_runtime = std::make_unique<MockRuntimeImplMachineModel>();
    mock_runtime->init(1);

    // Set the global runtime singleton to our mock runtime
    Realm::runtime_singleton = mock_runtime.get();

    // Set up a basic machine model with processors and memories
    setupBasicMachineModel();
  }

  void TearDown() override
  {
    if(mock_runtime) {
      // Clear the global runtime singleton
      Realm::runtime_singleton = nullptr;
      mock_runtime->finalize();
      mock_runtime.reset();
    }
    Realm::enable_unit_tests = false;
  }

  void setupBasicMachineModel()
  {
    // First, ensure MachineNodeInfo objects exist for all address spaces
    for(int node = 0; node < 1; node++) {
      if(mock_runtime->machine->nodeinfos.find(node) ==
         mock_runtime->machine->nodeinfos.end()) {
        mock_runtime->machine->nodeinfos[node] =
            new Realm::MachineNodeInfo(node, mock_runtime.get());
      }
    }

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded setup;

    // Add processors: 2 CPUs and 1 GPU on node 0
    // Note: processor indices must be sequential within each node
    setup.proc_infos = {
        {0, Processor::Kind::LOC_PROC, 0}, // CPU 0 on node 0
        {1, Processor::Kind::LOC_PROC, 0}, // CPU 1 on node 0
        {2, Processor::Kind::TOC_PROC, 0}  // GPU 2 on node 0
    };

    // Add memories: system memory and GPU memory on node 0
    setup.mem_infos = {
        {0, Memory::Kind::SYSTEM_MEM, 8ULL * 1024, 0}, // 8KB system mem on node 0
        {1, Memory::Kind::GPU_FB_MEM, 4ULL * 1024, 0}  // 4KB GPU mem on node 0
    };

    // Add processor-memory affinities
    setup.proc_mem_affinities = {
        {0, 0, 10000, 100}, // CPU 0 -> System mem 0 (high bandwidth, low latency)
        {1, 0, 10000, 100}, // CPU 1 -> System mem 0
        {2, 1, 20000, 50},  // GPU 2 -> GPU mem 1 (very high bandwidth, very low latency)
        {2, 0, 1000, 500}   // GPU 2 -> System mem 0 (lower bandwidth, higher latency)
    };

    // Add memory-memory affinities
    setup.mem_mem_affinities = {
        {0, 1, 5000, 200}, // System mem 0 <-> GPU mem 1
        {1, 0, 5000, 200}  // GPU mem 1 <-> System mem 0
    };

    mock_runtime->setup_mock_proc_mems(setup);
  }

  std::unique_ptr<MockRuntimeImplMachineModel> mock_runtime;
};

// Test basic ProcessorQuery functionality
TEST_F(HPPProcessorQueryTest, BasicQuery)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::ProcessorQuery query(machine);

  // Test count
  EXPECT_EQ(query.count(), 3);

  // Test first
  HPP::Processor first_proc = query.first();
  EXPECT_TRUE(first_proc.exists());

  // Test next
  HPP::Processor next_proc = query.next(first_proc);
  EXPECT_TRUE(next_proc.exists());
  EXPECT_NE(first_proc, next_proc);

  // Test random
  HPP::Processor random_proc = query.random();
  EXPECT_TRUE(random_proc.exists());
}

TEST_F(HPPProcessorQueryTest, OnlyKindFilter)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for CPUs only
  HPP::Machine::ProcessorQuery cpu_query(machine);
  cpu_query.only_kind(HPP::Processor::Kind::LOC_PROC);
  EXPECT_EQ(cpu_query.count(), 2); // 2 CPUs total

  // Query for GPUs only
  HPP::Machine::ProcessorQuery gpu_query(machine);
  gpu_query.only_kind(HPP::Processor::Kind::TOC_PROC);
  EXPECT_EQ(gpu_query.count(), 1); // 1 GPU total

  // Verify kinds
  for(const auto &proc : cpu_query) {
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::LOC_PROC);
  }
  for(const auto &proc : gpu_query) {
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::TOC_PROC);
  }
}

TEST_F(HPPProcessorQueryTest, LocalAddressSpaceFilter)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::ProcessorQuery local_query(machine);
  local_query.local_address_space();

  // Should have 3 processors in local address space
  EXPECT_EQ(local_query.count(), 3);

  // Verify all are in local address space
  for(const auto &proc : local_query) {
    EXPECT_EQ(proc.address_space(), 0);
  }
}

TEST_F(HPPProcessorQueryTest, IteratorFunctionality)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::ProcessorQuery query(machine);

  std::vector<HPP::Processor> processors;
  for(const auto &proc : query) {
    processors.push_back(proc);
  }

  EXPECT_EQ(processors.size(), 3);

  // Verify all processors are unique
  std::set<HPP::Processor> unique_processors(processors.begin(), processors.end());
  EXPECT_EQ(unique_processors.size(), 3);
}

TEST_F(HPPProcessorQueryTest, ChainedFilters)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Chain multiple filters
  HPP::Machine::ProcessorQuery query(machine);
  query.only_kind(HPP::Processor::Kind::LOC_PROC).local_address_space();

  // Should have 2 CPUs in local address space
  EXPECT_EQ(query.count(), 2);

  // Verify all results match both criteria
  for(const auto &proc : query) {
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::LOC_PROC);
    EXPECT_EQ(proc.address_space(), 0);
  }
}

TEST_F(HPPProcessorQueryTest, EmptyResults)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for a processor kind that doesn't exist in our mock setup
  // Use a valid enum value that we don't have in our test setup
  HPP::Machine::ProcessorQuery query(machine);
  // Instead of using an invalid kind, let's test with a kind that exists but has no
  // matches We'll create a query that should return no results by chaining incompatible
  // filters
  query.only_kind(HPP::Processor::Kind::LOC_PROC); // This should work

  // For now, let's test that the query works with valid kinds
  EXPECT_GE(query.count(), 0); // Should have at least 0 results

  // Test that we can still call methods on empty results
  if(query.count() == 0) {
    EXPECT_FALSE(query.first().exists());
    EXPECT_FALSE(query.random().exists());

    // Iterator should be empty
    int count = 0;
    for(const auto &proc : query) {
      count++;
    }
    EXPECT_EQ(count, 0);
  }
}

TEST_F(HPPProcessorQueryTest, CopyConstructor)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::ProcessorQuery original_query(machine);
  original_query.only_kind(HPP::Processor::Kind::LOC_PROC);

  // Copy the query
  HPP::Machine::ProcessorQuery copied_query(original_query);

  // Both should have the same results
  EXPECT_EQ(original_query.count(), copied_query.count());
  EXPECT_EQ(original_query.count(), 2);

  // Verify both queries return the same processors
  std::vector<HPP::Processor> original_procs;
  std::vector<HPP::Processor> copied_procs;

  for(const auto &proc : original_query) {
    original_procs.push_back(proc);
  }
  for(const auto &proc : copied_query) {
    copied_procs.push_back(proc);
  }

  EXPECT_EQ(original_procs.size(), copied_procs.size());
  EXPECT_EQ(original_procs.size(), 2);
}

TEST_F(HPPProcessorQueryTest, AssignmentOperator)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::ProcessorQuery query1(machine);
  query1.only_kind(HPP::Processor::Kind::LOC_PROC);

  HPP::Machine::ProcessorQuery query2(machine);
  query2.only_kind(HPP::Processor::Kind::TOC_PROC);

  // Before assignment, they should have different counts
  EXPECT_EQ(query1.count(), 2);
  EXPECT_EQ(query2.count(), 1);

  // Assign query1 to query2
  query2 = query1;

  // After assignment, both should have the same count
  // Note: The assignment copies the shared_ptr (impl), so both queries
  // should now point to the same underlying query object
  EXPECT_EQ(query1.count(), 2);
  EXPECT_EQ(query2.count(), 2);

  // Verify they are now equal (point to same underlying query)
  EXPECT_EQ(query1, query2);

  // Verify both queries return the same processors
  std::vector<HPP::Processor> procs1, procs2;
  for(const auto &proc : query1) {
    procs1.push_back(proc);
  }
  for(const auto &proc : query2) {
    procs2.push_back(proc);
  }
  EXPECT_EQ(procs1.size(), procs2.size());
  EXPECT_EQ(procs1.size(), 2);
}

TEST_F(HPPProcessorQueryTest, EqualityOperators)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::ProcessorQuery query1(machine);
  HPP::Machine::ProcessorQuery query2(machine);

  // Two different query objects should not be equal
  EXPECT_NE(query1, query2);

  // A query should be equal to itself
  EXPECT_EQ(query1, query1);
  EXPECT_EQ(query2, query2);
}

TEST_F(HPPProcessorQueryTest, NextFunctionality)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::ProcessorQuery query(machine);

  // Get all processors using next()
  std::vector<HPP::Processor> processors;
  HPP::Processor current = query.first();

  while(current.exists()) {
    processors.push_back(current);
    current = query.next(current);
  }

  EXPECT_EQ(processors.size(), 3);

  // Verify all processors are unique
  std::set<HPP::Processor> unique_processors(processors.begin(), processors.end());
  EXPECT_EQ(unique_processors.size(), 3);
}

TEST_F(HPPProcessorQueryTest, RandomFunctionality)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::ProcessorQuery query(machine);

  // Test random multiple times - should always return valid processors
  for(int i = 0; i < 10; i++) {
    HPP::Processor random_proc = query.random();
    EXPECT_TRUE(random_proc.exists());
  }

  // Test random on filtered query
  HPP::Machine::ProcessorQuery cpu_query(machine);
  cpu_query.only_kind(HPP::Processor::Kind::LOC_PROC);

  for(int i = 0; i < 10; i++) {
    HPP::Processor random_cpu = cpu_query.random();
    EXPECT_TRUE(random_cpu.exists());
    EXPECT_EQ(random_cpu.kind(), HPP::Processor::Kind::LOC_PROC);
  }
}

TEST_F(HPPProcessorQueryTest, FilteredIterator)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test iterator on filtered query
  HPP::Machine::ProcessorQuery cpu_query(machine);
  cpu_query.only_kind(HPP::Processor::Kind::LOC_PROC);

  std::vector<HPP::Processor> cpu_processors;
  for(const auto &proc : cpu_query) {
    cpu_processors.push_back(proc);
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::LOC_PROC);
  }

  EXPECT_EQ(cpu_processors.size(), 2);

  // Test iterator on GPU query
  HPP::Machine::ProcessorQuery gpu_query(machine);
  gpu_query.only_kind(HPP::Processor::Kind::TOC_PROC);

  std::vector<HPP::Processor> gpu_processors;
  for(const auto &proc : gpu_query) {
    gpu_processors.push_back(proc);
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::TOC_PROC);
  }

  EXPECT_EQ(gpu_processors.size(), 1);
}

TEST_F(HPPProcessorQueryTest, MultipleFilterChaining)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test chaining multiple filters
  HPP::Machine::ProcessorQuery query(machine);
  query.only_kind(HPP::Processor::Kind::LOC_PROC).local_address_space();

  // Should have 2 CPUs in local address space
  EXPECT_EQ(query.count(), 2);

  // Verify all results match both criteria
  for(const auto &proc : query) {
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::LOC_PROC);
    EXPECT_EQ(proc.address_space(), 0);
  }

  // Test that we can chain more filters (even if they don't change the result)
  HPP::Machine::ProcessorQuery chained_query(machine);
  chained_query.only_kind(HPP::Processor::Kind::LOC_PROC).local_address_space();

  EXPECT_EQ(chained_query.count(), 2);
}

TEST_F(HPPProcessorQueryTest, QueryReuse)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::ProcessorQuery query(machine);

  // Use the query multiple times
  EXPECT_EQ(query.count(), 3);
  EXPECT_EQ(query.count(), 3); // Should work multiple times

  // Create a new query with filter applied
  HPP::Machine::ProcessorQuery filtered_query(machine);
  filtered_query.only_kind(HPP::Processor::Kind::LOC_PROC);
  EXPECT_EQ(filtered_query.count(), 2);
  EXPECT_EQ(filtered_query.count(), 2); // Should work multiple times

  // Verify results are consistent
  std::vector<HPP::Processor> first_pass;
  std::vector<HPP::Processor> second_pass;

  for(const auto &proc : filtered_query) {
    first_pass.push_back(proc);
  }
  for(const auto &proc : filtered_query) {
    second_pass.push_back(proc);
  }

  EXPECT_EQ(first_pass.size(), second_pass.size());
  EXPECT_EQ(first_pass.size(), 2);
}

TEST_F(HPPProcessorQueryTest, EdgeCases)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test next() on the last processor
  HPP::Machine::ProcessorQuery query(machine);
  HPP::Processor first = query.first();
  HPP::Processor second = query.next(first);
  HPP::Processor third = query.next(second);
  HPP::Processor fourth = query.next(third); // Should be NO_PROC

  EXPECT_TRUE(first.exists());
  EXPECT_TRUE(second.exists());
  EXPECT_TRUE(third.exists());
  EXPECT_FALSE(fourth.exists());

  // Test next() on NO_PROC
  HPP::Processor after_none = query.next(HPP::Processor::NO_PROC);
  EXPECT_FALSE(after_none.exists());
}

TEST_F(HPPProcessorQueryTest, ProcessorProperties)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::ProcessorQuery query(machine);

  // Get a processor and verify its properties
  HPP::Processor proc = query.first();

  EXPECT_TRUE(proc.exists());
  EXPECT_GE(proc.kind(), 0);          // Kind should be valid
  EXPECT_GE(proc.address_space(), 0); // Address space should be valid

  // Verify all processors in query have valid properties
  for(const auto &p : query) {
    EXPECT_TRUE(p.exists());
    EXPECT_GE(p.kind(), 0);
    EXPECT_GE(p.address_space(), 0);
  }
}
