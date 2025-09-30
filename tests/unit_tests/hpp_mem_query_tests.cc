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

class HPPMemoryQueryTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
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

  static void TearDownTestSuite()
  {
    if(mock_runtime) {
      // Clear the global runtime singleton
      Realm::runtime_singleton = nullptr;
      mock_runtime->finalize();
      mock_runtime.reset();
    }
    Realm::enable_unit_tests = false;
  }

  static void setupBasicMachineModel()
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

  static std::unique_ptr<MockRuntimeImplMachineModel> mock_runtime;
};

// Static member definition
std::unique_ptr<MockRuntimeImplMachineModel> HPPMemoryQueryTest::mock_runtime;

// Test basic MemoryQuery functionality
TEST_F(HPPMemoryQueryTest, BasicQuery)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::MemoryQuery query(machine);

  // Test count
  EXPECT_EQ(query.count(), 2);

  // Test first
  HPP::Memory first_mem = query.first();
  EXPECT_TRUE(first_mem.exists());

  // Test next
  HPP::Memory next_mem = query.next(first_mem);
  EXPECT_TRUE(next_mem.exists());
  EXPECT_NE(first_mem, next_mem);

  // Test random
  HPP::Memory random_mem = query.random();
  EXPECT_TRUE(random_mem.exists());
}

TEST_F(HPPMemoryQueryTest, OnlyKindFilter)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for system memory only
  HPP::Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);
  EXPECT_EQ(sysmem_query.count(), 1); // 1 system memory total

  // Query for GPU memory only
  HPP::Machine::MemoryQuery gpumem_query(machine);
  gpumem_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM);
  EXPECT_EQ(gpumem_query.count(), 1); // 1 GPU memory total

  // Verify kinds
  for(const auto &mem : sysmem_query) {
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::SYSTEM_MEM);
  }
  for(const auto &mem : gpumem_query) {
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::GPU_FB_MEM);
  }
}

TEST_F(HPPMemoryQueryTest, LocalAddressSpaceFilter)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::MemoryQuery local_query(machine);
  local_query.local_address_space();

  // Should have 2 memories in local address space
  EXPECT_EQ(local_query.count(), 2);

  // Verify all are in local address space
  for(const auto &mem : local_query) {
    EXPECT_EQ(mem.address_space(), 0);
  }
}

TEST_F(HPPMemoryQueryTest, HasCapacityFilter)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for memories with at least 5GB capacity
  HPP::Machine::MemoryQuery large_query(machine);
  large_query.has_capacity(5ULL * 1024);
  EXPECT_EQ(large_query.count(), 1); // Only system memory (8KB) meets this

  // Query for memories with at least 2GB capacity
  HPP::Machine::MemoryQuery medium_query(machine);
  medium_query.has_capacity(2ULL * 1024);
  EXPECT_EQ(medium_query.count(), 2); // Both memories meet this

  // Query for memories with at least 10GB capacity
  HPP::Machine::MemoryQuery huge_query(machine);
  huge_query.has_capacity(10ULL * 1024);
  EXPECT_EQ(huge_query.count(), 0); // No memories meet this

  // Verify capacity constraints
  for(const auto &mem : large_query) {
    EXPECT_GE(mem.capacity(), 5ULL * 1024);
  }
  for(const auto &mem : medium_query) {
    EXPECT_GE(mem.capacity(), 2ULL * 1024);
  }
}

TEST_F(HPPMemoryQueryTest, IteratorFunctionality)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::MemoryQuery query(machine);

  std::vector<HPP::Memory> memories;
  for(const auto &mem : query) {
    memories.push_back(mem);
  }

  EXPECT_EQ(memories.size(), 2);

  // Verify all memories are unique
  std::set<HPP::Memory> unique_memories(memories.begin(), memories.end());
  EXPECT_EQ(unique_memories.size(), 2);
}

TEST_F(HPPMemoryQueryTest, ChainedFilters)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Chain multiple filters
  HPP::Machine::MemoryQuery query(machine);
  query.only_kind(HPP::Memory::Kind::SYSTEM_MEM).local_address_space();

  // Should have 1 system memory in local address space
  EXPECT_EQ(query.count(), 1);

  // Verify all results match both criteria
  for(const auto &mem : query) {
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::SYSTEM_MEM);
    EXPECT_EQ(mem.address_space(), 0);
  }
}

TEST_F(HPPMemoryQueryTest, EmptyResults)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for memories with very large capacity that don't exist
  HPP::Machine::MemoryQuery query(machine);
  query.has_capacity(100ULL * 1024); // 100KB

  // Should have no results
  EXPECT_EQ(query.count(), 0);

  // Test that we can still call methods on empty results
  EXPECT_FALSE(query.first().exists());
  EXPECT_FALSE(query.random().exists());

  // Iterator should be empty
  int count = 0;
  for(const auto &mem : query) {
    count++;
  }
  EXPECT_EQ(count, 0);
}

TEST_F(HPPMemoryQueryTest, CopyConstructor)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::MemoryQuery original_query(machine);
  original_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);

  // Copy the query
  HPP::Machine::MemoryQuery copied_query(original_query);

  // Both should have the same results
  EXPECT_EQ(original_query.count(), copied_query.count());
  EXPECT_EQ(original_query.count(), 1);

  // Verify both queries return the same memories
  std::vector<HPP::Memory> original_mems;
  std::vector<HPP::Memory> copied_mems;

  for(const auto &mem : original_query) {
    original_mems.push_back(mem);
  }
  for(const auto &mem : copied_query) {
    copied_mems.push_back(mem);
  }

  EXPECT_EQ(original_mems.size(), copied_mems.size());
  EXPECT_EQ(original_mems.size(), 1);
}

TEST_F(HPPMemoryQueryTest, AssignmentOperator)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::MemoryQuery query1(machine);
  query1.only_kind(HPP::Memory::Kind::SYSTEM_MEM);

  HPP::Machine::MemoryQuery query2(machine);
  query2.only_kind(HPP::Memory::Kind::GPU_FB_MEM);

  // Before assignment, they should have different counts
  EXPECT_EQ(query1.count(), 1);
  EXPECT_EQ(query2.count(), 1);

  // Assign query1 to query2
  query2 = query1;

  // After assignment, both should have the same count
  // Note: The assignment copies the shared_ptr (impl), so both queries
  // should now point to the same underlying query object
  EXPECT_EQ(query1.count(), 1);
  EXPECT_EQ(query2.count(), 1);

  // Verify they are now equal (point to same underlying query)
  EXPECT_EQ(query1, query2);

  // Verify both queries return the same memories
  std::vector<HPP::Memory> mems1, mems2;
  for(const auto &mem : query1) {
    mems1.push_back(mem);
  }
  for(const auto &mem : query2) {
    mems2.push_back(mem);
  }
  EXPECT_EQ(mems1.size(), mems2.size());
  EXPECT_EQ(mems1.size(), 1);
}

TEST_F(HPPMemoryQueryTest, EqualityOperators)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::MemoryQuery query1(machine);
  HPP::Machine::MemoryQuery query2(machine);

  // Two different query objects should not be equal
  EXPECT_NE(query1, query2);

  // A query should be equal to itself
  EXPECT_EQ(query1, query1);
  EXPECT_EQ(query2, query2);
}

TEST_F(HPPMemoryQueryTest, NextFunctionality)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::MemoryQuery query(machine);

  // Get all memories using next()
  std::vector<HPP::Memory> memories;
  HPP::Memory current = query.first();

  while(current.exists()) {
    memories.push_back(current);
    current = query.next(current);
  }

  EXPECT_EQ(memories.size(), 2);

  // Verify all memories are unique
  std::set<HPP::Memory> unique_memories(memories.begin(), memories.end());
  EXPECT_EQ(unique_memories.size(), 2);
}

TEST_F(HPPMemoryQueryTest, RandomFunctionality)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::MemoryQuery query(machine);

  // Test random multiple times - should always return valid memories
  for(int i = 0; i < 10; i++) {
    HPP::Memory random_mem = query.random();
    EXPECT_TRUE(random_mem.exists());
  }

  // Test random on filtered query
  HPP::Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);

  for(int i = 0; i < 10; i++) {
    HPP::Memory random_sysmem = sysmem_query.random();
    EXPECT_TRUE(random_sysmem.exists());
    EXPECT_EQ(random_sysmem.kind(), HPP::Memory::Kind::SYSTEM_MEM);
  }
}

TEST_F(HPPMemoryQueryTest, FilteredIterator)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test iterator on filtered query
  HPP::Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);

  std::vector<HPP::Memory> sysmem_memories;
  for(const auto &mem : sysmem_query) {
    sysmem_memories.push_back(mem);
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::SYSTEM_MEM);
  }

  EXPECT_EQ(sysmem_memories.size(), 1);

  // Test iterator on GPU memory query
  HPP::Machine::MemoryQuery gpumem_query(machine);
  gpumem_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM);

  std::vector<HPP::Memory> gpumem_memories;
  for(const auto &mem : gpumem_query) {
    gpumem_memories.push_back(mem);
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::GPU_FB_MEM);
  }

  EXPECT_EQ(gpumem_memories.size(), 1);
}

TEST_F(HPPMemoryQueryTest, MultipleFilterChaining)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test chaining multiple filters
  HPP::Machine::MemoryQuery query(machine);
  query.only_kind(HPP::Memory::Kind::SYSTEM_MEM).local_address_space();

  // Should have 1 system memory in local address space
  EXPECT_EQ(query.count(), 1);

  // Verify all results match both criteria
  for(const auto &mem : query) {
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::SYSTEM_MEM);
    EXPECT_EQ(mem.address_space(), 0);
  }

  // Test that we can chain more filters (even if they don't change the result)
  HPP::Machine::MemoryQuery chained_query(machine);
  chained_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM)
      .local_address_space()
      .has_capacity(1ULL * 1024); // At least 1KB

  EXPECT_EQ(chained_query.count(), 1);
}

TEST_F(HPPMemoryQueryTest, QueryReuse)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  HPP::Machine::MemoryQuery query(machine);

  // Use the query multiple times
  EXPECT_EQ(query.count(), 2);
  EXPECT_EQ(query.count(), 2); // Should work multiple times

  // Create a new query with filter applied
  HPP::Machine::MemoryQuery filtered_query(machine);
  filtered_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);
  EXPECT_EQ(filtered_query.count(), 1);
  EXPECT_EQ(filtered_query.count(), 1); // Should work multiple times

  // Verify results are consistent
  std::vector<HPP::Memory> first_pass;
  std::vector<HPP::Memory> second_pass;

  for(const auto &mem : filtered_query) {
    first_pass.push_back(mem);
  }
  for(const auto &mem : filtered_query) {
    second_pass.push_back(mem);
  }

  EXPECT_EQ(first_pass.size(), second_pass.size());
  EXPECT_EQ(first_pass.size(), 1);
}

TEST_F(HPPMemoryQueryTest, EdgeCases)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test next() on the last memory
  HPP::Machine::MemoryQuery query(machine);
  HPP::Memory first = query.first();
  HPP::Memory second = query.next(first);
  HPP::Memory third = query.next(second); // Should be NO_MEMORY

  EXPECT_TRUE(first.exists());
  EXPECT_TRUE(second.exists());
  EXPECT_FALSE(third.exists());

  // Test next() on NO_MEMORY
  HPP::Memory after_none = query.next(HPP::Memory::NO_MEMORY);
  EXPECT_FALSE(after_none.exists());
}

TEST_F(HPPMemoryQueryTest, MemoryProperties)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  HPP::Machine::MemoryQuery query(machine);

  // Get a memory and verify its properties
  HPP::Memory mem = query.first();

  EXPECT_TRUE(mem.exists());
  EXPECT_GE(mem.kind(), 0);          // Kind should be valid
  EXPECT_GE(mem.address_space(), 0); // Address space should be valid
  EXPECT_GT(mem.capacity(), 0);      // Capacity should be positive

  // Verify all memories in query have valid properties
  for(const auto &m : query) {
    EXPECT_TRUE(m.exists());
    EXPECT_GE(m.kind(), 0);
    EXPECT_GE(m.address_space(), 0);
    EXPECT_GT(m.capacity(), 0);
  }
}

TEST_F(HPPMemoryQueryTest, CapacityFilteringEdgeCases)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test with zero capacity (should return all memories)
  HPP::Machine::MemoryQuery zero_query(machine);
  zero_query.has_capacity(0);
  EXPECT_EQ(zero_query.count(), 2);

  // Test with capacity exactly matching a memory's capacity
  HPP::Machine::MemoryQuery exact_query(machine);
  exact_query.has_capacity(4ULL * 1024); // Exactly 4KB
  EXPECT_EQ(exact_query.count(), 2);     // Both memories have >= 4KB

  // Test with capacity between the two memory sizes
  HPP::Machine::MemoryQuery between_query(machine);
  between_query.has_capacity(6ULL * 1024); // 6KB
  EXPECT_EQ(between_query.count(), 1);     // Only system memory (8KB) meets this
}

TEST_F(HPPMemoryQueryTest, KindAndCapacityCombined)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test system memory with specific capacity
  HPP::Machine::MemoryQuery sysmem_large_query(machine);
  sysmem_large_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM)
      .has_capacity(5ULL * 1024);           // 5KB
  EXPECT_EQ(sysmem_large_query.count(), 1); // System memory (8KB) meets this

  // Test GPU memory with specific capacity
  HPP::Machine::MemoryQuery gpumem_small_query(machine);
  gpumem_small_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM)
      .has_capacity(2ULL * 1024);           // 2KB
  EXPECT_EQ(gpumem_small_query.count(), 1); // GPU memory (4KB) meets this

  // Test GPU memory with too large capacity requirement
  HPP::Machine::MemoryQuery gpumem_too_large_query(machine);
  gpumem_too_large_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM)
      .has_capacity(5ULL * 1024);               // 5KB
  EXPECT_EQ(gpumem_too_large_query.count(), 0); // GPU memory (4KB) doesn't meet this
}
