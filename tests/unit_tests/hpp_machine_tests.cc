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

/*
 * UNTESTED FUNCTIONALITY DUE TO MOCK RUNTIME LIMITATIONS:
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

class HPPMachineTest : public ::testing::Test {
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

// Test basic Machine functionality
TEST_F(HPPMachineTest, GetMachine)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  // Machine should be valid (we can't access private impl member directly)
  EXPECT_TRUE(true); // Basic test that get_machine() doesn't crash
}

TEST_F(HPPMachineTest, GetAllProcessors)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> processors;

  machine.get_all_processors(processors);

  // Should have 3 processors total (2 CPUs + 1 GPU on node 0)
  EXPECT_EQ(processors.size(), 3);

  // Verify all processors exist
  for(const auto &proc : processors) {
    EXPECT_TRUE(proc.exists());
  }
}

TEST_F(HPPMachineTest, GetLocalProcessors)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> local_processors;

  machine.get_local_processors(local_processors);

  // Should have 3 processors on local node (node 0)
  EXPECT_EQ(local_processors.size(), 3);

  // Verify all are in local address space
  for(const auto &proc : local_processors) {
    EXPECT_TRUE(proc.exists());
    EXPECT_EQ(proc.address_space(), 0); // Local address space
  }
}

TEST_F(HPPMachineTest, GetLocalProcessorsByKind)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> cpu_processors;

  machine.get_local_processors_by_kind(cpu_processors, HPP::Processor::Kind::LOC_PROC);
  // Verify kinds
  for(const auto &proc : cpu_processors) {
    EXPECT_EQ(proc.kind(), HPP::Processor::Kind::LOC_PROC);
  }
}

TEST_F(HPPMachineTest, GetAllMemories)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_all_memories(memories);

  // Should have 2 memories total
  EXPECT_EQ(memories.size(), 2);

  // Verify all memories exist
  for(const auto &mem : memories) {
    EXPECT_TRUE(mem.exists());
  }
}

TEST_F(HPPMachineTest, GetMemoriesByCapacity)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> large_memories;
  std::set<HPP::Memory> small_memories;

  // Get memories with at least 6KB capacity
  machine.get_memories_by_capacity(6ULL * 1024, large_memories);

  // Get memories with at least 2KB capacity
  machine.get_memories_by_capacity(2ULL * 1024, small_memories);

  // Should have 1 large memory (8KB system memory) and 2 small memories (all memories)
  EXPECT_EQ(large_memories.size(), 1);
  EXPECT_EQ(small_memories.size(), 2);
}

// Test ProcessorQuery functionality
TEST_F(HPPMachineTest, ProcessorQueryBasic)
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

TEST_F(HPPMachineTest, ProcessorQueryOnlyKind)
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

TEST_F(HPPMachineTest, ProcessorQueryLocalAddressSpace)
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

TEST_F(HPPMachineTest, ProcessorQueryIterator)
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

// Test MemoryQuery functionality
TEST_F(HPPMachineTest, MemoryQueryBasic)
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

TEST_F(HPPMachineTest, MemoryQueryOnlyKind)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for system memory only
  HPP::Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);
  EXPECT_EQ(sysmem_query.count(), 1); // 1 system memory

  // Query for GPU memory only
  HPP::Machine::MemoryQuery gpumem_query(machine);
  gpumem_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM);
  EXPECT_EQ(gpumem_query.count(), 1); // 1 GPU memory

  // Verify kinds
  for(const auto &mem : sysmem_query) {
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::SYSTEM_MEM);
  }
  for(const auto &mem : gpumem_query) {
    EXPECT_EQ(mem.kind(), HPP::Memory::Kind::GPU_FB_MEM);
  }
}

TEST_F(HPPMachineTest, MemoryQueryHasCapacity)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for memories with at least 6KB
  HPP::Machine::MemoryQuery large_query(machine);
  large_query.has_capacity(6ULL * 1024);
  EXPECT_EQ(large_query.count(), 1); // 1 memory with 8KB

  // Query for memories with at least 5KB
  HPP::Machine::MemoryQuery medium_query(machine);
  medium_query.has_capacity(5ULL * 1024);
  EXPECT_EQ(medium_query.count(), 1); // Still 1 memory (GPU has 4KB, so excluded)

  // Query for memories with at least 3KB
  HPP::Machine::MemoryQuery small_query(machine);
  small_query.has_capacity(3ULL * 1024);
  EXPECT_EQ(small_query.count(), 2); // All memories
}

TEST_F(HPPMachineTest, MemoryQueryLocalAddressSpace)
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

TEST_F(HPPMachineTest, MemoryQueryIterator)
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

// Test affinity functionality
TEST_F(HPPMachineTest, DISABLED_HasAffinityProcessorMemory)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Get a CPU and GPU processor
  HPP::Machine::ProcessorQuery cpu_query(machine);
  cpu_query.only_kind(HPP::Processor::Kind::LOC_PROC);
  HPP::Processor cpu = cpu_query.first();

  HPP::Machine::ProcessorQuery gpu_query(machine);
  gpu_query.only_kind(HPP::Processor::Kind::TOC_PROC);
  HPP::Processor gpu = gpu_query.first();

  // Get system and GPU memories
  HPP::Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);
  HPP::Memory sysmem = sysmem_query.first();

  HPP::Machine::MemoryQuery gpumem_query(machine);
  gpumem_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM);
  HPP::Memory gpumem = gpumem_query.first();

  // Test affinity checks
  HPP::Machine::AffinityDetails details;

  // CPU should have affinity to system memory
  EXPECT_TRUE(machine.has_affinity(cpu, sysmem, &details));
  EXPECT_GT(details.bandwidth, 0);
  EXPECT_GT(details.latency, 0);

  // GPU should have affinity to GPU memory
  EXPECT_TRUE(machine.has_affinity(gpu, gpumem, &details));
  EXPECT_GT(details.bandwidth, 0);
  EXPECT_GT(details.latency, 0);

  // GPU should also have affinity to system memory (but with different characteristics)
  EXPECT_TRUE(machine.has_affinity(gpu, sysmem, &details));
}

TEST_F(HPPMachineTest, DISABLED_HasAffinityMemoryMemory)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Get system and GPU memories
  HPP::Machine::MemoryQuery sysmem_query(machine);
  sysmem_query.only_kind(HPP::Memory::Kind::SYSTEM_MEM);
  HPP::Memory sysmem = sysmem_query.first();

  HPP::Machine::MemoryQuery gpumem_query(machine);
  gpumem_query.only_kind(HPP::Memory::Kind::GPU_FB_MEM);
  HPP::Memory gpumem = gpumem_query.first();

  // Test memory-memory affinity
  HPP::Machine::AffinityDetails details;

  // System memory and GPU memory should have affinity
  EXPECT_TRUE(machine.has_affinity(sysmem, gpumem, &details));
  EXPECT_GT(details.bandwidth, 0);
  EXPECT_GT(details.latency, 0);

  // Reverse direction should also work
  EXPECT_TRUE(machine.has_affinity(gpumem, sysmem, &details));
}

TEST_F(HPPMachineTest, DISABLED_GetProcMemAffinity)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  std::vector<HPP::Machine::ProcessorMemoryAffinity> affinities;
  int count = machine.get_proc_mem_affinity(affinities);

  // Should have 4 affinity relationships
  EXPECT_EQ(count, 4);
  EXPECT_EQ(affinities.size(), 4);

  // Verify all affinities have valid processors and memories
  for(const auto &affinity : affinities) {
    EXPECT_TRUE(affinity.p.exists());
    EXPECT_TRUE(affinity.m.exists());
    EXPECT_GT(affinity.bandwidth, 0);
    EXPECT_GT(affinity.latency, 0);
  }
}

TEST_F(HPPMachineTest, DISABLED_GetMemMemAffinity)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  std::vector<HPP::Machine::MemoryMemoryAffinity> affinities;
  int count = machine.get_mem_mem_affinity(affinities);

  // Should have 2 affinity relationships (bidirectional)
  EXPECT_EQ(count, 2);
  EXPECT_EQ(affinities.size(), 2);

  // Verify all affinities have valid memories
  for(const auto &affinity : affinities) {
    EXPECT_TRUE(affinity.m1.exists());
    EXPECT_TRUE(affinity.m2.exists());
    EXPECT_GT(affinity.bandwidth, 0);
    EXPECT_GT(affinity.latency, 0);
  }
}

// Test error handling and edge cases
TEST_F(HPPMachineTest, EmptyQueryResults)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for non-existent processor kind
  HPP::Machine::ProcessorQuery query(machine);
  query.only_kind(static_cast<HPP::Processor::Kind>(Processor::Kind::UTIL_PROC));

  EXPECT_EQ(query.count(), 0);
  EXPECT_FALSE(query.first().exists());
  EXPECT_FALSE(query.random().exists());

  // Iterator should be empty
  int count = 0;
  for(const auto &proc : query) {
    count++;
  }
  EXPECT_EQ(count, 0);
}

TEST_F(HPPMachineTest, MemoryCapacityEdgeCases)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Query for extremely large capacity
  HPP::Machine::MemoryQuery query(machine);
  query.has_capacity(100ULL * 1024 * 1024 * 1024 * 1024); // 100TB

  EXPECT_EQ(query.count(), 0);
  EXPECT_FALSE(query.first().exists());
}

TEST_F(HPPMachineTest, ProcessorMemoryProperties)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Get a processor and verify its properties
  HPP::Machine::ProcessorQuery query(machine);
  HPP::Processor proc = query.first();

  EXPECT_TRUE(proc.exists());
  EXPECT_GE(proc.kind(), 0);          // Kind should be valid
  EXPECT_GE(proc.address_space(), 0); // Address space should be valid
}

TEST_F(HPPMachineTest, MemoryProperties)
{
  HPP::Machine machine = HPP::Machine::get_machine();

  // Get a memory and verify its properties
  HPP::Machine::MemoryQuery query(machine);
  HPP::Memory mem = query.first();

  EXPECT_TRUE(mem.exists());
  EXPECT_GE(mem.kind(), 0);          // Kind should be valid
  EXPECT_GT(mem.capacity(), 0);      // Should have some capacity
  EXPECT_GE(mem.address_space(), 0); // Address space should be valid
}

// Test query chaining
TEST_F(HPPMachineTest, ProcessorQueryChaining)
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

TEST_F(HPPMachineTest, MemoryQueryChaining)
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
