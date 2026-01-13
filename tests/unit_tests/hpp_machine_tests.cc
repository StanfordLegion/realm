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

// ============================================================================
// Machine Model Configuration Structures
// ============================================================================

struct ProcessorInfo {
  int processorId;
  HPP::Processor::Kind kind;
  int node;
};

struct MemoryInfo {
  int memoryId;
  HPP::Memory::Kind kind;
  size_t capacity;
  int node;
};

struct ProcMemAffinityInfo {
  int processorId;
  int memoryId;
  int bandwidth;
  int latency;
};

struct MemMemAffinityInfo {
  int sourceMemId;
  int targetMemId;
  int bandwidth;
  int latency;
};

struct MachineModelConfig {
  std::string name;
  int numNodes;
  std::vector<ProcessorInfo> processors;
  std::vector<MemoryInfo> memories;
  std::vector<ProcMemAffinityInfo> procMemAffinities;
  std::vector<MemMemAffinityInfo> memMemAffinities;

  // Helper methods to query configuration
  size_t getTotalProcessorCount() const { return processors.size(); }

  size_t getLocalProcessorCount(int node) const
  {
    return std::count_if(processors.begin(), processors.end(),
                         [node](const ProcessorInfo &p) { return p.node == node; });
  }

  size_t getLocalProcessorCountByKind(int node, HPP::Processor::Kind kind) const
  {
    return std::count_if(processors.begin(), processors.end(),
                         [node, kind](const ProcessorInfo &p) {
                           return p.node == node && p.kind == kind;
                         });
  }

  size_t getTotalMemoryCount() const { return memories.size(); }

  size_t getMemoryCountByCapacity(size_t minCapacity) const
  {
    return std::count_if(
        memories.begin(), memories.end(),
        [minCapacity](const MemoryInfo &m) { return m.capacity >= minCapacity; });
  }
};

// ============================================================================
// Parameterized Test Fixture Base
// ============================================================================

class HPPMachineTest : public ::testing::TestWithParam<MachineModelConfig> {
protected:
  void SetUp() override
  {
    // Enable unit tests for the mock runtime
    Realm::enable_unit_tests = true;

    const MachineModelConfig &config = GetParam();

    // Initialize mock runtime with configured number of nodes
    mock_runtime = std::make_unique<MockRuntimeImplMachineModel>();
    mock_runtime->init(config.numNodes);

    // Set the global runtime singleton to our mock runtime
    Realm::runtime_singleton = mock_runtime.get();

    // Set up the machine model based on configuration
    setupMachineModel(config);
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

  void setupMachineModel(const MachineModelConfig &config)
  {
    // First, ensure MachineNodeInfo objects exist for all address spaces
    for(int node = 0; node < config.numNodes; node++) {
      if(mock_runtime->machine->nodeinfos.find(node) ==
         mock_runtime->machine->nodeinfos.end()) {
        mock_runtime->machine->nodeinfos[node] =
            new Realm::MachineNodeInfo(node, mock_runtime.get());
      }
    }

    MockRuntimeImplMachineModel::ProcessorMemoriesToBeAdded setup;

    // Convert configuration to mock runtime format
    for(const ProcessorInfo &proc : config.processors) {
      setup.proc_infos.push_back({static_cast<unsigned int>(proc.processorId),
                                  static_cast<Realm::Processor::Kind>(proc.kind),
                                  static_cast<realm_address_space_t>(proc.node)});
    }

    for(const MemoryInfo &mem : config.memories) {
      setup.mem_infos.push_back({static_cast<unsigned int>(mem.memoryId),
                                 static_cast<Realm::Memory::Kind>(mem.kind), mem.capacity,
                                 static_cast<realm_address_space_t>(mem.node)});
    }

    for(const ProcMemAffinityInfo &affinity : config.procMemAffinities) {
      setup.proc_mem_affinities.push_back(
          {static_cast<unsigned int>(affinity.processorId),
           static_cast<unsigned int>(affinity.memoryId),
           static_cast<unsigned int>(affinity.bandwidth),
           static_cast<unsigned int>(affinity.latency)});
    }

    for(const MemMemAffinityInfo &affinity : config.memMemAffinities) {
      setup.mem_mem_affinities.push_back({static_cast<unsigned int>(affinity.sourceMemId),
                                          static_cast<unsigned int>(affinity.targetMemId),
                                          static_cast<unsigned int>(affinity.bandwidth),
                                          static_cast<unsigned int>(affinity.latency)});
    }

    mock_runtime->setup_mock_proc_mems(setup);
  }

  std::unique_ptr<MockRuntimeImplMachineModel> mock_runtime;
};

// ============================================================================
// Machine Model Configurations
// ============================================================================

// Single node with basic CPU and GPU configuration
MachineModelConfig createBasicSingleNodeConfig()
{
  MachineModelConfig config;
  config.name = "BasicSingleNode";
  config.numNodes = 1;

  config.processors = {
      {0, HPP::Processor::Kind::LOC_PROC, 0}, // CPU 0 on node 0
      {1, HPP::Processor::Kind::LOC_PROC, 0}, // CPU 1 on node 0
      {2, HPP::Processor::Kind::TOC_PROC, 0}  // GPU 0 on node 0
  };

  config.memories = {
      {0, HPP::Memory::Kind::SYSTEM_MEM, 8192, 0}, // 8KB system mem on node 0
      {1, HPP::Memory::Kind::GPU_FB_MEM, 4096, 0}  // 4KB GPU mem on node 0
  };

  config.procMemAffinities = {
      {0, 0, 10000, 100}, // CPU 0 -> System mem
      {1, 0, 10000, 100}, // CPU 1 -> System mem
      {2, 1, 20000, 50},  // GPU -> GPU mem (high bandwidth)
      {2, 0, 1000, 500}   // GPU -> System mem (low bandwidth)
  };

  config.memMemAffinities = {{0, 1, 5000, 200}, {1, 0, 5000, 200}};

  return config;
}

// Single node with CPU only (no GPU)
MachineModelConfig createCPUOnlyConfig()
{
  MachineModelConfig config;
  config.name = "CPUOnly";
  config.numNodes = 1;

  config.processors = {{0, HPP::Processor::Kind::LOC_PROC, 0},
                       {1, HPP::Processor::Kind::LOC_PROC, 0},
                       {2, HPP::Processor::Kind::LOC_PROC, 0},
                       {3, HPP::Processor::Kind::LOC_PROC, 0}};

  config.memories = {
      {0, HPP::Memory::Kind::SYSTEM_MEM, 16384, 0} // 16KB system mem
  };

  config.procMemAffinities = {
      {0, 0, 10000, 100}, {1, 0, 10000, 100}, {2, 0, 10000, 100}, {3, 0, 10000, 100}};

  return config;
}

// Two nodes with different processor configurations
MachineModelConfig createTwoNodeConfig()
{
  MachineModelConfig config;
  config.name = "TwoNode";
  config.numNodes = 2;

  // Node 0: 2 CPUs + 1 GPU
  // Note: processor indices per node must start from 0
  config.processors = {{0, HPP::Processor::Kind::LOC_PROC, 0},
                       {1, HPP::Processor::Kind::LOC_PROC, 0},
                       {2, HPP::Processor::Kind::TOC_PROC, 0},
                       // Node 1: 1 CPU + 2 GPUs
                       {0, HPP::Processor::Kind::LOC_PROC, 1},
                       {1, HPP::Processor::Kind::TOC_PROC, 1},
                       {2, HPP::Processor::Kind::TOC_PROC, 1}};

  config.memories = {{0, HPP::Memory::Kind::SYSTEM_MEM, 8192, 0},
                     {1, HPP::Memory::Kind::GPU_FB_MEM, 4096, 0},
                     {0, HPP::Memory::Kind::SYSTEM_MEM, 12288, 1},
                     {1, HPP::Memory::Kind::GPU_FB_MEM, 6144, 1},
                     {2, HPP::Memory::Kind::GPU_FB_MEM, 6144, 1}};

  // Affinities use indices into the config vectors above (0-5 for processors, 0-4 for
  // memories)
  config.procMemAffinities = {
      // Node 0 affinities (processors 0-2 in vector, memories 0-1 in vector)
      {0, 0, 10000, 100}, // CPU 0 -> System mem 0
      {1, 0, 10000, 100}, // CPU 1 -> System mem 0
      {2, 1, 20000, 50},  // GPU 2 -> GPU mem 1
      {2, 0, 1000, 500},  // GPU 2 -> System mem 0
      // Node 1 affinities (processors 3-5 in vector, memories 2-4 in vector)
      {3, 2, 10000, 100}, // CPU 3 -> System mem 2
      {4, 3, 20000, 50},  // GPU 4 -> GPU mem 3
      {4, 2, 1000, 500},  // GPU 4 -> System mem 2
      {5, 4, 20000, 50},  // GPU 5 -> GPU mem 4
      {5, 2, 1000, 500}   // GPU 5 -> System mem 2
  };

  config.memMemAffinities = {
      {0, 1, 5000, 200}, // System mem 0 <-> GPU mem 1
      {1, 0, 5000, 200}, // GPU mem 1 <-> System mem 0
      {2, 3, 5000, 200}, // System mem 2 <-> GPU mem 3
      {3, 2, 5000, 200}, // GPU mem 3 <-> System mem 2
      {2, 4, 5000, 200}, // System mem 2 <-> GPU mem 4
      {4, 2, 5000, 200}  // GPU mem 4 <-> System mem 2
  };

  return config;
}

// Large memory configuration for capacity testing
MachineModelConfig createLargeMemoryConfig()
{
  MachineModelConfig config;
  config.name = "LargeMemory";
  config.numNodes = 1;

  config.processors = {{0, HPP::Processor::Kind::LOC_PROC, 0},
                       {1, HPP::Processor::Kind::TOC_PROC, 0}};

  config.memories = {
      {0, HPP::Memory::Kind::SYSTEM_MEM, 1024, 0}, // 1KB
      {1, HPP::Memory::Kind::SYSTEM_MEM, 4096, 0}, // 4KB
      {2, HPP::Memory::Kind::GPU_FB_MEM, 8192, 0}, // 8KB
      {3, HPP::Memory::Kind::SYSTEM_MEM, 16384, 0} // 16KB
  };

  config.procMemAffinities = {
      {0, 0, 10000, 100}, {0, 1, 10000, 100}, {0, 3, 10000, 100}, {1, 2, 20000, 50}};

  return config;
}

// GPU-only configuration (no CPUs) - tests edge case
MachineModelConfig createGPUOnlyConfig()
{
  MachineModelConfig config;
  config.name = "GPUOnly";
  config.numNodes = 1;

  config.processors = {{0, HPP::Processor::Kind::TOC_PROC, 0},
                       {1, HPP::Processor::Kind::TOC_PROC, 0}};

  config.memories = {{0, HPP::Memory::Kind::GPU_FB_MEM, 8192, 0}};

  config.procMemAffinities = {{0, 0, 20000, 50}, {1, 0, 20000, 50}};

  return config;
}

// Empty configuration - minimal valid machine (tests degenerate case)
MachineModelConfig createMinimalConfig()
{
  MachineModelConfig config;
  config.name = "Minimal";
  config.numNodes = 1;

  config.processors = {{0, HPP::Processor::Kind::LOC_PROC, 0}};

  config.memories = {{0, HPP::Memory::Kind::SYSTEM_MEM, 1024, 0}};

  config.procMemAffinities = {{0, 0, 10000, 100}};

  return config;
}

// Three nodes with asymmetric configuration - tests complex multi-node scenarios
MachineModelConfig createThreeNodeAsymmetricConfig()
{
  MachineModelConfig config;
  config.name = "ThreeNodeAsymmetric";
  config.numNodes = 3;

  // Node 0: 1 CPU
  // Node 1: 2 CPUs + 1 GPU
  // Node 2: 1 GPU only
  config.processors = {
      {0, HPP::Processor::Kind::LOC_PROC, 0}, // Node 0, proc index 0
      {0, HPP::Processor::Kind::LOC_PROC, 1}, // Node 1, proc index 0
      {1, HPP::Processor::Kind::LOC_PROC, 1}, // Node 1, proc index 1
      {2, HPP::Processor::Kind::TOC_PROC, 1}, // Node 1, proc index 2
      {0, HPP::Processor::Kind::TOC_PROC, 2}  // Node 2, proc index 0
  };

  config.memories = {
      {0, HPP::Memory::Kind::SYSTEM_MEM, 4096, 0}, // Node 0, mem index 0
      {0, HPP::Memory::Kind::SYSTEM_MEM, 8192, 1}, // Node 1, mem index 0
      {1, HPP::Memory::Kind::GPU_FB_MEM, 6144, 1}, // Node 1, mem index 1
      {0, HPP::Memory::Kind::GPU_FB_MEM, 12288, 2} // Node 2, mem index 0
  };

  config.procMemAffinities = {
      {0, 0, 10000, 100}, // Node 0 CPU (proc 0) -> Node 0 System mem (mem 0)
      {1, 1, 10000, 100}, // Node 1 CPU 0 (proc 1) -> Node 1 System mem (mem 1)
      {2, 1, 10000, 100}, // Node 1 CPU 1 (proc 2) -> Node 1 System mem (mem 1)
      {3, 2, 20000, 50},  // Node 1 GPU (proc 3) -> Node 1 GPU mem (mem 2)
      {4, 3, 20000, 50}   // Node 2 GPU (proc 4) -> Node 2 GPU mem (mem 3)
  };

  return config;
}

// ============================================================================
// Basic Machine Tests
// ============================================================================

TEST_P(HPPMachineTest, GetMachine)
{
  HPP::Machine machine = HPP::Machine::get_machine();
  // Verify we can get the machine instance without crashing
  std::set<HPP::Processor> processors;
  machine.get_all_processors(processors);
  // Should match the configuration
  EXPECT_EQ(processors.size(), GetParam().getTotalProcessorCount());
}

// ============================================================================
// Data-Driven Processor Query Tests
// ============================================================================

TEST_P(HPPMachineTest, QueryAllProcessors)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> processors;

  machine.get_all_processors(processors);

  // Verify count matches configuration
  EXPECT_EQ(processors.size(), config.getTotalProcessorCount());

  // Verify all processors exist
  for(HPP::Processor proc : processors) {
    EXPECT_TRUE(proc.exists());

    // Extract the processor index from the Realm ID
    Realm::ID proc_id(proc.id);
    unsigned proc_idx = proc_id.proc_proc_idx();

    // Find matching processor in configuration
    std::vector<ProcessorInfo>::const_iterator it =
        std::find_if(config.processors.begin(), config.processors.end(),
                     [proc_idx, &proc](const ProcessorInfo &pinfo) {
                       return pinfo.processorId == proc_idx &&
                              pinfo.kind == proc.kind() &&
                              pinfo.node == proc.address_space();
                     });

    // Verify this processor matches our configuration
    EXPECT_NE(it, config.processors.end())
        << "Processor with index " << proc_idx << ", kind "
        << static_cast<int>(proc.kind()) << " on node " << proc.address_space()
        << " not found in configuration";
  }
}

TEST_P(HPPMachineTest, QueryLocalProcessors)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> localProcessors;

  machine.get_local_processors(localProcessors);

  // Verify count matches configuration for node 0 (local node)
  EXPECT_EQ(localProcessors.size(), config.getLocalProcessorCount(0));

  // Verify all are in local address space
  for(HPP::Processor proc : localProcessors) {
    EXPECT_TRUE(proc.exists());
    EXPECT_EQ(proc.address_space(), 0);
  }
}

TEST_P(HPPMachineTest, QueryLocalProcessorsByKind)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();

  // Test all common processor kinds, including ones that may not exist in the
  // configuration
  std::vector<HPP::Processor::Kind> kindsToTest = {
      HPP::Processor::Kind::LOC_PROC,   HPP::Processor::Kind::TOC_PROC,
      HPP::Processor::Kind::UTIL_PROC,  HPP::Processor::Kind::IO_PROC,
      HPP::Processor::Kind::PROC_GROUP, HPP::Processor::Kind::PROC_SET,
      HPP::Processor::Kind::OMP_PROC,   HPP::Processor::Kind::PY_PROC};

  for(HPP::Processor::Kind kind : kindsToTest) {
    std::set<HPP::Processor> processors;
    machine.get_local_processors_by_kind(processors, kind);

    // Verify count matches configuration for this kind on node 0
    size_t expectedCount = config.getLocalProcessorCountByKind(0, kind);
    EXPECT_EQ(processors.size(), expectedCount)
        << "Mismatch for kind " << static_cast<int>(kind);

    // Build expected set of processor IDs from configuration
    std::set<int> expectedProcessorIds;
    for(const ProcessorInfo &pinfo : config.processors) {
      if(pinfo.node == 0 && pinfo.kind == kind) {
        expectedProcessorIds.insert(pinfo.processorId);
      }
    }

    // Verify all returned processors:
    // 1. Exist and are valid
    // 2. Have the correct kind
    // 3. Are in the local address space (node 0)
    // 4. Match expected processors from configuration
    std::set<int> actualProcessorIds;
    for(HPP::Processor proc : processors) {
      EXPECT_TRUE(proc.exists()) << "Processor should exist";
      EXPECT_EQ(proc.kind(), kind)
          << "Processor kind mismatch: expected " << static_cast<int>(kind) << " but got "
          << static_cast<int>(proc.kind());
      EXPECT_EQ(proc.address_space(), 0)
          << "Processor should be in local address space (0), but is in "
          << proc.address_space();
      actualProcessorIds.insert(Realm::ID(proc.id).proc_proc_idx());
    }

    // Verify we got exactly the processors we expected (no more, no less)
    EXPECT_EQ(actualProcessorIds, expectedProcessorIds)
        << "Returned processor IDs don't match configuration for kind "
        << static_cast<int>(kind);
  }
}

TEST_P(HPPMachineTest, QueryLocalProcessorsExcludesRemoteNodes)
{
  const MachineModelConfig &config = GetParam();

  // Skip this test if there's only one node (nothing to exclude)
  if(config.numNodes <= 1) {
    GTEST_SKIP() << "Test requires multiple nodes";
  }

  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Processor> localProcessors;

  machine.get_local_processors(localProcessors);

  // Collect all processor IDs from remote nodes (not node 0)
  std::set<int> remoteProcessorIds;
  for(const ProcessorInfo &pinfo : config.processors) {
    if(pinfo.node != 0) {
      remoteProcessorIds.insert(pinfo.processorId);
    }
  }

  // Verify none of the returned processors are from remote nodes
  for(HPP::Processor proc : localProcessors) {
    EXPECT_EQ(proc.address_space(), 0)
        << "Local processor query returned processor from remote node "
        << proc.address_space();

    // Additional verification: match processor against its config entry
    // by both index AND node (since processor indices are local to each node)
    bool isFromRemoteNode = false;
    unsigned proc_idx = Realm::ID(proc.id).proc_proc_idx();
    int proc_node = proc.address_space();
    for(const ProcessorInfo &pinfo : config.processors) {
      // Match by both processor index and node, since indices are per-node
      if(pinfo.processorId == proc_idx && pinfo.node == proc_node) {
        // Found the config entry for this processor, verify it's on node 0
        if(pinfo.node != 0) {
          isFromRemoteNode = true;
        }
        break;
      }
    }
    EXPECT_FALSE(isFromRemoteNode) << "Processor ID " << proc.id
                                   << " exists on remote node but was returned as local";
  }
}

// ============================================================================
// Data-Driven Memory Query Tests
// ============================================================================

TEST_P(HPPMachineTest, QueryAllMemories)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_all_memories(memories);

  // Verify count matches configuration
  EXPECT_EQ(memories.size(), config.getTotalMemoryCount());

  // Verify all memories exist
  for(HPP::Memory mem : memories) {
    EXPECT_TRUE(mem.exists());
  }
}

TEST_P(HPPMachineTest, QueryMemoriesWithZeroCapacity)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_memories_by_capacity(0, memories);

  // Should return all memories
  EXPECT_EQ(memories.size(), config.getMemoryCountByCapacity(0));

  for(HPP::Memory mem : memories) {
    EXPECT_TRUE(mem.exists());
  }
}

TEST_P(HPPMachineTest, QueryMemoriesWithSmallCapacity)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_memories_by_capacity(2048, memories); // 2KB

  // Verify count matches configuration
  EXPECT_EQ(memories.size(), config.getMemoryCountByCapacity(2048));

  // Verify all memories meet capacity requirement
  for(HPP::Memory mem : memories) {
    EXPECT_TRUE(mem.exists());
    EXPECT_GE(mem.capacity(), 2048U);
  }
}

TEST_P(HPPMachineTest, QueryMemoriesWithMediumCapacity)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_memories_by_capacity(6144, memories); // 6KB

  // Verify count matches configuration
  EXPECT_EQ(memories.size(), config.getMemoryCountByCapacity(6144));

  // Verify all memories meet capacity requirement
  for(HPP::Memory mem : memories) {
    EXPECT_TRUE(mem.exists());
    EXPECT_GE(mem.capacity(), 6144U);
  }
}

TEST_P(HPPMachineTest, QueryMemoriesWithLargeCapacity)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_memories_by_capacity(10240, memories); // 10KB

  // Verify count matches configuration
  EXPECT_EQ(memories.size(), config.getMemoryCountByCapacity(10240));

  // Verify all memories meet capacity requirement
  for(HPP::Memory mem : memories) {
    EXPECT_TRUE(mem.exists());
    EXPECT_GE(mem.capacity(), 10240U);
  }
}

TEST_P(HPPMachineTest, QueryMemoriesWithVeryLargeCapacity)
{
  const MachineModelConfig &config = GetParam();
  HPP::Machine machine = HPP::Machine::get_machine();
  std::set<HPP::Memory> memories;

  machine.get_memories_by_capacity(20480, memories); // 20KB

  // Verify count matches configuration
  EXPECT_EQ(memories.size(), config.getMemoryCountByCapacity(20480));

  // Verify all memories meet capacity requirement
  for(HPP::Memory mem : memories) {
    EXPECT_TRUE(mem.exists());
    EXPECT_GE(mem.capacity(), 20480U);
  }
}

// ============================================================================
// Test Suite Instantiations
// ============================================================================

// Test configurations cover various machine topologies:
// - Single node configurations (CPU-only, GPU-only, CPU+GPU, minimal)
// - Multi-node configurations (2 nodes symmetric, 3 nodes asymmetric)
// - Memory capacity variations
// Each configuration exercises different code paths and edge cases
INSTANTIATE_TEST_SUITE_P(
    MachineModelVariations, HPPMachineTest,
    ::testing::Values(
        createBasicSingleNodeConfig(), // 1 node: 2 CPUs + 1 GPU, mixed memory types
        createCPUOnlyConfig(), // 1 node: 4 CPUs, no GPU (tests missing processor kinds)
        createGPUOnlyConfig(), // 1 node: 2 GPUs, no CPU (tests missing LOC_PROC kind)
        createMinimalConfig(), // 1 node: 1 CPU minimal (tests degenerate case)
        createTwoNodeConfig(), // 2 nodes: heterogeneous (tests address space filtering)
        createThreeNodeAsymmetricConfig(), // 3 nodes: asymmetric distribution (complex
                                           // multi-node)
        createLargeMemoryConfig() // 1 node: varied memory capacities (tests capacity
                                  // queries)
        ),
    [](const ::testing::TestParamInfo<MachineModelConfig> &info) {
      return info.param.name;
    });
