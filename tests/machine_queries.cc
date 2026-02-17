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
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <set>
#include <map>
#include <vector>
#include <algorithm>

using namespace Realm;

Logger log_app("app");

// Task IDs
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

// Helper function to count processors of a specific kind
size_t count_processors(Machine machine, Processor::Kind kind)
{
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(kind);
  return pq.count();
}

// Helper function to count memories of a specific kind
size_t count_memories(Machine machine, Memory::Kind kind)
{
  Machine::MemoryQuery mq(machine);
  mq.only_kind(kind);
  return mq.count();
}

// Test basic ProcessorQuery functionality
void test_processor_query_basic(Machine machine, int &errors)
{
  log_app.print() << "Testing basic ProcessorQuery functionality...";

  // Test: Count all processors
  Machine::ProcessorQuery all_procs(machine);
  size_t total_procs = all_procs.count();
  log_app.print() << "  Total processors: " << total_procs;
  if(total_procs == 0) {
    log_app.error() << "ERROR: No processors found!";
    errors++;
  }

  // Test: Count by kind
  size_t cpu_count = count_processors(machine, Processor::LOC_PROC);
  size_t util_count = count_processors(machine, Processor::UTIL_PROC);
  size_t io_count = count_processors(machine, Processor::IO_PROC);
  log_app.print() << "  CPU processors: " << cpu_count;
  log_app.print() << "  Utility processors: " << util_count;
  log_app.print() << "  IO processors: " << io_count;

  // Test: first() and next()
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor first = pq.first();
  if(cpu_count > 0 && !first.exists()) {
    log_app.error() << "ERROR: first() returned NO_PROC for CPU processors!";
    errors++;
  }

  // Test: Iterate through all CPUs
  std::set<Processor> cpu_set;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    if(cpu_set.count(p)) {
      log_app.error() << "ERROR: Duplicate processor " << p << " in iteration!";
      errors++;
    }
    cpu_set.insert(p);
  }
  if(cpu_set.size() != cpu_count) {
    log_app.error() << "ERROR: Iteration count (" << cpu_set.size()
                    << ") != query count (" << cpu_count << ")!";
    errors++;
  }

  // Test: Iterator interface
  size_t iter_count = 0;
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
    iter_count++;
  }
  if(iter_count != cpu_count) {
    log_app.error() << "ERROR: Iterator count (" << iter_count << ") != query count ("
                    << cpu_count << ")!";
    errors++;
  }

  log_app.print() << "  Basic ProcessorQuery tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test basic MemoryQuery functionality
void test_memory_query_basic(Machine machine, int &errors)
{
  log_app.print() << "Testing basic MemoryQuery functionality...";

  // Test: Count all memories
  Machine::MemoryQuery all_mems(machine);
  size_t total_mems = all_mems.count();
  log_app.print() << "  Total memories: " << total_mems;
  if(total_mems == 0) {
    log_app.error() << "ERROR: No memories found!";
    errors++;
  }

  // Test: Count by kind
  size_t sysmem_count = count_memories(machine, Memory::SYSTEM_MEM);
  size_t regmem_count = count_memories(machine, Memory::REGDMA_MEM);
  log_app.print() << "  System memories: " << sysmem_count;
  log_app.print() << "  RegDMA memories: " << regmem_count;

  // Test: first() and next()
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory first = mq.first();
  if(sysmem_count > 0 && !first.exists()) {
    log_app.error() << "ERROR: first() returned NO_MEMORY for system memory!";
    errors++;
  }

  // Test: Iterate through all system memories
  std::set<Memory> mem_set;
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    if(mem_set.count(m)) {
      log_app.error() << "ERROR: Duplicate memory " << m << " in iteration!";
      errors++;
    }
    mem_set.insert(m);
  }
  if(mem_set.size() != sysmem_count) {
    log_app.error() << "ERROR: Iteration count (" << mem_set.size()
                    << ") != query count (" << sysmem_count << ")!";
    errors++;
  }

  log_app.print() << "  Basic MemoryQuery tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test query copy and independence
void test_query_copy(Machine machine, int &errors)
{
  log_app.print() << "Testing query copy and independence...";

  // Create original query
  Machine::ProcessorQuery pq1(machine);
  pq1.only_kind(Processor::LOC_PROC);
  size_t count1 = pq1.count();

  // Copy query
  Machine::ProcessorQuery pq2 = pq1;
  size_t count2 = pq2.count();

  if(count1 != count2) {
    log_app.error() << "ERROR: Copy query has different count!";
    errors++;
  }

  // Modify copy
  if(count_processors(machine, Processor::UTIL_PROC) > 0) {
    pq2.only_kind(Processor::UTIL_PROC);
    size_t count2_util = pq2.count();

    // Original should be unchanged
    size_t count1_after = pq1.count();
    if(count1_after != count1) {
      log_app.error() << "ERROR: Original query was modified after copy modification!";
      errors++;
    }

    if(count2_util == count1) {
      log_app.error() << "ERROR: Copy query was not modified!";
      errors++;
    }
  }

  log_app.print() << "  Query copy tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test has_affinity predicates
void test_has_affinity(Machine machine, int &errors)
{
  log_app.print() << "Testing has_affinity predicates...";

  // Find a CPU processor
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  if(!cpu.exists()) {
    log_app.print() << "  SKIP: No CPU processors available";
    return;
  }

  // Find memories with affinity to this CPU
  Machine::MemoryQuery mq(machine);
  mq.has_affinity_to(cpu);
  size_t affinity_count = mq.count();
  log_app.print() << "  Memories with affinity to CPU " << cpu << ": " << affinity_count;

  if(affinity_count == 0) {
    log_app.error() << "ERROR: No memories with affinity to CPU processor!";
    errors++;
  }

  // Verify each memory actually has affinity
  std::vector<Machine::ProcessorMemoryAffinity> affinities;
  machine.get_proc_mem_affinity(affinities, cpu);
  std::set<Memory> expected_mems;
  for(const auto &aff : affinities) {
    expected_mems.insert(aff.m);
  }

  std::set<Memory> query_mems;
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    query_mems.insert(m);
  }

  if(query_mems != expected_mems) {
    log_app.error() << "ERROR: has_affinity_to returned wrong memories!";
    log_app.error() << "  Expected: " << expected_mems.size()
                    << ", Got: " << query_mems.size();
    errors++;
  }

  // Test processor query with memory affinity
  Memory mem = mq.first();
  if(mem.exists()) {
    Machine::ProcessorQuery pq2(machine);
    pq2.has_affinity_to(mem);
    size_t proc_affinity_count = pq2.count();
    log_app.print() << "  Processors with affinity to Memory " << mem << ": "
                    << proc_affinity_count;

    if(proc_affinity_count == 0) {
      log_app.error() << "ERROR: No processors with affinity to memory!";
      errors++;
    }
  }

  log_app.print() << "  has_affinity tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test best_affinity_to for ProcessorQuery
void test_processor_best_affinity(Machine machine, int &errors)
{
  log_app.print() << "Testing ProcessorQuery best_affinity_to...";

  // Find a memory
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  if(!mem.exists()) {
    log_app.print() << "  SKIP: No system memory available";
    return;
  }

  // Find processor with best affinity to this memory
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);
  size_t best_count = pq.count();
  log_app.print() << "  Processors with best affinity to Memory " << mem << ": "
                  << best_count;

  if(best_count == 0) {
    log_app.error() << "ERROR: No processors with best affinity!";
    errors++;
    return;
  }

  // Get the best processor(s)
  std::vector<Processor> best_procs;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    best_procs.push_back(p);
  }

  // Manually compute expected best score
  Machine::ProcessorQuery all_cpu_pq(machine);
  all_cpu_pq.only_kind(Processor::LOC_PROC);
  int best_score = INT_MIN;
  bool found_any = false;

  for(Processor p = all_cpu_pq.first(); p.exists(); p = all_cpu_pq.next(p)) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, p);

    for(const auto &aff : affinities) {
      if(aff.m == mem) {
        // Default weights: bandwidth=1, latency=0
        int score = aff.bandwidth;
        if(!found_any || score > best_score) {
          best_score = score;
          found_any = true;
        }
        break;
      }
    }
  }

  // Verify all returned processors have the best score
  for(Processor p : best_procs) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, p);

    int proc_score = INT_MIN;
    for(const auto &aff : affinities) {
      if(aff.m == mem) {
        proc_score = aff.bandwidth;
        break;
      }
    }

    if(proc_score != best_score) {
      log_app.error() << "ERROR: Processor " << p << " has score " << proc_score
                      << " but best is " << best_score;
      errors++;
    }
  }

  // Test random() with best affinity
  Processor random_proc = pq.random();
  if(random_proc.exists()) {
    bool found = false;
    for(Processor p : best_procs) {
      if(p == random_proc) {
        found = true;
        break;
      }
    }
    if(!found) {
      log_app.error() << "ERROR: random() returned processor not in best affinity set!";
      errors++;
    }
  }

  log_app.print() << "  ProcessorQuery best_affinity tests: "
                  << (errors == 0 ? "PASS" : "FAIL");
}

// Test best_affinity_to for MemoryQuery
void test_memory_best_affinity(Machine machine, int &errors)
{
  log_app.print() << "Testing MemoryQuery best_affinity_to...";

  // Find a CPU processor
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  if(!cpu.exists()) {
    log_app.print() << "  SKIP: No CPU processor available";
    return;
  }

  // Find memory with best affinity to this processor
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  mq.best_affinity_to(cpu);
  size_t best_count = mq.count();
  log_app.print() << "  Memories with best affinity to Processor " << cpu << ": "
                  << best_count;

  if(best_count == 0) {
    log_app.error() << "ERROR: No memories with best affinity!";
    errors++;
    return;
  }

  // Get the best memory(s)
  std::vector<Memory> best_mems;
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    best_mems.push_back(m);
  }

  // Manually compute expected best score
  Machine::MemoryQuery all_sysmem_mq(machine);
  all_sysmem_mq.only_kind(Memory::SYSTEM_MEM);
  int best_score = INT_MIN;
  bool found_any = false;

  for(Memory m = all_sysmem_mq.first(); m.exists(); m = all_sysmem_mq.next(m)) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, cpu, m);

    for(const auto &aff : affinities) {
      // Default weights: bandwidth=1, latency=0
      int score = aff.bandwidth;
      if(!found_any || score > best_score) {
        best_score = score;
        found_any = true;
      }
    }
  }

  // Verify all returned memories have the best score
  for(Memory m : best_mems) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, cpu, m);

    int mem_score = INT_MIN;
    for(const auto &aff : affinities) {
      mem_score = aff.bandwidth;
    }

    if(affinities.size() > 0 && mem_score != best_score) {
      log_app.error() << "ERROR: Memory " << m << " has score " << mem_score
                      << " but best is " << best_score;
      errors++;
    }
  }

  // Test random() with best affinity
  Memory random_mem = mq.random();
  if(random_mem.exists()) {
    bool found = false;
    for(Memory m : best_mems) {
      if(m == random_mem) {
        found = true;
        break;
      }
    }
    if(!found) {
      log_app.error() << "ERROR: random() returned memory not in best affinity set!";
      errors++;
    }
  }

  log_app.print() << "  MemoryQuery best_affinity tests: "
                  << (errors == 0 ? "PASS" : "FAIL");
}

// Test best_affinity with custom weights
void test_best_affinity_weights(Machine machine, int &errors)
{
  log_app.print() << "Testing best_affinity with custom weights...";

  // Find a memory
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  if(!mem.exists()) {
    log_app.print() << "  SKIP: No system memory available";
    return;
  }

  // Test with latency-weighted affinity
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem, 0, 1); // bandwidth_weight=0, latency_weight=1
  size_t latency_best_count = pq.count();
  log_app.print() << "  Processors with best latency to Memory " << mem << ": "
                  << latency_best_count;

  if(latency_best_count == 0) {
    log_app.error() << "ERROR: No processors with best latency!";
    errors++;
    return;
  }

  // Get the best processor by latency
  Processor best_latency_proc = pq.first();

  // Manually verify
  Machine::ProcessorQuery all_cpu_pq(machine);
  all_cpu_pq.only_kind(Processor::LOC_PROC);
  int best_latency = INT_MIN;

  for(Processor p = all_cpu_pq.first(); p.exists(); p = all_cpu_pq.next(p)) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, p);

    for(const auto &aff : affinities) {
      if(aff.m == mem) {
        int score = aff.latency;
        if(best_latency == INT_MIN || score > best_latency) {
          best_latency = score;
        }
        break;
      }
    }
  }

  // Verify the returned processor has the best latency
  std::vector<Machine::ProcessorMemoryAffinity> best_affinities;
  machine.get_proc_mem_affinity(best_affinities, best_latency_proc);
  int returned_latency = INT_MIN;
  for(const auto &aff : best_affinities) {
    if(aff.m == mem) {
      returned_latency = aff.latency;
      break;
    }
  }

  if(returned_latency != best_latency) {
    log_app.error()
        << "ERROR: best_affinity_to with latency weight returned wrong processor!";
    log_app.error() << "  Expected latency: " << best_latency
                    << ", Got: " << returned_latency;
    errors++;
  }

  log_app.print() << "  best_affinity weight tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test best_affinity with multiple targets (MemoryQuery)
void test_memory_best_affinity_multiple_targets(Machine machine, int &errors)
{
  log_app.print() << "Testing MemoryQuery best_affinity with multiple targets...";

  // Find a CPU processor
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  if(!cpu.exists()) {
    log_app.print() << "  SKIP: No CPU processor available";
    return;
  }

  // Find a memory
  Machine::MemoryQuery temp_mq(machine);
  temp_mq.only_kind(Memory::SYSTEM_MEM);
  Memory target_mem = temp_mq.first();
  if(!target_mem.exists()) {
    log_app.print() << "  SKIP: No system memory available";
    return;
  }

  // Query for memories with best combined affinity to both processor and memory
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  mq.best_affinity_to(cpu);
  mq.best_affinity_to(target_mem);
  size_t combined_count = mq.count();
  log_app.print() << "  Memories with best combined affinity to Processor " << cpu
                  << " and Memory " << target_mem << ": " << combined_count;

  if(combined_count == 0) {
    log_app.error() << "ERROR: No memories with combined best affinity!";
    errors++;
    return;
  }

  // Verify the combined scoring works
  std::vector<Memory> result_mems;
  for(Memory m = mq.first(); m.exists(); m = mq.next(m)) {
    result_mems.push_back(m);
  }

  // Manually compute expected best combined score
  Machine::MemoryQuery all_sysmem_mq(machine);
  all_sysmem_mq.only_kind(Memory::SYSTEM_MEM);
  int best_combined_score = INT_MIN;

  for(Memory m = all_sysmem_mq.first(); m.exists(); m = all_sysmem_mq.next(m)) {
    int proc_score = INT_MIN;
    int mem_score = INT_MIN;

    // Get proc->memory affinity
    std::vector<Machine::ProcessorMemoryAffinity> proc_affinities;
    machine.get_proc_mem_affinity(proc_affinities, cpu, m);
    if(!proc_affinities.empty()) {
      proc_score = proc_affinities[0].bandwidth;
    }

    // Get memory->memory affinity
    std::vector<Machine::MemoryMemoryAffinity> mem_affinities;
    machine.get_mem_mem_affinity(mem_affinities, m, target_mem);
    if(!mem_affinities.empty()) {
      mem_score = mem_affinities[0].bandwidth;
    }

    if(proc_score != INT_MIN && mem_score != INT_MIN) {
      int combined_score = proc_score + mem_score;
      if(best_combined_score == INT_MIN || combined_score > best_combined_score) {
        best_combined_score = combined_score;
      }
    }
  }

  log_app.print() << "  Combined best_affinity tests: "
                  << (errors == 0 ? "PASS" : "FAIL");
}

// Test best_affinity with ties (multiple candidates with same best score)
void test_best_affinity_ties(Machine machine, int &errors)
{
  log_app.print() << "Testing best_affinity with ties...";

  // Find a memory
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  if(!mem.exists()) {
    log_app.print() << "  SKIP: No system memory available";
    return;
  }

  // Find all processors with best affinity
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  pq.best_affinity_to(mem);

  std::vector<Processor> best_procs;
  for(Processor p = pq.first(); p.exists(); p = pq.next(p)) {
    best_procs.push_back(p);
  }

  if(best_procs.empty()) {
    log_app.print() << "  SKIP: No processors with affinity";
    return;
  }

  // Check if all have the same score
  std::vector<Machine::ProcessorMemoryAffinity> first_affinities;
  machine.get_proc_mem_affinity(first_affinities, best_procs[0]);
  int first_score = INT_MIN;
  for(const auto &aff : first_affinities) {
    if(aff.m == mem) {
      first_score = aff.bandwidth;
      break;
    }
  }

  bool all_same = true;
  for(size_t i = 1; i < best_procs.size(); i++) {
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, best_procs[i]);
    int score = INT_MIN;
    for(const auto &aff : affinities) {
      if(aff.m == mem) {
        score = aff.bandwidth;
        break;
      }
    }
    if(score != first_score) {
      all_same = false;
      break;
    }
  }

  if(all_same && best_procs.size() > 1) {
    log_app.print() << "  Found " << best_procs.size()
                    << " processors with tied best affinity";
  } else if(best_procs.size() == 1) {
    log_app.print() << "  Single processor with best affinity (no tie)";
  } else {
    log_app.error() << "ERROR: Processors with different scores in best affinity result!";
    errors++;
  }

  log_app.print() << "  best_affinity tie tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test query with same_address_space_as
void test_same_address_space(Machine machine, int &errors)
{
  log_app.print() << "Testing same_address_space_as...";

  // Find a CPU processor
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  if(!cpu.exists()) {
    log_app.print() << "  SKIP: No CPU processor available";
    return;
  }

  // Find processors in same address space
  Machine::ProcessorQuery same_space_pq(machine);
  same_space_pq.same_address_space_as(cpu);
  size_t same_space_count = same_space_pq.count();
  log_app.print() << "  Processors in same address space as " << cpu << ": "
                  << same_space_count;

  if(same_space_count == 0) {
    log_app.error()
        << "ERROR: No processors in same address space (should include self)!";
    errors++;
  }

  // Verify the queried processor itself is in the result
  bool found_self = false;
  for(Processor p = same_space_pq.first(); p.exists(); p = same_space_pq.next(p)) {
    if(p == cpu) {
      found_self = true;
      break;
    }
  }

  if(!found_self) {
    log_app.error() << "ERROR: Processor not found in its own address space!";
    errors++;
  }

  log_app.print() << "  same_address_space tests: " << (errors == 0 ? "PASS" : "FAIL");
}

// Test combining multiple constraints
void test_combined_constraints(Machine machine, int &errors)
{
  log_app.print() << "Testing combined query constraints...";

  // Find a memory
  Machine::MemoryQuery mq(machine);
  mq.only_kind(Memory::SYSTEM_MEM);
  Memory mem = mq.first();
  if(!mem.exists()) {
    log_app.print() << "  SKIP: No system memory available";
    return;
  }

  // Find a CPU processor
  Machine::ProcessorQuery pq(machine);
  pq.only_kind(Processor::LOC_PROC);
  Processor cpu = pq.first();
  if(!cpu.exists()) {
    log_app.print() << "  SKIP: No CPU processor available";
    return;
  }

  // Combine: kind + same_address_space + has_affinity
  Machine::ProcessorQuery combined_pq(machine);
  combined_pq.only_kind(Processor::LOC_PROC);
  combined_pq.same_address_space_as(cpu);
  combined_pq.has_affinity_to(mem);
  size_t combined_count = combined_pq.count();
  log_app.print() << "  CPU processors in same address space with affinity to " << mem
                  << ": " << combined_count;

  // All results should satisfy all constraints
  for(Processor p = combined_pq.first(); p.exists(); p = combined_pq.next(p)) {
    // Check kind
    if(p.kind() != Processor::LOC_PROC) {
      log_app.error() << "ERROR: Processor " << p << " has wrong kind!";
      errors++;
    }

    // Check affinity
    std::vector<Machine::ProcessorMemoryAffinity> affinities;
    machine.get_proc_mem_affinity(affinities, p);
    bool has_affinity = false;
    for(const auto &aff : affinities) {
      if(aff.m == mem) {
        has_affinity = true;
        break;
      }
    }
    if(!has_affinity) {
      log_app.error() << "ERROR: Processor " << p << " doesn't have affinity to " << mem
                      << "!";
      errors++;
    }
  }

  log_app.print() << "  Combined constraint tests: " << (errors == 0 ? "PASS" : "FAIL");
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  log_app.print() << "Machine Query Test Starting...";

  int errors = 0;
  Machine machine = Machine::get_machine();

  // Run all tests
  test_processor_query_basic(machine, errors);
  test_memory_query_basic(machine, errors);
  test_query_copy(machine, errors);
  test_has_affinity(machine, errors);
  test_processor_best_affinity(machine, errors);
  test_memory_best_affinity(machine, errors);
  test_best_affinity_weights(machine, errors);
  test_memory_best_affinity_multiple_targets(machine, errors);
  test_best_affinity_ties(machine, errors);
  test_same_address_space(machine, errors);
  test_combined_constraints(machine, errors);

  // Summary
  log_app.print() << "========================================";
  if(errors == 0) {
    log_app.print() << "ALL TESTS PASSED!";
  } else {
    log_app.error() << "TESTS FAILED: " << errors << " error(s) detected!";
  }
  log_app.print() << "========================================";

  Runtime::get_runtime().shutdown(Event::NO_EVENT, errors > 0 ? 1 : 0);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // Select a processor to run the top level task
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // Start with a single processor to begin tests
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // Request shutdown once that task is complete
  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
