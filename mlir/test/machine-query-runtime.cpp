// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "realm.h"

#include <iostream>
#include <tuple>

std::tuple<Realm::Processor, bool> query_local_cpu();
std::tuple<Realm::Processor, bool> query_processor_for(Realm::Memory memory);
std::tuple<Realm::Memory, bool> query_system_memory(Realm::Processor cpu,
                                                    bool use_affinity);
std::tuple<Realm::Processor, bool> query_local_gpu();
std::tuple<Realm::Memory, bool> query_gpu_framebuffer(Realm::Processor gpu);

int main(int argc, char **argv)
{
  Realm::Runtime runtime;
  if(!runtime.init(&argc, &argv))
    return 1;

  auto [cpu, found_cpu] = query_local_cpu();
  auto [memory, found_memory] = query_system_memory(cpu, true);
  auto [nearby_cpu, found_nearby_cpu] = query_processor_for(memory);
  auto [oversized, found_oversized] = query_system_memory(cpu, false);
  auto [invalid, found_invalid] = query_system_memory(Realm::Processor::NO_PROC, true);
  Realm::Machine machine = Realm::Machine::get_machine();

  bool passed = found_cpu && cpu.exists() && cpu.kind() == Realm::Processor::LOC_PROC &&
                found_memory && memory.exists() &&
                memory.kind() == Realm::Memory::SYSTEM_MEM && memory.capacity() >= 4096 &&
                memory.address_space() == cpu.address_space() && found_nearby_cpu &&
                nearby_cpu.exists() &&
                nearby_cpu.address_space() == memory.address_space() &&
                machine.has_affinity(nearby_cpu, memory) && !found_oversized &&
                !oversized.exists() && !found_invalid && !invalid.exists();
  if(!passed) {
    std::cerr << std::boolalpha << "Unexpected Realm machine query results\n"
              << "CPU: " << cpu << " (found " << found_cpu << ")\n"
              << "System memory: " << memory << " (found " << found_memory << ")\n"
              << "Nearby CPU: " << nearby_cpu << " (found " << found_nearby_cpu << ")\n"
              << "Oversized memory request: " << oversized << " (found "
              << found_oversized << ")\n"
              << "Invalid CPU reference: " << invalid << " (found " << found_invalid
              << ")\n";
  }

  auto [gpu, found_gpu] = query_local_gpu();
  auto [framebuffer, found_framebuffer] = query_gpu_framebuffer(gpu);
  Realm::Processor expected_gpu = Realm::Machine::ProcessorQuery(machine)
                                      .only_kind(Realm::Processor::TOC_PROC)
                                      .local_address_space()
                                      .first();
  Realm::Memory expected_framebuffer = Realm::Memory::NO_MEMORY;
  if(expected_gpu.exists()) {
    expected_framebuffer = Realm::Machine::MemoryQuery(machine)
                               .only_kind(Realm::Memory::GPU_FB_MEM)
                               .same_address_space_as(expected_gpu)
                               .has_affinity_to(expected_gpu)
                               .first();
  }
  if(gpu != expected_gpu || found_gpu != expected_gpu.exists() ||
     framebuffer != expected_framebuffer ||
     found_framebuffer != expected_framebuffer.exists()) {
    std::cerr
        << "GPU processor or framebuffer query disagreed with the Realm machine model\n";
    passed = false;
  }

  runtime.shutdown(Realm::Event::NO_EVENT, passed ? 0 : 1);
  return runtime.wait_for_shutdown();
}
