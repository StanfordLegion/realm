// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func @query_local_cpu() -> (!realm.processor, i1) {
  %processor, %found = realm.machine.find_processor kind #realm.processor_kind<loc_proc> local
  return %processor, %found : !realm.processor, i1
}

func.func @query_processor_for(%memory: !realm.memory) -> (!realm.processor, i1) {
  %processor, %found = realm.machine.find_processor kind #realm.processor_kind<loc_proc>
      local same_address_space_as %memory with_affinity_to %memory
  return %processor, %found : !realm.processor, i1
}

func.func @query_system_memory(%cpu: !realm.processor, %use_affinity: i1) -> (!realm.memory, i1) {
  %memory, %found = scf.if %use_affinity -> (!realm.memory, i1) {
    %near, %ok = realm.machine.find_memory kind #realm.memory_kind<system_mem>
        same_address_space_as %cpu with_affinity_to %cpu min_capacity 4096
    scf.yield %near, %ok : !realm.memory, i1
  } else {
    %oversized, %ok = realm.machine.find_memory kind #realm.memory_kind<system_mem>
        local min_capacity 18446744073709551615
    scf.yield %oversized, %ok : !realm.memory, i1
  }
  return %memory, %found : !realm.memory, i1
}

func.func @query_local_gpu() -> (!realm.processor, i1) {
  %processor, %found = realm.machine.find_processor kind #realm.processor_kind<toc_proc> local
  return %processor, %found : !realm.processor, i1
}

func.func @query_gpu_framebuffer(%gpu: !realm.processor) -> (!realm.memory, i1) {
  %memory, %found = realm.machine.find_memory kind #realm.memory_kind<gpu_fb_mem>
      same_address_space_as %gpu with_affinity_to %gpu
  return %memory, %found : !realm.memory, i1
}
