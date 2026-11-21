// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func @find_local_cpu() -> (!realm.processor, i1) {
  %cpu, %found = realm.machine.find_processor kind #realm.processor_kind<loc_proc> local
  return %cpu, %found : !realm.processor, i1
}

func.func @find_cpu_memory(%cpu: !realm.processor) -> (!realm.memory, i1) {
  %memory, %found = realm.machine.find_memory kind #realm.memory_kind<system_mem>
      same_address_space_as %cpu with_affinity_to %cpu min_capacity 8
  return %memory, %found : !realm.memory, i1
}

func.func @run_pipeline(%cpu: !realm.processor, %memory: !realm.memory,
                        %layout: !realm.layout, %seed: !realm.task,
                        %double: !realm.task, %registered: !realm.event,
                        %start: !realm.event) -> (!realm.instance, !realm.event) {
  %input, %input_ready = realm.instance.create in %memory layout %layout
      after [%registered, %start]
  %has_input = realm.instance.exists %input
  %result, %finished = scf.if %has_input -> (!realm.instance, !realm.event) {
    %seeded = realm.spawn %seed(%input) on %cpu after [%input_ready]
    %output, %output_ready = realm.instance.create in %memory layout %layout
        after [%seeded]
    %has_output = realm.instance.exists %output
    %done = scf.if %has_output -> (!realm.event) {
      %doubled = realm.spawn %double(%input, %output) on %cpu after [%output_ready]
      scf.yield %doubled : !realm.event
    } else {
      scf.yield %output_ready : !realm.event
    }

    // Cover every input use, even if one precondition reports poison early.
    %settled = realm.event.join_settled [%input_ready, %seeded, %done]
    realm.instance.destroy %input after [%settled]
    scf.yield %output, %done : !realm.instance, !realm.event
  } else {
    scf.yield %input, %input_ready : !realm.instance, !realm.event
  }
  return %result, %finished : !realm.instance, !realm.event
}

func.func @await_pipeline(%finished: !realm.event) -> i1 {
  %poisoned = realm.event.await %finished
  return %poisoned : i1
}

func.func @release_result(%instance: !realm.instance, %finished: !realm.event) {
  %has_instance = realm.instance.exists %instance
  scf.if %has_instance {
    %settled = realm.event.join_settled [%finished]
    realm.instance.destroy %instance after [%settled]
  }
  return
}
