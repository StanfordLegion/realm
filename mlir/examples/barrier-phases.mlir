// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Each worker's completion permits one arrival. The second phase's workers
// are submitted up front, with the first phase event as their precondition.
// The gate and worker completions must succeed for these barrier arrivals.
func.func @arrive_two_phases(%cpu0: !realm.processor, %cpu1: !realm.processor,
                             %worker: !realm.task, %left: !realm.instance,
                             %right: !realm.instance, %first: !realm.barrier_phase,
                             %gate: !realm.event,
                             %a: i32, %b: i32, %c: i32, %d: i32) -> !realm.barrier_phase {
  %second = realm.barrier.advance %first
  %first_complete = realm.barrier.as_event %first
  %left0 = realm.spawn %worker(%left) on %cpu0
  %right0 = realm.spawn %worker(%right) on %cpu1 after [%gate]
  realm.barrier.arrive %first value %a after [%left0]
  realm.barrier.arrive %first value %b after [%right0]
  %left1 = realm.spawn %worker(%left) on %cpu0 after [%first_complete]
  %right1 = realm.spawn %worker(%right) on %cpu1 after [%first_complete]
  realm.barrier.arrive %second value %c after [%left1]
  realm.barrier.arrive %second value %d after [%right1]
  return %second : !realm.barrier_phase
}

func.func @read_phase(%phase: !realm.barrier_phase) -> (i32, i1) {
  %complete = realm.barrier.as_event %phase
  %poisoned = realm.event.await %complete
  %sum, %available = realm.barrier.get_result %phase
  return %sum, %available : i32, i1
}
