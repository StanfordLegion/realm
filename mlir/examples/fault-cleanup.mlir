// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// The caller cancels one gate and keeps the other pending until it has
// observed the fault. The temporary remains live for the pending task.
func.func @fault_cleanup(%cpu: !realm.processor, %seed: !realm.task,
                         %memory: !realm.memory, %layout: !realm.layout,
                         %cancel_gate: !realm.event, %slow_gate: !realm.event)
    -> (!realm.event, !realm.event) {
  %tmp, %ready = realm.instance.create in %memory layout %layout
  %exists = realm.instance.exists %tmp
  %reported, %safe = scf.if %exists -> (!realm.event, !realm.event) {
    %failed = realm.spawn %seed(%tmp) on %cpu after [%ready, %cancel_gate]
    %slow = realm.spawn %seed(%tmp) on %cpu after [%ready, %slow_gate]
    %reported = realm.event.merge [%failed, %slow]
    %safe = realm.event.join_settled [%ready, %failed, %slow]
    realm.instance.destroy %tmp after [%safe]
    scf.yield %reported, %safe : !realm.event, !realm.event
  } else {
    scf.yield %ready, %ready : !realm.event, !realm.event
  }
  return %reported, %safe : !realm.event, !realm.event
}
