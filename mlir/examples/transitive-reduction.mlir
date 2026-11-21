// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// When the tasks succeed, %b's completion already establishes %a's
// completion. The pass removes %a from the final launch's dependency list.
// CHECK-LABEL: func.func @reduce_event_edges(
// CHECK: %[[SEED:[a-zA-Z0-9]+]] = realm.spawn
// CHECK-NEXT: %[[INCREMENT:[a-zA-Z0-9]+]] = realm.spawn {{.*}} after [%[[SEED]]]
// CHECK-NEXT: {{.*}} = realm.spawn {{.*}} after [%[[INCREMENT]]]
// EMITC-LABEL: func.func @reduce_event_edges(
// EMITC: emitc.call_opaque "Realm::MLIR::readyEvent"
// EMITC-NOT: Realm::MLIR::mergeEvents
// EMITC: return
func.func @reduce_event_edges(%cpu: !realm.processor, %seed: !realm.task,
                              %increment: !realm.task, %buffer: !realm.instance)
    -> !realm.event attributes {realm.assume_success} {
  %a = realm.spawn %seed(%buffer) on %cpu
  %b = realm.spawn %increment(%buffer) on %cpu after [%a]
  %c = realm.spawn %increment(%buffer) on %cpu after [%a, %b]
  return %c : !realm.event
}
