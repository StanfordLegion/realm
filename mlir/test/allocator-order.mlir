// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// The second request can be funded by the pending free of the first
// instance. Preserve allocator issue order without adding a wait or passing
// the free's precondition to the next allocation.
//
// EMITC-LABEL: func.func @replace_buffer(
// EMITC: emitc.call_opaque "Realm::MLIR::createInstance"
// EMITC: emitc.call_opaque "Realm::MLIR::destroyInstance"
// EMITC-NOT: "Realm::MLIR::awaitEvent"
// EMITC: %[[READY:[a-zA-Z0-9]+]] = emitc.call_opaque "Realm::MLIR::readyEvent"()
// EMITC: emitc.call_opaque "Realm::MLIR::createInstance"(%arg0, %arg1, %[[READY]])
func.func @replace_buffer(%memory: !realm.memory, %layout: !realm.layout,
                          %last_use: !realm.event) -> (!realm.instance, !realm.event) {
  %old, %old_ready = realm.instance.create in %memory layout %layout
  %settled = realm.event.join_settled [%old_ready, %last_use]
  realm.instance.destroy %old after [%settled]
  %new, %new_ready = realm.instance.create in %memory layout %layout
  return %new, %new_ready : !realm.instance, !realm.event
}
