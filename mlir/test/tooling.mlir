// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// CANON-LABEL: func.func @empty_joins(
// CANON-NOT: realm.event.merge []
// CANON-NOT: realm.event.join_settled []
// CANON: %[[READY:[a-zA-Z0-9]+]] = realm.event.ready
// CANON-NEXT: %[[NATIVE:[a-zA-Z0-9]+]] = realm.event.merge [%{{[a-zA-Z0-9]+}}]
// CANON-NEXT: return %[[READY]], %[[READY]], %[[NATIVE]]
func.func @empty_joins(%incoming: !realm.event) -> (!realm.event, !realm.event, !realm.event) {
  %empty = realm.event.merge []
  %settled = realm.event.join_settled []
  %native = realm.event.merge [%incoming]
  return %empty, %settled, %native : !realm.event, !realm.event, !realm.event
}

// LOWER-LABEL: func.func @copy_and_arrive(
// LOWER-SAME: %[[SOURCE:[a-zA-Z0-9]+]]: !emitc.opaque<"Realm::RegionInstance">, %[[TARGET:[a-zA-Z0-9]+]]: !emitc.opaque<"Realm::RegionInstance">, %[[SPACE:[a-zA-Z0-9]+]]:
// LOWER-SAME: %[[PHASE:[a-zA-Z0-9]+]]: !emitc.opaque<"Realm::Barrier">, %[[ENTRY:[a-zA-Z0-9]+]]:
// LOWER: %[[COPIED:[a-zA-Z0-9]+]] = emitc.call_opaque "Realm::MLIR::copy"(%[[SPACE]], %[[SOURCE]], %[[TARGET]], %[[ENTRY]]) {args = [0 : index, 1 : index, 2 : index, 17 : i32, 29 : i32, 4, 3 : index]}
// LOWER-NEXT: %[[THREE:[a-zA-Z0-9_]+]] = arith.constant 3 : i32
// LOWER-NEXT: %[[SETTLED:[a-zA-Z0-9]+]] = emitc.call_opaque "Realm::MLIR::joinSettled"(%[[COPIED]])
// LOWER-NEXT: emitc.call_opaque "Realm::MLIR::arriveBarrier"(%[[PHASE]], %[[THREE]], %[[SETTLED]])
func.func @copy_and_arrive(%src: !realm.instance, %dst: !realm.instance,
                           %space: !realm.index_space<1, ui32>, %phase: !realm.barrier_phase,
                           %entry: !realm.event) {
  %copied = realm.copy %src field 17 to %dst field 29 over %space bytes 4
      after [%entry] : !realm.index_space<1, ui32>
  %three = arith.constant 3 : i32
  %settled = realm.event.join_settled [%copied]
  realm.barrier.arrive %phase value %three after [%settled]
  return
}

// LOWER-LABEL: func.func @machine_kinds(
// LOWER: #emitc.opaque<"Realm::Processor::TOC_PROC">
// LOWER: #emitc.opaque<"Realm::Memory::GPU_DYNAMIC_MEM">
// LOWER: #emitc.opaque<"Realm::Memory::LEVEL1_CACHE">
func.func @machine_kinds() {
  %gpu, %found = realm.machine.find_processor kind #realm.processor_kind<toc_proc> local
  %framebuffer, %has_framebuffer = realm.machine.find_memory kind #realm.memory_kind<gpu_dynamic_mem>
  %cache, %has_cache = realm.machine.find_memory kind #realm.memory_kind<level1_cache>
  return
}
