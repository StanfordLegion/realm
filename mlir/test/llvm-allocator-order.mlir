// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func private @realm_mlir_save_handles(!realm.instance, !realm.event, !llvm.ptr, !llvm.ptr)

// Submit the pending free before requesting the replacement, with no host wait.
// LLVM-LABEL: llvm.func @replace_buffer(
// LLVM: llvm.call @realm_mlir_create_dense
// LLVM: llvm.call @realm_region_instance_destroy
// LLVM-NOT: llvm.call @realm_event_wait
// LLVM: %[[READY:[a-zA-Z0-9]+]] = llvm.mlir.constant(0 : i64)
// LLVM: llvm.call @realm_mlir_create_dense({{.*}}, %[[READY]], {{[^,]+}}, {{[^,]+}})
func.func @replace_buffer(%memory: !realm.memory, %extent: i64, %gate: !realm.event,
                          %next: !llvm.ptr, %next_ready: !llvm.ptr) {
  %domain = realm.index_space.dense %extent
  %layout = realm.layout.dense %domain field_bytes 1
  %old, %old_ready = realm.instance.create in %memory layout %layout : !realm.dense_layout
  %settled = realm.event.join_settled [%old_ready, %gate]
  realm.instance.destroy %old after [%settled]
  %new, %ready = realm.instance.create in %memory layout %layout : !realm.dense_layout
  func.call @realm_mlir_save_handles(%new, %ready, %next, %next_ready)
    : (!realm.instance, !realm.event, !llvm.ptr, !llvm.ptr) -> ()
  return
}
