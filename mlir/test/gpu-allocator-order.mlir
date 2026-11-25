// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

module attributes {gpu.container_module, realm.task_initializer = "init_gpu_tasks"} {
  gpu.module @kernels attributes {sym_visibility = "private"} {
    gpu.func @touch(%buffer: memref<8xi32, 1>) kernel { gpu.return }
  }

  func.func private @launch(%buffer: !realm.instance,
      %gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64)
      attributes {realm.gpu_kernel = @kernels::@touch} {
    return
  }

  // The first use must be settled before freeing the first instance. Issue the
  // free ahead of the replacement request, and await the original task event.
  // REALM-LABEL: func.func @issue(
  // REALM: %[[OLD:[a-zA-Z0-9]+]], %[[READY:[a-zA-Z0-9]+]] = realm.instance.create
  // REALM: %[[TASK:[a-zA-Z0-9]+]], %[[REGISTERED:[a-zA-Z0-9]+]] = realm.task.register @launch id 70
  // REALM: %[[DONE:[a-zA-Z0-9]+]] = realm.spawn_typed %[[TASK]](%[[OLD]], {{.*}}) on %arg0 after [%[[READY]], %[[REGISTERED]]]
  // REALM: %[[SETTLED:[a-zA-Z0-9]+]] = realm.event.join_settled [%[[DONE]], %[[READY]]]
  // REALM: realm.instance.destroy %[[OLD]] after [%[[SETTLED]]]
  // REALM-NOT: realm.event.await
  // REALM: %[[NEW:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = realm.instance.create
  // REALM: realm.instance.destroy %[[NEW]]
  // REALM-NOT: realm.event.merge
  // REALM: realm.event.await %[[DONE]]
  //
  // LLVM-LABEL: llvm.func @init_gpu_tasks()
  // LLVM: llvm.call @realm_processor_register_task_by_kind(
  // LLVM-NOT: llvm.call @realm_processor_register_task_by_kind(
  // LLVM: llvm.return
  // LLVM-LABEL: llvm.func @issue(
  // LLVM: llvm.call @realm_mlir_create_dense(
  // LLVM-NOT: llvm.call @realm_event_merge
  // LLVM: llvm.call @realm_processor_spawn(
  // LLVM: llvm.call @realm_region_instance_destroy(
  // LLVM-NOT: llvm.call @realm_event_wait
  // LLVM: llvm.call @realm_mlir_create_dense(
  // LLVM: llvm.call @realm_event_wait(
  func.func @issue(%gpu: !realm.processor, %memory: !realm.memory)
      attributes {realm.gpu_entry} {
    %one = arith.constant 1 : index
    %old, %ready = gpu.alloc async () : memref<8xi32, 1>
    %done = gpu.launch_func async [%ready] @kernels::@touch
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%old : memref<8xi32, 1>)
        {realm.task_entry = @launch, realm.task_id = 70 : i32}
    %freed = gpu.dealloc async [%done] %old : memref<8xi32, 1>
    %new, %new_ready = gpu.alloc async () : memref<8xi32, 1>
    %freed_again = gpu.dealloc async [%new_ready] %new : memref<8xi32, 1>
    gpu.wait [%done]
    return
  }

  // REALM-LABEL: func.func @use_borrowed(
  // REALM: realm.task.register @launch id 70
  // REALM: realm.spawn_typed
  // LLVM-LABEL: llvm.func @use_borrowed(
  // LLVM-NOT: llvm.call @realm_processor_register_task_by_kind(
  // LLVM: llvm.call @realm_processor_spawn(
  func.func @use_borrowed(%gpu: !realm.processor, %memory: !realm.memory,
                          %borrowed: memref<8xi32, 1>) attributes {realm.gpu_entry} {
    %one = arith.constant 1 : index
    %done = gpu.launch_func async @kernels::@touch
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%borrowed : memref<8xi32, 1>)
        {realm.task_entry = @launch, realm.task_id = 70 : i32}
    gpu.wait [%done]
    return
  }
}
