// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

module attributes {gpu.container_module, realm.task_initializer = "realm_mlir_init_gpu_input_tasks"} {
  // The device code is packaged separately for this example. The host pass
  // preserves the kernel module and removes the gpu.launch_func references.
  gpu.module @kernels attributes {sym_visibility = "private"} {
    gpu.func @add_to_buffer(%buffer: memref<513xi32, 1>, %amount: i32) kernel {
      %block = gpu.block_id x
      %width = gpu.block_dim x
      %thread = gpu.thread_id x
      %first = arith.muli %block, %width : index
      %i = arith.addi %first, %thread : index
      %count = arith.constant 513 : index
      %active = arith.cmpi ult, %i, %count : index
      scf.if %active {
        %old = memref.load %buffer[%i] : memref<513xi32, 1>
        %new = arith.addi %old, %amount : i32
        memref.store %new, %buffer[%i] : memref<513xi32, 1>
      }
      gpu.return
    }
  }

  func.func private @realm_mlir_cuda_add_dims(
      !realm.instance, i64, i32, i64, i64, i64, i64, i64, i64)

  func.func private @launch_add(%instance: !realm.instance, %amount: i32,
      %gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64)
      attributes {realm.gpu_kernel = @kernels::@add_to_buffer} {
    %count = arith.constant 513 : i64
    func.call @realm_mlir_cuda_add_dims(%instance, %count, %amount, %gx, %gy, %gz, %bx, %by, %bz)
        : (!realm.instance, i64, i32, i64, i64, i64, i64, i64, i64) -> ()
    return
  }

  // The caller binds the processor, framebuffer, and initialized system-memory
  // instances. The buffer arguments become borrowed instance handles.
  func.func @realm_mlir_gpu_input(%gpu: !realm.processor, %memory: !realm.memory,
      %source: memref<513xi32>, %target: memref<513xi32>) attributes {realm.gpu_entry} {
    %one = arith.constant 1 : index
    %blocks = arith.constant 5 : index
    %threads = arith.constant 128 : index
    %increment = arith.constant 7 : i32
    %start = gpu.wait async
    %buffer, %ready = gpu.alloc async [%start] () : memref<513xi32, 1>
    %uploaded = gpu.memcpy async [%ready] %buffer, %source
        : memref<513xi32, 1>, memref<513xi32>
    %a = gpu.launch_func async [%uploaded] @kernels::@add_to_buffer
        blocks in (%blocks, %one, %one) threads in (%threads, %one, %one)
        args(%buffer : memref<513xi32, 1>, %increment : i32)
        {realm.task_entry = @launch_add, realm.task_id = 69 : i32}
    %b = gpu.launch_func async [%a] @kernels::@add_to_buffer
        blocks in (%blocks, %one, %one) threads in (%threads, %one, %one)
        args(%buffer : memref<513xi32, 1>, %increment : i32)
        {realm.task_entry = @launch_add, realm.task_id = 69 : i32}
    %downloaded = gpu.memcpy async [%b] %target, %buffer
        : memref<513xi32>, memref<513xi32, 1>
    %freed = gpu.dealloc async [%downloaded] %buffer : memref<513xi32, 1>
    gpu.wait [%downloaded]
    return
  }
}
