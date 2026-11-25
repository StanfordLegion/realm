// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

module attributes {realm.task_initializer = "init"} {
  func.func @observes_free(%gpu: !realm.processor, %memory: !realm.memory)
      attributes {realm.gpu_entry} {
    %buffer, %ready = gpu.alloc async () : memref<8xi32, 1>
    // expected-error @+1 {{observes deallocation completion, which Realm destroy does not return}}
    %freed = gpu.dealloc async [%ready] %buffer : memref<8xi32, 1>
    gpu.wait [%freed]
    return
  }
}

// -----

module attributes {realm.task_initializer = "init"} {
  func.func @frees_borrowed(%gpu: !realm.processor, %memory: !realm.memory,
                           %buffer: memref<8xi32, 1>) attributes {realm.gpu_entry} {
    // expected-error @+1 {{cannot destroy a borrowed buffer}}
    %freed = gpu.dealloc async %buffer : memref<8xi32, 1>
    return
  }
}

// -----

// expected-error @+1 {{requires a module realm.task_initializer}}
func.func @missing_initializer(%gpu: !realm.processor, %memory: !realm.memory)
    attributes {realm.gpu_entry} {
  return
}

// -----

module attributes {gpu.container_module, realm.task_initializer = "init"} {
  gpu.module @kernels {
    gpu.func @first() kernel { gpu.return }
    gpu.func @second() kernel { gpu.return }
  }
  func.func private @launch(%gx: i64, %gy: i64, %gz: i64, %bx: i64, %by: i64, %bz: i64)
      attributes {realm.gpu_kernel = @kernels::@first} {
    return
  }
  func.func @mismatched_kernel(%gpu: !realm.processor, %memory: !realm.memory)
      attributes {realm.gpu_entry} {
    %one = arith.constant 1 : index
    // expected-error @+1 {{requires the host launcher's realm.gpu_kernel to name @kernels::@second}}
    %done = gpu.launch_func async @kernels::@second
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        {realm.task_entry = @launch, realm.task_id = 70 : i32}
    return
  }
}
