// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// expected-error @+1 {{LLVM lowering does not support '!realm.layout'}}
func.func @import_layout(%ptr: !realm.layout) { return }

// -----

// expected-error @+1 {{LLVM lowering does not support '!realm.index_space<1, si64>'}}
func.func @import_space(%ptr: !realm.index_space<1, si64>) { return }

// -----

func.func @reservation() {
  // expected-error @+1 {{LLVM lowering does not support '!realm.reservation'}}
  %reservation = realm.reservation.create
  return
}

// -----

// expected-error @+1 {{LLVM lowering does not support '!realm.barrier_phase'}}
func.func private @import_barrier() -> !realm.barrier_phase

// -----

// expected-error @+1 {{device IR must be extracted or removed before Realm LLVM lowering of the host}}
gpu.module @kernels {
  gpu.func @kernel() kernel { gpu.return }
}

// -----

module attributes {realm.task_initializer = "init"} {
  func.func private @first() { return }
  func.func private @second() { return }
  func.func @entry1() {
    %binding, %ready = realm.task.register @first id 70
        kind #realm.processor_kind<toc_proc> as !realm.task_binding<() -> ()>
    return
  }
  func.func @entry2() {
    // expected-error @+1 {{conflicting task definitions for one ID and processor kind}}
    %binding, %ready = realm.task.register @second id 70
        kind #realm.processor_kind<toc_proc> as !realm.task_binding<() -> ()>
    return
  }
}
