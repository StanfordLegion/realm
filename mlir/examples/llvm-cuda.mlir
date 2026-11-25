// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

module attributes {realm.task_initializer = "realm_mlir_init_cuda_tasks"} {
  func.func private @realm_mlir_cuda_add(!realm.instance, i64, i32)

  func.func private @one_launch(%device: !realm.instance, %count: i64, %amount: i32) {
    func.call @realm_mlir_cuda_add(%device, %count, %amount) : (!realm.instance, i64, i32) -> ()
    return
  }

  func.func private @two_launches(%device: !realm.instance, %count: i64, %amount: i32) {
    func.call @realm_mlir_cuda_add(%device, %count, %amount) : (!realm.instance, i64, i32) -> ()
    func.call @realm_mlir_cuda_add(%device, %count, %amount) : (!realm.instance, i64, i32) -> ()
    return
  }

  // The host source and destination are already allocated. The GPU instance is
  // allocated and destroyed here, and copies cover both memory transfers.
  func.func private @gpu_roundtrip(
      %gpu: !realm.processor, %memory: !realm.memory, %source: !realm.instance,
      %target: !realm.instance, %count: i64,
      %single: !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
      %pair: !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
      %registered: !realm.event, %combined: i1) -> i1 {
    %space = realm.index_space.dense %count
    %layout = realm.layout.dense %space field_bytes 4
    %device, %ready = realm.instance.create in %memory layout %layout : !realm.dense_layout
    %exists = realm.instance.exists %device
    %failed = scf.if %exists -> i1 {
      %uploaded = realm.copy %source field 0 to %device field 0 over %space bytes 4
          after [%ready] : !realm.dense_index_space
      %step = arith.constant 7 : i32
      %computed = scf.if %combined -> !realm.event {
        %done = realm.spawn_typed %pair(%device, %count, %step) on %gpu
            after [%registered, %uploaded] : !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
            !realm.instance, i64, i32
        scf.yield %done : !realm.event
      } else {
        %first = realm.spawn_typed %single(%device, %count, %step) on %gpu
            after [%registered, %uploaded] : !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
            !realm.instance, i64, i32
        %done = realm.spawn_typed %single(%device, %count, %step) on %gpu
            after [%first] : !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
            !realm.instance, i64, i32
        scf.yield %done : !realm.event
      }
      %downloaded = realm.copy %device field 0 to %target field 0 over %space bytes 4
          after [%computed] : !realm.dense_index_space
      %settled = realm.event.join_settled [%ready, %uploaded, %computed, %downloaded]
      realm.instance.destroy %device after [%settled]
      %ignored = realm.event.await %settled
      %poisoned = realm.event.await %downloaded
      scf.yield %poisoned : i1
    } else {
      %poisoned = realm.event.await %ready
      %missing = arith.constant true
      scf.yield %missing : i1
    }
    return %failed : i1
  }

  // Export a scalar C ABI entry point for the CUDA setup and result checker.
  func.func @realm_mlir_cuda_examples(
      %gpu: !realm.processor, %memory: !realm.memory, %source: !realm.instance,
      %separate: !realm.instance, %combined: !realm.instance, %count: i64) -> i32 {
    %single, %single_reg = realm.task.register @one_launch id 66
        kind #realm.processor_kind<toc_proc> as !realm.task_binding<(!realm.instance, i64, i32) -> ()>
    %pair, %pair_reg = realm.task.register @two_launches id 67
        kind #realm.processor_kind<toc_proc> as !realm.task_binding<(!realm.instance, i64, i32) -> ()>
    %reg = realm.event.merge [%single_reg, %pair_reg]
    %no = arith.constant false
    %yes = arith.constant true
    %a = func.call @gpu_roundtrip(%gpu, %memory, %source, %separate,
        %count, %single, %pair, %reg, %no)
        : (!realm.processor, !realm.memory, !realm.instance, !realm.instance, i64,
           !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
           !realm.task_binding<(!realm.instance, i64, i32) -> ()>, !realm.event, i1) -> i1
    %b = func.call @gpu_roundtrip(%gpu, %memory, %source, %combined,
        %count, %single, %pair, %reg, %yes)
        : (!realm.processor, !realm.memory, !realm.instance, !realm.instance, i64,
           !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
           !realm.task_binding<(!realm.instance, i64, i32) -> ()>, !realm.event, i1) -> i1
    %failed = arith.ori %a, %b : i1
    %status = arith.extui %failed : i1 to i32
    return %status : i32
  }

  // The caller supplies a borrowed CUDA allocation and its readiness event.
  func.func @realm_mlir_cuda_external(
      %gpu: !realm.processor, %buffer: !realm.instance, %count: i64,
      %ready: !realm.event) -> !realm.event {
    %task, %registered = realm.task.register @one_launch id 68
        kind #realm.processor_kind<toc_proc> as !realm.task_binding<(!realm.instance, i64, i32) -> ()>
    %step = arith.constant 7 : i32
    %done = realm.spawn_typed %task(%buffer, %count, %step) on %gpu
        after [%ready, %registered] : !realm.task_binding<(!realm.instance, i64, i32) -> ()>,
        !realm.instance, i64, i32
    return %done : !realm.event
  }
}
