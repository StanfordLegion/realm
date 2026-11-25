// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// REALM-LABEL: func.func private @run_async(
// REALM-SAME: %[[CPU:[a-zA-Z0-9]+]]: !realm.processor, %[[INSTANCE:[a-zA-Z0-9]+]]: !realm.instance)
// REALM: %[[TASK:[a-zA-Z0-9]+]], %[[REG:[a-zA-Z0-9]+]] = realm.task.register @__realm_async_run_async_128 id 128
// REALM: %[[FIRST:[a-zA-Z0-9]+]] = realm.spawn_typed %[[TASK]](%[[INSTANCE]], {{.*}}) on %[[CPU]] after [%[[REG]]]
// REALM: %[[NEXT_TASK:[a-zA-Z0-9_]+]], %[[NEXT_REG:[a-zA-Z0-9_]+]] = realm.task.register @__realm_async_run_async_129 id 129
// REALM: %[[NEXT:[a-zA-Z0-9]+]] = realm.spawn_typed %[[NEXT_TASK]](%[[INSTANCE]], {{.*}}) on %[[CPU]] after [%[[FIRST]], %[[NEXT_REG]]]
// REALM: realm.event.await %[[NEXT]]
// REALM: cf.assert
//
// LLVM: llvm.func @realm_mlir_instance_address(i64, i64, i64) -> !llvm.ptr
// LLVM-LABEL: llvm.func internal @write_value(
// LLVM-LABEL: llvm.func internal @run_async(
// LLVM-NOT: llvm.call @realm_processor_register_task_by_kind(
// LLVM-NOT: llvm.call @realm_event_merge(
// LLVM: llvm.call @realm_processor_spawn(
// LLVM-NOT: llvm.call @realm_event_merge(
// LLVM: llvm.call @realm_processor_spawn(
// LLVM: llvm.call @realm_event_wait(
// LLVM-LABEL: llvm.func internal @run(
// LLVM-LABEL: llvm.func @realm_mlir_init_cpu_tasks() -> i64 {
// LLVM-COUNT-4: llvm.call @realm_processor_register_task_by_kind(
// LLVM-NOT: llvm.call @realm_processor_register_task_by_kind(
// LLVM-LABEL: llvm.func @main(
// LLVM-LABEL: llvm.func internal @__realm_async_run_async_128(
// LLVM-LABEL: llvm.func internal @__realm_async_run_async_129(
// LLVM-LABEL: llvm.func internal @__realm_task___realm_async_run_async_128(
// LLVM: %[[EXPECTED:[a-zA-Z0-9]+]] = llvm.mlir.constant(16 : i64)
// LLVM-NOT: llvm.load
// LLVM: llvm.call @realm_mlir_check_task_payload(%arg0, %arg1, %[[EXPECTED]])
// LLVM: llvm.load {{.*}} {alignment = 1 : i64}

module attributes {realm.task_initializer = "realm_mlir_init_cpu_tasks"} {
  func.func private @realm_mlir_instance_address(!realm.instance, i64, i64) -> !llvm.ptr

  func.func private @write_value(%buffer: !realm.instance, %value: i32) {
    %offset = arith.constant 0 : i64
    %bytes = arith.constant 4 : i64
    %ptr = func.call @realm_mlir_instance_address(%buffer, %offset, %bytes)
        : (!realm.instance, i64, i64) -> !llvm.ptr
    llvm.store %value, %ptr : i32, !llvm.ptr
    return
  }

  func.func private @add(%buffer: !realm.instance, %increment: f32) {
    %offset = arith.constant 0 : i64
    %bytes = arith.constant 4 : i64
    %ptr = func.call @realm_mlir_instance_address(%buffer, %offset, %bytes)
        : (!realm.instance, i64, i64) -> !llvm.ptr
    %value = llvm.load %ptr : !llvm.ptr -> i32
    %amount = arith.fptosi %increment : f32 to i32
    %sum = arith.addi %value, %amount : i32
    llvm.store %sum, %ptr : i32, !llvm.ptr
    return
  }

  // The token-only Async conversion outlines these as two typed Realm tasks.
  func.func private @run_async(%cpu: !realm.processor, %buffer: !realm.instance)
      attributes {realm.async_task_base = 128 : i32} {
    %one = arith.constant 1.0 : f32
    %first = async.execute {
      func.call @add(%buffer, %one) : (!realm.instance, f32) -> ()
      async.yield
    }
    %second = async.execute [%first] {
      func.call @add(%buffer, %one) : (!realm.instance, f32) -> ()
      async.yield
    }
    async.await %second : !async.token
    return
  }

  func.func private @run(%cpu: !realm.processor, %memory: !realm.memory, %n: i64) -> i32 {
    %one = arith.constant 1 : i32
    %zero = arith.constant 0 : i32
    %write, %write_registered = realm.task.register @write_value id 64
        kind #realm.processor_kind<loc_proc> as !realm.task_binding<(!realm.instance, i32) -> ()>
    %add, %add_registered = realm.task.register @add id 65
        kind #realm.processor_kind<loc_proc> as !realm.task_binding<(!realm.instance, f32) -> ()>
    %space = realm.index_space.dense %n
    %layout = realm.layout.dense %space field_bytes 4
    %first, %first_ready = realm.instance.create in %memory layout %layout : !realm.dense_layout
    %second, %second_ready = realm.instance.create in %memory layout %layout : !realm.dense_layout
    %has_first = realm.instance.exists %first
    %has_second = realm.instance.exists %second
    %both = arith.andi %has_first, %has_second : i1
    %status = scf.if %both -> i32 {
      %initial = arith.constant 20 : i32
      %increment = arith.constant 22.0 : f32
      %written = realm.spawn_typed %write(%first, %initial) on %cpu
          after [%first_ready, %write_registered] : !realm.task_binding<(!realm.instance, i32) -> ()>,
          !realm.instance, i32
      %copied = realm.copy %first field 0 to %second field 0 over %space bytes 4
          after [%written, %second_ready] : !realm.dense_index_space
      %done = realm.spawn_typed %add(%second, %increment) on %cpu
          after [%copied, %add_registered] : !realm.task_binding<(!realm.instance, f32) -> ()>,
          !realm.instance, f32

      %settled = realm.event.join_settled [%first_ready, %second_ready, %written, %copied, %done]
      realm.instance.destroy %first after [%settled]
      %poisoned = realm.event.await %done
      %result = scf.if %poisoned -> i32 {
        scf.yield %one : i32
      } else {
        func.call @run_async(%cpu, %second) : (!realm.processor, !realm.instance) -> ()
        %offset = arith.constant 0 : i64
        %bytes = arith.constant 4 : i64
        %ptr = func.call @realm_mlir_instance_address(%second, %offset, %bytes)
            : (!realm.instance, i64, i64) -> !llvm.ptr
        %actual = llvm.load %ptr : !llvm.ptr -> i32
        %expected = arith.constant 44 : i32
        %correct = arith.cmpi eq, %actual, %expected : i32
        %code = arith.select %correct, %zero, %one : i32
        scf.yield %code : i32
      }
      realm.instance.destroy %second after [%settled]
      %ignored = realm.event.await %settled
      scf.yield %result : i32
    } else {
      scf.if %has_first {
        %safe = realm.event.join_settled [%first_ready]
        realm.instance.destroy %first after [%safe]
        %ignored = realm.event.await %safe
      }
      scf.if %has_second {
        %safe = realm.event.join_settled [%second_ready]
        realm.instance.destroy %second after [%safe]
        %ignored = realm.event.await %safe
      }
      scf.yield %one : i32
    }
    return %status : i32
  }

  func.func private @realm_runtime_create(!llvm.ptr) -> i32
  func.func private @realm_runtime_init(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
  func.func private @realm_runtime_signal_shutdown(!llvm.ptr, i64, i32) -> i32
  func.func private @realm_mlir_init_cpu_tasks() -> !realm.event
  func.func private @realm_mlir_collective_start(!realm.event) -> i32
  func.func private @realm_mlir_wait_for_shutdown() -> i32
  func.func private @realm_mlir_check_status(i32)

  func.func @main(%argc: i32, %argv: !llvm.ptr) -> i32 {
    %one = arith.constant 1 : i64
    %zero = arith.constant 0 : i64
    %argc_slot = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %argv_slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    %runtime_slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %argc, %argc_slot : i32, !llvm.ptr
    llvm.store %argv, %argv_slot : !llvm.ptr, !llvm.ptr
    %created = func.call @realm_runtime_create(%runtime_slot) : (!llvm.ptr) -> i32
    func.call @realm_mlir_check_status(%created) : (i32) -> ()
    %runtime = llvm.load %runtime_slot : !llvm.ptr -> !llvm.ptr
    %started = func.call @realm_runtime_init(%runtime, %argc_slot, %argv_slot)
        : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
    func.call @realm_mlir_check_status(%started) : (i32) -> ()
    %registrations = func.call @realm_mlir_init_cpu_tasks() : () -> !realm.event
    %local = func.call @realm_mlir_collective_start(%registrations) : (!realm.event) -> i32
    %zero_status = arith.constant 0 : i32
    %root = arith.cmpi ne, %local, %zero_status : i32
    scf.if %root {
      %cpu, %has_cpu = realm.machine.find_processor kind #realm.processor_kind<loc_proc> local
      %memory, %has_memory = realm.machine.find_memory kind #realm.memory_kind<system_mem>
          same_address_space_as %cpu with_affinity_to %cpu
      %found = arith.andi %has_cpu, %has_memory : i1
      %status = scf.if %found -> i32 {
        %first = func.call @run(%cpu, %memory, %one) : (!realm.processor, !realm.memory, i64) -> i32
        %second = func.call @run(%cpu, %memory, %one) : (!realm.processor, !realm.memory, i64) -> i32
        %either_failed = arith.ori %first, %second : i32
        scf.yield %either_failed : i32
      } else {
        %missing = arith.constant 1 : i32
        scf.yield %missing : i32
      }
      %stopped = func.call @realm_runtime_signal_shutdown(%runtime, %zero, %status)
          : (!llvm.ptr, i64, i32) -> i32
      func.call @realm_mlir_check_status(%stopped) : (i32) -> ()
    }
    %result = func.call @realm_mlir_wait_for_shutdown() : () -> i32
    return %result : i32
  }
}
