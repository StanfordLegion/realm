// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func @dense(%count: i64) {
  // expected-error @+1 {{EmitC lowering does not support '!realm.dense_index_space'}}
  %space = realm.index_space.dense %count
  return
}

// -----

func.func private @task() { return }
func.func @register() {
  // expected-error @+1 {{EmitC lowering does not support '!realm.task_binding<() -> ()>'}}
  %binding, %ready = realm.task.register @task id 70
      kind #realm.processor_kind<loc_proc> as !realm.task_binding<() -> ()>
  return
}
