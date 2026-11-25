// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

module attributes {realm.task_initializer = "init"} {
  func.func @produces_value(%cpu: !realm.processor)
      attributes {realm.async_task_base = 100 : i32} {
    // expected-error @+1 {{requires token-only dependencies and no async values}}
    %done, %value = async.execute -> !async.value<i32> {
      %one = arith.constant 1 : i32
      async.yield %one : i32
    }
    return
  }
}

// -----

func.func @missing_initializer(%cpu: !realm.processor)
    attributes {realm.async_task_base = 100 : i32} {
  // expected-error @+1 {{requires a module realm.task_initializer}}
  %done = async.execute { async.yield }
  return
}
