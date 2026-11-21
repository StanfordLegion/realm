// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Independent tasks each produce one value. A third task consumes both values.
func.func @fork_join(%cpu0: !realm.processor, %cpu1: !realm.processor,
                     %seed: !realm.task, %sum: !realm.task,
                     %left: !realm.instance, %right: !realm.instance,
                     %result: !realm.instance, %entry: !realm.event)
    -> (!realm.event, !realm.event) {
  %a = realm.spawn %seed(%left) on %cpu0 after [%entry]
  %b = realm.spawn %seed(%right) on %cpu1 after [%entry]
  %done = realm.spawn %sum(%left, %right, %result) on %cpu0 after [%a, %b]
  %settled = realm.event.join_settled [%a, %b, %done]
  return %done, %settled : !realm.event, !realm.event
}
