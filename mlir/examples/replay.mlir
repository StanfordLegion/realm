// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func private @replay_graph(%cpu0: !realm.processor, %cpu1: !realm.processor,
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

// Rebind the same graph to a second set of instances and carry the first
// completion into the second invocation as its precondition.
func.func @replay_twice(%cpu0: !realm.processor, %cpu1: !realm.processor,
                        %seed: !realm.task, %sum: !realm.task,
                        %a0: !realm.instance, %b0: !realm.instance, %out0: !realm.instance,
                        %a1: !realm.instance, %b1: !realm.instance, %out1: !realm.instance,
                        %entry: !realm.event) -> (!realm.event, !realm.event) {
  %frontier, %settled0 = func.call @replay_graph(%cpu0, %cpu1, %seed, %sum, %a0, %b0, %out0, %entry)
      : (!realm.processor, !realm.processor, !realm.task, !realm.task,
         !realm.instance, !realm.instance, !realm.instance, !realm.event)
      -> (!realm.event, !realm.event)
  %done, %settled1 = func.call @replay_graph(%cpu0, %cpu1, %seed, %sum, %a1, %b1, %out1, %frontier)
      : (!realm.processor, !realm.processor, !realm.task, !realm.task,
         !realm.instance, !realm.instance, !realm.instance, !realm.event)
      -> (!realm.event, !realm.event)
  %settled = realm.event.join_settled [%settled0, %settled1]
  return %done, %settled : !realm.event, !realm.event
}
