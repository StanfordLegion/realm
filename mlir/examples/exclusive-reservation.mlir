// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Both tasks are submitted immediately. Their independent grant events
// serialize the updates, and each successful acquisition gets one release.
func.func @reservation_counter(%cpu0: !realm.processor, %cpu1: !realm.processor,
                                %increment: !realm.task, %buffer: !realm.instance)
    -> !realm.event {
  %lock = realm.reservation.create
  %grant0 = realm.reservation.acquire %lock
  %a = realm.spawn %increment(%buffer) on %cpu0 after [%grant0]
  %settled0 = realm.event.join_settled [%a]
  realm.reservation.release %lock after [%settled0]

  %grant1 = realm.reservation.acquire %lock
  %b = realm.spawn %increment(%buffer) on %cpu1 after [%grant1]
  %settled1 = realm.event.join_settled [%b]
  realm.reservation.release %lock after [%settled1]
  %settled = realm.event.join_settled [%a, %b]
  realm.reservation.destroy %lock after [%settled]
  return %settled : !realm.event
}
