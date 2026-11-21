// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func @deferred_copy(%memory: !realm.memory, %layout: !realm.layout,
                        %src: !realm.instance, %dst: !realm.instance,
                        %space: !realm.index_space<1, si64>,
                        %entry: !realm.event) -> !realm.event {
  %tmp, %allocated = realm.instance.create in %memory layout %layout after [%entry]
  %has_handle = realm.instance.exists %tmp
  %done = scf.if %has_handle -> (!realm.event) {
    %staged = realm.copy %src field 0 to %tmp field 0 over %space bytes 4
        after [%allocated] : !realm.index_space<1, si64>
    %copied = realm.copy %tmp field 0 to %dst field 0 over %space bytes 4
        after [%staged] : !realm.index_space<1, si64>
    %settled = realm.event.join_settled [%allocated, %staged, %copied]
    realm.instance.destroy %tmp after [%settled]
    scf.yield %copied : !realm.event
  } else {
    scf.yield %allocated : !realm.event
  }
  return %done : !realm.event
}

func.func @wait(%event: !realm.event) -> i1 {
  %ready = realm.event.ready
  %merged = realm.event.merge [%event, %ready]
  %poisoned = realm.event.await %merged
  return %poisoned : i1
}
