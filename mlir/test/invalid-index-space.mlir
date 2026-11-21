// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// expected-error @+1 {{expected a positive index space rank}}
func.func private @zero_rank(!realm.index_space<0, si64>)

// -----

// expected-error @+1 {{expected a signed or unsigned coordinate type}}
func.func private @signless_coordinate(!realm.index_space<1, i64>)

// -----

func.func @unsupported_coordinate(%src: !realm.instance, %dst: !realm.instance,
                                   %space: !realm.index_space<1, ui8>) -> !realm.event {
  // expected-error @+1 {{requires si32, ui32, or si64 coordinates for Realm lowering}}
  %event = realm.copy %src field 0 to %dst field 0 over %space bytes 4
      : !realm.index_space<1, ui8>
  return %event : !realm.event
}
