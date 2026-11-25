// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

func.func @invalid_copy(%src: !realm.instance, %dst: !realm.instance,
                        %space: !realm.index_space<1, si64>) -> !realm.event {
  // expected-error @+1 {{requires a positive field size in bytes}}
  %done = realm.copy %src field 0 to %dst field 0 over %space bytes 0
      : !realm.index_space<1, si64>
  return %done : !realm.event
}

// -----

func.func @reserved_source(%src: !realm.instance, %dst: !realm.instance,
                            %space: !realm.index_space<1, si64>) -> !realm.event {
  // expected-error @+1 {{reserves field ID -1 for fills}}
  %done = realm.copy %src field -1 to %dst field 0 over %space bytes 16
      : !realm.index_space<1, si64>
  return %done : !realm.event
}

// -----

func.func @reserved_target(%src: !realm.instance, %dst: !realm.instance,
                            %space: !realm.index_space<1, si64>) -> !realm.event {
  // expected-error @+1 {{reserves field ID -1 for fills}}
  %done = realm.copy %src field 0 to %dst field -1 over %space bytes 16
      : !realm.index_space<1, si64>
  return %done : !realm.event
}
