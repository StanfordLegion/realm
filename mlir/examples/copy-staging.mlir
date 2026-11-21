// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// Copy through a temporary, then compare with the optimized direct transfer.
// CHECK-LABEL: func.func @staged_copy(
// CHECK: %[[STAGED:[a-zA-Z0-9]+]] = realm.copy
// CHECK-NEXT: {{.*}} = realm.copy {{.*}} after [%[[STAGED]]]
func.func @staged_copy(%src: !realm.instance, %tmp: !realm.instance,
                       %dst: !realm.instance, %space: !realm.index_space<1, si64>,
                       %ready: !realm.event) -> !realm.event {
  %staged = realm.copy %src field 0 to %tmp field 0 over %space bytes 4
      after [%ready] : !realm.index_space<1, si64>
  %copied = realm.copy %tmp field 0 to %dst field 0 over %space bytes 4
      after [%staged] : !realm.index_space<1, si64>
  return %copied : !realm.event
}

// The caller owns three disjoint instances, grants exclusive use of %tmp,
// guarantees a legal direct copy into a ready destination, and keeps %src
// unchanged until the result event triggers.
// CHECK-LABEL: func.func @elide_staging(
// CHECK-SAME: %[[SOURCE:[a-zA-Z0-9]+]]: !realm.instance, %{{[a-zA-Z0-9]+}}: !realm.instance, %[[TARGET:[a-zA-Z0-9]+]]:
// CHECK-NEXT: %[[DIRECT:[a-zA-Z0-9]+]] = realm.copy %[[SOURCE]] field 0 to %[[TARGET]] field 0
// CHECK-NEXT: return %[[DIRECT]] : !realm.event
func.func @elide_staging(%src: !realm.instance, %tmp: !realm.instance,
                         %dst: !realm.instance, %space: !realm.index_space<1, si64>,
                         %ready: !realm.event) -> !realm.event
    attributes {realm.assume_success, realm.private_staging} {
  %staged = realm.copy %src field 0 to %tmp field 0 over %space bytes 4
      after [%ready] : !realm.index_space<1, si64>
  %copied = realm.copy %tmp field 0 to %dst field 0 over %space bytes 4
      after [%staged] : !realm.index_space<1, si64>
  return %copied : !realm.event
}
