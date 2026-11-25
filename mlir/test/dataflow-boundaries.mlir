// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

// CHECK-LABEL: func.func @owned_staging(
// CHECK: %[[TMP:[a-zA-Z0-9]+]], %[[ALLOCATED:[a-zA-Z0-9]+]] = realm.instance.create
// CHECK-NEXT: {{.*}} = realm.instance.exists %[[TMP]]
// CHECK-NEXT: %[[DIRECT:[a-zA-Z0-9]+]] = realm.copy {{.*}} field 3 to {{.*}} field 5 {{.*}} after [%[[ALLOCATED]]]
// CHECK-NEXT: %[[SETTLED:[a-zA-Z0-9]+]] = realm.event.join_settled [%[[ALLOCATED]], %[[DIRECT]]]
// CHECK-NEXT: realm.instance.destroy %[[TMP]] after [%[[SETTLED]]]
func.func @owned_staging(%src: !realm.instance, %dst: !realm.instance,
                         %space: !realm.index_space<1, si64>, %memory: !realm.memory,
                         %layout: !realm.layout) -> !realm.event
    attributes {realm.assume_success, realm.private_staging} {
  %tmp, %allocated = realm.instance.create in %memory layout %layout
  %exists = realm.instance.exists %tmp
  %staged = realm.copy %src field 3 to %tmp field 4 over %space bytes 4
      after [%allocated] : !realm.index_space<1, si64>
  %done = realm.copy %tmp field 4 to %dst field 5 over %space bytes 4
      after [%staged] : !realm.index_space<1, si64>
  %settled = realm.event.join_settled [%allocated, %staged, %done]
  realm.instance.destroy %tmp after [%settled]
  return %done : !realm.event
}

// CHECK-LABEL: func.func @observable_first_copy(
// CHECK: %[[FIRST:[a-zA-Z0-9]+]] = realm.copy
// CHECK-NEXT: %[[SECOND:[a-zA-Z0-9]+]] = realm.copy {{.*}} after [%[[FIRST]]]
// CHECK-NEXT: return %[[FIRST]], %[[SECOND]]
func.func @observable_first_copy(%src: !realm.instance, %tmp: !realm.instance,
                                  %dst: !realm.instance, %space: !realm.index_space<1, si64>)
    -> (!realm.event, !realm.event) attributes {realm.assume_success, realm.private_staging} {
  %first = realm.copy %src field 0 to %tmp field 0 over %space bytes 4
      : !realm.index_space<1, si64>
  %second = realm.copy %tmp field 0 to %dst field 0 over %space bytes 4
      after [%first] : !realm.index_space<1, si64>
  return %first, %second : !realm.event, !realm.event
}

// CHECK-LABEL: func.func @cleanup_does_not_cover_second(
// CHECK: %[[STAGED:[a-zA-Z0-9]+]] = realm.copy
// CHECK-NEXT: %[[DONE:[a-zA-Z0-9]+]] = realm.copy {{.*}} after [%[[STAGED]]]
// CHECK-NEXT: {{.*}} = realm.event.join_settled [{{.*}}, %[[STAGED]]]
func.func @cleanup_does_not_cover_second(%src: !realm.instance, %dst: !realm.instance,
                                         %space: !realm.index_space<1, si64>, %memory: !realm.memory,
                                         %layout: !realm.layout) -> (!realm.event, !realm.event)
    attributes {realm.assume_success, realm.private_staging} {
  %tmp, %allocated = realm.instance.create in %memory layout %layout
  %staged = realm.copy %src field 0 to %tmp field 0 over %space bytes 4
      after [%allocated] : !realm.index_space<1, si64>
  %done = realm.copy %tmp field 0 to %dst field 0 over %space bytes 4
      after [%staged] : !realm.index_space<1, si64>
  %staging_only = realm.event.join_settled [%allocated, %staged]
  return %done, %staging_only : !realm.event, !realm.event
}

// CHECK-LABEL: func.func @settled_is_opaque(
// CHECK: %[[FIRST:[a-zA-Z0-9]+]] = realm.spawn
// CHECK: %[[LATER:[a-zA-Z0-9]+]] = realm.spawn {{.*}} after [%{{[a-zA-Z0-9]+}}]
// CHECK-NEXT: {{.*}} = realm.spawn {{.*}} after [%[[FIRST]], %[[LATER]]]
func.func @settled_is_opaque(%cpu: !realm.processor, %task: !realm.task)
    -> !realm.event attributes {realm.assume_success} {
  %first = realm.spawn %task() on %cpu
  %settled = realm.event.join_settled [%first]
  %later = realm.spawn %task() on %cpu after [%settled]
  %done = realm.spawn %task() on %cpu after [%first, %later]
  return %done : !realm.event
}

// CHECK-LABEL: func.func @nested_spawn(
// CHECK: %[[FIRST:[a-zA-Z0-9]+]] = realm.spawn
// CHECK: %[[LATER:[a-zA-Z0-9]+]] = realm.spawn {{.*}} after [%[[FIRST]]]
// CHECK-NEXT: {{.*}} = realm.spawn {{.*}} after [%[[LATER]]]
func.func @nested_spawn(%cpu: !realm.processor, %task: !realm.task, %condition: i1)
    attributes {realm.assume_success} {
  %first = realm.spawn %task() on %cpu
  scf.if %condition {
    %later = realm.spawn %task() on %cpu after [%first]
    %done = realm.spawn %task() on %cpu after [%first, %later]
  }
  return
}
