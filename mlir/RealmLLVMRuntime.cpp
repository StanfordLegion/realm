// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "RealmMachineQueries.h"

#include "realm/instance.h"
#include "realm/realm_c.h"
#include "realm/runtime.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>

static_assert(sizeof(size_t) == 8, "Realm MLIR's LLVM ABI requires a 64-bit target");
static_assert(sizeof(int) == 4 && sizeof(realm_id_t) == 8);

extern "C" {
void realm_mlir_check_status(realm_status_t status)
{
  if(status != REALM_SUCCESS) {
    std::fprintf(stderr, "Realm MLIR runtime call failed with status %d\n", status);
    std::abort();
  }
}

void realm_mlir_check_task_payload(const void *data, size_t actual, size_t expected)
{
  if(actual != expected || (expected != 0 && !data)) {
    std::fprintf(stderr, "Invalid Realm MLIR task payload (%zu bytes, expected %zu)\n",
                 actual, expected);
    std::abort();
  }
}

realm_runtime_t realm_mlir_runtime()
{
  realm_runtime_t runtime = nullptr;
  realm_mlir_check_status(realm_runtime_get_runtime(&runtime));
  return runtime;
}

// Every address space calls this with the readiness of its local registrations.
// The collective NOP gives the host launcher a common startup rendezvous.
int realm_mlir_collective_start(realm_event_t registered)
{
  Realm::Processor root = Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
                              .only_kind(Realm::Processor::LOC_PROC)
                              .first();
  if(!root.exists())
    realm_mlir_check_status(REALM_PROCESSOR_ERROR_INVALID_PROCESSOR);
  auto runtime = realm_mlir_runtime();
  realm_event_t ready;
  realm_mlir_check_status(realm_runtime_collective_spawn(
      runtime, root.id, REALM_TASK_ID_PROCESSOR_NOP, nullptr, 0, registered, 0, &ready));
  int poisoned = 0;
  realm_mlir_check_status(
      realm_event_wait(runtime, ready, REALM_WAIT_INFINITE, &poisoned));
  if(poisoned)
    realm_mlir_check_status(REALM_ERROR);
  realm_runtime_attr_t attribute = REALM_RUNTIME_ATTR_LOCAL_ADDRESS_SPACE;
  uint64_t local;
  realm_mlir_check_status(realm_runtime_get_attributes(runtime, &attribute, &local, 1));
  return local == root.address_space();
}

int realm_mlir_wait_for_shutdown()
{
  return Realm::Runtime::get_runtime().wait_for_shutdown();
}

realm_processor_t realm_mlir_find_processor(int kind, int local, int has_same,
                                            realm_memory_t same, int has_affinity,
                                            realm_memory_t affinity)
{
  auto [processor, found] = Realm::MLIR::findProcessor(
      static_cast<Realm::Processor::Kind>(kind), local,
      has_same ? std::optional(Realm::Memory(same)) : std::nullopt,
      has_affinity ? std::optional(Realm::Memory(affinity)) : std::nullopt);
  return processor.id;
}

realm_memory_t realm_mlir_find_memory(int kind, int local, int has_same,
                                      realm_processor_t same, int has_affinity,
                                      realm_processor_t affinity, uint64_t capacity)
{
  auto [memory, found] = Realm::MLIR::findMemory(
      static_cast<Realm::Memory::Kind>(kind), local,
      has_same ? std::optional(Realm::Processor(same)) : std::nullopt,
      has_affinity ? std::optional(Realm::Processor(affinity)) : std::nullopt, capacity);
  return memory.id;
}

realm_status_t realm_mlir_create_dense(realm_runtime_t runtime, realm_memory_t memory,
                                       long long extent, size_t field_bytes,
                                       realm_event_t wait, realm_region_instance_t *inst,
                                       realm_event_t *ready)
{
  if(extent < 0 || field_bytes == 0 ||
     static_cast<unsigned long long>(extent) >
         std::numeric_limits<size_t>::max() / field_bytes)
    return REALM_REGION_INSTANCE_ERROR_INVALID_PARAMS;
  long long lower = 0, upper = extent - 1;
  realm_field_id_t field = 0;
  realm_region_instance_create_params_t params{};
  params.memory = memory;
  params.lower_bound = &lower;
  params.upper_bound = &upper;
  params.num_dims = 1;
  params.coord_type = REALM_COORD_TYPE_LONG_LONG;
  params.field_ids = &field;
  params.field_sizes = &field_bytes;
  params.num_fields = 1;
  params.block_size = 1;
  return realm_region_instance_create(runtime, &params, nullptr, wait, inst, ready);
}

realm_status_t realm_mlir_copy_dense(realm_runtime_t runtime, realm_region_instance_t src,
                                     realm_field_id_t src_field,
                                     realm_region_instance_t dst,
                                     realm_field_id_t dst_field, long long extent,
                                     size_t bytes, realm_event_t wait,
                                     realm_event_t *done)
{
  if(extent < 0 || bytes == 0 ||
     static_cast<unsigned long long>(extent) > std::numeric_limits<size_t>::max() / bytes)
    return REALM_REGION_INSTANCE_ERROR_INVALID_PARAMS;
  long long lower = 0, upper = extent - 1;
  realm_copy_src_dst_field_t source{src, src_field, bytes}, target{dst, dst_field, bytes};
  realm_region_instance_copy_params_t params{};
  params.srcs = &source;
  params.dsts = &target;
  params.lower_bound = &lower;
  params.upper_bound = &upper;
  params.num_dims = 1;
  params.coord_type = REALM_COORD_TYPE_LONG_LONG;
  params.num_fields = 1;
  return realm_region_instance_copy(runtime, &params, nullptr, wait, 0, done);
}

// Valid only for a ready, locally addressable instance with matching layout.
void *realm_mlir_instance_address(realm_region_instance_t instance, size_t byte_offset,
                                  size_t byte_count)
{
  Realm::RegionInstance handle(instance);
  const size_t extent = handle.get_layout()->bytes_used;
  if(byte_offset > extent || byte_count > extent - byte_offset) {
    std::fputs("Realm MLIR instance address range is out of bounds\n", stderr);
    std::abort();
  }
  void *ptr = handle.pointer_untyped(byte_offset, byte_count);
  if(!ptr) {
    std::fputs("Realm MLIR requires a directly addressable instance\n", stderr);
    std::abort();
  }
  return ptr;
}
}
