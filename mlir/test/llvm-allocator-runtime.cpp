// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "realm.h"
#include "realm/realm_c.h"

#include <cstdio>
#include <cstdlib>

extern "C" void replace_buffer(realm_memory_t memory, int64_t extent, realm_event_t gate,
                               realm_region_instance_t *out, realm_event_t *out_ready);

extern "C" void realm_mlir_save_handles(realm_region_instance_t instance,
                                        realm_event_t ready, realm_region_instance_t *out,
                                        realm_event_t *out_ready)
{
  *out = instance;
  *out_ready = ready;
}

int main(int argc, char **argv)
{
  using namespace Realm;
  Runtime runtime;
  if(!runtime.init(&argc, &argv))
    return 1;
  constexpr size_t mib = 1024 * 1024;
  Memory memory = Machine::MemoryQuery(Machine::get_machine())
                      .local_address_space()
                      .only_kind(Memory::SYSTEM_MEM)
                      .first();
  if(!memory.exists() || memory.capacity() != 4 * mib) {
    std::fputs("Expected a 4 MiB system memory\n", stderr);
    std::abort();
  }
  UserEvent gate = UserEvent::create_user_event();
  RegionInstance next;
  Event ready;
  replace_buffer(memory.id, 3 * mib, gate.id, &next.id, &ready.id);
  bool poisoned = false;
  if(!next.exists() || ready.has_triggered_faultaware(poisoned)) {
    std::fputs("Replacement allocation should be pending\n", stderr);
    std::abort();
  }
  gate.trigger();
  ready.wait_faultaware(poisoned);
  if(poisoned) {
    std::fputs("Replacement allocation failed after the pending free\n", stderr);
    std::abort();
  }
  next.destroy();
  runtime.shutdown();
  return runtime.wait_for_shutdown();
}
