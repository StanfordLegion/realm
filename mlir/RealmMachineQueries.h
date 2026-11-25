// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REALM_MLIR_MACHINE_QUERIES_H
#define REALM_MLIR_MACHINE_QUERIES_H

#include <cstdint>
#include <optional>
#include <tuple>

#include "realm/machine.h"

namespace Realm::MLIR {

  // Apply the filters common to processor and memory discovery. An absent
  // reference omits a filter. A present sentinel cannot match any resource.
  template <typename Query, typename Reference>
  bool applyMachineFilters(Query &query, bool local,
                           std::optional<Reference> same_address_space,
                           std::optional<Reference> affinity)
  {
    if((same_address_space && !same_address_space->exists()) ||
       (affinity && !affinity->exists()))
      return false;
    if(local)
      query.local_address_space();
    if(same_address_space)
      query.same_address_space_as(*same_address_space);
    if(affinity)
      query.has_affinity_to(*affinity);
    return true;
  }

  inline std::tuple<Processor, bool>
  findProcessor(Processor::Kind kind, bool local,
                std::optional<Memory> same_address_space, std::optional<Memory> affinity)
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    if(!applyMachineFilters(query, local, same_address_space, affinity))
      return {Processor::NO_PROC, false};
    Processor processor = query.only_kind(kind).first();
    return {processor, processor.exists()};
  }

  inline std::tuple<Memory, bool> findMemory(Memory::Kind kind, bool local,
                                             std::optional<Processor> same_address_space,
                                             std::optional<Processor> affinity,
                                             uint64_t min_capacity)
  {
    Machine::MemoryQuery query(Machine::get_machine());
    if(!applyMachineFilters(query, local, same_address_space, affinity))
      return {Memory::NO_MEMORY, false};
    query.only_kind(kind);

    // Older Realm releases can return an undersized memory from first()
    // even when a capacity filter was supplied.
    for(Memory memory : query) {
      if(min_capacity == 0 || memory.capacity() >= min_capacity)
        return {memory, true};
    }
    return {Memory::NO_MEMORY, false};
  }

} // namespace Realm::MLIR

#endif // REALM_MLIR_MACHINE_QUERIES_H
