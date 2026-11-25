// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REALM_MLIR_RUNTIME_H
#define REALM_MLIR_RUNTIME_H

#include "realm/event.h"
#include "realm/indexspace.h"
#include "realm/inst_layout.h"
#include "realm/instance.h"
#include "realm/memory.h"
#include "realm/processor.h"
#include "realm/profiling.h"
#include "realm/reservation.h"

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace Realm::MLIR {
  inline Event readyEvent() { return Event::NO_EVENT; }

  template <typename... Events>
  Event mergeEvents(Events... events)
  {
    std::array<Event, sizeof...(Events)> inputs{events...};
    return Event::merge_events(inputs.data(), inputs.size());
  }

  template <typename... Events>
  Event joinSettled(Events... events)
  {
    std::array<Event, sizeof...(Events)> inputs{events...};
    return Event::merge_events_ignorefaults(inputs.data(), inputs.size());
  }

  inline bool awaitEvent(Event event)
  {
    bool poisoned = false;
    event.wait_faultaware(poisoned);
    return poisoned;
  }

  inline std::tuple<RegionInstance, Event>
  createInstance(Memory memory, const InstanceLayoutGeneric *layout, Event wait_on)
  {
    RegionInstance instance;
    Event ready = RegionInstance::create_instance(instance, memory, *layout,
                                                  ProfilingRequestSet(), wait_on);
    return {instance, ready};
  }

  inline bool instanceExists(RegionInstance instance) { return instance.exists(); }

  inline void destroyInstance(RegionInstance instance, Event wait_on)
  {
    instance.destroy(wait_on);
  }

  inline Event copy(const IndexSpaceGeneric *space, RegionInstance source,
                    RegionInstance target, FieldID source_field, FieldID target_field,
                    size_t bytes, Event wait_on)
  {
    std::vector<CopySrcDstField> sources(1), targets(1);
    sources.front().set_field(source, source_field, bytes);
    targets.front().set_field(target, target_field, bytes);
    return space->copy(sources, targets, ProfilingRequestSet(), wait_on);
  }

  inline Reservation createReservation() { return Reservation::create_reservation(); }

  inline Event acquireReservation(Reservation reservation, Event wait_on)
  {
    return reservation.acquire(0, true, wait_on);
  }

  inline void releaseReservation(Reservation reservation, Event wait_on)
  {
    reservation.release(wait_on);
  }

  inline void destroyReservation(Reservation reservation, Event wait_on)
  {
    reservation.destroy_reservation(wait_on);
  }

  inline Barrier advanceBarrier(Barrier phase) { return phase.advance_barrier(); }

  inline Event barrierEvent(Barrier phase) { return phase; }

  inline void arriveBarrier(Barrier phase, int32_t value, Event wait_on)
  {
    phase.arrive(1, wait_on, &value, sizeof(value));
  }

  inline std::tuple<int32_t, bool> barrierResult(Barrier phase)
  {
    int32_t value = 0;
    bool available = phase.get_result(&value, sizeof(value));
    return {value, available};
  }

  template <typename... Instances>
  Event spawn(Processor processor, Processor::TaskFuncID task, Event wait_on,
              Instances... instances)
  {
    std::array<RegionInstance::id_t, sizeof...(Instances)> args{instances.id...};
    return processor.spawn(task, args.data(), args.size() * sizeof(RegionInstance::id_t),
                           wait_on);
  }
} // namespace Realm::MLIR

#endif // REALM_MLIR_RUNTIME_H
