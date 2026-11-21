// Copyright 2026 Stanford University, NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "realm.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

using Realm::Barrier;
using Realm::Event;
using Realm::IndexSpaceGeneric;
using Realm::InstanceLayoutGeneric;
using Realm::Memory;
using Realm::Processor;
using Realm::RegionInstance;
using Realm::UserEvent;
using TaskID = Processor::TaskFuncID;

std::tuple<Event, Event> fork_join(Processor cpu0, Processor cpu1, TaskID seed,
                                   TaskID sum, RegionInstance left, RegionInstance right,
                                   RegionInstance result, Event entry);
std::tuple<Event, Event> replay_twice(Processor cpu0, Processor cpu1, TaskID seed,
                                      TaskID sum, RegionInstance left0,
                                      RegionInstance right0, RegionInstance out0,
                                      RegionInstance left1, RegionInstance right1,
                                      RegionInstance out1, Event entry);
std::tuple<Event, Event> fault_cleanup(Processor cpu, TaskID seed, Memory memory,
                                       const InstanceLayoutGeneric *layout,
                                       Event cancel_gate, Event slow_gate);
Event staged_copy(RegionInstance source, RegionInstance staging, RegionInstance target,
                  const IndexSpaceGeneric *space, Event ready);
Event elide_staging(RegionInstance source, RegionInstance staging, RegionInstance target,
                    const IndexSpaceGeneric *space, Event ready);
Event reduce_event_edges(Processor cpu, TaskID seed, TaskID increment,
                         RegionInstance buffer);
Event reservation_counter(Processor cpu0, Processor cpu1, TaskID increment,
                          RegionInstance buffer);
Barrier arrive_two_phases(Processor cpu0, Processor cpu1, TaskID worker,
                          RegionInstance left, RegionInstance right, Barrier first,
                          Event gate, int32_t a, int32_t b, int32_t c, int32_t d);
std::tuple<int32_t, bool> read_phase(Barrier phase);
bool await_pipeline(Event finished);

namespace {
  struct Buffers {
    std::array<RegionInstance, 6> instances{};

    bool create(Memory memory, const InstanceLayoutGeneric &layout)
    {
      for(RegionInstance &instance : instances) {
        Event ready = RegionInstance::create_instance(instance, memory, layout,
                                                      Realm::ProfilingRequestSet());
        if(await_pipeline(ready) || !instance.exists())
          return false;
      }
      return true;
    }

    ~Buffers()
    {
      for(RegionInstance instance : instances)
        if(instance.exists())
          instance.destroy();
    }
  };

  bool report(bool passed, const char *description)
  {
    if(passed)
      std::cout << description << '\n';
    else
      std::cerr << "Check failed: " << description << '\n';
    return passed;
  }

  bool pending(Event event)
  {
    bool poisoned = false;
    return !event.has_triggered_faultaware(poisoned);
  }

  struct SumReduction {
    using LHS = int32_t;
    using RHS = int32_t;
    static constexpr RHS identity = 0;

    template <bool Exclusive>
    static void apply(LHS &lhs, RHS rhs)
    {
      if constexpr(!Exclusive) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> guard(mutex);
        lhs += rhs;
        return;
      }
      lhs += rhs;
    }

    template <bool Exclusive>
    static void fold(RHS &lhs, RHS rhs)
    {
      apply<Exclusive>(lhs, rhs);
    }
  };

  bool checkBarriers(Realm::Runtime &runtime, Processor cpu, Processor other,
                     TaskID worker, RegionInstance left, RegionInstance right)
  {
    constexpr Realm::ReductionOpID sum_id = 1;
    if(!runtime.register_reduction<SumReduction>(sum_id))
      return false;

    int32_t initial = 0;
    Barrier first = Barrier::create_barrier(2, sum_id, &initial, sizeof(initial));
    UserEvent gate = UserEvent::create_user_event();
    Barrier second =
        arrive_two_phases(cpu, other, worker, left, right, first, gate, 3, 7, 11, 15);
    bool phases_pending = pending(first) && pending(second);
    gate.trigger();
    auto [sum0, available0] = read_phase(first);
    auto [sum1, available1] = read_phase(second);
    first.destroy_barrier();
    return report(phases_pending && available0 && available1 && sum0 == 10 &&
                      sum1 == 26 && left.read<int32_t>(0) == 21 &&
                      right.read<int32_t>(0) == 21,
                  "Barrier reduction: phase results were 10 and 26");
  }

  bool checkCopies(const IndexSpaceGeneric &space, RegionInstance source,
                   RegionInstance staging, RegionInstance baseline,
                   RegionInstance optimized)
  {
    for(size_t i = 0; i < 4; ++i)
      source.write<int32_t>(i * sizeof(int32_t), static_cast<int32_t>(10 + i));

    bool poisoned_baseline =
        await_pipeline(staged_copy(source, staging, baseline, &space, Event::NO_EVENT));
    staging.write<int32_t>(0, -1);
    bool poisoned_optimized = await_pipeline(
        elide_staging(source, staging, optimized, &space, Event::NO_EVENT));

    bool passed =
        !poisoned_baseline && !poisoned_optimized && staging.read<int32_t>(0) == -1;
    for(size_t i = 0; passed && i < 4; ++i) {
      size_t offset = i * sizeof(int32_t);
      int32_t actual = baseline.read<int32_t>(offset);
      passed = actual == static_cast<int32_t>(10 + i) &&
               actual == optimized.read<int32_t>(offset);
    }
    return report(passed, "Copy elimination: both destinations matched, optimized "
                          "staging stayed untouched");
  }
} // namespace

bool run_feature_examples(Realm::Runtime &runtime, Processor cpu, Memory memory,
                          TaskID seed, TaskID sum, TaskID increment)
{
  Processor other = Realm::Machine::ProcessorQuery(Realm::Machine::get_machine())
                        .only_kind(Processor::LOC_PROC)
                        .local_address_space()
                        .next(cpu);
  if(!other.exists())
    other = cpu;

  Realm::Rect<1, long long> bounds(Realm::Point<1, long long>(0),
                                   Realm::Point<1, long long>(3));
  Realm::IndexSpace<1, long long> space(bounds);
  std::map<Realm::FieldID, size_t> fields{{0, sizeof(int32_t)}};
  Realm::InstanceLayoutConstraints constraints(fields, 1);
  int dimension_order[] = {0};
  std::unique_ptr<InstanceLayoutGeneric> layout(
      InstanceLayoutGeneric::choose_instance_layout(space, constraints, dimension_order));
  Buffers buffers;
  if(!buffers.create(memory, *layout))
    return report(false, "Allocate example instances");
  auto [a0, b0, out0, a1, b1, out1] = buffers.instances;

  auto [joined, fork_safe] =
      fork_join(cpu, other, seed, sum, a0, b0, out0, Event::NO_EVENT);
  bool fork_poisoned = await_pipeline(joined);
  bool fork_settled = !await_pipeline(fork_safe);
  bool passed = report(!fork_poisoned && fork_settled && out0.read<int32_t>(0) == 42,
                       "Fork/join: two branches produced 21 and the join produced 42");

  UserEvent cancelled = UserEvent::create_user_event();
  UserEvent slow = UserEvent::create_user_event();
  auto [reported, safe] = fault_cleanup(cpu, seed, memory, layout.get(), cancelled, slow);
  cancelled.cancel();
  bool fault_reported = await_pipeline(reported);
  bool cleanup_pending = pending(safe);
  slow.trigger();
  bool cleanup_settled = !await_pipeline(safe);
  passed &=
      report(fault_reported && cleanup_pending && cleanup_settled,
             "Fault cleanup: merge reported poison before destruction became eligible");

  IndexSpaceGeneric generic_space(space);
  passed &= checkCopies(generic_space, a0, b0, out0, out1);

  Event incremented = reduce_event_edges(cpu, seed, increment, a0);
  passed &= report(!await_pipeline(incremented) && a0.read<int32_t>(0) == 23,
                   "Transitive reduction: two increments after the seed produced 23");

  a0.write<int32_t>(0, 0);
  Event counted = reservation_counter(cpu, other, increment, a0);
  passed &= report(!await_pipeline(counted) && a0.read<int32_t>(0) == 2,
                   "Reservation: two exclusive increments produced 2");

  passed &= checkBarriers(runtime, cpu, other, seed, a0, b0);

  out0.write<int32_t>(0, 0);
  out1.write<int32_t>(0, 0);
  UserEvent gate = UserEvent::create_user_event();
  auto [replayed, replay_safe] =
      replay_twice(cpu, other, seed, sum, a0, b0, out0, a1, b1, out1, gate);
  bool replay_pending = pending(replayed);
  gate.trigger();
  bool replay_poisoned = await_pipeline(replayed);
  bool replay_settled = !await_pipeline(replay_safe);
  passed &= report(replay_pending && !replay_poisoned && replay_settled &&
                       out0.read<int32_t>(0) == 42 && out1.read<int32_t>(0) == 42,
                   "Replay: both bindings produced 42 with a carried completion event");
  return passed;
}
