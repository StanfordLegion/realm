/*
 * Copyright 2026 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm.h"

#include <cstdlib>
#include <vector>

using namespace Realm;

namespace {
  Logger log_app("app");

  enum
  {
    TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE,
    CHECK_PREIMAGE_1D_TASK,
    CHECK_PREIMAGE_2D_TASK,
  };

  template <int N>
  struct CheckArgs {
    IndexSpace<N> preimage;
    Rect<N> expected;
  };

  template <int N>
  void check_preimage(const void *args, size_t, const void *, size_t, Processor)
  {
    const auto &check = *static_cast<const CheckArgs<N> *>(args);
    check.preimage.make_valid(/*precise=*/true).wait();
    if(check.preimage.dense() || check.preimage.volume() != check.expected.volume() ||
       (!check.expected.empty() && !check.preimage.contains_all(check.expected))) {
      log_app.error() << "unexpected preimage " << check.preimage << ", expected "
                      << check.expected;
      std::abort();
    }
  }

  template <int N, typename Field>
  FieldDataDescriptor<IndexSpace<N>, Field> make_field(Memory memory,
                                                       const Rect<N> &bounds, Field value)
  {
    IndexSpace<N> space(bounds);
    RegionInstance inst;
    RegionInstance::create_instance(inst, memory, space, {sizeof(Field)}, 0,
                                    ProfilingRequestSet())
        .wait();
    space
        .fill({CopySrcDstField().set_field(inst, 0, sizeof(Field))},
              ProfilingRequestSet(), &value, sizeof(value))
        .wait();
    return {space, inst, 0};
  }

  template <int N, typename Field>
  void test_preimage(Memory data_memory, Processor verifier,
                     const FieldDataDescriptor<IndexSpace<N>, int> &colors,
                     Field out_of_range, Field in_range)
  {
    const Rect<N> good_bounds(Point<N>(2), Point<N>(3));
    const IndexSpace<N> parent(Rect<N>(Point<N>(0), Point<N>(3)));
    const auto misses =
        make_field(data_memory, Rect<N>(Point<N>(0), Point<N>(1)), out_of_range);
    const auto hits = make_field(data_memory, good_bounds, in_range);

    for(bool sparse_target : {false, true}) {
      for(bool mixed : {false, true}) {
        UserEvent populate_target = UserEvent::NO_USER_EVENT;
        Event target_ready = Event::NO_EVENT;
        std::vector<IndexSpace<N>> targets;
        if(sparse_target) {
          populate_target = UserEvent::create_user_event();
          target_ready = parent.template create_subspaces_by_field<int>(
              {colors}, {1}, targets, ProfilingRequestSet(), populate_target);
        } else {
          targets.emplace_back(good_bounds);
        }

        std::vector<FieldDataDescriptor<IndexSpace<N>, Field>> data{misses};
        if(mixed)
          data.push_back(hits);
        std::vector<IndexSpace<N>> preimages;
        Event done = parent.create_subspaces_by_preimage(data, targets, preimages,
                                                         ProfilingRequestSet());
        // With one worker, the local image micro-op gets priority over the
        // partitioning operation that populates the sparse target.
        if(sparse_target)
          populate_target.trigger();
        Event::merge_events(done, target_ready).wait();

        const CheckArgs<N> check{preimages[0],
                                 mixed ? good_bounds : Rect<N>::make_empty()};
        verifier
            .spawn(N == 1 ? CHECK_PREIMAGE_1D_TASK : CHECK_PREIMAGE_2D_TASK, &check,
                   sizeof(check))
            .wait();
        preimages[0].destroy();
        if(sparse_target)
          targets[0].destroy();
      }
    }
    hits.inst.destroy();
    misses.inst.destroy();
  }

  template <int N>
  void test_dimension(Memory local_memory, Memory data_memory, Processor verifier)
  {
    const auto colors = make_field(local_memory, Rect<N>(Point<N>(2), Point<N>(3)), 1);
    test_preimage(data_memory, verifier, colors, Point<N>(10), Point<N>(2));
    test_preimage(data_memory, verifier, colors, Rect<N>(Point<N>(10), Point<N>(11)),
                  Rect<N>(Point<N>(2), Point<N>(3)));
    colors.inst.destroy();
  }

  void top_level_task(const void *, size_t, const void *, size_t, Processor local)
  {
    Machine machine = Machine::get_machine();
    Machine::ProcessorQuery cpus(machine);
    cpus.only_kind(Processor::LOC_PROC);
    Machine::MemoryQuery memories(machine);
    memories.only_kind(Memory::SYSTEM_MEM);
    Memory local_memory =
        Machine::MemoryQuery(memories).same_address_space_as(local).first();
    Processor remote = Processor::NO_PROC;
    for(Processor cpu : cpus) {
      if(cpu.address_space() != local.address_space()) {
        remote = cpu;
        break;
      }
    }

    // A different rank verifies the locally owned preimages when one is available.
    const Processor verifier = remote.exists() ? remote : local;
    test_dimension<1>(local_memory, local_memory, verifier);
#if REALM_MAX_DIM >= 2
    test_dimension<2>(local_memory, local_memory, verifier);
#endif

    // With remote fields, the dense and sparse targets place their outputs on
    // different ranks. Validate on the initiator to fetch the dense-target result.
    if(remote.exists()) {
      Memory remote_memory =
          Machine::MemoryQuery(memories).same_address_space_as(remote).first();
      test_dimension<1>(local_memory, remote_memory, local);
#if REALM_MAX_DIM >= 2
      test_dimension<2>(local_memory, remote_memory, local);
#endif
    }
    Runtime::get_runtime().shutdown(Processor::get_current_finish_event());
  }
} // namespace

int main(int argc, char **argv)
{
  Runtime runtime;
  if(!runtime.init(&argc, &argv))
    return 1;
  runtime.register_task(TOP_LEVEL_TASK, top_level_task);
  runtime.register_task(CHECK_PREIMAGE_1D_TASK, check_preimage<1>);
#if REALM_MAX_DIM >= 2
  runtime.register_task(CHECK_PREIMAGE_2D_TASK, check_preimage<2>);
#endif
  Processor cpu = Machine::ProcessorQuery(Machine::get_machine())
                      .only_kind(Processor::LOC_PROC)
                      .first();
  runtime.collective_spawn(cpu, TOP_LEVEL_TASK, nullptr, 0);
  return runtime.wait_for_shutdown();
}
