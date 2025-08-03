/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
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

#include <time.h>

#include <cassert>

#include "realm/cmdline.h"
#include "realm/runtime.h"
#include "realm.h"
#include "realm/id.h"
#include "realm/network.h"
#include "osdep.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  NODE_TASK,
};

constexpr int NUM_INSTS = 1;

struct TaskArgs {
  IndexSpace<1> sources[NUM_INSTS];
  FieldDataDescriptor<IndexSpace<1>, Point<1>> ptr_data[NUM_INSTS];
  IndexSpace<1> parent;
};

namespace TestConfig {
  bool remote_create = false;
};

void node_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  TaskArgs &task_args = *(TaskArgs *)args;

  std::vector<Point<1>> colors{0, 1};
  {
    std::vector<IndexSpace<1>> subspaces;
    Event e2 = task_args.parent.create_subspaces_by_field(
        std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>>{
            std::begin(task_args.ptr_data), std::end(task_args.ptr_data)},
        colors, subspaces, ProfilingRequestSet());
    for(size_t i = 0; i < subspaces.size(); i++) {
      subspaces[i].destroy(e2);
    }
    e2.wait();
  }
}

void main_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
               Processor p)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(5)));
  rects.push_back(Rect<1>(Point<1>(10), Point<1>(15)));

  Machine machine = Machine::get_machine();
  std::vector<Memory> memories;
  for(Machine::MemoryQuery::iterator it = Machine::MemoryQuery(machine).begin(); it;
      ++it) {
    Memory m = *it;
    if(m.kind() == Memory::SYSTEM_MEM) {
      memories.push_back(m);
    }
  }

  std::vector<Event> events;
  for(std::vector<Memory>::const_iterator it = memories.begin(); it != memories.end();
      ++it) {
    Processor proc = *Machine::ProcessorQuery(machine)
                          .only_kind(Processor::LOC_PROC)
                          .same_address_space_as(*it)
                          .begin();

    IndexSpace<1, int> root1(rects[0]);
    IndexSpace<1> parent(rects[0]);

    std::vector<IndexSpace<1>> sources;
    root1.create_equal_subspaces(NUM_INSTS, 1, sources, Realm::ProfilingRequestSet())
        .wait();

    std::vector<size_t> field_sizes;
    field_sizes.push_back(sizeof(int));
    field_sizes.push_back(sizeof(Point<1>));

    std::vector<FieldDataDescriptor<IndexSpace<1>, Point<1>>> ptr_data(NUM_INSTS);

    for(int i = 0; i < NUM_INSTS; i++) {
      int mem_idx = i % memories.size();
      RegionInstance ri;
      RegionInstance::create_instance(ri, memories[mem_idx], sources[i], field_sizes, 0,
                                      Realm::ProfilingRequestSet())
          .wait();
      ptr_data[i].index_space = sources[i];
      ptr_data[i].inst = ri;
      ptr_data[i].field_offset = 0;
      AffineAccessor<int, 1> a_vals(ri, 0);
      for(PointInRectIterator<1, int> pir(root1.bounds); pir.valid; pir.step()) {
        a_vals.write(pir.p, pir.p[0]);
      }
    }

    TaskArgs args;
    for(int i = 0; i < NUM_INSTS; i++) {
      args.sources[0] = sources[0];
      args.ptr_data[0] = ptr_data[0];
    }
    args.parent = parent;
    Event e = proc.spawn(NODE_TASK, &args, sizeof(args));
    events.push_back(e);
  }

  Event::merge_events(events).wait();
  usleep(100000);
  Runtime::get_runtime().shutdown(Processor::get_current_finish_event(), 0);
}

int main(int argc, char **argv)
{
  Runtime rt;

  CommandLineParser cp;
  cp.add_option_int_units("-remote_create", TestConfig::remote_create);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.init(&argc, &argv);
  rt.register_task(MAIN_TASK, main_task);
  Processor::register_task_by_kind(Processor::LOC_PROC, /*!global=*/false, NODE_TASK,
                                   CodeDescriptor(node_task), ProfilingRequestSet(), 0, 0)
      .wait();

  ModuleConfig *core = Runtime::get_runtime().get_module_config("core");
  assert(core->set_property("report_sparsity_leaks", 1));

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, MAIN_TASK, 0, 0);
  int ret = rt.wait_for_shutdown();
  return ret;
}
