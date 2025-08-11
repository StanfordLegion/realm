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

#include "common.h"
#include "realm.hpp"

using namespace REALM_NAMESPACE;

enum
{
  TOP_LEVEL_TASK = REALM_TASK_ID_FIRST_AVAILABLE + 0,
  HELLO_TASK,
};

void REALM_FNPTR hello_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, realm_processor_t proc)
{
  std::cout << "hello_task on proc " << proc << std::endl;
}

void REALM_FNPTR top_level_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, realm_processor_t proc_in)
{
  std::cout << "top_level_task on proc " << proc_in << std::endl;

  // get all cpu procs
  Machine::ProcessorQuery proc_query(Machine::get_machine());
  proc_query.only_kind(Processor::LOC_PROC);

  // spawn tasks on all cpu procs
  std::set<Event> events;
  for(Machine::ProcessorQuery::iterator it = proc_query.begin(); it; ++it) {
    Processor proc = *it;
    if(!proc.exists()) {
      continue;
    }

    std::cout << "spawning hello_task on proc " << proc << std::endl;
    Event event = proc.spawn(HELLO_TASK, nullptr, 0);
    assert(event != Event::NO_EVENT);
    events.insert(event);
  }
  Event merged = Event::merge_events(events);
  assert(merged != Event::NO_EVENT);
  std::cout << "waiting for merged event " << merged.id << std::endl;
  merged.wait();

#ifdef REALM_USE_CUDA
  Machine::ProcessorQuery proc_query_gpu(Machine::get_machine());
  proc_query_gpu.only_kind(Processor::TOC_PROC);

  Processor proc_gpu(proc_query_gpu.first());
  assert(proc_gpu.exists());

  Event gpuEvent = proc_gpu.spawn(HELLO_TASK, nullptr, 0);
  assert(gpuEvent != Event::NO_EVENT);
  gpuEvent.wait();
#endif
}

int main(int argc, char **argv)
{
  Runtime runtime;
  assert(runtime.init(&argc, &argv));

  Event register_task_event = Event::NO_EVENT;
  realm_id_t eventId = register_task_event.id;

  // TODO: replace with C++ API once CodeDesc and ProfileRequest classes are implemented
  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, TOP_LEVEL_TASK, top_level_task, 0,
      0, &eventId));
  register_task_event.id = eventId;
  register_task_event.wait();

  // TODO: replace with C++ API once CodeDesc and ProfileRequest classes are implemented
  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, LOC_PROC, REALM_REGISTER_TASK_DEFAULT, HELLO_TASK, hello_task, 0, 0,
      &eventId));
  register_task_event.id = eventId;
  register_task_event.wait();

  // TODO: replace with C++ API once CodeDesc and ProfileRequest classes are implemented
  CHECK_REALM(realm_processor_register_task_by_kind(
      runtime, TOC_PROC, REALM_REGISTER_TASK_DEFAULT, HELLO_TASK, hello_task, 0, 0,
      &eventId));
  register_task_event.id = eventId;
  register_task_event.wait();

  Processor proc = Machine::ProcessorQuery(Machine::get_machine())
                       .only_kind(Processor::LOC_PROC)
                       .first();
  assert(proc.exists());

  Event spawn_event = runtime.collective_spawn(proc, TOP_LEVEL_TASK, nullptr, 0);
  assert(spawn_event != Event::NO_EVENT);

  runtime.shutdown(spawn_event, 0);
  runtime.wait_for_shutdown();
  std::cout << "exiting" << std::endl;
  return 0;
}