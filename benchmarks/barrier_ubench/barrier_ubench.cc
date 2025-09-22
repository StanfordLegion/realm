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

#include "realm.h"
#include "realm/cmdline.h"
#include "realm/id.h"

#include <time.h>

using namespace Realm;

#include "../realm_ubench_common.h"

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  BENCHMARK_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  ARRIVE_TASK,
};

namespace TestConfig {
  int output_config = 1;
  bool n2n = false;
  int num_phases = 2;
  size_t num_tasks_per_proc = 1;
  int num_groups = 1;
  int num_samples = 1;
  int num_warmup_samples = 1;
}; // namespace TestConfig

struct ArriveArgs {
  size_t id;
  Barrier barrier;
  size_t num_arrives;
  UserEvent trigger_start_event;
  UserEvent trigger_end_event;
};

/*
  In this benchmark, we create a barrier on the top level task (benchmark_task), and then
  launch N tasks inside the benchmark_task to run barrier.arrive(), where
  N = cpu size * TestConfig::num_barriers_per_proc. Here is an example of task-process
  mapping when num_barriers_per_proc = 2: Task0:P0, Task1:P0, Task2:P1, Task3:P1, ...

  Since the caller of barrier.arrive can be arbitrary, we divide all tasks into M groups,
  which is controlled by TestConfig::num_groups. Therefore, only one group will
  participate the arrive in each iteration. Here is an example of task-group mapping when
  num_groups = 2: Task0:0, Task1:0, Task2:1, Task3:1, ...

  This benchmark has 2 modes:
  1. N-to-N, enabled by -nton. In this mode, there are N concurrent arrive tasks, in each
  phase, all the arrive calls depends on the barrier of the previous phase. In this case,
  there are N arrives and N waits. Therefore, the event graph is as follows if there are 2
  arrive tasks and 2 phases:

  start_event -> B1.arrive -> B1 complete -> B2.arrive -> B2 complete -> end_event
              -> B1.arrive ->             -> B2.arrive ->

  2. N-to-1. In this mode, there are also N concurrent arrive tasks, however, in each
  phase, only the arrive calls inside the 1st arrive task depends on the barrier of the
  previous phase. Therefore, there are N arrives and 1 wait.
*/

void arrive_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  assert(arglen == sizeof(ArriveArgs));
  const ArriveArgs &arrive_args = *reinterpret_cast<const ArriveArgs *>(args);
  int group_id = arrive_args.id / arrive_args.num_arrives;
  int id_in_group = arrive_args.id % arrive_args.num_arrives;
  log_app.info("Start Barrier task %zu, on proc:%llx, group %d, id_in_group %d",
               arrive_args.id, p.id, group_id, id_in_group);

  Barrier barrier = arrive_args.barrier;
  Stat time_stat;
  Event wait_on = arrive_args.trigger_start_event;
  Barrier previous_barrier = barrier;
  for(int i = 0; i < TestConfig::num_phases; i++) {
    int current_group_id = i % TestConfig::num_groups;
    if(group_id == current_group_id) {
      barrier.arrive(1, wait_on);
      log_app.info("Arrive task %zu, on proc:%llx, group id %d, iter %d", arrive_args.id,
                   p.id, group_id, i);
    }
    previous_barrier = barrier;
    // in 1-to-N mode, only the first one in the group depends on the previous barrier
    if(TestConfig::n2n || (!TestConfig::n2n && id_in_group == 0)) {
      wait_on = previous_barrier;
      log_app.info("Arrive task %zu, on proc:%llx, iter %d on wait", arrive_args.id, p.id, i);
    }
    barrier = barrier.advance_barrier();
  }
  arrive_args.trigger_end_event.trigger(previous_barrier);
}

void benchmark_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  // output configuration
  if (TestConfig::output_config) {
    output_machine_config();
    printf("BENCHMARK_CONFIGURATION {num_phases:%d, n2n:%d, num_groups:%d, num_tasks_per_processor:%zu}\n", TestConfig::num_phases, TestConfig::n2n, TestConfig::num_groups, TestConfig::num_tasks_per_proc);
  }

  std::vector<Processor> cpu_procs;
  {
    Machine::ProcessorQuery query(Machine::get_machine());
    query.only_kind(Processor::LOC_PROC);
    cpu_procs.insert(cpu_procs.end(), query.begin(), query.end());
  }

  size_t num_tasks = cpu_procs.size() * TestConfig::num_tasks_per_proc;
  size_t num_arrives =
      cpu_procs.size() * TestConfig::num_tasks_per_proc / TestConfig::num_groups;

  Stat stat_time_per_arrive;

  for (int s = 0; s < TestConfig::num_samples + TestConfig::num_warmup_samples; s++) {
    Barrier origin_barrier = Barrier::create_barrier(num_arrives);
    Barrier barrier = origin_barrier;

    // Launch arrive and wait tasks
    std::vector<Event> arrive_events(num_tasks, Event::NO_EVENT);
    std::vector<Event> wait_events(num_tasks, Event::NO_EVENT);

    Stat time_stat;
    size_t cpu_idx = 0;

    ArriveArgs arrive_args;
    arrive_args.barrier = barrier;
    arrive_args.num_arrives = num_arrives;
    arrive_args.trigger_start_event = UserEvent::create_user_event();

    for(size_t i = 0; i < num_tasks; i++) {
      arrive_args.id = i;
      arrive_args.trigger_end_event = UserEvent::create_user_event();
      wait_events[i] = arrive_args.trigger_end_event;
      arrive_events[i] =
          cpu_procs[cpu_idx].spawn(ARRIVE_TASK, &arrive_args, sizeof(ArriveArgs));
      if(i % TestConfig::num_tasks_per_proc == TestConfig::num_tasks_per_proc - 1) {
        cpu_idx++;
      }
    }

    // make sure we issue the arrive ahead
    Event::merge_events(arrive_events).wait();
    log_app.info(
        "Start barrier benchmark, number of arrive tasks: %zu, number of arrives: %zu",
        num_tasks, num_arrives);

    arrive_args.trigger_start_event.trigger();
    Event merged_event = Event::merge_events(wait_events);
    long long start_time = Clock::current_time_in_microseconds();
    merged_event.wait();
    long long end_time = Clock::current_time_in_microseconds();
    double time_per_arrive = static_cast<double>(end_time - start_time) / TestConfig::num_phases;
    if (s >= TestConfig::num_warmup_samples) {
      stat_time_per_arrive.sample(time_per_arrive);
    }
    log_app.info("time_per_arrive:%.2f, sample %d", time_per_arrive, s);
    origin_barrier.destroy_barrier();
  }
#ifdef BENCHMARK_USE_JSON_FORMAT
  std::cout << "RESULT {name:barrier_time_per_arrive, " << stat_time_per_arrive << ", unit:-us}" << std::endl;
#else
  std::cout << "RESULT barrier_time_per_arrive=" << stat_time_per_arrive << ", -us" << std::endls;
#endif
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-config", TestConfig::output_config)
      .add_option_bool("-n2n", TestConfig::n2n)
      .add_option_int("-np", TestConfig::num_phases)
      .add_option_int("-ntpp", TestConfig::num_tasks_per_proc)
      .add_option_int("-ng", TestConfig::num_groups)
      .add_option_int("-s", TestConfig::num_samples)
      .add_option_int("-warmup", TestConfig::num_warmup_samples);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  // when ng > 1, we only support N-to-N
  if (!TestConfig::n2n && TestConfig::num_groups > 1) {
    log_app.error("Do not support N-to-1 with number of groups > 1");
    return 0;
  }

  rt.register_task(BENCHMARK_TASK, benchmark_task);
  rt.register_task(ARRIVE_TASK, arrive_task);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = rt.collective_spawn(p, BENCHMARK_TASK, 0, 0);

  // request shutdown once that task is complete
  rt.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  rt.wait_for_shutdown();

  return 0;
}
