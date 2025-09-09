/* Copyright 2024 Stanford University
 * Copyright 2024 NVIDIA Corp
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

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <realm.h>
#include <realm/cmdline.h>

using namespace Realm;

#include "../realm_ubench_common.h"

Logger log_app("app");

enum
{
  BENCH_LAUNCHER_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  BENCH_TIMING_TASK,
  DUMMY_TASK_LAUNCHER,
  DUMMY_TASK
};

struct BenchTimingTaskArgs {
  int output_configs = 1;
  size_t num_launcher_tasks = 1;
  size_t num_dummy_tasks = 100;
  size_t num_samples = 1;
  size_t num_warmup_samples = 1;
  size_t arg_size = 0;
  bool chain = false;
  bool test_gpu = false;
  bool use_proc_group = false;
  int remote_mode = 2; // 0: local, 1: remote, 2: local and remote
};

struct DummyTaskLauncherArgs {
  UserEvent dummy_task_trigger_event;
  UserEvent dummy_task_wait_event;
  size_t num_child_tasks;
  size_t arg_size;
  bool chain;
};

#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
void dummy_gpu_task(const void *args, size_t arglen, 
		                const void *userdata, size_t userlen, Processor p);
#endif

static void output_configuration(const BenchTimingTaskArgs &args)
{
  printf("BENCHMARK_CONFIGURATION {num_launcher_tasks:%zu, num_dummy_tasks:%zu, "
         "num_samples:%zu, arg_size:%zu, chain:%d, test_gpu:%d, "
         "use_proc_group:%d, remote_mode:%d}\n",
         args.num_launcher_tasks, args.num_dummy_tasks, args.num_samples,
         args.arg_size, args.chain, args.test_gpu, args.use_proc_group,
         args.remote_mode);
}

static void dummy_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{}

static void dummy_task_launcher(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, Processor p)
{
  // Launches N dummy tasks, waiting for the given user event and signaling another user
  // event once complete
  const DummyTaskLauncherArgs &self_args =
      *reinterpret_cast<const DummyTaskLauncherArgs *>(args);
  assert(arglen == sizeof(DummyTaskLauncherArgs));
  std::vector<char> task_args(self_args.arg_size);
  std::vector<Event> task_events(self_args.num_child_tasks, Event::NO_EVENT);
  UserEvent trigger_event = UserEvent::create_user_event();
  Event depends_event = trigger_event;
  for(size_t i = 0; i < task_events.size(); i++) {
    task_events[i] =
        p.spawn(DUMMY_TASK, task_args.data(), task_args.size(), depends_event);
    if(self_args.chain) {
      depends_event = task_events[i];
    }
  }
  self_args.dummy_task_wait_event.trigger(
      self_args.chain ? task_events.back() : Event::merge_events(task_events));
  trigger_event.trigger(self_args.dummy_task_trigger_event);
}

static void bench_timing_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor proc)
{
  const BenchTimingTaskArgs &self_args =
      *reinterpret_cast<const BenchTimingTaskArgs *>(args);
  assert(arglen == sizeof(BenchTimingTaskArgs));

  // output configuration
  if (self_args.output_configs) {
    output_machine_config();
    output_configuration(self_args);
  }

  Stat spawn_time, completion_time;

  DummyTaskLauncherArgs launcher_args;
  launcher_args.arg_size = self_args.arg_size;
  launcher_args.num_child_tasks = self_args.num_dummy_tasks;
  launcher_args.chain = self_args.chain;

  Processor::Kind proc_kind = Processor::Kind::NO_KIND;
  if (self_args.test_gpu) {
    proc_kind = Processor::TOC_PROC;
  } else {
    proc_kind = Processor::LOC_PROC;
  }

  std::vector<Processor> processors;
  size_t proc_num = 0;
  Machine::ProcessorQuery processors_to_test =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(proc_kind);

  if(self_args.remote_mode == 0) {
    processors_to_test = processors_to_test.local_address_space();
    processors.assign(processors_to_test.begin(), processors_to_test.end());
  } else if(self_args.remote_mode == 1) {
    std::unordered_map<AddressSpace, std::vector<Processor>> proc_map;
    for(Processor proc : processors_to_test) {
      proc_map[proc.address_space()].push_back(proc);
    }
    std::unordered_map<AddressSpace, std::vector<Processor>>::iterator it;
    for(it = proc_map.begin(); it != proc_map.end(); it++) {
      if(it->first != proc.address_space()) {
        processors.insert(processors.end(), it->second.begin(), it->second.end());
      }
    }
  } else {
    processors.assign(processors_to_test.begin(), processors_to_test.end());
  }

  {
    proc_num = processors.size();

    if (self_args.use_proc_group) {
      ProcessorGroup proc_group = ProcessorGroup::create_group(processors);
      processors.clear();
      processors.push_back(proc_group);
    }
  }

  std::vector<Event> task_events(self_args.num_launcher_tasks * proc_num,
                                 Event::NO_EVENT);
  std::vector<Event> child_task_events(self_args.num_launcher_tasks * proc_num,
                                       Event::NO_EVENT);

  for(size_t s = 0; s < self_args.num_samples + self_args.num_warmup_samples; s++) {
    UserEvent trigger_event = UserEvent::create_user_event();
    launcher_args.dummy_task_trigger_event = UserEvent::create_user_event();

    for(size_t p = 0; p < proc_num; p++) {
      Processor target_processor = processors[p % processors.size()];
      log_app.info("Proc:%llx launches tasks onto target proc:%llx", proc.id,
                   target_processor.id);
      for(size_t t = 0; t < self_args.num_launcher_tasks; t++) {
        launcher_args.dummy_task_wait_event = UserEvent::create_user_event();
        task_events[p * self_args.num_launcher_tasks + t] = target_processor.spawn(
            DUMMY_TASK_LAUNCHER, &launcher_args, sizeof(launcher_args), trigger_event);
        child_task_events[p * self_args.num_launcher_tasks + t] =
            launcher_args.dummy_task_wait_event;
      }
    }

    // Make sure the launcher tasks have completed (their child tasks are all queued)
    Event wait_event = Event::merge_events(task_events);
    {
      // Time the spawn
      size_t start_time = Clock::current_time_in_microseconds();
      trigger_event.trigger();
      wait_event.wait();
      size_t end_time = Clock::current_time_in_microseconds();
      if(s >= self_args.num_warmup_samples) {
        spawn_time.sample(double(proc_num * self_args.num_launcher_tasks *
                                self_args.num_dummy_tasks * 1e6) /
                          double(end_time - start_time));
        log_app.info() << "Spawn sample (us): " << end_time - start_time;
      }
    }

    wait_event = Event::merge_events(child_task_events);
    {
      // Time the completion of the dummy tasks
      size_t start_time = Clock::current_time_in_microseconds();
      launcher_args.dummy_task_trigger_event.trigger();
      wait_event.wait();
      size_t end_time = Clock::current_time_in_microseconds();
      if(s >= self_args.num_warmup_samples) {
        completion_time.sample(double(proc_num * self_args.num_launcher_tasks *
                                      self_args.num_dummy_tasks * 1e6) /
                              double(end_time - start_time));
        log_app.info() << "Completion sample (us): " << end_time - start_time;
      }
    }
  }

#ifdef BENCHMARK_USE_JSON_FORMAT
  std::cout << "RESULT {name:spawn_rate, " << spawn_time << ", unit:+tasks/s}" << std::endl;
  std::cout << "RESULT {name:completion_rate, " << completion_time
                  << ", unit:+tasks/s}" << std::endl;
#else
  std::cout << "RESULT spawn_rate=/" << spawn_time << " +tasks/s" << std::endl;
  std::cout << "RESULT completion_rate=/" << completion_time << " +tasks/s" << std::endl;
#endif
  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime r;
  CommandLineParser cp;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(BENCH_TIMING_TASK, bench_timing_task);
  Processor::register_task_by_kind(Processor::LOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK_LAUNCHER,
				   CodeDescriptor(dummy_task_launcher),
				   ProfilingRequestSet()).wait();
  Processor::register_task_by_kind(Processor::LOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK,
				   CodeDescriptor(dummy_task),
				   ProfilingRequestSet()).wait();
#if defined(REALM_USE_CUDA) || defined(REALM_USE_HIP)
  Processor::register_task_by_kind(Processor::TOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK_LAUNCHER,
				   CodeDescriptor(dummy_task_launcher),
				   ProfilingRequestSet()).wait();
  Processor::register_task_by_kind(Processor::TOC_PROC,
				   false /*!global*/,
				   DUMMY_TASK,
				   CodeDescriptor(dummy_gpu_task),
				   ProfilingRequestSet()).wait();
#endif

  BenchTimingTaskArgs args;

  cp.add_option_int("-config", args.output_configs);
  cp.add_option_int("-a", args.arg_size);
  cp.add_option_int("-s", args.num_samples);
  cp.add_option_int("-warmup", args.num_warmup_samples);
  cp.add_option_int("-tpp", args.num_launcher_tasks);
  cp.add_option_int("-n", args.num_dummy_tasks);
  cp.add_option_bool("-c", args.chain);
  cp.add_option_bool("-gpu", args.test_gpu);
  cp.add_option_bool("-g", args.use_proc_group);
  cp.add_option_int("-remote", args.remote_mode);

  ok = cp.parse_command_line(argc, (const char **)argv);
  assert(ok);

  Machine::ProcessorQuery processors_to_test =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  r.collective_spawn(p, BENCH_TIMING_TASK, &args, sizeof(args));

  return r.wait_for_shutdown();
}
