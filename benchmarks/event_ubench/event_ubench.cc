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

#include <iomanip>
#include <iostream>
#include <realm.h>
#include <realm/cmdline.h>
#include <realm/network.h>
using namespace Realm;

#include "../realm_ubench_common.h"

Logger log_app("app");

enum
{
  BENCH_LAUNCHER_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  BENCH_SETUP_FAN_TASK,
  BENCH_SETUP_CHAIN_TASK,
  BENCH_TIMING_TASK
};

enum TestFlags
{
  FAN_TEST = 1 << 1,
  CHAIN_TEST = 1 << 2
};

struct BenchLauncherTaskArgs {
  int output_configs = 1;
  uint64_t enabled_tests = 0;
  bool measure_latency = false;
  bool skip_same_proc = false;
  size_t min_usecs = 1000000;
  size_t num_samples = 0;
  size_t min_test_size = 1024;
  size_t max_test_size = 1024;
};

struct BenchSetupFanTaskArgs {
  size_t num_samples;
  size_t num_events_per_sample;
  struct SampleEventPair {
    Event start_event;
    UserEvent wait_event;
  } events[1];
};

struct BenchSetupChainTaskArgs {
  size_t num_samples;
  size_t num_events_per_sample;
  UserEvent chain_events[1];
};

struct BenchTimingTaskArgs {
  uint64_t enabled_tests;
  bool measure_latency;
  size_t min_usecs;
  size_t num_samples;
  size_t min_num_events;
  size_t max_num_events;
  Processor dst_proc;
};

Stat inter_node_fan_events_rate;
Stat intra_node_fan_events_rate;
Stat inter_node_chain_events_rate;
Stat intra_node_chain_events_rate;

static void display_processor_info(Processor p) {}

static void output_configuration(const BenchLauncherTaskArgs &args)
{
  std::cout << "BENCHMARK_CONFIGURATION {enabled_tests:" << std::hex << args.enabled_tests
            << std::dec << ", measure_latency:" << args.measure_latency
            << ", min_usecs:" << args.min_usecs << ", num_samples:" << args.num_samples
            << ", min_test_size:" << args.min_test_size
            << ", max_test_size:" << args.max_test_size << "}" << std::endl;
}

// We want the fanned out events to be allocated on the remote processor for every sample,
// so setup the measurement DAGs and the start and end event for each sample DAG, but then
// spawn the fan eetup task to fill in the fan-out events.  This allows us to measure how
// fanned out events communicate across processors (it should be just one active message
// per sample if the processor is not in the local address space)
static void setup_fan_test(const BenchTimingTaskArgs &src_args, size_t num_samples,
                           Processor current_processor, size_t num_events,
                           UserEvent &trigger_event, Event &wait_event)
{
  std::vector<char> task_arg_buffer(sizeof(BenchSetupFanTaskArgs) +
                                    (num_samples - 1) *
                                        sizeof(BenchSetupFanTaskArgs::SampleEventPair));
  BenchSetupFanTaskArgs &task_args =
      *reinterpret_cast<BenchSetupFanTaskArgs *>(task_arg_buffer.data());
  std::vector<Event> event_array(num_samples, Event::NO_EVENT);

  trigger_event = UserEvent::create_user_event();
  Event sample_trigger_event = trigger_event;

  task_args.num_samples = num_samples;
  task_args.num_events_per_sample = num_events;
  for(size_t j = 0; j < num_samples; j++) {
    UserEvent start_sample_event = UserEvent::create_user_event();
    task_args.events[j].start_event = start_sample_event;
    task_args.events[j].wait_event = UserEvent::create_user_event();

    start_sample_event.trigger(sample_trigger_event);
    event_array[j] = task_args.events[j].wait_event;
    if(src_args.measure_latency)
      sample_trigger_event = task_args.events[j].wait_event;
  }
  wait_event = Event::merge_events(event_array);

  // Pass all the sample events to the setup task and have it set up the fan DAG
  src_args.dst_proc.spawn(BENCH_SETUP_FAN_TASK, &task_args, task_arg_buffer.size())
      .wait();
}

static void setup_chain_test(const BenchTimingTaskArgs &src_args, size_t num_samples,
                             Processor current_processor, size_t chain_length,
                             UserEvent &trigger_event, Event &wait_event)
{
  std::vector<char> task_arg_buffer(sizeof(BenchSetupChainTaskArgs) +
                                    (src_args.num_samples * chain_length - 1) *
                                        sizeof(UserEvent));
  BenchSetupChainTaskArgs &task_args =
      *reinterpret_cast<BenchSetupChainTaskArgs *>(task_arg_buffer.data());

  trigger_event = UserEvent::create_user_event();
  Event sample_trigger_event = trigger_event;

  std::vector<Event> end_of_samples_events(num_samples, Event::NO_EVENT);

  task_args.num_samples = num_samples;
  task_args.num_events_per_sample = chain_length;

  for(size_t i = 0; i < num_samples; i++) {
    for(size_t j = 0; j < chain_length; j++) {
      task_args.chain_events[i * chain_length + j] = UserEvent::create_user_event();
    }

    task_args.chain_events[i * chain_length].trigger(sample_trigger_event);
    end_of_samples_events[i] = task_args.chain_events[(i + 1) * chain_length - 1];
    if(src_args.measure_latency)
      sample_trigger_event = task_args.chain_events[(i + 1) * chain_length - 1];
  }
  wait_event = Event::merge_events(end_of_samples_events);

  // Pass all the sample events to the setup task and have it set up the fan DAG
  src_args.dst_proc.spawn(BENCH_SETUP_CHAIN_TASK, &task_args, task_arg_buffer.size())
      .wait();
}

static float report_timing(std::string name, Processor from_proc, Processor to_proc,
                           float usecs, bool measure_latency, size_t num_samples,
                           size_t num_events_per_sample)
{
  double result = 0.0f;
  if(measure_latency) {
    result = (usecs) / num_samples;
    log_app.print() << "RESULT " << name << " " << from_proc << "->" << to_proc << "=/"
                    << result << " -us";
    log_app.info() << "(" << num_samples << " samples)>";
  } else {
    result = (num_samples * num_events_per_sample * 1e6) / usecs;
#ifdef BENCHMARK_USE_JSON_FORMAT
    std::cout << "RESULT {name:" << name << "_from_proc_" << from_proc << "_to_proc_"
              << to_proc << "_events_" << num_events_per_sample
              << ", avarage:" << std::scientific << std::setprecision(2) << result
              << ", unit:+events/s}" << std::endl;
#else
    std::cout << "RESULT " << name << " " << from_proc << "->" << to_proc << "=/"
              << std::scientific << std::setprecision(2) << result << " +events/s"
              << std::endl;
#endif
    log_app.info() << "(" << num_samples << " samples, total=" << usecs << " us)";
  }
  return result;
}

static double time_dag(UserEvent &trigger_event, Event &wait_event)
{
  double start_time = Clock::current_time_in_microseconds();
  trigger_event.trigger();
  wait_event.wait();
  double end_time = Clock::current_time_in_microseconds();
  return end_time - start_time;
}

//
// Each sample is set up as follows:
/* clang-format off */
// FAN: (for num_events_per_sample number of inner_events)
//                      +--> inner_event --+
// sample_start_event --+--> inner_event --+--> sample_finish_event
//                      +--> inner_event --+
// CHAIN: (where N is the chain depth)
// start_sample_event --> local_event1 --> remote_event1 --> local_event2 --> remote_event2 --> ... -> local_eventN sample_finish_event
/* clang-format on */

// This task just sets up the triggering events and spawns the task to set up the DAG,
// then performs the measurement by triggering and waiting for the DAG to complete and
// aggregates the result.
// We want to spawn a task to setup the DAGs in order to ensure the sub-DAG events are
// local to the target remote processor rather than the local one. The following layout is
// how each test's measurement should be setup

/* clang-format off */
// Bandwidth: (for num_samples number of samples)
//                 +--> SAMPLE --+
// trigger_event --+--> SAMPLE --+--> wait_event
//                 +--> SAMPLE --+
//
// Latency: (for num_samples number of samples)
// trigger_event --> SAMPLE --> SAMPLE --> SAMPLE --> wait_event
/* clang-format on */

static void bench_timing_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p)
{
  assert(arglen == sizeof(BenchTimingTaskArgs));
  const BenchTimingTaskArgs &src_args = *static_cast<const BenchTimingTaskArgs *>(args);

  // run the fan test
  if(src_args.enabled_tests & FAN_TEST) {
    UserEvent trigger_event;
    Event wait_event;
    size_t num_samples = src_args.num_samples;

    for(size_t i = src_args.min_num_events; i <= src_args.max_num_events; i <<= 1) {
      double usecs = 0.0;
      if(src_args.num_samples == 0) {
        if(src_args.min_usecs > 0) {
          setup_fan_test(src_args, 10, p, i, trigger_event, wait_event);
          usecs = time_dag(trigger_event, wait_event);
          if(usecs < src_args.min_usecs) {
            // Dynamically figure out the number of samples to fill the minimum time for
            // this test
            num_samples = std::max(10.0, (10.0 * src_args.min_usecs) / usecs);
          }
        }
      }
      setup_fan_test(src_args, num_samples, p, i, trigger_event, wait_event);
      usecs = time_dag(trigger_event, wait_event);
      double result = report_timing("fan_event_rate", p, src_args.dst_proc, usecs,
                                    src_args.measure_latency, num_samples, i + 2);
      if(p.address_space() == src_args.dst_proc.address_space()) {
        intra_node_fan_events_rate.sample(result);
      } else {
        inter_node_fan_events_rate.sample(result);
      }
    }
  }

  // run the chain test
  if(src_args.enabled_tests & CHAIN_TEST) {
    UserEvent trigger_event;
    Event wait_event;
    size_t num_samples = src_args.num_samples;

    for(size_t i = src_args.min_num_events; i <= src_args.max_num_events; i <<= 1) {
      setup_chain_test(src_args, num_samples, p, i, trigger_event, wait_event);
      double usecs = time_dag(trigger_event, wait_event);
      double result = report_timing("chain_event_rate", p, src_args.dst_proc, usecs,
                                    src_args.measure_latency, num_samples, 2 * i + 2);
      if(p.address_space() == src_args.dst_proc.address_space()) {
        intra_node_chain_events_rate.sample(result);
      } else {
        inter_node_chain_events_rate.sample(result);
      }
    }
  }
}

static void bench_setup_fan_task(const void *args, size_t arglen, const void *userdata,
                                 size_t userlen, Processor p)
{
  const BenchSetupFanTaskArgs &src_args =
      *static_cast<const BenchSetupFanTaskArgs *>(args);
  assert(arglen ==
         sizeof(src_args) + (src_args.num_samples - 1) * sizeof(src_args.events));
  std::vector<Event> events(src_args.num_events_per_sample, Event::NO_EVENT);
  for(size_t i = 0; i < src_args.num_samples; i++) {
    for(size_t j = 0; j < src_args.num_events_per_sample; j++) {
      UserEvent e = UserEvent::create_user_event();
      e.trigger(src_args.events[i].start_event);
      events[j] = e;
    }
    src_args.events[i].wait_event.trigger(Event::merge_events(events));
  }
}

static void bench_setup_chain_task(const void *args, size_t arglen, const void *userdata,
                                   size_t userlen, Processor p)
{
  const BenchSetupChainTaskArgs &src_args =
      *static_cast<const BenchSetupChainTaskArgs *>(args);
  assert(arglen ==
         sizeof(src_args) + (src_args.num_samples * src_args.num_events_per_sample - 1) *
                                sizeof(src_args.chain_events));
  for(size_t i = 0; i < src_args.num_samples; i++) {
    for(size_t j = 0; j < src_args.num_events_per_sample - 1; j++) {
      UserEvent chain_link_event = UserEvent::create_user_event();
      chain_link_event.trigger(
          src_args.chain_events[i * src_args.num_events_per_sample + j]);
      src_args.chain_events[i * src_args.num_events_per_sample + j + 1].trigger(
          chain_link_event);
    }
  }
}

// This task launches a task for each pair of processors in the machine.
static void bench_launcher(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  BenchTimingTaskArgs task_args;
  Event e = Event::NO_EVENT;
  const BenchLauncherTaskArgs &src_args =
      *static_cast<const BenchLauncherTaskArgs *>(args);

  assert(arglen == sizeof(BenchLauncherTaskArgs));

  // output configuration
  if(src_args.output_configs) {
    output_machine_config();
    output_configuration(src_args);
  }

  Machine::ProcessorQuery q1 =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Processor p : q1) {
    display_processor_info(p);
  }

  task_args.enabled_tests = src_args.enabled_tests;
  task_args.num_samples = src_args.num_samples;
  task_args.min_usecs = src_args.min_usecs;
  task_args.min_num_events = src_args.min_test_size;
  task_args.max_num_events = src_args.max_test_size;
  task_args.measure_latency = src_args.measure_latency;

  q1 = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  Machine::ProcessorQuery q2 =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Processor p1 : q1) {
    for(Processor p2 : q2) {
      if(src_args.skip_same_proc && p1 == p2) {
        continue;
      }
      task_args.dst_proc = p2;
      e = p1.spawn(BENCH_TIMING_TASK, &task_args, sizeof(task_args), e);
    }
  }

  e.wait();
}

int main(int argc, char **argv)
{
  Runtime r;
  CommandLineParser cp;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(BENCH_LAUNCHER_TASK, bench_launcher);
  r.register_task(BENCH_SETUP_FAN_TASK, bench_setup_fan_task);
  r.register_task(BENCH_SETUP_CHAIN_TASK, bench_setup_chain_task);
  r.register_task(BENCH_TIMING_TASK, bench_timing_task);

  BenchLauncherTaskArgs args;
  std::vector<std::string> enabled_tests;

  cp.add_option_int("-config", args.output_configs);
  cp.add_option_int("-s", args.num_samples);
  cp.add_option_int("-m", args.min_test_size);
  cp.add_option_int("-n", args.max_test_size);
  cp.add_option_bool("-L", args.measure_latency);
  cp.add_option_bool("-skip", args.skip_same_proc);
  cp.add_option_stringlist("-t", enabled_tests);
  ok = cp.parse_command_line(argc, (const char **)argv);

  if(args.min_test_size > args.max_test_size) {
    args.max_test_size = args.min_test_size;
  }

  if(enabled_tests.size() == 0) {
    args.enabled_tests = ~0ULL;
  } else {
    for(size_t i = 0; i < enabled_tests.size(); i++) {
      if(enabled_tests[i] == "FAN")
        args.enabled_tests |= (uint64_t)FAN_TEST;
      else if(enabled_tests[i] == "CHAIN")
        args.enabled_tests |= (uint64_t)CHAIN_TEST;
      else
        abort();
    }
  }

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();

  Event e = r.collective_spawn(p, BENCH_LAUNCHER_TASK, &args, sizeof(args));
  e.wait();
  if(Network::my_node_id == 0) {
    std::vector<Stat> inter_node_fan_events_rate_all;
    std::vector<Stat> intra_node_fan_events_rate_all;
    std::vector<Stat> inter_node_chain_events_rate_all;
    std::vector<Stat> intra_node_chain_events_rate_all;
    Network::gather<Stat>(0, inter_node_fan_events_rate, inter_node_fan_events_rate_all);
    Network::gather<Stat>(0, intra_node_fan_events_rate, intra_node_fan_events_rate_all);
    Network::gather<Stat>(0, inter_node_chain_events_rate,
                          inter_node_chain_events_rate_all);
    Network::gather<Stat>(0, intra_node_chain_events_rate,
                          intra_node_chain_events_rate_all);
    {
      Stat global_stat;
      for(const Stat &stat : inter_node_fan_events_rate_all) {
        if (stat.get_count() > 0) {
          global_stat.accumulate(stat);
        }
      }
      if (global_stat.get_count() > 0) {
#ifdef BENCHMARK_USE_JSON_FORMAT
      std::cout << "RESULT {name:inter_node_fan_events_rate, "
                << global_stat << ", unit:+events/s}" << std::endl;
#else
      std::cout << "RESULT inter_node_fan_events_rate=" << global_stat
                << ", +events/s" << std::endl;
#endif
      }
    }
    {
      Stat global_stat;
      for(const Stat &stat : intra_node_fan_events_rate_all) {
        if (stat.get_count() > 0) {
          global_stat.accumulate(stat);
        }
      }
      if (global_stat.get_count() > 0) {
#ifdef BENCHMARK_USE_JSON_FORMAT
      std::cout << "RESULT {name:intra_node_fan_events_rate, "
                << global_stat << ", unit:+events/s}" << std::endl;
#else
      std::cout << "RESULT intra_node_fan_events_rate=" << global_stat
                << ", +events/s" << std::endl;
#endif
      }
    }
    {
      Stat global_stat;
      for(const Stat &stat : inter_node_chain_events_rate_all) {
        if (stat.get_count() > 0) {
          global_stat.accumulate(stat);
        }
      }
      if (global_stat.get_count() > 0) {
#ifdef BENCHMARK_USE_JSON_FORMAT
      std::cout << "RESULT {name:inter_node_chain_events_rate, "
                << global_stat << ", unit:+events/s}" << std::endl;
#else
      std::cout << "RESULT inter_node_chain_events_rate=" << global_stat
                << ", +events/s" << std::endl;
#endif
      }
    }
    {
      Stat global_stat;
      for(const Stat &stat : intra_node_chain_events_rate_all) {
        if (stat.get_count() > 0) {
          global_stat.accumulate(stat);
        }
      }
      if (global_stat.get_count() > 0) {
#ifdef BENCHMARK_USE_JSON_FORMAT
      std::cout << "RESULT {name:intra_node_chain_events_rate, "
                << global_stat << ", unit:+events/s}" << std::endl;
#else
      std::cout << "RESULT intra_node_chain_events_rate=" << global_stat
                << ", +events/s" << std::endl;
#endif
      }
    }
  } else {
    Network::gather<Stat>(0, inter_node_fan_events_rate);
    Network::gather<Stat>(0, intra_node_fan_events_rate);
    Network::gather<Stat>(0, inter_node_chain_events_rate);
    Network::gather<Stat>(0, intra_node_chain_events_rate);
  }

  r.shutdown();
  return r.wait_for_shutdown();
}
