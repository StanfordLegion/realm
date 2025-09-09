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

/*
  In this benchmark, we create N reservations in total, where N = locks_per_processor *
  number of LOC processors. Reservations are created by make_locks_task, such that we can
  make sure that the N reservations are created on different processors.

  In the chain mode, the dependency graph looks like:
    trigger_event -> lock[0].lock -> DUMMY_TASK on cpus[0] -> lock[0].unlock ->
  lock[0].lock -> DUMMY_TASK on cpus[1] -> lock[0].unlock -> ... -> final_events[0]
                  -> lock[1].lock -> DUMMY_TASK on cpus[0] -> lock[1].unlock ->
  lock[1].lock -> DUMMY_TASK on cpus[1] -> lock[1].unlock -> ... -> final_events[1]
                  ...

  In the fan out mode, the dependency graph looks like:
    trigger_event -> lock[0].lock -> DUMMY_TASK on cpus[0] -> lock[0].unlock -> final_events[0]
                  -> lock[1].lock -> DUMMY_TASK on cpus[0] -> lock[1].unlock ->
                  ...
                  -> lock[0].lock -> DUMMY_TASK on cpus[1] -> lock[0].unlock -> final_events[1]
                  -> lock[1].lock -> DUMMY_TASK on cpus[1] -> lock[1].unlock ->
                  ...
  It is noted that DUMMY_TASK is only used for debugging, so in actual run, lock is
  followed by unlock.
*/

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <time.h>

#include <realm.h>
#include <realm/cmdline.h>

//#define USE_DUMMY_TASK

using namespace Realm;

#include "../realm_ubench_common.h"

Logger log_app("app");

// TASK IDs
enum
{
  BENCHMARK_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 1,
  MAKE_LOCKS_TASK,
  RETURN_LOCKS_TASK,
  CHAIN_LOCK_TASK,
  FAN_LOCK_TASK,
  DUMMY_TASK,
};

enum TestFlags
{
  FAN_TEST = 1 << 1,
  CHAIN_TEST = 1 << 2
};

namespace TestConfig {
  int output_config = 1;
  uint64_t enabled_tests = 0;
  int locks_per_processor = 16;
  int tasks_per_processor_per_lock = 1;
  int num_samples = 10;
  int num_warmup_samples = 1;
}; // namespace TestConfig

struct MakeLockTaskArgs {
  Processor orig_proc;
};

struct ReturnLockTaskArgs {
  Reservation reservations[1];
};

struct ChainTaskArgs {
  Reservation lock;
  Event precondition;
  UserEvent final_event;
  int depth;
};

struct FanTaskArgs {
  Event start_event;
  UserEvent final_event;
  size_t num_locks;
  Reservation reservations[1];
};

// only used for debug
struct DummyTaskArgs {
  Reservation res;
};

static std::set<Reservation> lock_set;

// forward declaration
void chain_locks_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p);

Processor get_next_processor(Processor cur)
{
  std::vector<Processor> all_procs;
  Machine::ProcessorQuery processors_to_test =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  all_procs.assign(processors_to_test.begin(), processors_to_test.end());
  for(std::vector<Processor>::const_iterator it = all_procs.begin();
      it != all_procs.end(); it++) {
    if(*it == cur) {
      // Advance the iterator once to get the next, handle
      // the wrap around case too
      it++;
      if(it == all_procs.end()) {
        return *(all_procs.begin());
      } else {
        return *it;
      }
    }
  }
  // Should always find one
  assert(false);
  return Processor::NO_PROC;
}

void report_timing(Stat &stat_grants_per_sec, UserEvent &start_event,
                   Event &final_event, int procs_size, bool warmup)
{
  double start, stop;
  start = Realm::Clock::current_time_in_microseconds();
  // Trigger the start event
  start_event.trigger();
  // Wait for the final event
  final_event.wait();
  stop = Realm::Clock::current_time_in_microseconds();
  double latency = stop - start;
  log_app.info("Total time: %7.3f us", latency);
  double grants_per_sec = TestConfig::locks_per_processor *
                          TestConfig::tasks_per_processor_per_lock * procs_size / latency * 1000;
  log_app.info("Reservation Grants/s: %7.3f", grants_per_sec);
  if(!warmup) { // no record for warmup
    stat_grants_per_sec.sample(grants_per_sec);
  }
}

void benchmark_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  std::vector<Processor> all_procs;
  Machine::ProcessorQuery processors_to_test =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  all_procs.assign(processors_to_test.begin(), processors_to_test.end());
  // Send a request to each processor to make the given number of locks
  {
    MakeLockTaskArgs make_lock_task_args;
    make_lock_task_args.orig_proc = p;
    for(Processor &copy_proc : all_procs) {
      Event wait_event =
          copy_proc.spawn(MAKE_LOCKS_TASK, &make_lock_task_args, sizeof(MakeLockTaskArgs));
      wait_event.wait();
    }
  }
  assert(lock_set.size() == all_procs.size() * TestConfig::locks_per_processor);

  // output configuration
  if (TestConfig::output_config) {
    output_machine_config();
    std::cout << "BENCHMARK_CONFIGURATION {enabled_tests:" << TestConfig::enabled_tests
              << ", locks per processor:" << TestConfig::locks_per_processor
              << ", num_samples:" << TestConfig::num_samples
              << ", tasks per lock per processor (acquire operations per task per lock):" << TestConfig::tasks_per_processor_per_lock
              << ", num_locks:" << lock_set.size()
              << ", num_procs:" << all_procs.size() << "}" << std::endl;
  }

  // run chain test
  if(TestConfig::enabled_tests & CHAIN_TEST) {
    Stat stat_grants_per_sec;
    for(int rep = 0; rep < TestConfig::num_samples + TestConfig::num_warmup_samples; rep++) {
      UserEvent start_event = UserEvent::create_user_event();
      std::vector<Event> final_events;
      // For each lock in the lock set, stripe it through all the processors with
      // dependences
      int lock_depth = TestConfig::tasks_per_processor_per_lock * all_procs.size();
      for(const Reservation &lock : lock_set) {
        UserEvent event = UserEvent::create_user_event();
        ChainTaskArgs chain_task_args = {lock, start_event, event, lock_depth};
        // We can just call it locally here to start on our processor
        chain_locks_task(&chain_task_args, sizeof(ChainTaskArgs), 0, 0, p);
        final_events.push_back(event);
      }
      Event final_event = Event::merge_events(final_events);
      assert(final_event.exists());
      log_app.info("Running chain, iteration:%d...", rep);
      bool warmup = (rep < TestConfig::num_warmup_samples) ? true : false;
      report_timing(stat_grants_per_sec, start_event, final_event,
                    all_procs.size(), warmup);
    }
#ifdef BENCHMARK_USE_JSON_FORMAT
    std::cout << "RESULT {name:chain_throughput, " << stat_grants_per_sec << ", unit:+reservations/s}" << std::endl;
#else
    std::cout << "RESULT chain_throughput=/" << stat_grants_per_sec << " +reservations/s" << std::endl;
#endif
  }

  // run fan test
  if(TestConfig::enabled_tests & FAN_TEST) {
    Stat stat_grants_per_sec;
    for(int rep = 0; rep < TestConfig::num_samples + TestConfig::num_warmup_samples; rep++) {
      UserEvent start_event = UserEvent::create_user_event();
      std::vector<Event> final_events;
      // Package up all the locks and tell the processor how many tasks to register for
      // each
      std::vector<char> task_arg_buffer(sizeof(FanTaskArgs) +
                                        (lock_set.size() - 1) * sizeof(Reservation));
      FanTaskArgs &fan_task_args =
          *reinterpret_cast<FanTaskArgs *>(task_arg_buffer.data());
      fan_task_args.start_event = start_event;
      fan_task_args.num_locks = lock_set.size();
      size_t idx = 0;
      for(const Reservation &lock : lock_set) {
        fan_task_args.reservations[idx] = lock;
        idx++;
      }
      // Send the message to all the processors
      for(Processor &target : all_procs) {
        fan_task_args.final_event = UserEvent::create_user_event();
        Event wait_event =
            target.spawn(FAN_LOCK_TASK, &fan_task_args, task_arg_buffer.size());
        final_events.push_back(fan_task_args.final_event);
        wait_event.wait();
      }
      Event final_event = Event::merge_events(final_events);
      assert(final_event.exists());
      log_app.info("Running fan, iteration:%d...", rep);
      bool warmup = (rep < TestConfig::num_warmup_samples) ? true : false;
      report_timing(stat_grants_per_sec, start_event, final_event,
                    all_procs.size(), warmup);
    }
#ifdef BENCHMARK_USE_JSON_FORMAT
    std::cout << "RESULT {name:fan_throughput, " << stat_grants_per_sec << ", unit:+reservations/s}" << std::endl;
#else
    std::cout << "RESULT fan_throughput=/" << stat_grants_per_sec << " +reservations/s" << std::endl;
#endif
  }

  log_app.info("Cleaning up...");
  for(Reservation lock : lock_set) {
    lock.destroy_reservation();
  }
}

void make_locks_task(const void *args, size_t arglen, const void *userdata,
                     size_t userlen, Processor p)
{
  assert(arglen == sizeof(MakeLockTaskArgs));
  const MakeLockTaskArgs *task_args = static_cast<const MakeLockTaskArgs *>(args);
  Processor orig = task_args->orig_proc;

  std::vector<char> task_arg_buffer(sizeof(ReturnLockTaskArgs) +
                                    (TestConfig::locks_per_processor - 1) *
                                        sizeof(Reservation));
  ReturnLockTaskArgs &return_lock_task_args =
      *reinterpret_cast<ReturnLockTaskArgs *>(task_arg_buffer.data());

  for(int idx = 0; idx < TestConfig::locks_per_processor; idx++) {
    return_lock_task_args.reservations[idx] = Realm::Reservation::create_reservation();
  }
  Event wait_event =
      orig.spawn(RETURN_LOCKS_TASK, &return_lock_task_args, task_arg_buffer.size());
  wait_event.wait();
}

void return_locks_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{
  const ReturnLockTaskArgs *return_locks_task_args =
      static_cast<const ReturnLockTaskArgs *>(args);
  assert(arglen == sizeof(Reservation) * TestConfig::locks_per_processor);
  for(int idx = 0; idx < TestConfig::locks_per_processor; idx++) {
    lock_set.insert(return_locks_task_args->reservations[idx]);
  }
}

void chain_locks_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  assert(arglen == sizeof(ChainTaskArgs));
  const ChainTaskArgs *chain_task_args = static_cast<const ChainTaskArgs *>(args);
  log_app.info("chain Processor %llx, lock %llx, depth %d", p.id, chain_task_args->lock.id, chain_task_args->depth);

  // Chain the lock acquistion, task call, lock release
  Event lock_event =
      chain_task_args->lock.acquire(0, true, chain_task_args->precondition);
#ifdef USE_DUMMY_TASK
  DummyTaskArgs dummy_task_args = {chain_task_args->lock};
  lock_event = p.spawn(DUMMY_TASK, &dummy_task_args, sizeof(DummyTaskArgs), lock_event);
#endif
  chain_task_args->lock.release(lock_event);
  if(chain_task_args->depth == 1) {
    chain_task_args->final_event.trigger(lock_event);
  } else {
    ChainTaskArgs next_struct = {chain_task_args->lock, lock_event,
                                 chain_task_args->final_event,
                                 chain_task_args->depth - 1};
    Processor next_proc = get_next_processor(p);
    Event wait_event =
        next_proc.spawn(CHAIN_LOCK_TASK, &next_struct, sizeof(ChainTaskArgs));
    wait_event.wait();
  }
}

void fan_locks_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  const FanTaskArgs *fan_task_args = static_cast<const FanTaskArgs *>(args);
  assert(arglen ==
         (sizeof(FanTaskArgs) + (fan_task_args->num_locks - 1) * sizeof(Reservation)));

  Event precondition = fan_task_args->start_event;
  size_t num_locks = fan_task_args->num_locks;
  std::set<Event> wait_for_events;
  for(size_t i = 0; i < num_locks; i++) {
    Reservation lock = fan_task_args->reservations[i];
    Event lock_event = precondition;
    for (int j = 0; j < TestConfig::tasks_per_processor_per_lock; j++) {
      lock_event = lock.acquire(0, true, precondition);
      //log_app.print("lock %llx", lock.id);
#ifdef USE_DUMMY_TASK
      DummyTaskArgs dummy_task_args = {lock};
      lock_event =
          p.spawn(DUMMY_TASK, &dummy_task_args, sizeof(DummyTaskArgs), lock_event);
#endif
      lock.release(lock_event);
    }
    wait_for_events.insert(lock_event);
  }
  // Merge all the wait for events together and send back the result
  Event final_event = Event::merge_events(wait_for_events);
  fan_task_args->final_event.trigger(final_event);
}

#ifdef USE_DUMMY_TASK
void dummy_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                Processor p)
{
  // Do nothing
  const DummyTaskArgs *dummy_task_args = static_cast<const DummyTaskArgs *>(args);
  log_app.info("dummy task on proc:%llx, reservation:%llx", p.id,
                dummy_task_args->res.id);
}
#endif

int main(int argc, char **argv)
{
  Runtime r;

  bool ok = r.init(&argc, &argv);
  assert(ok);

  r.register_task(BENCHMARK_TASK, benchmark_task);
  r.register_task(MAKE_LOCKS_TASK, make_locks_task);
  r.register_task(RETURN_LOCKS_TASK, return_locks_task);
  r.register_task(CHAIN_LOCK_TASK, chain_locks_task);
  r.register_task(FAN_LOCK_TASK, fan_locks_task);
#ifdef USE_DUMMY_TASK
  r.register_task(DUMMY_TASK, dummy_task);
#endif

  std::vector<std::string> enabled_tests;

  // parse argv
  CommandLineParser cp;
  cp.add_option_int("-config", TestConfig::output_config);
  cp.add_option_int("-s", TestConfig::num_samples);
  cp.add_option_int("-warmup", TestConfig::num_warmup_samples);
  cp.add_option_int("-lpp", TestConfig::locks_per_processor);
  cp.add_option_int("-tpppl", TestConfig::tasks_per_processor_per_lock);
  cp.add_option_stringlist("-t", enabled_tests);

  ok = cp.parse_command_line(argc, (const char **)argv);
  assert(ok);

  if(enabled_tests.size() == 0) {
    TestConfig::enabled_tests = ~0ULL;
  } else {
    for(size_t i = 0; i < enabled_tests.size(); i++) {
      if(enabled_tests[i] == "FAN")
        TestConfig::enabled_tests |= (uint64_t)FAN_TEST;
      else if(enabled_tests[i] == "CHAIN")
        TestConfig::enabled_tests |= (uint64_t)CHAIN_TEST;
      else
        abort();
    }
  }

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single task - everybody gets the same finish event
  Event e = r.collective_spawn(p, BENCHMARK_TASK, 0, 0);

  // request shutdown once that task is complete
  r.shutdown(e);

  // now sleep this thread until that shutdown actually happens
  r.wait_for_shutdown();

  return 0;
}
