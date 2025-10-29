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
#include "realm/id.h"
#include "realm/cmdline.h"
#include "realm/network.h"

#include <deque>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  VERIFY_TASK,
  ALLOC_PROF_TASK,
  ALLOC_INST_TASK,
  INST_STATUS_PROF_TASK,
  MUSAGE_PROF_TASK,
};

struct VerifyArgs {
  RegionInstance inst = RegionInstance::NO_INST;
  Rect<1> bounds;
  int expected = 0xDEADBEEF;
};

struct ProfMusageResult {
  UserEvent done;
  int expected_usage;
};

struct ProfTimelResult {
  UserEvent done;
};

struct ProfAllocResult {
  UserEvent done;
};

static void musage_profiling_task(const void *args, size_t arglen, const void *userdata,
                                  size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ProfMusageResult));
  const ProfMusageResult *result =
      static_cast<const ProfMusageResult *>(resp.user_data());
  ProfilingMeasurements::InstanceMemoryUsage memory_usage;
  assert((resp.get_measurement(memory_usage)));

  // TODO(apryakhin@): Verify timestamps
  assert(memory_usage.bytes != 0);
  assert(memory_usage.instance != RegionInstance::NO_INST);
  log_app.error("musage - Expected: %d, Used: %zd", result->expected_usage, memory_usage.bytes);
  assert(result->expected_usage == memory_usage.bytes);

  result->done.trigger();
}

static void inst_profiling_task(const void *args, size_t arglen, const void *userdata,
                                size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ProfTimelResult));
  const ProfTimelResult *result = static_cast<const ProfTimelResult *>(resp.user_data());
  ProfilingMeasurements::InstanceTimeline inst_time;
  assert((resp.get_measurement(inst_time)));

  // TODO(apryakhin@): Verify timestamps
  assert(inst_time.create_time != 0);
  assert(inst_time.ready_time != 0);
  assert(inst_time.instance != 0);  // TODO: add instance correlation

  log_app.error() << "timel " << inst_time.instance;

  result->done.trigger();
}

static void inst_status_profiling_task(const void *args, size_t arglen,
                                       const void *userdata, size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ProfAllocResult));
  const ProfAllocResult *result = static_cast<const ProfAllocResult *>(resp.user_data());
  ProfilingMeasurements::InstanceStatus inst_status;
  assert((resp.get_measurement(inst_status)));
  assert(inst_status.error_code == 0); // FAILURE CASE
  assert(inst_status.inst != 0); // TODO(cperry): add instance correlation

  result->done.trigger();
}

static void alloc_profiling_task(const void *args, size_t arglen, const void *userdata,
                                 size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ProfAllocResult));
  const ProfAllocResult *result = static_cast<const ProfAllocResult *>(resp.user_data());
  ProfilingMeasurements::InstanceAllocResult inst_alloc;
  assert((resp.get_measurement(inst_alloc)));
  assert(inst_alloc.success); // FAILURE CASE

  result->done.trigger();
}

static void verify_task(const void *args, size_t arglen, const void *userdata,
                        size_t userlen, Processor p)
{
  const VerifyArgs &vargs = *static_cast<const VerifyArgs *>(args);
  std::vector<int> values(vargs.bounds.volume(), 0);
  vargs.inst.read_untyped(0, values.data(), sizeof(int) * values.size());
  for(size_t i = 0; i < values.size(); i++) {
    assert(values[i] == vargs.expected && "Unexpected value");
  }
  log_app.error() << "Verified inst " << vargs.inst;
}

template <int N>
InstanceLayoutGeneric *create_layout(Rect<N> bounds)
{
  InstanceLayoutConstraints ilc({sizeof(int)}, 1);
  int dim_order[N] = {0};
  InstanceLayoutGeneric *ilg =
      InstanceLayoutGeneric::choose_instance_layout<N, int>(bounds, ilc, dim_order);
  return ilg;
}

static Event setup_prs(ProfilingRequestSet &prs, Processor tgt, int expected_sz)
{
  UserEvent musage_e = UserEvent::create_user_event();
  UserEvent inst_prof_e = UserEvent::create_user_event();
  UserEvent inst_status_prof_e = UserEvent::create_user_event();
  UserEvent alloc_prof_e = UserEvent::create_user_event();
  {
    ProfMusageResult result;
    result.done = musage_e;
    result.expected_usage = expected_sz;
    prs.add_request(tgt, MUSAGE_PROF_TASK, &result, sizeof(result))
        .add_measurement<ProfilingMeasurements::InstanceMemoryUsage>();
  }
  {
    ProfTimelResult result;
    result.done = inst_prof_e;
    prs.add_request(tgt, ALLOC_INST_TASK, &result, sizeof(result))
        .add_measurement<ProfilingMeasurements::InstanceTimeline>();
  }
  {
    ProfAllocResult result;
    result.done = inst_status_prof_e;
    prs.add_request(tgt, INST_STATUS_PROF_TASK, &result, sizeof(result))
        .add_measurement<ProfilingMeasurements::InstanceStatus>();
  }
  {
    ProfAllocResult result;
    result.done = alloc_prof_e;
    prs.add_request(tgt, ALLOC_PROF_TASK, &result, sizeof(result))
        .add_measurement<ProfilingMeasurements::InstanceAllocResult>();
  }

  return Event::merge_events(musage_e, inst_prof_e, inst_status_prof_e, alloc_prof_e);
}

static void run_test(Processor proc, RegionInstance inst, Rect<1> bounds, int min_chunk_sz, int test_value, bool external_inst = false)
{
  UserEvent trigger = UserEvent::create_user_event();
  Event wait_on = trigger;
  Event prof_event = Event::NO_EVENT;
  std::vector<RegionInstance> instances(1, inst);
  std::vector<Event> prof_events;

  Processor verify_proc = Machine::ProcessorQuery(Machine::get_machine())
                              .same_address_space_as(inst.get_location())
                              .first();

  for(size_t chunk_sz = bounds.hi >> 1; chunk_sz > min_chunk_sz;
      chunk_sz >>= 1) {

    std::vector<RegionInstance> split_instances(instances.size() * 2, RegionInstance::NO_INST);
    Event next_wait_on = wait_on;
    for(size_t i = 0; i < instances.size(); i++) {
      // Split each instance in half.
      const int start = 2 * i * chunk_sz;
      const int end = 2 * (i + 1) * chunk_sz;
      const InstanceLayoutGeneric *layouts[] = {
          create_layout(Rect<1>(start, start + chunk_sz - 1)),
          create_layout(Rect<1>(start + chunk_sz, end - 1))};

      Realm::ProfilingRequestSet prs[2];
      prof_events.push_back(setup_prs(prs[0], proc, chunk_sz * sizeof(int)));
      prof_events.push_back(setup_prs(prs[1], proc, chunk_sz * sizeof(int)));

      // Wait to redistrict until the parent instance's operations are complete
      Event redistrict_e = instances[i].redistrict(split_instances.data() + 2 * i, layouts, 2, prs,
                                        wait_on);
      // Clean up the layouts
      delete layouts[0];
      delete layouts[1];

      // Verify the data hasn't changed in the new instances
      Event fin_verif[2];
      for(size_t j = 0; j < 2; j++) {
        VerifyArgs vargs{split_instances[2 * i + j],
                         Rect<1>(chunk_sz * j, chunk_sz * (j + 1) - 1), test_value};
        log_app.error() << "Testing inst " << split_instances[2 * i + j];
        fin_verif[j] = verify_proc.spawn(VERIFY_TASK, &vargs, sizeof(vargs), redistrict_e);
      }
      // Accumulate all the verifications for these split instances to wait on next
      next_wait_on = Event::merge_events(next_wait_on, fin_verif[0], fin_verif[1]);
    }
    wait_on = next_wait_on;
    // Update the set of instances to split for the next iteration
    instances.swap(split_instances);
  }
  for (RegionInstance inst : instances) {
    inst.destroy(wait_on);
  }

  trigger.trigger();

  bool poisoned = false;
  wait_on.wait_faultaware(poisoned);
  assert(!poisoned);
  Event::merge_events(prof_events).wait_faultaware(poisoned);
  assert(!poisoned);
}

static void main_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  Machine::MemoryQuery mq(Machine::get_machine());
  const size_t num = 1048576;
  mq.has_capacity(num * sizeof(int));
  for (Memory m : mq) {
    RegionInstance inst;
    Rect<1> bounds(0, num);
    std::vector<int> test_value(bounds.volume(), 0xDEADBEEF + m.id);
    std::vector<size_t> fields{sizeof(int)};
    ProfilingRequestSet prs;
    Event prof_e = setup_prs(prs, p, bounds.volume() * sizeof(int));

    RegionInstance::create_instance(inst, m, bounds, fields, 1, prs).wait();
    inst.write_untyped(0, test_value.data(), sizeof(int) * test_value.size());
    log_app.error() << "==== Testing memory: " << m << " ====";
    run_test(p, inst, bounds, 1024, 0xDEADBEEF + m.id);
    prof_e.wait();
    log_app.error("============");
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  CommandLineParser cp;

  rt.init(&argc, &argv);

  // cp.add_option_int("-i", num_iterations);
  // cp.add_option_int("-needs_oom", needs_oom);
  // cp.add_option_int("-needs_ext", needs_ext);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  rt.register_task(MAIN_TASK, main_task);
  rt.register_task(VERIFY_TASK, verify_task);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   ALLOC_PROF_TASK, CodeDescriptor(alloc_profiling_task),
                                   ProfilingRequestSet())
      .wait();

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/,
                                   ALLOC_INST_TASK, CodeDescriptor(inst_profiling_task),
                                   ProfilingRequestSet())
      .wait();

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/, MUSAGE_PROF_TASK,
      CodeDescriptor(musage_profiling_task), ProfilingRequestSet())
      .wait();

  Processor::register_task_by_kind(
      Processor::LOC_PROC, false /*!global*/, INST_STATUS_PROF_TASK,
      CodeDescriptor(inst_status_profiling_task), ProfilingRequestSet())
      .wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);
  return rt.wait_for_shutdown();
}
