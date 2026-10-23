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

// Tests batched delivery of profiling responses
// (ProfilingRequest::set_batched_delivery): many short tasks on several
// processors ask for responses on a processor group or on a single
// processor, mixed with requests that are not batched; some responses are
// slow, so deliveries run out of their time slice and hand over; and some
// response tasks make further batched requests to the queue they are being
// delivered from. Every response must arrive exactly once, intact.

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "realm.h"
#include "realm/profiling.h"

#include "osdep.h"

using namespace Realm;
using namespace Realm::ProfilingMeasurements;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  CHILD_TASK,
  RESPONSE_TASK,
};

struct ResponseArgs {
  int index;
  int magic;
  int chain_index; // >= 0: the response task makes another batched request
  Processor target;
};

static const int MAGIC = 0x5eed1e55;
static const int NUM_TASKS = 20000;
static const int CHAIN_EVERY = 199; // every so many batched tasks chain
static const int MAX_CHAINED = NUM_TASKS / CHAIN_EVERY + 1;
static const int TOTAL = NUM_TASKS + MAX_CHAINED;

static std::atomic<int> seen[TOTAL];
static std::atomic<int> received(0);
static Processor chain_proc;

static void child_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{}

static void add_request(ProfilingRequestSet &prs, int index, Processor target,
                        bool batched, int chain_index)
{
  ResponseArgs ra;
  memset(&ra, 0, sizeof(ra));
  ra.index = index;
  ra.magic = MAGIC;
  ra.chain_index = chain_index;
  ra.target = target;
  ProfilingRequest &pr = prs.add_request(target, RESPONSE_TASK, &ra, sizeof(ra));
  pr.add_measurement<OperationTimeline>()
      .add_measurement<OperationProcessorUsage>()
      .add_measurement<OperationFinishEvent>();
  if(batched)
    pr.set_batched_delivery();
}

static void response_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(ResponseArgs));
  ResponseArgs ra;
  memcpy(&ra, resp.user_data(), sizeof(ra));
  if((ra.magic != MAGIC) || (ra.index < 0) || (ra.index >= TOTAL)) {
    log_app.fatal() << "corrupt response";
    abort();
  }
  OperationTimeline tl;
  OperationProcessorUsage pu;
  OperationFinishEvent fe;
  if(!resp.get_measurement(tl) || !resp.get_measurement(pu) ||
     !resp.get_measurement(fe) || !tl.is_valid() || !fe.finish_event.exists()) {
    log_app.fatal() << "missing measurements in response " << ra.index;
    abort();
  }
  // runs inside a task of its own or inside a delivery task
  assert(Processor::get_current_finish_event().exists());
  if(seen[ra.index].fetch_add(1) != 0) {
    log_app.fatal() << "duplicate response " << ra.index;
    abort();
  }
  // some responses are slow, so that deliveries use up their time slice
  if((ra.index % 97) == 0) {
    long long t_end = Clock::current_time_in_nanoseconds() + 30000;
    while(Clock::current_time_in_nanoseconds() < t_end) {
    }
  }
  if(ra.chain_index >= 0) {
    ProfilingRequestSet prs;
    add_request(prs, NUM_TASKS + ra.chain_index, ra.target, true /*batched*/, -1);
    chain_proc.spawn(CHILD_TASK, nullptr, 0, prs);
  }
  received.fetch_add(1);
}

static void top_level_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  std::vector<Processor> cpus;
  Machine::ProcessorQuery q(Machine::get_machine());
  q.local_address_space().only_kind(Processor::LOC_PROC);
  for(Processor c : q)
    if(c != p)
      cpus.push_back(c);
  if(cpus.size() < 2) {
    log_app.fatal() << "need at least 3 CPUs";
    abort();
  }
  chain_proc = cpus[0];
  // responses go to a group of every CPU (as Legion's profiler's do) or to
  //  one CPU
  std::vector<Processor> members(cpus);
  members.push_back(p);
  Processor group = ProcessorGroup::create_group(members);
  for(int i = 0; i < TOTAL; i++)
    seen[i].store(0);

  int chains = 0;
  for(int i = 0; i < NUM_TASKS; i++) {
    const Processor target = ((i % 3) == 0) ? cpus.back() : group;
    const bool batched = ((i % 5) != 0); // one in five is not batched
    const int chain_index = (batched && ((i % CHAIN_EVERY) == 0)) ? chains++ : -1;
    ProfilingRequestSet prs;
    add_request(prs, i, target, batched, chain_index);
    cpus[i % cpus.size()].spawn(CHILD_TASK, nullptr, 0, prs);
  }
  assert(chains <= MAX_CHAINED);

  // wait for every response
  const int expected = NUM_TASKS + chains;
  const long long t_limit = Clock::current_time_in_nanoseconds() + 120000000000LL;
  while(received.load() < expected) {
    if(Clock::current_time_in_nanoseconds() > t_limit) {
      log_app.fatal() << "timed out: " << received.load() << " of " << expected
                      << " responses";
      abort();
    }
    usleep(1000);
  }
  for(int i = 0; i < TOTAL; i++) {
    const int want = (i < expected) ? 1 : 0;
    if(seen[i].load() != want) {
      log_app.fatal() << "response " << i << " seen " << seen[i].load() << " times";
      abort();
    }
  }
  log_app.print() << "all " << expected << " responses received exactly once ("
                  << chains << " from chained requests)";
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);
  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, CHILD_TASK,
                                   CodeDescriptor(child_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, RESPONSE_TASK,
                                   CodeDescriptor(response_task), ProfilingRequestSet())
      .external_wait();
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e);
  return rt.wait_for_shutdown();
}
