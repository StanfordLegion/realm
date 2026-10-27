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

// Guards the test harness itself: when the integration tests are launched on N ranks
//  (tests/CMakeLists.txt sets REALM_TEST_EXPECTED_RANKS to the launcher's rank count),
//  this fails unless Realm really sees N ranks.  Without it, a launcher that silently
//  stops applying - an old CTest that ignores the TEST_LAUNCHER property, a network
//  module that falls back to loopback, a launcher that starts N independent
//  single-rank jobs - turns every "multi-rank" integration test into a single-rank one
//  while the suite stays green.

#include "realm.h"

#include <cstdio>
#include <cstdlib>

using namespace Realm;

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

static int expected_ranks = 0;
static int observed_ranks = 0;

static void top_level_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  // count the address spaces that have processors, which is what the tests that
  //  rely on remote ranks actually need
  observed_ranks = 0;
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(Processor::LOC_PROC);
  AddressSpace max_space = 0;
  bool any = false;
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
    if(!any || ((*it).address_space() > max_space))
      max_space = (*it).address_space();
    any = true;
  }
  observed_ranks = (any ? int(max_space) + 1 : 0);
  printf("rank_count_check: expected %d rank(s), runtime sees %d\n", expected_ranks,
         observed_ranks);
  fflush(stdout);
}

int main(int argc, char **argv)
{
  const char *e = getenv("REALM_TEST_EXPECTED_RANKS");
  expected_ranks = ((e != nullptr) && (*e != '\0')) ? atoi(e) : 1;

  Runtime rt;
  rt.init(&argc, &argv);

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task), ProfilingRequestSet())
      .external_wait();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  Event e2 = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e2);
  int ret = rt.wait_for_shutdown();
  if(ret != 0)
    return ret;

  // only the rank that ran the task knows the count; the others report success and
  //  leave the verdict to it
  if(observed_ranks == 0)
    return 0;
  if(observed_ranks != expected_ranks) {
    fprintf(stderr,
            "rank_count_check: FAILED - the test harness asked for %d rank(s) but the "
            "runtime has %d\n",
            expected_ranks, observed_ranks);
    return 1;
  }
  return 0;
}
