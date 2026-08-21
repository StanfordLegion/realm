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

// Barrier performance benchmark (tla/barrier/SCALE_TEST_PLAN.md section 4, the
// T4 A/B comparison).  Uses ONLY the public Barrier API so the identical
// source compiles against both this branch and main - A/B is by build, never
// by runtime toggle.  Results go to stdout via printf: the compile-time log
// floor defaults to WARNING, and a benchmark's output must not depend on the
// logging configuration.
//
// Two timing modes per repetition, each on a fresh barrier:
//
//   serial - arrive -> wait -> advance, fully synchronous.  The per-generation
//     time is one complete arrive/trigger/notify/wake round, measured locally
//     on rank 0, so it needs no cross-rank clock.  The first generations of
//     rep 0 are printed individually (the ADOPTION CURVE): on the scalable
//     path the eager->planned transition is visible as a step down, and its
//     absence means a plan silently failed to install.
//
//   pipe - arrivals run ahead of waits by a fixed window.  Once the plan has
//     converged every generation is pure tree aggregation, so this is the
//     repeated-pattern fast path under test.  Reported as generations/second
//     over the post-warmup span.
//
// Two arrival patterns, fixed across generations (a REPEATED pattern is the
// point - the plan must be learned once and then only aggregated):
//
//   uniform - every rank arrives once per generation.
//   half    - the lower half of the ranks arrive twice, the upper half only
//     wait.  This puts real quotas in the plan and disjoint arriver/waiter
//     sets in the notification tree.
//
// Wall-clock numbers say whether it is fast; the fast-path PROOF is the
// counter cross-check (new build only): rerun with '-level barrier=2' and
// check reports_received at the owner is ~plan_radix per generation - flat in
// N - with plan_rebuilds=1 and flush_episodes confined to startup.

#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <vector>

#include "realm.h"
#include "realm/cmdline.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WORKER_TASK,
};

enum Mode
{
  M_SERIAL = 0,
  M_PIPE,
};

enum Pattern
{
  PAT_UNIFORM = 0,
  PAT_HALF,
  PAT_NUM
};

static const char *pattern_names[PAT_NUM] = {"uniform", "half"};

namespace TestConfig {
  int sgens = 256;  // generations per serial rep
  int pgens = 1024; // generations per pipe rep
  int window = 32;  // run-ahead window for pipe mode
  int warmup = 16;  // generations excluded from steady-state stats
  int reps = 3;     // repetitions (fresh barrier each) per mode+pattern
  int adopt = 32;   // adoption-curve generations printed from serial rep 0
  int pattern = -1; // -1 = both, else a single Pattern
}; // namespace TestConfig

struct WorkerArgs {
  Barrier barrier;
  int mode;
  int pattern;
  int gens;
  int rep;
  int rank;
  int nranks;
};

// arrivals rank r owes each generation; the sum over r is ALWAYS nranks
static int count_for(int pattern, int nranks, int r)
{
  if((pattern == PAT_UNIFORM) || (nranks < 2)) {
    return 1;
  }
  // PAT_HALF: lower half carries doubled arrivals, upper half only waits
  if(r < (nranks / 2)) {
    return 2;
  }
  if(((nranks % 2) != 0) && (r == (nranks - 1))) {
    return 1; // odd rank count: the last rank tops the total up to nranks
  }
  return 0;
}

static void worker_task(const void *args, size_t arglen, const void *userdata,
                        size_t userlen, Processor p)
{
  const WorkerArgs wa = *(const WorkerArgs *)args;
  const int N = wa.nranks, G = wa.gens, R = wa.rank;
  const int count = count_for(wa.pattern, N, R);
  const bool timer = (R == 0); // per-gen rounds are local, any rank would do

  Barrier b = wa.barrier;

  if(wa.mode == M_SERIAL) {
    std::vector<double> gen_us;
    if(timer) {
      gen_us.reserve(G);
    }
    for(int g = 0; g < G; g++) {
      long long t0 = timer ? Clock::current_time_in_nanoseconds() : 0;
      if(count > 0) {
        b.arrive(count);
      }
      b.wait();
      if(timer) {
        long long t1 = Clock::current_time_in_nanoseconds();
        gen_us.push_back((double)(t1 - t0) * 1e-3);
      }
      b = b.advance_barrier();
      assert(b.exists());
    }
    if(timer) {
      if((wa.rep == 0) && (TestConfig::adopt > 0)) {
        printf("adoption_us: pattern=%s", pattern_names[wa.pattern]);
        for(int g = 0; (g < TestConfig::adopt) && (g < G); g++) {
          printf(" %.1f", gen_us[g]);
        }
        printf("\n");
      }
      std::vector<double> steady(gen_us.begin() + std::min(TestConfig::warmup, G - 1),
                                 gen_us.end());
      std::sort(steady.begin(), steady.end());
      const size_t n = steady.size();
      // the p99/max tail matters as much as the median: a rare multi-ms stall
      //  barely moves p95 but eats hundreds of generations of pipe throughput
      printf("serial: pattern=%s rep=%d gens=%d median_us=%.2f p95_us=%.2f "
             "p99_us=%.2f max_us=%.2f min_us=%.2f\n",
             pattern_names[wa.pattern], wa.rep, G, steady[n / 2],
             steady[(size_t)(0.95 * (n - 1))], steady[(size_t)(0.99 * (n - 1))],
             steady[n - 1], steady[0]);
      fflush(stdout);
    }
    return;
  }

  // M_PIPE.  Generation 1 is the start-sync: every rank arrives AND waits on
  //  it, so the measured span begins with all ranks released together.
  std::vector<Barrier> handles;
  handles.reserve(G);
  long long t_start = 0, t_warm = 0, t_end = 0;
  const int W = TestConfig::window;
  const int warm = std::min(TestConfig::warmup, G - 1);
  for(int g = 0; g < G; g++) {
    handles.push_back(b);
    if(g == 0) {
      if(count > 0) {
        b.arrive(count);
      }
      b.wait();
      t_start = Clock::current_time_in_nanoseconds();
    } else {
      if(count > 0) {
        b.arrive(count);
      }
      if(g >= W) {
        handles[g - W].wait();
        if((g - W) == warm) {
          t_warm = Clock::current_time_in_nanoseconds();
        }
      }
    }
    b = b.advance_barrier();
    assert(b.exists());
  }
  // drain the windowed waits
  for(int wg = std::max(1, G - W); wg < G; wg++) {
    handles[wg].wait();
    if(wg == warm) {
      t_warm = Clock::current_time_in_nanoseconds();
    }
  }
  t_end = Clock::current_time_in_nanoseconds();

  if(timer) {
    // gens 1..G-1 complete inside [t_start, t_end]; gens warm+1..G-1 inside
    //  [t_warm, t_end]
    double total = (double)(G - 1) / ((double)(t_end - t_start) * 1e-9);
    double steady = (double)(G - 1 - warm) / ((double)(t_end - t_warm) * 1e-9);
    printf("pipe: pattern=%s rep=%d gens=%d window=%d gens_per_sec=%.0f "
           "total_gens_per_sec=%.0f\n",
           pattern_names[wa.pattern], wa.rep, G, W, steady, total);
    fflush(stdout);
  }
}

static void main_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  // one worker processor per address space
  std::map<AddressSpace, Processor> per_space;
  Machine::ProcessorQuery pq =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it) {
    AddressSpace s = (*it).address_space();
    if(per_space.find(s) == per_space.end()) {
      per_space[s] = *it;
    }
  }
  const int N = (int)per_space.size();
  printf("barrier_bench: ranks=%d sgens=%d pgens=%d window=%d warmup=%d reps=%d\n", N,
         TestConfig::sgens, TestConfig::pgens, TestConfig::window, TestConfig::warmup,
         TestConfig::reps);
  fflush(stdout);

  for(int pat = 0; pat < PAT_NUM; pat++) {
    if((TestConfig::pattern >= 0) && (pat != TestConfig::pattern)) {
      continue;
    }
    for(int mode = 0; mode <= M_PIPE; mode++) {
      for(int rep = 0; rep < TestConfig::reps; rep++) {
        Barrier bar = Barrier::create_barrier(N);
        std::vector<Event> done;
        int rank = 0;
        for(std::map<AddressSpace, Processor>::iterator it = per_space.begin();
            it != per_space.end(); ++it, ++rank) {
          WorkerArgs wa;
          wa.barrier = bar;
          wa.mode = mode;
          wa.pattern = pat;
          wa.gens = (mode == M_SERIAL) ? TestConfig::sgens : TestConfig::pgens;
          wa.rep = rep;
          wa.rank = rank;
          wa.nranks = N;
          done.push_back(it->second.spawn(WORKER_TASK, &wa, sizeof(wa)));
        }
        Event::merge_events(done).wait();
        bar.destroy_barrier();
      }
    }
  }
  printf("barrier_bench: DONE\n");
  fflush(stdout);
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  CommandLineParser cp;
  cp.add_option_int("-sgens", TestConfig::sgens);
  cp.add_option_int("-pgens", TestConfig::pgens);
  cp.add_option_int("-window", TestConfig::window);
  cp.add_option_int("-warmup", TestConfig::warmup);
  cp.add_option_int("-reps", TestConfig::reps);
  cp.add_option_int("-adopt", TestConfig::adopt);
  cp.add_option_int("-pattern", TestConfig::pattern);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);
  // the pipe-mode warm timestamp is recorded at the wait of generation
  //  'warmup', which the windowed loop only reaches if this holds
  assert(TestConfig::pgens > (TestConfig::window + TestConfig::warmup));

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, MAIN_TASK,
                                   CodeDescriptor(main_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, WORKER_TASK,
                                   CodeDescriptor(worker_task), ProfilingRequestSet())
      .external_wait();

  Event e = rt.collective_spawn(p, MAIN_TASK, 0, 0);
  rt.shutdown(e);
  return rt.wait_for_shutdown();
}
