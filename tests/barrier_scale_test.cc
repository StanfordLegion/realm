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

// The DEVIATION DRIVER for the scalable-barrier protocols
// (tla/SCALE_TEST_PLAN.md).  The existing barrier tests exercise steady state
// only - none of the deviation machinery (ARRIVAL_PROTOCOL rules 2/3/5/8/10)
// ever executes under them, which the counters prove.  Each phase here forces
// one protocol case; the exercise-proof is the counter dump each barrier
// emits when it is destroyed (build with -DREALM_LOG_LEVEL=INFO and run with
// -level barrier=2 to see it) plus the protocol's own info-level logs.
//
// Every rank computes the same seeded schedule, so the expected arrival total
// per generation is known everywhere and conservation is maintained by
// construction: a phase never changes the total (except ALTER, which changes
// it through the API whose persistence it is testing).
//
// The oracle here is local (v1): in-order exactly-once wakes per rank, poison
// observed exactly where scripted, and phase completion (a hang is caught by
// the ctest timeout).  The cross-rank no-early-trigger ledger of
// SCALE_TEST_PLAN.md section 2 is a follow-up.

#include <assert.h>

#include "realm.h"
#include "osdep.h"
#include "realm/cmdline.h"
#include "realm/network.h"

using namespace Realm;

Logger log_app("app");

enum
{
  MAIN_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WORKER_TASK,
};

enum Phase
{
  P_STEADY = 0, // rule 1: plan learned once, then pure aggregation
  P_OVER,       // rule 2: rotating over-arrival (+1 / -1 elsewhere)
  P_OUTSIDER,   // rule 3: rotating outsider (a rank with 0 gets 1)
  P_RUNAHEAD,   // rule 10.1/10.4: arrivals sprint ahead of waits
  P_CHURN,      // rules 5/10.2/10.3: pattern rotates EVERY generation
  P_ALTER,      // rules 8/9: persistent count change mid-phase
  P_SPLITWAIT,  // notification: waiters disjoint from arrivers + far-future
  P_POISON,     // Q4: poisoned arrival poisons exactly its generation
  P_CHAOS,      // everything above, seeded random
  P_STEP,       // ONE persistent pattern shift at mid-phase: the realistic
                //  Legion case.  Every post-shift generation deviates from the
                //  stale plan the SAME way, so this is the probe for whether
                //  the owner can ever re-learn a plan whose deviation is
                //  discovered mid-generation (the partial-evidence question).
  P_NUM_PHASES
};

static const char *phase_names[P_NUM_PHASES] = {"steady", "over",  "outsider", "runahead",
                                                "churn",  "alter", "split",    "poison",
                                                "chaos",  "step"};

namespace TestConfig {
  int gens = 32;       // generations per phase
  int runahead = 4;    // max generations arrivals may lead waits (P_RUNAHEAD)
  int phase_mask = -1; // bitmask of phases to run, -1 = all
  int seed = 12345;
}; // namespace TestConfig

struct WorkerArgs {
  Barrier barrier;
  int phase;
  int rank; // this worker's index == address space
  int nranks;
  int gens;
  int seed;
};

// deterministic per-(seed, phase, gen) mix - every rank computes the same value
static uint64_t mix(uint64_t a, uint64_t b, uint64_t c)
{
  uint64_t x = a * 0x9E3779B97F4A7C15ull + b * 0xBF58476D1CE4E5B9ull +
               c * 0x94D049BB133111EBull + 0x2545F4914F6CDD1Dull;
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  return x;
}

// How many arrivals rank r owes for generation g of this phase.  The sum over
// r is ALWAYS nranks (the create-time expected count), except in P_ALTER where
// the alteration adds one from its generation onward - through the API, which
// is the point.
static int arrivals_for(int phase, int seed, int nranks, int gens, int g, int r)
{
  if(nranks == 1) {
    return 1; // every deviation needs a second rank; degenerate to steady
  }
  switch(phase) {
  case P_OVER:
  {
    // odd generations: one rank +1, its neighbour -1
    if(g % 2 == 0)
      return 1;
    int over = (int)(mix(seed, phase, g) % nranks);
    int under = (over + 1) % nranks;
    if(r == over)
      return 2;
    if(r == under)
      return 0;
    return 1;
  }
  case P_OUTSIDER:
  {
    // rank (g%nranks) is silent for a WINDOW of generations, then returns as
    //  an outsider while its neighbour goes quiet - the plan learned during
    //  the window did not predict it
    int quiet = (g / 4) % nranks;    // rotates every 4 generations
    int loud = (quiet + 1) % nranks; // carries the quiet rank's arrival
    if(r == quiet)
      return 0;
    if(r == loud)
      return 2;
    return 1;
  }
  case P_CHURN:
  {
    // a random pair swaps roles EVERY generation: constant plan rebuilds,
    //  which is what makes the park/dead-plan race windows probable
    int a = (int)(mix(seed, phase * 977, g) % nranks);
    int b = (a + 1 + (int)(mix(seed, phase, g * 31) % (nranks - 1))) % nranks;
    if(r == a)
      return 2;
    if(r == b)
      return 0;
    return 1;
  }
  case P_ALTER:
    return 1; // the altering rank adds its extra arrival explicitly (below)
  case P_SPLITWAIT:
  {
    // first half of the ranks carry all arrivals; second half only waits
    int arrivers = (nranks + 1) / 2;
    if(r >= arrivers)
      return 0;
    // distribute nranks arrivals over 'arrivers' ranks
    int base = nranks / arrivers;
    int extra = nranks % arrivers;
    return base + ((r < extra) ? 1 : 0);
  }
  case P_STEP:
  {
    // pattern A (everyone 1) for the first half; pattern B (rank 0 silent,
    //  rank 1 doubled) for the second - constant within each half, total
    //  conserved.  A healthy plan lifecycle re-learns ONCE at the shift and
    //  aggregates thereafter; a broken one declines and stays eager forever.
    if(g < gens / 2)
      return 1;
    if(r == 0)
      return 0;
    if(r == 1)
      return 2;
    return 1;
  }
  case P_CHAOS:
  {
    // random redistribution with the total conserved: walk the ranks with a
    //  seeded shuffle of +1/-1 pairs
    int v = 1;
    uint64_t h = mix(seed, phase, g);
    int plus = (int)(h % nranks);
    int minus = (plus + 1 + (int)((h >> 16) % (nranks - 1))) % nranks;
    if(r == plus)
      v += 1;
    if(r == minus)
      v -= 1;
    return v;
  }
  case P_STEADY:
  case P_RUNAHEAD:
  case P_POISON:
  default:
    return 1;
  }
}

static bool poisoned_gen(int phase, int gens, int g)
{
  // one poisoned generation, mid-phase
  return (phase == P_POISON) && (g == gens / 2);
}

static void worker_task(const void *args, size_t arglen, const void *userdata,
                        size_t userlen, Processor p)
{
  const WorkerArgs wa = *(const WorkerArgs *)args;
  const int N = wa.nranks, G = wa.gens, R = wa.rank;

  // two independent walks over the generation chain: 'ab' is where arrivals
  //  are issued, 'wb' where waits happen.  In P_RUNAHEAD 'ab' leads 'wb' by a
  //  rank-staggered lag, which is what creates run-ahead: arrivals on
  //  generations the previous plan is still current for.
  Barrier ab = wa.barrier;
  int lag = 0;
  if(wa.phase == P_RUNAHEAD && N > 1) {
    lag = 1 + (R % TestConfig::runahead);
  }

  // P_ALTER: rank 0 alters +1 at the midpoint and owes one extra arrival per
  //  generation from then on - the PERSISTENCE is the thing under test
  //  (event.h:271, the pre-existing bug C6 fixed).
  const int alter_at = G / 2;
  const bool i_alter = (wa.phase == P_ALTER) && (R == 0) && (N > 1);

  // arrivals first walk
  std::vector<Barrier> wait_handles;
  wait_handles.reserve(G);
  for(int g = 0; g < G; g++) {
    wait_handles.push_back(ab);

    int count = arrivals_for(wa.phase, wa.seed, N, G, g, R);
    if(i_alter && (g == alter_at)) {
      // the contract: hold an unissued arrival from the pre-alteration count
      //  while altering, then use the returned (timestamped) handle
      assert(count > 0);
      ab = ab.alter_arrival_count(1);
      wait_handles.back() = ab; // wait on the timestamped branch too
    }
    if(i_alter && (g >= alter_at)) {
      count += 1; // persistent: one extra every generation from the alteration on
    }

    if(poisoned_gen(wa.phase, G, g) && (R == (g % N)) && (count > 0)) {
      // one arrival rides a poisoned precondition; Q4 says the generation is
      //  poisoned for every waiter
      UserEvent ue = UserEvent::create_user_event();
      ab.arrive(1, ue);
      ue.cancel(); // poisons the precondition
      count -= 1;
    }
    if(count > 0) {
      ab.arrive(count);
    }

    // waits: everyone waits every generation, except the split phase (only
    //  the non-arrivers wait) and run-ahead (waits lag behind arrivals)
    bool do_wait = true;
    if(wa.phase == P_SPLITWAIT && N > 1) {
      do_wait = (R >= (N + 1) / 2);
    }
    if(do_wait && (g >= lag)) {
      const int wg = g - lag;
      bool poisoned = false;
      wait_handles[wg].wait_faultaware(poisoned);
      const bool expect = poisoned_gen(wa.phase, G, wg);
      if(poisoned != expect) {
        log_app.fatal() << "poison mismatch: phase=" << phase_names[wa.phase]
                        << " rank=" << R << " gen=" << wg << " got=" << poisoned
                        << " want=" << expect;
        abort();
      }
    }

    ab = ab.advance_barrier();
    assert(ab.exists());
  }

  // drain the lagged waits
  if(lag > 0) {
    for(int wg = G - lag; wg < G; wg++) {
      bool poisoned = false;
      wait_handles[wg].wait_faultaware(poisoned);
      assert(poisoned == poisoned_gen(wa.phase, G, wg));
    }
  }

  // P_SPLITWAIT far-future check: the last rank ALSO waits on the final
  //  generation up front next phase - modelled here simply by re-waiting the
  //  last handle (already triggered: exercises the has_triggered fast path)
  bool p2 = false;
  wait_handles[G - 1].wait_faultaware(p2);
  assert(p2 == poisoned_gen(wa.phase, G, G - 1));
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
  log_app.print() << "barrier_scale_test: ranks=" << N << " gens=" << TestConfig::gens
                  << " seed=" << TestConfig::seed;

  for(int phase = 0; phase < P_NUM_PHASES; phase++) {
    if((TestConfig::phase_mask >= 0) && !((TestConfig::phase_mask >> phase) & 1)) {
      continue;
    }
    // expected per generation == N by construction of arrivals_for()
    Barrier b = Barrier::create_barrier(N);

    std::vector<Event> done;
    int rank = 0;
    for(std::map<AddressSpace, Processor>::iterator it = per_space.begin();
        it != per_space.end(); ++it, ++rank) {
      WorkerArgs wa;
      wa.barrier = b;
      wa.phase = phase;
      wa.rank = rank;
      wa.nranks = N;
      wa.gens = TestConfig::gens;
      wa.seed = TestConfig::seed;
      done.push_back(it->second.spawn(WORKER_TASK, &wa, sizeof(wa)));
    }
    Event::merge_events(done).wait();

    // destruction is what fires the owner-side counter dump ("destroyed") -
    //  the exercise-proof of SCALE_TEST_PLAN section 1
    b.destroy_barrier();
    log_app.print() << "phase " << phase_names[phase] << ": PASS";
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  CommandLineParser cp;
  cp.add_option_int("-gens", TestConfig::gens);
  cp.add_option_int("-seed", TestConfig::seed);
  cp.add_option_int("-runahead", TestConfig::runahead);
  cp.add_option_int("-phases", TestConfig::phase_mask);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

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
