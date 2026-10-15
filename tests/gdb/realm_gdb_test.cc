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

// Target program for the realm-gdb.py tests (run_realm_gdb_tests.py).
//
// On a single CPU processor it parks a known set of tasks on user threads:
//   * RGT_NUM_CHAIN tasks, each blocked three calls deep in
//     rgt_chain_level1 -> rgt_chain_level2 -> rgt_chain_level3;
//   * one task blocked at the bottom of an RGT_RECURSE_DEPTH-deep recursion;
//   * one task blocked in rgt_hot_wait, an optimized frame that keeps values
//     in callee-saved registers;
//   * one task blocked in rgt_opt_wait, an optimized frame with values that
//     are optimized out;
//   * the top-level task, waiting for all of the above.
// Then a probe task stops for the debugger (how depends on -mode, below).
// Once released, every task checks its own frame, so if the debugger left
// anything corrupted the program reports FAIL and exits non-zero.
//
// Modes:
//   stop    call realm_gdb_test_stop(), where gdb sets a breakpoint
//   attach  print a READY line and spin until gdb sets realm_gdb_test_release
//   crash   raise SIGSEGV (for kernel-written core files)
//   freeze  raise SIGSEGV with REALM_FREEZE_ON_ERROR=1 in the environment,
//           so realm_freeze() holds the process for gdb to attach to

#include "realm_gdb_test.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(__GNUC__)
#include <alloca.h>
#endif

using namespace Realm;

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  CHAIN_TASK,
  RECURSE_TASK,
  HOT_TASK,
  OPT_TASK,
  PROBE_TASK,
  PARKED_TASK,
};

enum
{
  MODE_STOP,
  MODE_ATTACH,
  MODE_CRASH,
  MODE_FREEZE,
};

extern "C" {
RealmGdbTestState realm_gdb_test_state;
volatile int realm_gdb_test_release = 0;
volatile int realm_gdb_test_hold = 0;
int realm_gdb_test_mode = MODE_STOP;
}

static int attach_timeout_seconds = 300;

// --------------------------------------------------------------------------
// Functions for gdb to call (inferior calls) or stop in
// --------------------------------------------------------------------------

struct RealmGdbTestWidget {
  int id;
  int values[4];

  RGT_NOINLINE int sum() const
  {
    int s = id;
    for(int i = 0; i < 4; i++)
      s += values[i];
    return s;
  }
};

extern "C" {

// gdb puts a breakpoint here in -mode stop.
RGT_NOINLINE void realm_gdb_test_stop(void) { __asm__ __volatile__("" ::: "memory"); }

RGT_NOINLINE long realm_gdb_test_opaque(long x)
{
  __asm__ __volatile__("" : "+r"(x));
  return (x * 2654435761L) ^ 0x5DEECE66DL;
}

RGT_NOINLINE int realm_gdb_test_marker(int index) { return 4242 + 1000 * index; }

RGT_NOINLINE int realm_gdb_test_add(int a, int b)
{
  int sum = a + b;
  // (a breakpoint here sees a and b)
  realm_gdb_test_opaque(sum);
  return sum;
}

RGT_NOINLINE int realm_gdb_test_widget_sum(const RealmGdbTestWidget *w)
{
  return w->sum();
}

// Uses `kb` KiB of stack, to see whose stack an inferior call runs on.
RGT_NOINLINE long realm_gdb_test_stack_hog(long kb)
{
  size_t bytes = size_t(kb) * 1024;
  volatile char *buf = static_cast<volatile char *>(alloca(bytes));
  long sum = 0;
  for(size_t i = 0; i < bytes; i += 512)
    buf[i] = char(i >> 9);
  for(size_t i = 0; i < bytes; i += 512)
    sum += buf[i];
  return sum;
}

// Exercises malloc/stdio inside an inferior call.
RGT_NOINLINE char *realm_gdb_test_describe(int value)
{
  char *s = static_cast<char *>(malloc(64));
  snprintf(s, 64, "value=%d", value);
  return s;
}

// Faults immediately, for inferior calls that are interrupted by a signal.
RGT_NOINLINE int realm_gdb_test_fault(void)
{
  return *static_cast<volatile int *>(nullptr);
}

// A slow inferior call, during which other threads keep running.
RGT_NOINLINE int realm_gdb_test_nap(int usec)
{
  usleep(usec);
  return usec;
}

RGT_NOINLINE void realm_gdb_test_wait_for_debugger(void)
{
  switch(realm_gdb_test_mode) {
  case MODE_STOP:
    realm_gdb_test_stop();
    // A test can keep this thread here when gdb lets it run (e.g. during
    // a function call made in another inferior).
    while(realm_gdb_test_hold)
      usleep(1000);
    break;
  case MODE_ATTACH:
  {
    printf("REALM_GDB_TEST: READY %d\n", int(getpid()));
    fflush(stdout);
    for(long i = 0; !realm_gdb_test_release; i++) {
      if(i >= attach_timeout_seconds * 1000L) {
        printf("REALM_GDB_TEST: FAIL timed out waiting for the debugger\n");
        fflush(stdout);
        _exit(3);
      }
      usleep(1000);
    }
    break;
  }
  case MODE_CRASH:
  case MODE_FREEZE:
    printf("REALM_GDB_TEST: CRASHING %d\n", int(getpid()));
    fflush(stdout);
    raise(SIGSEGV);
    break;
  }
}
}

static void record_failure(const char *what)
{
  printf("REALM_GDB_TEST: FAIL %s\n", what);
  fflush(stdout);
  __sync_fetch_and_add(&realm_gdb_test_state.failures, 1);
}

// --------------------------------------------------------------------------
// Parked tasks
// --------------------------------------------------------------------------

struct ParkedArgs {
  int index;
  UserEvent release;
  Barrier parked;
};

static void park(const ParkedArgs &args)
{
  __sync_fetch_and_add(&realm_gdb_test_state.num_parked, 1);
  args.parked.arrive();
  args.release.wait();
}

RGT_NOINLINE int rgt_chain_level3(const ParkedArgs &args, RealmGdbTestWidget &widget)
{
  int chain_marker = realm_gdb_test_marker(args.index);
  char chain_tag[32];
  snprintf(chain_tag, sizeof(chain_tag), "chain-%d", args.index);
  realm_gdb_test_state.chain_markers[args.index] = chain_marker;
  realm_gdb_test_state.chain_locals[args.index] = &chain_marker;
  realm_gdb_test_state.widget_sums[args.index] = widget.sum();

  park(args);

  char expected_tag[32];
  snprintf(expected_tag, sizeof(expected_tag), "chain-%d", args.index);
  int bad = 0;
  if(chain_marker != realm_gdb_test_marker(args.index))
    bad++;
  if(strcmp(chain_tag, expected_tag) != 0)
    bad++;
  if(widget.sum() != realm_gdb_test_state.widget_sums[args.index])
    bad++;
  return bad;
}

// rgt_chain_level2 returns a long double and rgt_chain_level1 a double, so
// that forcing a return from their frames with a value (MI's -exec-return)
// sets floating-point registers (x87 and SSE ones on x86-64).
RGT_NOINLINE long double rgt_chain_level2(const ParkedArgs &args)
{
  // A canary below the innermost frames: an inferior call or register
  // write that lands on the parked stack shows up as damage here.
  unsigned long canary[RGT_CANARY_WORDS];
  for(int i = 0; i < RGT_CANARY_WORDS; i++)
    canary[i] = 0xC0FFEE00UL + unsigned(i);
  RealmGdbTestWidget widget;
  widget.id = args.index;
  for(int i = 0; i < 4; i++)
    widget.values[i] = 10 * (args.index + 1) + i;

  int bad = rgt_chain_level3(args, widget);

  for(int i = 0; i < RGT_CANARY_WORDS; i++)
    if(canary[i] != 0xC0FFEE00UL + unsigned(i))
      bad++;
  return bad;
}

RGT_NOINLINE double rgt_chain_level1(const ParkedArgs &args)
{
  volatile int chain_depth = 1;
  long double bad = rgt_chain_level2(args);
  if(chain_depth != 1)
    bad++;
  return double(bad);
}

static void chain_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{
  assert(arglen == sizeof(ParkedArgs));
  const ParkedArgs &pargs = *static_cast<const ParkedArgs *>(args);
  if(rgt_chain_level1(pargs) != 0)
    record_failure("chain task frame changed while parked");
  __sync_fetch_and_add(&realm_gdb_test_state.num_finished, 1);
}

RGT_NOINLINE int rgt_recurse(const ParkedArgs &args, int depth)
{
  volatile int recurse_depth = depth;
  int bad = 0;
  if(depth < RGT_RECURSE_DEPTH) {
    bad = rgt_recurse(args, depth + 1);
  } else {
    realm_gdb_test_state.recurse_local = const_cast<int *>(&recurse_depth);
    park(args);
  }
  if(recurse_depth != depth)
    bad++;
  return bad;
}

static void recurse_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p)
{
  assert(arglen == sizeof(ParkedArgs));
  const ParkedArgs &pargs = *static_cast<const ParkedArgs *>(args);
  if(rgt_recurse(pargs, 1) != 0)
    record_failure("recursive task frame changed while parked");
  __sync_fetch_and_add(&realm_gdb_test_state.num_finished, 1);
}

// Inlined into hot_task, with more to do once rgt_hot_wait returns, so that
// it returns into the middle of an inlined function: were the thread moved
// to that return address, its innermost frame would be an inline one.
static inline __attribute__((always_inline)) int rgt_inline_hot_wait(const ParkedArgs &pargs)
{
  int bad = rgt_hot_wait(1000, pargs.release, pargs.parked);
  realm_gdb_test_opaque(bad);
  return bad;
}

static void hot_task(const void *args, size_t arglen, const void *userdata,
                     size_t userlen, Processor p)
{
  assert(arglen == sizeof(ParkedArgs));
  const ParkedArgs &pargs = *static_cast<const ParkedArgs *>(args);
  volatile int hot_task_local = 1;
  realm_gdb_test_state.hot_local = const_cast<int *>(&hot_task_local);
  if(rgt_inline_hot_wait(pargs) != 0)
    record_failure("callee-saved registers changed in a parked task");
  __sync_fetch_and_add(&realm_gdb_test_state.num_finished, 1);
}

// --------------------------------------------------------------------------
// Probe and top-level tasks
// --------------------------------------------------------------------------

// Wakes every parked task, once (the probe does it; gdb may do it sooner).
static UserEvent release_event;
static int released = 0;

extern "C" RGT_NOINLINE void realm_gdb_test_release_parked(void)
{
  if(__sync_bool_compare_and_swap(&released, 0, 1))
    release_event.trigger();
}

struct ProbeArgs {
  int unused;
};

// Runs on the parked tasks' processor once they have all arrived at their
// barrier, and so only once the last one has parked.
static void parked_task(const void *args, size_t arglen, const void *userdata,
                        size_t userlen, Processor p)
{}

static void probe_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{
  assert(arglen == sizeof(ProbeArgs));
  volatile int probe_marker = 0x5eed;
  realm_gdb_test_state.probe_local = const_cast<int *>(&probe_marker);
  RealmGdbTestWidget probe_widget = {7, {1, 2, 3, 4}};

  if(rgt_hot_probe(2000) != 0)
    record_failure("callee-saved registers changed in the stopped thread");
  if(probe_marker != 0x5eed)
    record_failure("probe task frame changed");
  if(probe_widget.sum() != 17)
    record_failure("probe task frame changed");

  realm_gdb_test_release_parked();
}

static void opt_task(const void *args, size_t arglen, const void *userdata,
                     size_t userlen, Processor p)
{
  assert(arglen == sizeof(ParkedArgs));
  const ParkedArgs &pargs = *static_cast<const ParkedArgs *>(args);
  volatile int opt_task_local = 1;
  realm_gdb_test_state.opt_local = const_cast<int *>(&opt_task_local);
  RgtPair given = {realm_gdb_test_opaque(2000), realm_gdb_test_opaque(2001)};
  if(rgt_opt_wait(2000, given, pargs.release, pargs.parked) != 0)
    record_failure("an optimized parked frame changed");
  __sync_fetch_and_add(&realm_gdb_test_state.num_finished, 1);
}

static void top_level_task(const void *args, size_t arglen, const void *userdata,
                           size_t userlen, Processor p)
{
  UserEvent release = UserEvent::create_user_event();
  release_event = release;
  // chain tasks + recursive task + hot task + opt task
  const int num_parked = RGT_NUM_CHAIN + 3;
  Barrier parked = Barrier::create_barrier(num_parked);

  // All of these run on this processor, so each one parks at its barrier
  // arrival before the next starts.  The probe runs only once every one of
  // them is parked (after parked_task): here too, or on a second CPU
  // processor if there is one (-ll:cpu 2), so that the parked tasks can run
  // while the probe's thread is stopped.
  std::vector<Event> done;
  for(int i = 0; i < RGT_NUM_CHAIN; i++) {
    ParkedArgs pargs = {i, release, parked};
    done.push_back(p.spawn(CHAIN_TASK, &pargs, sizeof(pargs)));
  }
  {
    ParkedArgs pargs = {0, release, parked};
    done.push_back(p.spawn(RECURSE_TASK, &pargs, sizeof(pargs)));
    done.push_back(p.spawn(HOT_TASK, &pargs, sizeof(pargs)));
    done.push_back(p.spawn(OPT_TASK, &pargs, sizeof(pargs)));
  }
  Processor probe_proc = p;
  Machine::ProcessorQuery cpus(Machine::get_machine());
  cpus.only_kind(Processor::LOC_PROC);
  for(Machine::ProcessorQuery::iterator it = cpus.begin(); it != cpus.end(); ++it)
    if(*it != p) {
      probe_proc = *it;
      break;
    }
  Event all_parked = p.spawn(PARKED_TASK, 0, 0, parked);
  ProbeArgs probe_args = {0};
  done.push_back(
      probe_proc.spawn(PROBE_TASK, &probe_args, sizeof(probe_args), all_parked));

  Event::merge_events(done).wait();

  if(realm_gdb_test_state.num_parked != num_parked)
    record_failure("not every task parked");
}

int main(int argc, char **argv)
{
#if defined(__linux__) && defined(PR_SET_PTRACER)
  // Let a debugger that isn't our parent attach (Yama ptrace_scope 1).
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif

  Runtime rt;
  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-mode") && (i + 1 < argc)) {
      const char *m = argv[++i];
      if(!strcmp(m, "stop"))
        realm_gdb_test_mode = MODE_STOP;
      else if(!strcmp(m, "attach"))
        realm_gdb_test_mode = MODE_ATTACH;
      else if(!strcmp(m, "crash"))
        realm_gdb_test_mode = MODE_CRASH;
      else if(!strcmp(m, "freeze"))
        realm_gdb_test_mode = MODE_FREEZE;
      else {
        fprintf(stderr, "unknown mode: %s\n", m);
        return 2;
      }
    } else if(!strcmp(argv[i], "-timeout") && (i + 1 < argc)) {
      attach_timeout_seconds = atoi(argv[++i]);
    }
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(CHAIN_TASK, chain_task);
  rt.register_task(RECURSE_TASK, recurse_task);
  rt.register_task(HOT_TASK, hot_task);
  rt.register_task(OPT_TASK, opt_task);
  rt.register_task(PROBE_TASK, probe_task);
  rt.register_task(PARKED_TASK, parked_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  rt.shutdown(e);
  rt.wait_for_shutdown();

  if(realm_gdb_test_state.failures != 0) {
    printf("REALM_GDB_TEST: FAIL %d problem(s)\n", realm_gdb_test_state.failures);
    return 1;
  }
  printf("REALM_GDB_TEST: PASS\n");
  return 0;
}
