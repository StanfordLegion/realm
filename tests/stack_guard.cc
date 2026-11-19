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

// Realm test for the guard region below a user thread's stack: a task that
// recurses past the bottom of its stack, with small frames or with frames
// bigger than a page, must fault in the guard region, even with writable
// memory mapped just below it, while a task that uses all of its stack must
// not fault at all.  The guard must be at least 64 KiB: aarch64's
// -fstack-clash-protection assumes as much, and a frame smaller than that may
// touch the stack first near its bottom.
//
// Linux only (it reads /proc/self/maps), and only where the task runs on a
// user thread: elsewhere it says why it's skipped, and passes.  It can't tell
// whose PROT_NONE mapping lies below the stack, so it would pass, without a
// guard of Realm's own, if some other mapping of 64 KiB or more happened to
// be there; a newly made stack normally has nothing below it.

#include <realm.h>

#include <stdio.h>
#include <string>
#include <vector>

using namespace Realm;

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  CHECK_TASK,
};

Logger log_app("app");

// the stack size asked for, in MB (-ll:stacksize)
static const size_t STACK_MB = 1;

#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
#define STACK_GUARD_SANITIZED
#endif
#if defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(address_sanitizer)
#define STACK_GUARD_SANITIZED
#endif
#endif

#if defined(REALM_ON_LINUX) && defined(REALM_USE_USER_THREADS) &&                        \
    !defined(STACK_GUARD_SANITIZED)

#include <errno.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

struct Mapping {
  uintptr_t lo, hi;
  char perms[5];
};

static std::vector<Mapping> read_maps(void)
{
  std::vector<Mapping> maps;
  FILE *f = fopen("/proc/self/maps", "r");
  if(f == nullptr)
    return maps;
  char line[4096];
  while(fgets(line, sizeof(line), f) != nullptr) {
    Mapping m;
    unsigned long lo, hi;
    if(sscanf(line, "%lx-%lx %4s", &lo, &hi, m.perms) == 3) {
      m.lo = lo;
      m.hi = hi;
      maps.push_back(m);
    }
  }
  fclose(f);
  return maps;
}

static const Mapping *find_mapping(const std::vector<Mapping> &maps, uintptr_t addr)
{
  for(const Mapping &m : maps)
    if((m.lo <= addr) && (addr < m.hi))
      return &m;
  return nullptr;
}

// the guard region below the stack must be at least this big
static const size_t MIN_GUARD = 64 << 10;

static sigjmp_buf fault_env;
static volatile uintptr_t fault_addr = 0;
static volatile int fault_signal = 0;
static volatile uintptr_t deepest = 0;
// faults are caught only at these addresses, those the check can reach
static volatile uintptr_t catch_lo = 0, catch_hi = 0;
static struct sigaction old_segv, old_bus;

static void fault_handler(int sig, siginfo_t *info, void *)
{
  uintptr_t addr = reinterpret_cast<uintptr_t>(info->si_addr);
  if((addr < catch_lo) || (addr >= catch_hi)) {
    // not the check's fault: put the old handler back for when it recurs
    sigaction(sig, (sig == SIGBUS) ? &old_bus : &old_segv, nullptr);
    return;
  }
  fault_signal = sig;
  fault_addr = addr;
  siglongjmp(fault_env, 1);
}

// Recurse until a frame is at or below `limit`.  Each of descend_small's
// frames takes a little over 512 bytes of stack, well under a page, so it
// can't step over a guard; each of descend_big's takes 16 KiB, of which it
// touches (unprobed) only the top and frame[0] at the bottom, so it steps over
// a guard too small unless the frame happens to straddle it (which is why the
// guard's size is checked first).
__attribute__((noinline)) static int descend_small(uintptr_t limit)
{
  volatile char frame[512];
  frame[0] = 1;
  uintptr_t here = reinterpret_cast<uintptr_t>(&frame[0]);
  deepest = here;
  if(here <= limit)
    return frame[0];
  return descend_small(limit) + frame[0];
}

__attribute__((noinline)) static int descend_big(uintptr_t limit)
{
  volatile char frame[16 << 10];
  frame[0] = 1;
  uintptr_t here = reinterpret_cast<uintptr_t>(&frame[0]);
  deepest = here;
  if(here <= limit)
    return frame[0];
  return descend_big(limit) + frame[0];
}

// Recurses down to `limit` with SIGSEGV and SIGBUS caught at addresses in
// [lo, hi) (on an alternate signal stack, as the faulting stack has no room).
// Returns true if that faulted, and the address in fault_addr.
static bool descend_catching_faults(int (*descend)(uintptr_t), uintptr_t limit,
                                    uintptr_t lo, uintptr_t hi)
{
  static char altstack_mem[256 << 10];
  stack_t altstack, old_altstack;
  altstack.ss_sp = altstack_mem;
  altstack.ss_size = sizeof(altstack_mem);
  altstack.ss_flags = 0;
  sigaltstack(&altstack, &old_altstack);

  catch_lo = lo;
  catch_hi = hi;
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_sigaction = fault_handler;
  action.sa_flags = SA_SIGINFO | SA_ONSTACK;
  sigemptyset(&action.sa_mask);
  sigaction(SIGSEGV, &action, &old_segv);
  sigaction(SIGBUS, &action, &old_bus);

  fault_signal = 0;
  fault_addr = 0;
  bool faulted = false;
  if(sigsetjmp(fault_env, 1) == 0)
    descend(limit);
  else
    faulted = true;

  sigaction(SIGSEGV, &old_segv, nullptr);
  sigaction(SIGBUS, &old_bus, nullptr);
  sigaltstack(&old_altstack, nullptr);
  return faulted;
}

// Maps `size` writable bytes ending at `top`, only if nothing is mapped there.
static char *map_below(uintptr_t top, size_t size)
{
  void *want = reinterpret_cast<void *>(top - size);
  void *got = mmap(want, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  if(got == MAP_FAILED)
    return nullptr;
  if(got != want) { // a kernel older than MAP_FIXED_NOREPLACE took it as a hint
    munmap(got, size);
    return nullptr;
  }
  return static_cast<char *>(got);
}

static bool on_host_stack(uintptr_t sp)
{
  pthread_attr_t attr;
  if(pthread_getattr_np(pthread_self(), &attr) != 0)
    return false;
  void *addr;
  size_t size;
  int ret = pthread_attr_getstack(&attr, &addr, &size);
  pthread_attr_destroy(&attr);
  if(ret != 0)
    return false;
  uintptr_t lo = reinterpret_cast<uintptr_t>(addr);
  return (lo <= sp) && (sp < lo + size);
}

static int check_stack_guard(void)
{
  volatile char marker = 0;
  const uintptr_t sp = reinterpret_cast<uintptr_t>(&marker);

  if(on_host_stack(sp)) {
    printf("stack_guard: skipped: the task is running on a kernel thread, not a user "
           "thread\n");
    return 0;
  }

  std::vector<Mapping> maps = read_maps();
  const Mapping *stack_map = find_mapping(maps, sp);
  if(stack_map == nullptr) {
    printf("stack_guard: skipped: can't find the task's stack in /proc/self/maps\n");
    return 0;
  }
  const Mapping stack = *stack_map;
  const Mapping *below = find_mapping(maps, stack.lo - 1);
  // the guard region: PROT_NONE, just below the stack
  const bool guarded =
      (below != nullptr) && (below->hi == stack.lo) && !strcmp(below->perms, "---p");
  const uintptr_t guard_lo = guarded ? below->lo : stack.lo;
  printf("stack %#lx-%#lx %s (%lu KiB); below it: %s", (unsigned long)stack.lo,
         (unsigned long)stack.hi, stack.perms,
         (unsigned long)((stack.hi - stack.lo) >> 10),
         below ? below->perms : "nothing mapped");
  if(guarded)
    printf(" (%lu KiB)", (unsigned long)((stack.lo - guard_lo) >> 10));
  printf("\n");

  int errors = 0;
  // (the mapping can be bigger than asked for, where it merged with a writable
  //  one above it)
  if((stack.hi - stack.lo) < (STACK_MB << 20)) {
    log_app.error() << "FAIL: the stack is smaller than the " << STACK_MB
                    << " MB asked for";
    errors++;
  }

  // 1) All of the stack can be used: recursing to within 16 KiB of its bottom
  //  doesn't fault
  if(descend_catching_faults(descend_small, stack.lo + (16 << 10), stack.lo, stack.hi)) {
    log_app.error() << "FAIL: faulted within the stack (signal " << fault_signal
                    << ", address " << std::hex << fault_addr << std::dec << ", "
                    << (fault_addr - stack.lo) << " bytes above its bottom)";
    errors++;
  } else
    printf("whole stack used without a fault: deepest frame %lu bytes above the "
           "bottom\n",
           (unsigned long)(deepest - stack.lo));

  // 2) An overflow faults in the guard region, with writable memory (the
  //  "victim") mapped just below it, which the overflow mustn't reach; or just
  //  below the stack if there's no guard, for the overflow to write into
  //  rather than into memory of the runtime's.  Nothing is tried where an
  //  overflow could write over something else.
  if(!guarded && (below != nullptr)) {
    log_app.error() << "FAIL: no guard region below the stack, but a " << below->perms
                    << " mapping, which an overflow would write over (not tried)";
    return errors + 1;
  }
  if(guarded && ((stack.lo - guard_lo) < MIN_GUARD)) {
    log_app.error() << "FAIL: the guard region below the stack is only "
                    << ((stack.lo - guard_lo) >> 10) << " KiB, less than "
                    << (MIN_GUARD >> 10) << " KiB (an overflow wasn't tried)";
    return errors + 1;
  }
  const size_t victim_size = 128 << 10;
  char *victim = map_below(guard_lo, victim_size);
  if(victim != nullptr)
    printf("writable victim mapped at %#lx-%#lx, %lu KiB below the stack\n",
           (unsigned long)reinterpret_cast<uintptr_t>(victim), (unsigned long)guard_lo,
           (unsigned long)((stack.lo - guard_lo) >> 10));
  else if(!guarded) {
    log_app.error() << "FAIL: no guard region below the stack, and the memory there "
                       "can't be mapped for the overflow to write into (not tried)";
    return errors + 1;
  } else
    printf("no victim: the memory just below the guard region is taken\n");

  struct {
    const char *name;
    int (*descend)(uintptr_t);
  } kinds[] = {{"512-byte frames", descend_small}, {"16 KiB frames", descend_big}};
  for(const auto &kind : kinds) {
    if(victim != nullptr)
      memset(victim, 0x5a, victim_size);
    // overflow to a quarter of the way into the victim (or where it would be);
    //  a 16 KiB frame overshoots that by less than another quarter
    bool faulted = descend_catching_faults(kind.descend, guard_lo - victim_size / 4,
                                           guard_lo - victim_size, stack.hi);
    size_t clobbered = 0;
    if(victim != nullptr)
      for(size_t i = 0; i < victim_size; i++)
        if(victim[i] != 0x5a)
          clobbered++;

    if(!faulted) {
      log_app.error() << "FAIL: an overflow with " << kind.name
                      << " wasn't caught: it went " << (stack.lo - deepest)
                      << " bytes below the stack, changing " << clobbered
                      << " bytes of the writable mapping below it";
      errors++;
    } else if((fault_addr < guard_lo) || (fault_addr >= stack.lo)) {
      log_app.error() << "FAIL: an overflow with " << kind.name << " faulted at "
                      << (stack.lo - fault_addr)
                      << " bytes below the stack, outside the guard region (signal "
                      << fault_signal << ")";
      errors++;
    } else if(clobbered != 0) {
      log_app.error() << "FAIL: an overflow with " << kind.name << " wrote " << clobbered
                      << " bytes below the guard region";
      errors++;
    } else
      printf("overflow with %s caught in the guard region: signal %d at %lu bytes below "
             "the stack\n",
             kind.name, (int)fault_signal, (unsigned long)(stack.lo - fault_addr));
  }
  if(victim != nullptr)
    munmap(victim, victim_size);

  return errors;
}

#else

static int check_stack_guard(void)
{
  printf("stack_guard: skipped: needs Linux and user threads, without sanitizers\n");
  return 0;
}

#endif

static int check_errors = -1;

void check_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                Processor p)
{
  check_errors = check_stack_guard();
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  // The checks run in a second task, which this one's wait makes Realm start
  //  on a new user thread: the newest stack, so most likely with nothing yet
  //  mapped just below it.  (Below an older stack there may be a newer
  //  mapping, a stack of another thread, say.)
  p.spawn(CHECK_TASK, 0, 0).wait();
  if(check_errors == 0)
    printf("stack_guard: PASS\n");
  Runtime::get_runtime().shutdown(Event::NO_EVENT, check_errors ? 1 : 0);
}

int main(int argc, const char **argv)
{
  // ask for a known stack size, small enough to overflow quickly
  std::vector<const char *> args(argv, argv + argc);
  std::string stack_mb = std::to_string(STACK_MB);
  args.push_back("-ll:stacksize");
  args.push_back(stack_mb.c_str());
  int nargs = static_cast<int>(args.size());
  args.push_back(nullptr); // argv[argc]
  const char **pargs = args.data();

  Runtime rt;
  rt.init(&nargs, (char ***)&pargs);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, CHECK_TASK,
                                   CodeDescriptor(check_task), ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  return rt.wait_for_shutdown();
}
