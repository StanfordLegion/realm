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

// Realm test for deferred allocation in remote memories.  Four shapes per
// memory:
//
// 1. The funding-cycle regression (tla/allocation/bugs/BUG-8.md): fill the
//    memory with instance d, create a same-sized instance c from a non-owner
//    node, and have another node destroy d with c's ready event as the
//    precondition.  The client is legal - the destroy depends only on an
//    earlier-requested creation - but if the owner learns of the destroy
//    before it admits c's creation, it may plan to satisfy c out of d's
//    space, creating a cycle (c waits on d's release, which waits on c's
//    ready event) that hangs silently.  The taint fix marks c's ready event
//    as depending on the pending creation of c itself (TAINT_INST), so the
//    owner refuses to fund c from a release gated on that event - the
//    creation resolves honestly (funded some other way or failed) instead of
//    deadlocking.  A hang here is the bug (the alarm watchdog catches it in
//    single-process runs; under mpiexec SIGALRM is not delivered, so the
//    test-harness timeout is the backstop).  NOTE: detecting a regression
//    requires >= 3 ranks - with only 2, the dependent destroy travels behind
//    the creation request on the same source-destination pair and per-pair
//    transport ordering shields the race; 1- and 2-rank runs verify
//    functionality only.
//
// 2. The handoff pattern that must keep working: destroy d deferred on a
//    user event, then create c while that release is still in flight.
//    Whichever way the race resolves is legal - c may be funded by the
//    pending release or may fail honestly - but a bounded number of retries
//    after the release completes must succeed, and nothing may hang.
//
// 3. The taint headline (tla/allocation/DIST-DESIGN.md 5c): a cross-node
//    deletion of d gated on the finish event of a task that is STILL
//    RUNNING, then a remote creation of c that can only be satisfied out of
//    d's space.  The task was launched with no precondition, so its finish
//    event's taint resolves clean (TAINT_NONE) at the memory's owner via
//    the subscription ack, and the owner must admit c with DEFERRED success
//    - observed through an InstanceAllocResult profiling response - while
//    the task is provably still running (its gate event is untriggered).  A
//    blunter fix (refusing to fund from any release with an untriggered
//    precondition) would fail this shape.
//
// 4. The copy-gated BUG-8 variant: another node copies d's data into c
//    (gated on c's readiness - the canonical migration idiom) and destroys d
//    gated on the copy's completion.  The destroy's precondition depends on
//    c's own creation THROUGH THE COPY, so the owner must refuse to fund c
//    from d's release and the creation must fail honestly.  Without the
//    copy-completion taint union (transfer.cc copy_impl) the copy-finish
//    event reads as clean at the owner and the funding cycle is planned - a
//    hang.  Like shape 1, the racing arrival order needs >= 3 ranks.
//
// 5. C-API user-event laundering: like shape 1, but the dependence carrier
//    is a user event minted through realm_c.h and later bound (also through
//    the C API) to c's readiness.  The C mint path must record TOP exactly
//    like UserEvent::create_user_event - with a clean-by-default C mint the
//    owner funds c from a deletion that comes to depend on it (hang).
//
// 6. Preconditioned-create root ring: c2 is created gated on c1's readiness,
//    and the deletion of d is gated on c2's readiness.  c2's creation event
//    must carry INST(c2) UNION taint(e_c1) = TOP; with a plain INST(c2) root
//    the owner deems d's release safe for c1 (c1 != c2) and funds it,
//    closing the ring c1 <- d-release <- e_c2 <- c2 <- e_c1 <- c1 (hang).
//
// 7. Early destroy racing its own creation request from a third node: the
//    (tiny) destroy can reach the owner before the (large) creation request
//    and must be parked and spliced back in AFTER the creation's outcome is
//    published - splicing earlier lets the release re-park forever and
//    silently lose the deletion (detected here as a reclaim failure).  Both
//    instant outcomes are exercised: success (empty memory) and honest
//    failure (full memory).
//
// 8. Cross-memory funding ring: two memories with distinct owners, crossed
//    deletions each gated on the OTHER memory's pending creation.  A taint
//    of INST(j) where j lives in a different memory must be TOP-equivalent;
//    honoring the "different instance" exemption locally would let both
//    owners fund and deadlock the pair across memories.  Requires >= 2
//    distinct memory-owner ranks (fully exercised at 3, where both deletion
//    issuers are third parties); skips with a warning below that.

#include <realm.h>
#include <realm/cmdline.h>
#include <realm/realm_c.h>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  DEP_DESTROY_TASK,
  SLEEPY_TASK,
  ALLOC_RESULT_TASK,
  COPY_DESTROY_TASK,
};

namespace {
  int rounds = 4;
  int max_retries = 20;
  // many small fields make the creation request's serialized layout large,
  //  which widens the window in which a (tiny) dependent-destroy message
  //  from a third node can beat it to the owner - the race the round-trip
  //  fix closes
  int num_fields = 512;
  bool test_ok = true;
  // written by alloc_result_task, read after its 'decided' event triggers -
  //  the event ordering makes this race-free (both on the top-level rank)
  bool alloc_decided_success = false;
} // namespace

struct DepDestroyArgs {
  RegionInstance inst;
  Event precondition;
};

// issues a deferred destroy of an instance we did not create, gated on an
//  event handed to us by another node - the BUG-8 dependence carrier
void dep_destroy_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  assert(arglen == sizeof(DepDestroyArgs));
  const DepDestroyArgs *da = static_cast<const DepDestroyArgs *>(args);
  log_app.info() << "dependent destroy: proc=" << p << " inst=" << da->inst
                 << " pre=" << da->precondition;
  da->inst.destroy(da->precondition);
}

// issues a copy of src's data into dst gated on 'pre' (dst's readiness),
//  then a deferred destroy of src gated on the copy's completion - shape 4's
//  migration idiom, whose deletion depends on dst's creation through the copy
struct CopyDestroyArgs {
  RegionInstance src, dst;
  Event pre;
  Rect<1> bounds;
};

void copy_destroy_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{
  assert(arglen == sizeof(CopyDestroyArgs));
  const CopyDestroyArgs *ca = static_cast<const CopyDestroyArgs *>(args);
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(ca->src, 0, sizeof(uint64_t));
  dsts[0].set_field(ca->dst, 0, sizeof(uint64_t));
  IndexSpace<1> is(ca->bounds);
  Event copy_done = is.copy(srcs, dsts, ProfilingRequestSet(), ca->pre);
  log_app.info() << "copy-gated destroy: proc=" << p << " src=" << ca->src
                 << " dst=" << ca->dst << " copy_done=" << copy_done;
  ca->src.destroy(copy_done);
}

// a task that stays running until an event we control triggers - shape 3's
//  "still-running task" whose finish event gates the cross-node deletion.
//  It is launched with NO precondition, so the finish event's taint is clean
//  by construction; the internal wait is invisible to taint (frozen at
//  launch), which is exactly the semantics shape 3 pins down.
struct SleepyArgs {
  UserEvent gate;
};

void sleepy_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  assert(arglen == sizeof(SleepyArgs));
  const SleepyArgs *sa = static_cast<const SleepyArgs *>(args);
  log_app.info() << "sleepy task running: proc=" << p << " gate=" << sa->gate;
  sa->gate.wait();
}

// profiling response task: reports whether the allocation decision was
//  success (possibly deferred) or failure, then signals the waiting driver
struct AllocResultArgs {
  UserEvent decided;
};

void alloc_result_task(const void *args, size_t arglen, const void *userdata,
                       size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  assert(resp.user_data_size() == sizeof(AllocResultArgs));
  const AllocResultArgs *ra = static_cast<const AllocResultArgs *>(resp.user_data());
  ProfilingMeasurements::InstanceAllocResult result;
  alloc_decided_success = resp.get_measurement(result) && result.success;
  ra->decided.trigger();
}

// a profiling request set that keeps a failed allocation from being fatal -
//  we test failure paths on purpose
static ProfilingRequestSet failure_tolerant_prs()
{
  ProfilingRequestSet prs;
  prs.add_request(Processor::NO_PROC, 0 /*ignore*/)
      .add_measurement<ProfilingMeasurements::InstanceStatus>();
  return prs;
}

// like failure_tolerant_prs, but also asks for the allocation decision
//  itself (InstanceAllocResult), delivered to alloc_result_task on
//  resp_proc as soon as the owner decides - for a deferred success this is
//  BEFORE the instance becomes ready, which is what lets shape 3 observe
//  "admitted while the funding release's precondition is still pending"
static ProfilingRequestSet alloc_result_prs(UserEvent decided, Processor resp_proc)
{
  ProfilingRequestSet prs;
  AllocResultArgs ra;
  ra.decided = decided;
  // the alloc-result measurement must be in a request of its own: a
  //  request's response is only sent once ALL its measurements are ready,
  //  and InstanceStatus is not known until the instance is freed - the
  //  whole point here is to observe the decision early
  prs.add_request(resp_proc, ALLOC_RESULT_TASK, &ra, sizeof(ra))
      .add_measurement<ProfilingMeasurements::InstanceAllocResult>();
  // separate status request keeps a failed allocation from being fatal
  prs.add_request(Processor::NO_PROC, 0 /*ignore*/)
      .add_measurement<ProfilingMeasurements::InstanceStatus>();
  return prs;
}

static Event create_full_instance(RegionInstance &inst, Memory m,
                                  const ProfilingRequestSet &prs,
                                  Event wait_on = Event::NO_EVENT)
{
  const size_t field_size = 8;
  std::vector<size_t> field_sizes(num_fields, field_size);
  size_t elements = m.capacity() / (field_size * num_fields);
  assert(elements > 0);
  Rect<1> bounds(0, static_cast<int>(elements) - 1);
  return RegionInstance::create_instance(inst, m, bounds, field_sizes, 0 /*SOA*/, prs,
                                         wait_on);
}

// fill the memory with an instance that must allocate cleanly
static RegionInstance make_occupant(Memory m)
{
  RegionInstance d;
  Event e = create_full_instance(d, m, ProfilingRequestSet());
  alarm(30);
  e.wait();
  alarm(0);
  return d;
}

// the memory must become fully reclaimable at this point - the destroy we
//  just issued may still be in flight to the owner (the documented
//  same-source race), so a probe allocation can honestly fail until the
//  release has been applied; retry a bounded number of times
static void check_reclaimed(Memory m, int round, const char *phase)
{
  int retry = 0;
  while(true) {
    RegionInstance probe;
    Event e = create_full_instance(probe, m, failure_tolerant_prs());
    alarm(30);
    bool poisoned = false;
    e.wait_faultaware(poisoned);
    alarm(0);
    if(!poisoned) {
      probe.destroy();
      return;
    }
    if(++retry > max_retries) {
      log_app.error() << "memory not reclaimed: mem=" << m << " phase=" << phase
                      << " round=" << round;
      test_ok = false;
      return;
    }
  }
}

// shape 1: the BUG-8 funding-cycle regression
static void test_dependent_destroy(Memory m, Processor dep_proc, int round)
{
  RegionInstance d = make_occupant(m);

  // create a same-sized instance - it can only be satisfied out of d's space
  RegionInstance c;
  Event e_c = create_full_instance(c, m, failure_tolerant_prs());

  // hand (d, e_c) to another node, which destroys d gated on c's readiness -
  //  this is the legal client shape that used to be able to race the
  //  creation request to the owner and fund c from d's own space
  DepDestroyArgs da;
  da.inst = d;
  da.precondition = e_c;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da, sizeof(da)).wait();
  alarm(0);

  // the creation must resolve either way - a hang here is the BUG-8 cycle
  alarm(30);
  bool poisoned = false;
  e_c.wait_faultaware(poisoned);
  alarm(0);

  if(poisoned) {
    // honest failure: the dependent destroy's precondition is poisoned, so
    //  the deletion was dropped and d is still allocated
    d.destroy();
  } else {
    // c was funded by d's release (possible when the destroy was ordered
    //  after the creation was admitted and the events resolved in order)
    c.destroy();
  }

  check_reclaimed(m, round, "dependent-destroy");
}

// shape 2: remote deferred creation funded by an in-flight deletion
static void test_handoff(Memory m, int round, int &first_try_funded)
{
  RegionInstance d = make_occupant(m);

  UserEvent u = UserEvent::create_user_event();
  d.destroy(u);

  // race the creation against the in-flight release
  RegionInstance c;
  Event e_c = create_full_instance(c, m, failure_tolerant_prs());

  u.trigger();

  alarm(30);
  bool poisoned = false;
  e_c.wait_faultaware(poisoned);
  alarm(0);

  if(!poisoned) {
    first_try_funded++;
  } else {
    // honest miss - the release arrived after the creation was admitted;
    //  once the release completes, a retry must succeed
    int retry = 0;
    while(true) {
      if(++retry > max_retries) {
        log_app.error() << "handoff retries exhausted: mem=" << m << " round=" << round;
        test_ok = false;
        return;
      }
      e_c = create_full_instance(c, m, failure_tolerant_prs());
      alarm(30);
      e_c.wait_faultaware(poisoned);
      alarm(0);
      if(!poisoned)
        break;
    }
  }

  c.destroy();
  check_reclaimed(m, round, "handoff");
}

// shape 3: a creation funded by a deletion gated on a still-running task
static void test_taint_funded_create(Memory m, Processor dep_proc, Processor resp_proc,
                                     int round)
{
  RegionInstance d = make_occupant(m);

  // a task that is genuinely still running while the owner decides: it
  //  blocks on 'gate', which we do not trigger until the decision has been
  //  observed.  Launched with no precondition, so its finish event's taint
  //  resolves clean at the memory's owner.
  UserEvent gate = UserEvent::create_user_event();
  SleepyArgs sa;
  sa.gate = gate;
  Event e_task = dep_proc.spawn(SLEEPY_TASK, &sa, sizeof(sa));

  // cross-node deletion of d gated on the running task's finish event
  DepDestroyArgs da;
  da.inst = d;
  da.precondition = e_task;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da, sizeof(da)).wait();
  alarm(0);

  // create c - it can only be satisfied out of d's space, and d's release is
  //  gated on the running task.  The owner must resolve the finish event's
  //  taint (clean) and admit c with DEFERRED success.  The release may still
  //  be in flight to the owner when the creation arrives (the same
  //  same-source race shape 2 tolerates), in which case the creation fails
  //  honestly - retry a bounded number of times.
  RegionInstance c;
  Event e_c = Event::NO_EVENT;
  bool funded = false;
  for(int retry = 0; !funded && (retry <= max_retries); retry++) {
    UserEvent decided = UserEvent::create_user_event();
    alloc_decided_success = false;
    e_c = create_full_instance(c, m, alloc_result_prs(decided, resp_proc));
    alarm(30);
    decided.wait();
    alarm(0);
    funded = alloc_decided_success;
    if(!funded)
      usleep(100000);
  }
  if(!funded) {
    log_app.error() << "taint-funded create retries exhausted: mem=" << m
                    << " round=" << round;
    test_ok = false;
    gate.trigger();
    return;
  }

  // the headline assertions: the decision came back success while the task
  //  is provably still running (we have not triggered its gate, so its
  //  finish event cannot have fired), and the success is deferred (d still
  //  occupies the memory, so c cannot be ready yet)
  if(e_task.has_triggered()) {
    log_app.error() << "task finished before funding decision observed: mem=" << m
                    << " round=" << round;
    test_ok = false;
  }
  if(e_c.has_triggered()) {
    log_app.error() << "creation ready while occupant still allocated: mem=" << m
                    << " round=" << round;
    test_ok = false;
  }

  // let the task finish: e_task triggers, d's release applies, c gets funded
  gate.trigger();
  alarm(30);
  bool poisoned = false;
  e_c.wait_faultaware(poisoned);
  alarm(0);
  if(poisoned) {
    log_app.error() << "deferred-success creation poisoned: mem=" << m
                    << " round=" << round;
    test_ok = false;
    return;
  }

  c.destroy();
  check_reclaimed(m, round, "taint-funded");
}

// shape 4: the copy-gated BUG-8 variant
static void test_copy_gated_destroy(Memory m, Processor dep_proc, int round)
{
  RegionInstance d = make_occupant(m);

  // create c - it can only be satisfied out of d's space
  RegionInstance c;
  Event e_c = create_full_instance(c, m, failure_tolerant_prs());

  // hand (d, c, e_c) to another node: it copies d into c gated on c's
  //  readiness, then destroys d gated on the copy's completion.  The
  //  deletion now depends on c's creation through the copy-finish event.
  const size_t field_size = 8;
  size_t elements = m.capacity() / (field_size * num_fields);
  CopyDestroyArgs ca;
  ca.src = d;
  ca.dst = c;
  ca.pre = e_c;
  ca.bounds = Rect<1>(0, static_cast<int>(elements) - 1);
  alarm(30);
  dep_proc.spawn(COPY_DESTROY_TASK, &ca, sizeof(ca)).wait();
  alarm(0);

  // the creation must fail honestly: the only deletion that could fund it
  //  transitively depends on it.  A hang here is the copy-shaped BUG-8 cycle
  //  (c waits on d's release, d's release waits on the copy, the copy waits
  //  on c).
  alarm(30);
  bool poisoned = false;
  e_c.wait_faultaware(poisoned);
  alarm(0);

  if(!poisoned) {
    log_app.error() << "copy-gated creation was wrongly funded: mem=" << m
                    << " round=" << round;
    test_ok = false;
    c.destroy();
    check_reclaimed(m, round, "copy-gated");
    return;
  }

  // the deletion was dropped with its poisoned precondition - d is still
  //  allocated; clean up for the next round
  d.destroy();
  check_reclaimed(m, round, "copy-gated");
}

// shape 5: BUG-8 laundered through a C-API user event
static void test_capi_laundering(Memory m, Processor dep_proc, int round)
{
  RegionInstance d = make_occupant(m);

  // mint the dependence carrier through the C API - the mint path that
  //  bypasses UserEvent::create_user_event and must record TOP itself
  realm_runtime_t c_rt = nullptr;
  realm_status_t st = realm_runtime_get_runtime(&c_rt);
  assert(st == REALM_SUCCESS);
  realm_user_event_t c_ue = 0;
  st = realm_user_event_create(c_rt, &c_ue);
  assert(st == REALM_SUCCESS);
  Event u;
  u.id = c_ue;

  // create c - only satisfiable out of d's space - BEFORE the deletion is
  //  issued, so the client is legal (the deletion will only ever depend on
  //  an earlier-requested creation)
  RegionInstance c;
  Event e_c = create_full_instance(c, m, failure_tolerant_prs());

  // cross-node deletion of d gated on the (still unbound) user event
  DepDestroyArgs da;
  da.inst = d;
  da.precondition = u;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da, sizeof(da)).wait();
  alarm(0);

  // bind the user event to c's readiness through the C API - the deletion
  //  now depends on c, which the owner only refuses if the C mint said TOP
  st = realm_user_event_trigger(c_rt, c_ue, e_c.id, 0 /*!ignore_faults*/);
  assert(st == REALM_SUCCESS);

  // the creation must fail honestly - a hang here is the laundered cycle
  alarm(30);
  bool poisoned = false;
  e_c.wait_faultaware(poisoned);
  alarm(0);
  if(!poisoned) {
    log_app.error() << "laundered creation was wrongly funded: mem=" << m
                    << " round=" << round;
    test_ok = false;
    c.destroy();
    check_reclaimed(m, round, "capi-launder");
    return;
  }

  d.destroy();
  check_reclaimed(m, round, "capi-launder");
}

// shape 6: preconditioned-create root ring (root = INST(self) u taint(pre))
static void test_root_ring(Memory m, Processor dep_proc, int round)
{
  RegionInstance d = make_occupant(m);

  // c1: only satisfiable out of d's space
  RegionInstance c1;
  Event e_c1 = create_full_instance(c1, m, failure_tolerant_prs());

  // c2: gated on c1's readiness, so e_c2's taint root must widen to TOP
  RegionInstance c2;
  Event e_c2 = create_full_instance(c2, m, failure_tolerant_prs(), e_c1);

  // cross-node deletion of d gated on c2's readiness.  With a plain
  //  INST(c2) root the owner deems d's release safe for c1 and funds it -
  //  the ring c1 <- d-release <- e_c2 <- c2 <- e_c1 <- c1 hangs.
  DepDestroyArgs da;
  da.inst = d;
  da.precondition = e_c2;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da, sizeof(da)).wait();
  alarm(0);

  alarm(30);
  bool poisoned = false;
  e_c1.wait_faultaware(poisoned);
  alarm(0);
  if(!poisoned) {
    log_app.error() << "ring creation c1 was wrongly funded: mem=" << m
                    << " round=" << round;
    test_ok = false;
  }
  // c2 is cancelled along with its poisoned precondition
  alarm(30);
  e_c2.wait_faultaware(poisoned);
  alarm(0);
  if(!poisoned) {
    log_app.error() << "ring creation c2 was wrongly funded: mem=" << m
                    << " round=" << round;
    test_ok = false;
  }

  d.destroy();
  check_reclaimed(m, round, "root-ring");
}

// shape 7: early destroy racing its own instance's creation request
static void test_early_destroy(Memory m, Processor dep_proc, int round)
{
  // part 1: instant success - the racing destroy may park at the owner and
  //  must be spliced back in after the outcome is published; a lost deletion
  //  shows up as a reclaim failure below
  RegionInstance c;
  Event e_c = create_full_instance(c, m, failure_tolerant_prs());
  DepDestroyArgs da;
  da.inst = c;
  da.precondition = Event::NO_EVENT;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da, sizeof(da)).wait();
  alarm(0);
  alarm(30);
  bool poisoned = false;
  e_c.wait_faultaware(poisoned);
  alarm(0);
  if(poisoned) {
    log_app.error() << "creation failed on empty memory: mem=" << m << " round=" << round;
    test_ok = false;
    return;
  }
  check_reclaimed(m, round, "early-destroy");

  // part 2: instant honest failure (memory full) with the same race - the
  //  parked destroy of the failed instance must be handled, not lost or
  //  crashed on
  RegionInstance d = make_occupant(m);
  RegionInstance c2;
  Event e_c2 = create_full_instance(c2, m, failure_tolerant_prs());
  DepDestroyArgs da2;
  da2.inst = c2;
  da2.precondition = Event::NO_EVENT;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da2, sizeof(da2)).wait();
  alarm(0);
  alarm(30);
  e_c2.wait_faultaware(poisoned);
  alarm(0);
  if(!poisoned) {
    log_app.error() << "creation succeeded on full memory: mem=" << m
                    << " round=" << round;
    test_ok = false;
    c2.destroy();
  }
  d.destroy();
  check_reclaimed(m, round, "early-destroy-fail");

  // part 3: held-window variant - a contract-legal destroy (gated on the
  //  instance's own ready event) racing the DECISION publication of a HELD
  //  admission.  While the admission sits in the held queue the destroy
  //  takes the normal (untriggered) release path; if it lands between the
  //  decision's queue-pop and its outcome publication, it parks and must be
  //  spliced when the decision completes - a missed splice loses the
  //  deletion (reclaim failure below).
  RegionInstance d3 = make_occupant(m);
  UserEvent gate = UserEvent::create_user_event();
  SleepyArgs sa;
  sa.gate = gate;
  Event e_task = dep_proc.spawn(SLEEPY_TASK, &sa, sizeof(sa));
  DepDestroyArgs da3;
  da3.inst = d3;
  da3.precondition = e_task;
  alarm(30);
  dep_proc.spawn(DEP_DESTROY_TASK, &da3, sizeof(da3)).wait();
  alarm(0);
  // the admission is held while the owner resolves the finish event's taint;
  //  destroy c3 (gated on its own readiness) immediately, racing the
  //  decision that the taint ack triggers
  RegionInstance c3;
  Event e_c3 = create_full_instance(c3, m, failure_tolerant_prs());
  c3.destroy(e_c3);
  gate.trigger();
  alarm(30);
  bool poisoned3 = false;
  e_c3.wait_faultaware(poisoned3);
  alarm(0);
  // both outcomes are legal (funded by d3's release, or an honest miss if
  //  the creation was decided before the release arrived); either way both
  //  d3 and c3 must be fully reclaimed - d3's release applies when the task
  //  finishes, and c3's deferred destroy applies (or is dropped with its
  //  poisoned precondition) with it
  check_reclaimed(m, round, "held-window");
}

// shape 8: cross-memory funding ring (foreign INST must be TOP-equivalent)
static void test_cross_memory_ring(const std::vector<Memory> &mems,
                                   const std::vector<Processor> &all_procs)
{
  // two memories with distinct owners
  Memory m_a = Memory::NO_MEMORY, m_b = Memory::NO_MEMORY;
  for(size_t i = 0; (i < mems.size()) && !m_b.exists(); i++)
    for(size_t j = i + 1; j < mems.size(); j++)
      if(mems[i].address_space() != mems[j].address_space()) {
        m_a = mems[i];
        m_b = mems[j];
        break;
      }
  if(!m_b.exists()) {
    log_app.warning()
        << "cross-memory shape skipped (needs two memories with distinct owners)";
    return;
  }
  // each deletion is issued from the OTHER memory's owner space, so it races
  //  its target's creation request from an independent node
  Processor p_a = Processor::NO_PROC, p_b = Processor::NO_PROC;
  for(size_t i = 0; i < all_procs.size(); i++) {
    if(all_procs[i].address_space() == m_b.address_space())
      p_a = all_procs[i];
    if(all_procs[i].address_space() == m_a.address_space())
      p_b = all_procs[i];
  }
  assert(p_a.exists() && p_b.exists());

  for(int round = 0; round < rounds; round++) {
    RegionInstance d_a = make_occupant(m_a);
    RegionInstance d_b = make_occupant(m_b);
    RegionInstance c_a, c_b;
    Event e_a = create_full_instance(c_a, m_a, failure_tolerant_prs());
    Event e_b = create_full_instance(c_b, m_b, failure_tolerant_prs());

    // crossed deletions: each depends on the OTHER memory's pending
    //  creation.  If each owner honored the "different instance" exemption
    //  for a foreign instance id, both would fund and deadlock as a pair.
    DepDestroyArgs da;
    da.inst = d_a;
    da.precondition = e_b;
    DepDestroyArgs db;
    db.inst = d_b;
    db.precondition = e_a;
    alarm(30);
    Event s_a = p_a.spawn(DEP_DESTROY_TASK, &da, sizeof(da));
    Event s_b = p_b.spawn(DEP_DESTROY_TASK, &db, sizeof(db));
    Event::merge_events(s_a, s_b).wait();
    alarm(0);

    // both creations must fail honestly - a hang here is the cross ring
    bool pa = false, pb = false;
    alarm(60);
    e_a.wait_faultaware(pa);
    e_b.wait_faultaware(pb);
    alarm(0);
    if(!pa || !pb) {
      log_app.error() << "cross-memory creation wrongly funded: round=" << round;
      test_ok = false;
    }
    if(pa)
      d_a.destroy();
    else
      c_a.destroy();
    if(pb)
      d_b.destroy();
    else
      c_b.destroy();
    check_reclaimed(m_a, round, "cross-memory-a");
    check_reclaimed(m_b, round, "cross-memory-b");
  }
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  std::vector<Processor> all_procs;
  Machine::ProcessorQuery pq =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
    all_procs.push_back(*it);
  assert(!all_procs.empty());

  std::vector<Memory> mems;
  Machine::MemoryQuery mq =
      Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM);
  for(Machine::MemoryQuery::iterator it = mq.begin(); it != mq.end(); ++it)
    if((*it).capacity() > 0)
      mems.push_back(*it);

  log_app.warning() << "deferred_remote_alloc: nodes="
                    << Machine::get_machine().get_address_space_count()
                    << " procs=" << all_procs.size() << " memories=" << mems.size();

  for(size_t i = 0; i < mems.size(); i++) {
    Memory m = mems[i];

    // issue the dependent destroy from a node that is neither the creator
    //  nor the memory's owner when one exists - its (tiny) destroy request
    //  then races the creator's (large) creation request from an
    //  independent source, the widest race in the unfixed protocol; fall
    //  back to the owner, then to any other node
    Processor dep_proc = all_procs[0];
    for(size_t j = 0; j < all_procs.size(); j++) {
      AddressSpace s = all_procs[j].address_space();
      AddressSpace d = dep_proc.address_space();
      bool best = (s != p.address_space()) && (s != m.address_space());
      bool cur_best = (d != p.address_space()) && (d != m.address_space());
      bool good = (s == m.address_space()) && (s != p.address_space());
      bool cur_good = (d == m.address_space()) && (d != p.address_space());
      if(best || (good && !cur_best) ||
         (!cur_best && !cur_good && (s != p.address_space())))
        dep_proc = all_procs[j];
    }

    log_app.info() << "exercising memory " << m << " owner=" << m.address_space()
                   << " dep_proc=" << dep_proc;

    int first_try_funded = 0;
    for(int round = 0; round < rounds; round++) {
      test_dependent_destroy(m, dep_proc, round);
      test_handoff(m, round, first_try_funded);
      test_taint_funded_create(m, dep_proc, p, round);
      test_copy_gated_destroy(m, dep_proc, round);
      test_capi_laundering(m, dep_proc, round);
      test_root_ring(m, dep_proc, round);
      test_early_destroy(m, dep_proc, round);
    }
    log_app.info() << "memory " << m << ": handoff funded on first try "
                   << first_try_funded << "/" << rounds;
  }

  // shape 8 spans two memories with distinct owners
  test_cross_memory_ring(mems, all_procs);

  if(!test_ok) {
    log_app.fatal() << "FAILED";
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 1);
    return;
  }
  log_app.warning() << "PASSED";
  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
}

int main(int argc, const char **argv)
{
  Runtime rt;
  rt.init(&argc, (char ***)&argv);

  CommandLineParser clp;
  clp.add_option_int("-rounds", rounds);
  clp.add_option_int("-retries", max_retries);
  bool ok = clp.parse_command_line(argc, argv);
  assert(ok);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(p.kind(), false /*!global*/, TOP_LEVEL_TASK,
                                   CodeDescriptor(top_level_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, DEP_DESTROY_TASK,
                                   CodeDescriptor(dep_destroy_task),
                                   ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, SLEEPY_TASK,
                                   CodeDescriptor(sleepy_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, ALLOC_RESULT_TASK,
                                   CodeDescriptor(alloc_result_task),
                                   ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(p.kind(), false /*!global*/, COPY_DESTROY_TASK,
                                   CodeDescriptor(copy_destroy_task),
                                   ProfilingRequestSet())
      .external_wait();

  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  return rt.wait_for_shutdown();
}
