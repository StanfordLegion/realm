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

// Always compiled with optimization (see CMakeLists.txt), so that hot0..hot4
// live in callee-saved registers across the calls below, and on aarch64
// fhot0..fhot3 in its callee-saved floating-point registers (d8..d15; on
// x86-64 none is callee-saved, so they are spilled to the stack).  A
// debugger showing a parked thread has to recover those registers from the
// thread's saved context; one that changes the stopped thread's registers
// has to put them all back.  Each function checks its values afterwards.

#include "realm_gdb_test.h"

RGT_NOINLINE int rgt_hot_wait(long seed, Realm::Event release, Realm::Barrier parked)
{
  long hot0 = realm_gdb_test_opaque(seed + 0);
  long hot1 = realm_gdb_test_opaque(seed + 1);
  long hot2 = realm_gdb_test_opaque(seed + 2);
  long hot3 = realm_gdb_test_opaque(seed + 3);
  long hot4 = realm_gdb_test_opaque(seed + 4);
  realm_gdb_test_state.hot_wait_expected[0] = hot0;
  realm_gdb_test_state.hot_wait_expected[1] = hot1;
  realm_gdb_test_state.hot_wait_expected[2] = hot2;
  realm_gdb_test_state.hot_wait_expected[3] = hot3;
  realm_gdb_test_state.hot_wait_expected[4] = hot4;
  __sync_fetch_and_add(&realm_gdb_test_state.num_parked, 1);

  parked.arrive();
  release.wait();

  int bad = 0;
  bad += (hot0 != realm_gdb_test_opaque(seed + 0));
  bad += (hot1 != realm_gdb_test_opaque(seed + 1));
  bad += (hot2 != realm_gdb_test_opaque(seed + 2));
  bad += (hot3 != realm_gdb_test_opaque(seed + 3));
  bad += (hot4 != realm_gdb_test_opaque(seed + 4));
  return bad;
}

// Parked with floating-point values live across the wait: on aarch64 they
// are kept in its callee-saved floating-point registers (d8..d15).  (A
// function of its own, so as not to change how rgt_hot_wait's values are
// kept.)
RGT_NOINLINE int rgt_fhot_wait(double seed, Realm::Event release, Realm::Barrier parked)
{
  double fhot0 = realm_gdb_test_fopaque(seed + 0.25);
  double fhot1 = realm_gdb_test_fopaque(seed + 1.25);
  double fhot2 = realm_gdb_test_fopaque(seed + 2.25);
  double fhot3 = realm_gdb_test_fopaque(seed + 3.25);
  realm_gdb_test_state.fhot_wait_expected[0] = fhot0;
  realm_gdb_test_state.fhot_wait_expected[1] = fhot1;
  realm_gdb_test_state.fhot_wait_expected[2] = fhot2;
  realm_gdb_test_state.fhot_wait_expected[3] = fhot3;
  __sync_fetch_and_add(&realm_gdb_test_state.num_parked, 1);

  parked.arrive();
  release.wait();

  int bad = 0;
  bad += (fhot0 != realm_gdb_test_fopaque(seed + 0.25));
  bad += (fhot1 != realm_gdb_test_fopaque(seed + 1.25));
  bad += (fhot2 != realm_gdb_test_fopaque(seed + 2.25));
  bad += (fhot3 != realm_gdb_test_fopaque(seed + 3.25));
  return bad;
}

// Parked with values that are optimized out where it waits: `gone`
// entirely, `half` in part (its first is kept for afterwards), and
// (depending on the compiler) the argument `given`.  gdb's own printer
// prints those that are entirely optimized out without asking their
// pretty-printer.
RGT_NOINLINE int rgt_opt_wait(long seed, RgtPair given, Realm::Event release,
                              Realm::Barrier parked)
{
  RgtPair gone = {realm_gdb_test_opaque(seed + 10), realm_gdb_test_opaque(seed + 11)};
  RgtPair half = {realm_gdb_test_opaque(seed + 12), realm_gdb_test_opaque(seed + 13)};
  realm_gdb_test_state.opt_sum =
      gone.first + gone.second + given.first + given.second + half.second;
  __sync_fetch_and_add(&realm_gdb_test_state.num_parked, 1);

  parked.arrive();
  release.wait();

  return half.first != realm_gdb_test_opaque(seed + 12);
}

// Inlined into rgt_hot_probe, so that a breakpoint here stops with an inline
// frame innermost.
static inline __attribute__((always_inline)) long rgt_inline_stop(long x)
{
  return realm_gdb_test_opaque(x);
}

RGT_NOINLINE int rgt_hot_probe(long seed)
{
  long hot0 = realm_gdb_test_opaque(seed + 0);
  long hot1 = realm_gdb_test_opaque(seed + 1);
  long hot2 = realm_gdb_test_opaque(seed + 2);
  long hot3 = realm_gdb_test_opaque(seed + 3);
  long hot4 = realm_gdb_test_opaque(seed + 4);
  double fhot0 = realm_gdb_test_fopaque(seed + 0.25);
  double fhot1 = realm_gdb_test_fopaque(seed + 1.25);
  double fhot2 = realm_gdb_test_fopaque(seed + 2.25);
  double fhot3 = realm_gdb_test_fopaque(seed + 3.25);
  realm_gdb_test_state.hot_probe_expected[0] = hot0;
  realm_gdb_test_state.hot_probe_expected[1] = hot1;
  realm_gdb_test_state.hot_probe_expected[2] = hot2;
  realm_gdb_test_state.hot_probe_expected[3] = hot3;
  realm_gdb_test_state.hot_probe_expected[4] = hot4;
  realm_gdb_test_state.fhot_probe_expected[0] = fhot0;
  realm_gdb_test_state.fhot_probe_expected[1] = fhot1;
  realm_gdb_test_state.fhot_probe_expected[2] = fhot2;
  realm_gdb_test_state.fhot_probe_expected[3] = fhot3;

  rgt_inline_stop(seed);
  realm_gdb_test_wait_for_debugger();

  int bad = 0;
  bad += (hot0 != realm_gdb_test_opaque(seed + 0));
  bad += (hot1 != realm_gdb_test_opaque(seed + 1));
  bad += (hot2 != realm_gdb_test_opaque(seed + 2));
  bad += (hot3 != realm_gdb_test_opaque(seed + 3));
  bad += (hot4 != realm_gdb_test_opaque(seed + 4));
  bad += (fhot0 != realm_gdb_test_fopaque(seed + 0.25));
  bad += (fhot1 != realm_gdb_test_fopaque(seed + 1.25));
  bad += (fhot2 != realm_gdb_test_fopaque(seed + 2.25));
  bad += (fhot3 != realm_gdb_test_fopaque(seed + 3.25));
  return bad;
}
