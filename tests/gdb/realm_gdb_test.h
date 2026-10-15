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

// Shared declarations for the program driven by the realm-gdb.py tests (see
// run_realm_gdb_tests.py).  Everything gdb needs to find has C linkage and a
// realm_gdb_test_ prefix so that the harness can name it without caring how
// the compiler mangles C++.

#ifndef REALM_GDB_TEST_H
#define REALM_GDB_TEST_H

#include "realm.h"

#if defined(__clang__)
#define RGT_NOINLINE __attribute__((noinline, used))
#else
#define RGT_NOINLINE __attribute__((noinline, noclone, used))
#endif

#define RGT_NUM_CHAIN 3
#define RGT_NUM_HOT 5
#define RGT_NUM_FHOT 4
#define RGT_RECURSE_DEPTH 40
#define RGT_CANARY_WORDS 64

// A type with a pretty-printer that, like libstdc++'s, can't print a value
// that is optimized out (realm_gdb_tests.install_pair_printer).
struct RgtPair {
  long first, second;
};

// The program's state, published for the gdb harness.  The harness uses the
// *_local addresses to work out which parked user thread is which (by
// checking whose stack contains them), and compares what the debugger shows
// against the expected values recorded here.
struct RealmGdbTestState {
  int num_parked;                       // tasks that arrived at the parked barrier
  int chain_markers[RGT_NUM_CHAIN];     // chain_marker in rgt_chain_level3
  void *chain_locals[RGT_NUM_CHAIN];    // &chain_marker in rgt_chain_level3
  void *recurse_local;                  // &recurse_depth in the innermost rgt_recurse
  void *hot_local;                      // a local in rgt_hot_task's frame
  void *fhot_local;                     // a local in rgt_fhot_task's frame
  void *opt_local;                      // a local in opt_task's frame
  long opt_sum;                         // computed by rgt_opt_wait
  long hot_wait_expected[RGT_NUM_HOT];  // hot0..hot4 in rgt_hot_wait
  long hot_probe_expected[RGT_NUM_HOT]; // hot0..hot4 in rgt_hot_probe
  double fhot_wait_expected[RGT_NUM_FHOT];  // fhot0..fhot3 in rgt_fhot_wait
  double fhot_probe_expected[RGT_NUM_FHOT]; // fhot0..fhot3 in rgt_hot_probe
  void *probe_local;                    // a local in rgt_probe_task's frame
  int widget_sums[RGT_NUM_CHAIN];       // widget.sum() of each chain task
  int num_finished;                     // parked tasks that have finished
  int failures;                         // problems found after resuming
};

extern "C" {
extern RealmGdbTestState realm_gdb_test_state;
extern volatile int realm_gdb_test_release;
extern volatile int realm_gdb_test_hold;
extern int realm_gdb_test_mode;

long realm_gdb_test_opaque(long x);
double realm_gdb_test_fopaque(double x);
void realm_gdb_test_wait_for_debugger(void);
void realm_gdb_test_release_parked(void);
}

// Defined in realm_gdb_test_opt.cc, which is always optimized so that these
// frames keep values in callee-saved registers across calls (or lose them).
int rgt_hot_wait(long seed, Realm::Event release, Realm::Barrier parked);
int rgt_fhot_wait(double seed, Realm::Event release, Realm::Barrier parked);
int rgt_opt_wait(long seed, RgtPair given, Realm::Event release, Realm::Barrier parked);
int rgt_hot_probe(long seed);

#endif
