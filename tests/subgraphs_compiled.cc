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

// The original subgraph test (subgraphs.cc) ported to what the compiled
// subgraph implementation supports: the same writer task, copy, reader task,
// barrier arrival with a reduced interpolated value, external precondition
// and postcondition, argument interpolation, four chained instantiations and
// a final instantiation plus destroy from a (possibly remote) processor.
//
// Two things are inlined or replaced:
//  - the nested instantiation of an inner subgraph that performed the
//    barrier arrival is now an arrival operation of the outer graph;
//  - the reservation acquire/release pair that serialized the instantiations
//    is gone: the (default) INSTANTIATION_ORDER mode orders them, and the
//    external precondition of each instantiation is chained on the previous
//    one's external postcondition instead of the reverse order the
//    reservation test used.

#include "realm.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <csignal>
#include <cmath>

#include <time.h>

#include "osdep.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WRITER_TASK,
  READER_TASK,
  CLEANUP_TASK,
};

enum
{
  FID_DATA = 100,
};

enum
{
  REDOP_INT_ADD = 60,
};

class ReductionOpIntAdd {
public:
  typedef int LHS;
  typedef int RHS;

  template <bool EXCL>
  static void apply(LHS &lhs, RHS rhs)
  {
    lhs += rhs;
  }

  // both of these are optional
  static const RHS identity;

  template <bool EXCL>
  static void fold(RHS &rhs1, RHS rhs2)
  {
    rhs1 += rhs2;
  }
};

const ReductionOpIntAdd::RHS ReductionOpIntAdd::identity = 0;

struct WriterTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst;
  int wrval;
};

struct ReaderTaskArgs {
  IndexSpace<1> is;
  RegionInstance inst;
  int rdval;
};

void writer_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  const WriterTaskArgs &wargs = *reinterpret_cast<const WriterTaskArgs *>(args);
  AffineAccessor<int, 1> acc(wargs.inst, FID_DATA);
  for(IndexSpaceIterator<1> it(wargs.is); it.valid; it.step())
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      acc[it2.p] = it2.p[0] + wargs.wrval;
    }
}

int correct = 0;

void reader_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  const ReaderTaskArgs &rargs = *reinterpret_cast<const ReaderTaskArgs *>(args);
  AffineAccessor<int, 1> acc(rargs.inst, FID_DATA);
  for(IndexSpaceIterator<1> it(rargs.is); it.valid; it.step())
    for(PointInRectIterator<1> it2(it.rect); it2.valid; it2.step()) {
      int expval = it2.p[0] + rargs.rdval;
      int actval = acc[it2.p];
      if(expval == actval)
        correct++;
      else
        log_app.error() << "MISMATCH: " << it2.p << ": " << actval << " != " << expval;
    }
}

struct CleanupTaskArgs {
  Event precond;
  Subgraph subgraph;
  UserEvent cleanup_done;
};

struct SubgraphArgs {
  int rdwr_val;
  int arr_inc;
};

void cleanup_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                  Processor p)
{
  const CleanupTaskArgs &cta = *static_cast<const CleanupTaskArgs *>(args);

  SubgraphArgs sg_args;
  sg_args.rdwr_val = 200;
  sg_args.arr_inc = 10;

  // the whole instantiation uses a global precondition; the definition's
  //  external precondition is satisfied from the start
  std::vector<Event> preconditions(1, Event::NO_EVENT);
  std::vector<Event> postconditions(1);
  Event e = cta.subgraph.instantiate(&sg_args, sizeof(sg_args), ProfilingRequestSet(),
                                     preconditions, postconditions, cta.precond);

  // technically e already dominates postconditions[0], but do this to make
  //  sure the external postcondition actually triggers
  e = Event::merge_events(e, postconditions[0]);

  // destroy the subgraph and trigger the done event upon completion of this
  //  instance (from a remote node, this exercises the ordering of the
  //  instantiate and destroy messages)
  cta.cleanup_done.trigger(cta.subgraph.destroy(e));
}

// clang doesn't permit use of offsetof when there are non-POD types involved
// and gcc doesn't permit computing the address of a dereference of a nullptr
template <typename T, typename U>
static size_t compute_offset(U T::*field_ptr)
{
  T thing;
  size_t ofs =
      (reinterpret_cast<size_t>(&(thing.*field_ptr)) - reinterpret_cast<size_t>(&thing));
  return ofs;
}

#define OFFSETOF(type, field) compute_offset<type>(&type::field)

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  log_app.print() << "Realm compiled subgraphs test";

  // do everything on this processor - get a good memory to use
  Memory m = Machine::MemoryQuery(Machine::get_machine()).has_affinity_to(p).first();
  assert(m.exists());

  IndexSpace<1> is = Rect<1>(0, 9);
  RegionInstance inst1, inst2;
  std::map<FieldID, size_t> field_sizes;
  field_sizes[FID_DATA] = sizeof(int);

  Event e = RegionInstance::create_instance(inst1, m, is, field_sizes,
                                            0 /*block_size=SOA*/, ProfilingRequestSet());

  e = RegionInstance::create_instance(inst2, m, is, field_sizes, 0 /*block_size=SOA*/,
                                      ProfilingRequestSet(), e);

  // immediate mode - no subgraph
  WriterTaskArgs w_args;
  w_args.is = is;
  w_args.inst = inst1;
  w_args.wrval = 5;
  e = p.spawn(WRITER_TASK, &w_args, sizeof(w_args), e);

  std::vector<CopySrcDstField> copy_src(1), copy_dst(1);
  copy_src[0].set_field(inst1, FID_DATA, sizeof(int));
  copy_dst[0].set_field(inst2, FID_DATA, sizeof(int));
  e = is.copy(copy_src, copy_dst, ProfilingRequestSet(), e);

  ReaderTaskArgs r_args;
  r_args.is = is;
  r_args.inst = inst2;
  r_args.rdval = 5;
  e = p.spawn(READER_TASK, &r_args, sizeof(r_args), e);

  e.wait();

  int ival = 77;
  Barrier barrier = Barrier::create_barrier(5, REDOP_INT_ADD, &ival, sizeof(ival));

  // the concurrency mode is left at its default (INSTANTIATION_ORDER)
  SubgraphDefinition sd;
  sd.tasks.resize(2);
  sd.tasks[0].proc = p;
  sd.tasks[0].task_id = WRITER_TASK;
  w_args.wrval = 6;
  sd.tasks[0].args.set(&w_args, sizeof(w_args));
  sd.tasks[1].proc = p;
  sd.tasks[1].task_id = READER_TASK;
  r_args.rdval = 7;
  sd.tasks[1].args.set(&r_args, sizeof(r_args));

  sd.copies.resize(1);
  sd.copies[0].space = is;
  sd.copies[0].srcs = copy_src;
  sd.copies[0].dsts = copy_dst;

  // the arrival the original test's inner subgraph performed, with the same
  //  base value (7); each instantiation adds its arr_inc by interpolation
  sd.arrivals.resize(1);
  sd.arrivals[0].barrier = barrier;
  sd.arrivals[0].count = 1;
  int base_value = 7;
  sd.arrivals[0].reduce_value.set(&base_value, sizeof(base_value));

  sd.dependencies.resize(5);
  // ext precond 0 -> writer -> copy -> reader; writer -> arrival;
  //  writer -> ext postcond 0 (triggers mid-graph, like the acquire did)
  sd.dependencies[0].src_op_kind = SubgraphDefinition::OPKIND_EXT_PRECOND;
  sd.dependencies[0].src_op_index = 0;
  sd.dependencies[0].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[0].tgt_op_index = 0;

  sd.dependencies[1].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[1].src_op_index = 0;
  sd.dependencies[1].tgt_op_kind = SubgraphDefinition::OPKIND_COPY;
  sd.dependencies[1].tgt_op_index = 0;

  sd.dependencies[2].src_op_kind = SubgraphDefinition::OPKIND_COPY;
  sd.dependencies[2].src_op_index = 0;
  sd.dependencies[2].tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[2].tgt_op_index = 1;

  sd.dependencies[3].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[3].src_op_index = 0;
  sd.dependencies[3].tgt_op_kind = SubgraphDefinition::OPKIND_ARRIVAL;
  sd.dependencies[3].tgt_op_index = 0;

  sd.dependencies[4].src_op_kind = SubgraphDefinition::OPKIND_TASK;
  sd.dependencies[4].src_op_index = 0;
  sd.dependencies[4].tgt_op_kind = SubgraphDefinition::OPKIND_EXT_POSTCOND;
  sd.dependencies[4].tgt_op_index = 0;

  sd.interpolations.resize(3);
  sd.interpolations[0].offset = OFFSETOF(SubgraphArgs, rdwr_val);
  sd.interpolations[0].bytes = sizeof(int);
  sd.interpolations[0].target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  sd.interpolations[0].target_index = 0;
  sd.interpolations[0].target_offset = OFFSETOF(WriterTaskArgs, wrval);

  sd.interpolations[1].offset = OFFSETOF(SubgraphArgs, rdwr_val);
  sd.interpolations[1].bytes = sizeof(int);
  sd.interpolations[1].target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  sd.interpolations[1].target_index = 1;
  sd.interpolations[1].target_offset = OFFSETOF(ReaderTaskArgs, rdval);

  sd.interpolations[2].offset = OFFSETOF(SubgraphArgs, arr_inc);
  sd.interpolations[2].bytes = sizeof(int);
  sd.interpolations[2].target_kind =
      SubgraphDefinition::Interpolation::TARGET_ARRIVAL_VALUE;
  sd.interpolations[2].target_index = 0;
  sd.interpolations[2].target_offset = 0;
  sd.interpolations[2].redop_id = REDOP_INT_ADD;

  Subgraph sg;
  e = Subgraph::create_subgraph(sg, sd, ProfilingRequestSet());

  std::vector<UserEvent> start_events(4);
  std::vector<Event> written_events(4);
  std::vector<Event> finish_events(4);
  for(int i = 0; i < 4; i++) {
    SubgraphArgs sg_args;
    sg_args.rdwr_val = 100 * (i + 1);
    sg_args.arr_inc = (i + 1);
    start_events[i] = UserEvent::create_user_event();
    std::vector<Event> preconds(1, start_events[i]);
    std::vector<Event> postconds(1);
    finish_events[i] = sg.instantiate(&sg_args, sizeof(sg_args), ProfilingRequestSet(),
                                      preconds, postconds, e);
    written_events[i] = postconds[0];
  }
  e = Event::merge_events(finish_events);

  // do one more invocation on a remote processor if we can, and have it
  //  do the subgraph cleanup
  Machine::ProcessorQuery pq(Machine::get_machine());
  pq.only_kind(Processor::LOC_PROC);
  Processor lastp = Processor::NO_PROC;
  for(Machine::ProcessorQuery::iterator it = pq.begin(); it != pq.end(); ++it)
    lastp = *it;
  assert(lastp.exists());

  // pass the merged finish event to this task as an arg, not a precondition
  //  of the task itself
  CleanupTaskArgs cleanup_args;
  cleanup_args.precond = e;
  cleanup_args.subgraph = sg;
  cleanup_args.cleanup_done = UserEvent::create_user_event();
  lastp.spawn(CLEANUP_TASK, &cleanup_args, sizeof(CleanupTaskArgs),
              ProfilingRequestSet());

  // connect up start events: each instantiation's precondition is the
  //  previous one's writer having run (the instantiations are ordered anyway)
  start_events[0].trigger();
  start_events[1].trigger(written_events[0]);
  start_events[2].trigger(written_events[1]);
  start_events[3].trigger(written_events[2]);

  cleanup_args.cleanup_done.wait();

  bool ok = true;

  // make sure the barrier got triggered with the right values
  barrier.wait();
  int exp_bar_value = 77 + (7 + 1) + (7 + 2) + (7 + 3) + (7 + 4) + (7 + 10);
  int act_bar_value = 0;
  {
    bool got = barrier.get_result(&act_bar_value, sizeof(act_bar_value));
    assert(got);
  }
  if(exp_bar_value != act_bar_value) {
    log_app.error() << "barrier value mismatch: exp=" << exp_bar_value
                    << " act=" << act_bar_value;
    ok = false;
  }

  int expcorrect = 60; // counting the cleanup task
  if(expcorrect != correct) {
    log_app.error() << correct << " correct comparisons (out of " << expcorrect << ")";
    ok = false;
  }

  Runtime::get_runtime().shutdown(Event::NO_EVENT, ok ? 0 : 1);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(WRITER_TASK, writer_task);
  rt.register_task(READER_TASK, reader_task);
  rt.register_task(CLEANUP_TASK, cleanup_task);

  rt.register_reduction<ReductionOpIntAdd>(REDOP_INT_ADD);

  // select a processor to run the top level task on
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  // collective launch of a single main task
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  // main task will call shutdown - wait for that and return the exit code
  return rt.wait_for_shutdown();
}
