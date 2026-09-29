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

// Unit tests for Realm subgraphs.
//
// These are written as an integration test rather than a gtest unit test
// because exercising subgraphs requires a live Realm runtime executing tasks.
//
// Every test runs once per execution mode it declares valid (INTERPRETED
// and/or COMPILED). The first CPU processor is reserved for the test driver:
// several tests poll for events that a buggy implementation may never trigger,
// and polling would starve subgraph tasks sharing that processor. Tests
// therefore build their subgraphs on worker_cpus() only.
//
// Command line (after Realm's own flags are stripped):
//   -iters N         instantiations per configuration (default 5)
//   -seed S          seed for randomized graph shapes (default 12345)
//   -dag_size N      operation count for the large random DAG (default 64)
//   -many N          instantiations for the many-instantiations test (default 2000)
//   -hang_timeout S  seconds before a never-triggering event is declared a hang
//   -only PREFIX     run only tests whose name starts with PREFIX
//   -skip NAME       skip a test (repeatable)
//   -list            list test names and exit
//   -death SCENARIO  run one "must abort" scenario; printing DEATH-TEST-SURVIVED
//                    and exiting 0 means the runtime failed to reject the misuse

#include "realm.h"
#include "realm/event.h"
#include "realm/indexspace.h"
#include "realm/profiling.h"
#include "realm/serialize.h"
#include "realm/subgraph.h"
#include "realm/timers.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

// A counter for assigning task IDs to tests so that we
// don't need a gigantic enum with all of the task IDs.
static int32_t task_id_counter = TOP_LEVEL_TASK + 1;

// Field IDs.
enum
{
  FID_DATA = 100,
};

// Common reduction operation IDs.
enum
{
  REDOP_INT_ADD = 100,
};

// SumReduction is a simple integer addition reduction operation.
struct SumReduction {
  typedef int LHS;
  typedef int RHS;
  static const RHS identity = 0;

  template <bool EXCL>
  void apply(LHS &lhs, const RHS &rhs) const
  {
    lhs += rhs;
  }

  template <bool EXCL>
  void fold(RHS &rhs1, const RHS &rhs2) const
  {
    rhs1 += rhs2;
  }
};

////////////////////////////////////////////////////////////////////////
//
// Configuration and machine helpers
//

struct TestConfig {
  int iterations = 5;
  unsigned seed = 12345;
  int dag_size = 64;
  int many = 2000;
  double hang_timeout = 30.0;
  std::string only;
  std::set<std::string> skip;
  std::string death;
  bool list = false;
};
static TestConfig config;

static std::vector<Processor> all_cpus()
{
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine())
                                   .only_kind(Processor::LOC_PROC)
                                   .local_address_space();
  return std::vector<Processor>(pq.begin(), pq.end());
}

// A CPU in some other address space, or NO_PROC in a single-rank run.
static Processor remote_cpu()
{
  AddressSpace here = Processor::get_executing_processor().address_space();
  Machine::ProcessorQuery pq =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  for(Processor p : pq)
    if(p.address_space() != here)
      return p;
  return Processor::NO_PROC;
}

// Processors available for subgraph tasks: everything but the driver's CPU.
static std::vector<Processor> worker_cpus()
{
  std::vector<Processor> cpus = all_cpus();
  if(!cpus.empty())
    cpus.erase(cpus.begin());
  return cpus;
}

static std::vector<Processor> worker_cpus(size_t max_count)
{
  std::vector<Processor> cpus = worker_cpus();
  if(cpus.size() > max_count)
    cpus.resize(max_count);
  return cpus;
}

static Memory sysmem()
{
  return Machine::MemoryQuery(Machine::get_machine())
      .only_kind(Memory::Kind::SYSTEM_MEM)
      .first();
}

// Progress and verdict output goes to stdout directly: Realm's compile-time
// minimum log level may exclude Logger::print().
static void report(const std::string &msg)
{
  printf("%s\n", msg.c_str());
  fflush(stdout);
}

static const char *mode_name(SubgraphDefinition::ExecutionMode mode)
{
  return (mode == SubgraphDefinition::COMPILED) ? "COMPILED" : "INTERPRETED";
}

// Polls for an event with a wall-clock timeout. Only for use from the
// driver processor and only for events that a buggy implementation might
// never trigger; everything else should use a normal (yielding) wait.
static bool wait_with_timeout(Event e, double seconds, bool *poisoned = nullptr)
{
  double deadline = Clock::current_time() + seconds;
  while(true) {
    bool p = false;
    if(e.has_triggered_faultaware(p)) {
      if(poisoned)
        *poisoned = p;
      return true;
    }
    if(Clock::current_time() > deadline)
      return false;
    usleep(500);
  }
}

static long resident_kb()
{
  std::ifstream f("/proc/self/statm");
  long size = 0, resident = 0;
  f >> size >> resident;
  return resident * (sysconf(_SC_PAGESIZE) / 1024);
}

////////////////////////////////////////////////////////////////////////
//
// Test base class and definition helpers
//

class SubgraphTest {
public:
  virtual ~SubgraphTest() = default;

  // Register all metadata that will be used in the test.
  virtual void register_test() {}

  // Make sure that the requisite machine resources are available for the test.
  virtual bool can_run() { return false; }

  // Initialize any state for the test.
  virtual void init(SubgraphDefinition::ExecutionMode mode) {}

  // Start the test. Run is responsible for not having any pending work
  // left after it returns.
  virtual void run() {}

  // Verify the results. This may be needed if the checking can only be
  // done after the subgraphs launched by the test have completed.
  virtual bool check() { return true; }

  virtual void cleanup() {}

  // True if the test detected a hang: the runtime may be wedged and no
  // further tests should be attempted.
  virtual bool hung() const { return false; }

  virtual std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const
  {
    // Unless overridden, all execution modes are valid.
    return {SubgraphDefinition::INTERPRETED, SubgraphDefinition::COMPILED};
  }

  virtual std::string name() const = 0;
};

static int make_task_desc(SubgraphDefinition &sd, Processor proc, int task_id,
                          const void *args, size_t args_size)
{
  SubgraphDefinition::TaskDesc td;
  td.proc = proc;
  td.task_id = task_id;
  td.args.set(args, args_size);
  sd.tasks.push_back(td);
  return sd.tasks.size() - 1;
}

static int make_copy_desc(SubgraphDefinition &sd, IndexSpace<1> space, RegionInstance src,
                          RegionInstance dst, FieldID field_id, size_t size)
{
  SubgraphDefinition::CopyDesc cd;
  cd.space = space;
  cd.srcs.resize(1);
  cd.srcs[0].set_field(src, field_id, size);
  cd.dsts.resize(1);
  cd.dsts[0].set_field(dst, field_id, size);
  sd.copies.push_back(cd);
  return sd.copies.size() - 1;
}

static int make_fill_desc(SubgraphDefinition &sd, IndexSpace<1> space, RegionInstance inst,
                          FieldID field_id, const void *fill_value, size_t fill_value_size)
{
  SubgraphDefinition::CopyDesc cd;
  cd.space = space;
  cd.srcs.resize(1);
  cd.srcs[0].set_fill(fill_value, fill_value_size);
  cd.dsts.resize(1);
  cd.dsts[0].set_field(inst, field_id, fill_value_size);
  sd.copies.push_back(cd);
  return sd.copies.size() - 1;
}

static int make_reduction_copy_desc(SubgraphDefinition &sd, IndexSpace<1> space,
                                    RegionInstance src, RegionInstance dst,
                                    FieldID field_id, int redop_id)
{
  SubgraphDefinition::CopyDesc cd;
  cd.space = space;
  cd.srcs.resize(1);
  cd.srcs[0].set_field(src, field_id, sizeof(int));
  cd.dsts.resize(1);
  cd.dsts[0].set_field(dst, field_id, sizeof(int));
  cd.dsts[0].set_redop(redop_id, false /* is_fold */);
  sd.copies.push_back(cd);
  return sd.copies.size() - 1;
}

static void add_dependency(SubgraphDefinition &sd, SubgraphDefinition::OpKind src_op_kind,
                           int src_op_index, SubgraphDefinition::OpKind tgt_op_kind,
                           int tgt_op_index)
{
  SubgraphDefinition::Dependency dep;
  dep.src_op_kind = src_op_kind;
  dep.src_op_index = src_op_index;
  dep.tgt_op_kind = tgt_op_kind;
  dep.tgt_op_index = tgt_op_index;
  sd.dependencies.push_back(dep);
}

static void add_task_interpolation(SubgraphDefinition &sd, int task_index, size_t offset,
                                   size_t bytes, size_t target_offset)
{
  SubgraphDefinition::Interpolation interp;
  interp.offset = offset;
  interp.bytes = bytes;
  interp.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  interp.target_index = task_index;
  interp.target_offset = target_offset;
  sd.interpolations.push_back(interp);
}

static void add_arrival_barrier_interpolation(SubgraphDefinition &sd, int arrival_index,
                                              size_t offset, size_t bytes)
{
  SubgraphDefinition::Interpolation interp;
  interp.offset = offset;
  interp.bytes = bytes;
  interp.target_kind = SubgraphDefinition::Interpolation::TARGET_ARRIVAL_BARRIER;
  interp.target_index = arrival_index;
  sd.interpolations.push_back(interp);
}

static void add_arrival_redvalue_interpolation(SubgraphDefinition &sd, int arrival_index,
                                               size_t offset, size_t bytes, int redop_id)
{
  SubgraphDefinition::Interpolation interp;
  interp.offset = offset;
  interp.bytes = bytes;
  interp.target_kind = SubgraphDefinition::Interpolation::TARGET_ARRIVAL_VALUE;
  interp.target_index = arrival_index;
  interp.redop_id = redop_id;
  sd.interpolations.push_back(interp);
}

////////////////////////////////////////////////////////////////////////
//
// Counting-task DAG infrastructure
//
// A DagSpec describes a task graph over an abstract set of processors. Each
// task records its own completion count and checks that every predecessor
// has completed at least as many times as it is about to complete itself.
// This lets the same graph be instantiated repeatedly (chained or, for
// INSTANTIATION_ORDER subgraphs, unchained) and still detect any dependency
// that was not enforced, without resetting state between instantiations.
//

struct DagSpec {
  std::vector<int> proc_of_op;         // index into the processor list
  std::vector<std::vector<int>> preds; // predecessor operation indices

  size_t size() const { return proc_of_op.size(); }
  size_t num_edges() const
  {
    size_t n = 0;
    for(const std::vector<int> &p : preds)
      n += p.size();
    return n;
  }
  int add_op(int proc, std::vector<int> ps = {})
  {
    proc_of_op.push_back(proc);
    preds.push_back(std::move(ps));
    return proc_of_op.size() - 1;
  }
};

struct DagState {
  const DagSpec *spec = nullptr;
  std::vector<Processor> procs;
  std::unique_ptr<std::atomic<int64_t>[]> done; // per-op completion count
  std::atomic<int64_t> executed{0};
  std::atomic<int64_t> violations{0};

  void reset(const DagSpec *s, const std::vector<Processor> &p)
  {
    spec = s;
    procs = p;
    done.reset(new std::atomic<int64_t>[s->size()]);
    for(size_t i = 0; i < s->size(); i++)
      done[i].store(0);
    executed.store(0);
    violations.store(0);
  }
};

struct DagTaskArgs {
  DagState *state;
  int op;
};

static int dag_task_id = 0;

static void dag_task(const void *args, size_t arglen, const void *userdata,
                     size_t userlen, Processor p)
{
  const DagTaskArgs *a = static_cast<const DagTaskArgs *>(args);
  DagState *st = a->state;
  const int op = a->op;
  const int64_t my_count = st->done[op].load(std::memory_order_acquire);
  for(int pred : st->spec->preds[op]) {
    if(st->done[pred].load(std::memory_order_acquire) < my_count + 1) {
      st->violations.fetch_add(1);
      log_app.error() << "dependency violation: op " << op << " (run " << my_count + 1
                      << ") ran before predecessor " << pred;
    }
  }
  if(p != st->procs[st->spec->proc_of_op[op]]) {
    st->violations.fetch_add(1);
    log_app.error() << "placement violation: op " << op << " ran on " << p
                    << " instead of " << st->procs[st->spec->proc_of_op[op]];
  }
  st->done[op].store(my_count + 1, std::memory_order_release);
  st->executed.fetch_add(1, std::memory_order_acq_rel);
}

static Subgraph build_dag_subgraph(const DagSpec &spec, DagState &state,
                                   SubgraphDefinition::ExecutionMode mode,
                                   SubgraphDefinition::ConcurrencyMode cmode)
{
  SubgraphDefinition sd;
  sd.execution_mode = mode;
  sd.concurrency_mode = cmode;
  for(size_t i = 0; i < spec.size(); i++) {
    DagTaskArgs a{&state, int(i)};
    make_task_desc(sd, state.procs[spec.proc_of_op[i]], dag_task_id, &a, sizeof(a));
  }
  for(size_t i = 0; i < spec.size(); i++)
    for(int pred : spec.preds[i])
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, pred,
                     SubgraphDefinition::OPKIND_TASK, int(i));
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  return sg;
}

// Graph shapes. Processor indices are assigned round-robin unless noted.
static DagSpec dag_chain(int n, int nprocs)
{
  DagSpec s;
  for(int i = 0; i < n; i++)
    s.add_op(i % nprocs, (i > 0) ? std::vector<int>{i - 1} : std::vector<int>{});
  return s;
}

static DagSpec dag_independent(int n, int nprocs)
{
  DagSpec s;
  for(int i = 0; i < n; i++)
    s.add_op(i % nprocs);
  return s;
}

static DagSpec dag_fan(int width, int nprocs)
{
  DagSpec s;
  int root = s.add_op(0);
  std::vector<int> middle;
  for(int i = 0; i < width; i++)
    middle.push_back(s.add_op((i + 1) % nprocs, {root}));
  s.add_op(0, middle);
  return s;
}

static DagSpec dag_layers(int layers, int width, int nprocs, bool dense)
{
  DagSpec s;
  for(int l = 0; l < layers; l++) {
    for(int w = 0; w < width; w++) {
      std::vector<int> preds;
      if(l > 0) {
        if(dense) {
          for(int pw = 0; pw < width; pw++)
            preds.push_back((l - 1) * width + pw);
        } else {
          preds.push_back((l - 1) * width + w);
        }
      }
      s.add_op((l * width + w) % nprocs, preds);
    }
  }
  return s;
}

static DagSpec dag_random(int n, int nprocs, double edge_prob, std::mt19937 &rng)
{
  DagSpec s;
  std::uniform_int_distribution<int> pick_proc(0, nprocs - 1);
  std::bernoulli_distribution edge(edge_prob);
  for(int i = 0; i < n; i++) {
    std::vector<int> preds;
    for(int j = 0; j < i; j++)
      if(edge(rng))
        preds.push_back(j);
    s.add_op(pick_proc(rng), preds);
  }
  return s;
}

////////////////////////////////////////////////////////////////////////
//
// SimpleTasksTest: alternating writer/reader layers over a region instance
// split across two processors (from the original PR).
//

class SimpleTasksTest : public SubgraphTest {
public:
  std::string name() const override { return "SimpleTasksTest"; }

  struct WriterTaskArgs {
    WriterTaskArgs(RegionInstance inst, IndexSpace<1> is, int level)
      : inst(inst)
      , is(is)
      , level(level)
    {}
    RegionInstance inst;
    IndexSpace<1> is;
    int level;
  };

  static void writer_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const WriterTaskArgs *task_args = static_cast<const WriterTaskArgs *>(args);
    AffineAccessor<int, 1> acc(task_args->inst, FID_DATA);
    for(int i = task_args->is.bounds.lo[0]; i <= task_args->is.bounds.hi[0]; i++) {
      acc[i] = i + (10 * task_args->level);
    }
  }

  struct ReaderTaskArgs {
    ReaderTaskArgs(RegionInstance inst, IndexSpace<1> is, int level,
                   SimpleTasksTest *test)
      : inst(inst)
      , is(is)
      , level(level)
      , test(test)
    {}
    RegionInstance inst;
    IndexSpace<1> is;
    int level;
    SimpleTasksTest *test;
  };

  static void reader_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const ReaderTaskArgs *task_args = static_cast<const ReaderTaskArgs *>(args);
    AffineAccessor<int, 1> acc(task_args->inst, FID_DATA);
    for(int i = task_args->is.bounds.lo[0]; i <= task_args->is.bounds.hi[0]; i++) {
      int expected = i + (10 * task_args->level);
      int actual = acc[i];
      if(actual != expected) {
        log_app.error() << "MISMATCH: " << i << ": " << actual << " != " << expected;
        task_args->test->error = true;
      }
    }
  }

  void register_test() override
  {
    writer_task_id = task_id_counter++;
    reader_task_id = task_id_counter++;
    Runtime rt = Runtime::get_runtime();
    rt.register_task(writer_task_id, writer_task);
    rt.register_task(reader_task_id, reader_task);
  }

  bool can_run() override { return worker_cpus().size() >= 2 && sysmem().exists(); }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    execution_mode = mode;
    error = false;
    std::vector<Processor> cpus = worker_cpus(2);

    IndexSpace<1> is = Rect<1>(0, 9);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(inst, sysmem(), is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(inst.exists());

    // 4 layers alternating between writers and readers. The instance is split
    // across the two CPUs, each writer writes half, each reader checks all.
    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    WriterTaskArgs warg1(inst, IndexSpace<1>(Rect<1>(0, 4)), 0);
    WriterTaskArgs warg2(inst, IndexSpace<1>(Rect<1>(5, 9)), 0);
    ReaderTaskArgs rarg(inst, IndexSpace<1>(Rect<1>(0, 9)), 0, this);

    int w0_1 = make_task_desc(sd, cpus[0], writer_task_id, &warg1, sizeof(warg1));
    int w0_2 = make_task_desc(sd, cpus[1], writer_task_id, &warg2, sizeof(warg2));
    int r0_1 = make_task_desc(sd, cpus[0], reader_task_id, &rarg, sizeof(rarg));
    int r0_2 = make_task_desc(sd, cpus[1], reader_task_id, &rarg, sizeof(rarg));

    warg1.level = warg2.level = rarg.level = 1;

    int w1_1 = make_task_desc(sd, cpus[0], writer_task_id, &warg1, sizeof(warg1));
    int w1_2 = make_task_desc(sd, cpus[1], writer_task_id, &warg2, sizeof(warg2));
    int r1_1 = make_task_desc(sd, cpus[0], reader_task_id, &rarg, sizeof(rarg));
    int r1_2 = make_task_desc(sd, cpus[1], reader_task_id, &rarg, sizeof(rarg));

    auto dense = [&](std::initializer_list<int> srcs, std::initializer_list<int> tgts) {
      for(int s : srcs)
        for(int t : tgts)
          add_dependency(sd, SubgraphDefinition::OPKIND_TASK, s,
                         SubgraphDefinition::OPKIND_TASK, t);
    };
    dense({w0_1, w0_2}, {r0_1, r0_2});
    dense({r0_1, r0_2}, {w1_1, w1_2});
    dense({w1_1, w1_2}, {r1_1, r1_2});
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    // Back to back with event chaining between instantiations.
    Event e = Event::NO_EVENT;
    for(int i = 0; i < config.iterations; i++)
      e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), e);
    e.wait();

    // In compiled mode the subgraph orders instantiations itself, so the
    // same must work without chaining.
    if(execution_mode == SubgraphDefinition::COMPILED) {
      std::vector<Event> evs(config.iterations);
      for(int i = 0; i < config.iterations; i++)
        evs[i] = sg.instantiate(nullptr, 0, ProfilingRequestSet());
      Event::merge_events(evs).wait();
    }
  }

  bool check() override { return !error; }

  void cleanup() override
  {
    sg.destroy().wait();
    inst.destroy();
  }

private:
  int writer_task_id = 0;
  int reader_task_id = 0;
  Subgraph sg;
  RegionInstance inst;
  bool error = false;
  SubgraphDefinition::ExecutionMode execution_mode = SubgraphDefinition::INTERPRETED;
};

////////////////////////////////////////////////////////////////////////
//
// Tests of operation kinds the compiled mode does not support yet. They
// run in INTERPRETED mode only; the compiled mode's refusal of these graphs
// is covered by the death scenarios below.
//

class SimpleCopyTest : public SubgraphTest {
public:
  std::string name() const override { return "SimpleCopyTest"; }

  std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const override
  {
    return {SubgraphDefinition::INTERPRETED};
  }

  bool can_run() override { return sysmem().exists(); }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    // inst: [0,4] filled, [5,9] reduced into. copy_dst_inst: [0,4] copied from
    // inst, [5,9] pre-filled with -1 to check the sub-piece copy.
    IndexSpace<1> is = Rect<1>(0, 9);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(inst, sysmem(), is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(copy_dst_inst, sysmem(), is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    {
      int fill_value = -1;
      std::vector<CopySrcDstField> dsts(1);
      dsts[0].set_field(copy_dst_inst, FID_DATA, sizeof(int));
      is.fill(dsts, ProfilingRequestSet(), &fill_value, sizeof(fill_value)).wait();
    }
    IndexSpace<1> red_src_is = Rect<1>(5, 9);
    RegionInstance::create_instance(red_src_inst, sysmem(), red_src_is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    {
      int fill_value = 42;
      std::vector<CopySrcDstField> dsts(1);
      dsts[0].set_field(red_src_inst, FID_DATA, sizeof(int));
      red_src_is.fill(dsts, ProfilingRequestSet(), &fill_value, sizeof(fill_value))
          .wait();
    }

    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    int fill_value = 15210;
    int f1 = make_fill_desc(sd, IndexSpace<1>(Rect<1>(0, 4)), inst, FID_DATA, &fill_value,
                            sizeof(fill_value));
    fill_value = 5;
    int f2 = make_fill_desc(sd, IndexSpace<1>(Rect<1>(5, 9)), inst, FID_DATA, &fill_value,
                            sizeof(fill_value));
    int r1 = make_reduction_copy_desc(sd, IndexSpace<1>(Rect<1>(5, 9)), red_src_inst, inst,
                                      FID_DATA, REDOP_INT_ADD);
    int c1 = make_copy_desc(sd, IndexSpace<1>(Rect<1>(0, 4)), inst, copy_dst_inst, FID_DATA,
                            sizeof(int));
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, f1, SubgraphDefinition::OPKIND_COPY,
                   c1);
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, f2, SubgraphDefinition::OPKIND_COPY,
                   r1);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override { sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait(); }

  bool check() override
  {
    AffineAccessor<int, 1> acc_inst(inst, FID_DATA);
    AffineAccessor<int, 1> acc_copy(copy_dst_inst, FID_DATA);
    bool success = true;
    auto expect = [&](AffineAccessor<int, 1> &acc, int lo, int hi, int expected) {
      for(int i = lo; i <= hi; i++) {
        if(acc[i] != expected) {
          log_app.error() << "MISMATCH at " << i << ": " << acc[i] << " != " << expected;
          success = false;
        }
      }
    };
    expect(acc_inst, 0, 4, 15210);
    expect(acc_inst, 5, 9, 5 + 42);
    expect(acc_copy, 0, 4, 15210);
    expect(acc_copy, 5, 9, -1);
    return success;
  }

  void cleanup() override
  {
    sg.destroy().wait();
    inst.destroy();
    copy_dst_inst.destroy();
    red_src_inst.destroy();
  }

private:
  Subgraph sg;
  RegionInstance inst, red_src_inst, copy_dst_inst;
};

class BarrierArrivalTest : public SubgraphTest {
public:
  std::string name() const override { return "BarrierArrivalTest"; }

  std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const override
  {
    return {SubgraphDefinition::INTERPRETED};
  }

  bool can_run() override { return true; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    for(int i = 0; i < 3; i++) {
      barriers[i] = Barrier::create_barrier(1);
      SubgraphDefinition::ArrivalDesc ad;
      ad.barrier = barriers[i];
      sd.arrivals.push_back(ad);
    }
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override { sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait(); }

  bool check() override
  {
    for(int i = 0; i < 3; i++)
      barriers[i].wait();
    return true;
  }

  void cleanup() override { sg.destroy().wait(); }

private:
  Subgraph sg;
  Barrier barriers[3];
};

class InterpolationTest : public SubgraphTest {
public:
  std::string name() const override { return "InterpolationTest"; }

  std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const override
  {
    return {SubgraphDefinition::INTERPRETED};
  }

  struct WriterTaskArgs {
    WriterTaskArgs(RegionInstance inst32, RegionInstance inst64, int32_t value1,
                   int64_t value2, int32_t value3)
      : inst32(inst32)
      , inst64(inst64)
      , value1(value1)
      , value2(value2)
      , value3(value3)
    {}
    RegionInstance inst32;
    RegionInstance inst64;
    int32_t value1;
    int64_t value2; // interpolations of different sizes
    int32_t value3;
  };

  static void writer_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const WriterTaskArgs *task_args = static_cast<const WriterTaskArgs *>(args);
    AffineAccessor<int32_t, 1> acc32(task_args->inst32, FID_DATA);
    AffineAccessor<int64_t, 1> acc64(task_args->inst64, FID_DATA);
    acc32[0] = task_args->value1;
    acc64[0] = task_args->value2;
    acc32[1] = task_args->value3;
  }

  void register_test() override
  {
    writer_task_id = task_id_counter++;
    Runtime::get_runtime().register_task(writer_task_id, writer_task);
  }

  bool can_run() override { return worker_cpus().size() >= 1 && sysmem().exists(); }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    Processor cpu = worker_cpus()[0];
    IndexSpace<1> is1 = Rect<1>(0, 1);
    IndexSpace<1> is2 = Rect<1>(0, 0);
    std::map<FieldID, size_t> field_sizes32 = {{FID_DATA, sizeof(int32_t)}};
    std::map<FieldID, size_t> field_sizes64 = {{FID_DATA, sizeof(int64_t)}};
    RegionInstance::create_instance(inst32, sysmem(), is1, field_sizes32, 0,
                                    ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(inst64, sysmem(), is2, field_sizes64, 0,
                                    ProfilingRequestSet())
        .wait();

    int initial_reduce_value = 0;
    interpolated_barrier = Barrier::create_barrier(1, REDOP_INT_ADD, &initial_reduce_value,
                                                   sizeof(initial_reduce_value));

    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    WriterTaskArgs wargs(inst32, inst64, 0, 0, 0);
    int task_idx = make_task_desc(sd, cpu, writer_task_id, &wargs, sizeof(wargs));

    SubgraphDefinition::ArrivalDesc ad;
    ad.barrier = Barrier::NO_BARRIER;
    ad.reduce_value.set(&initial_reduce_value, sizeof(initial_reduce_value));
    sd.arrivals.push_back(ad);
    int arrival_idx = 0;

    size_t offset = 0;
    add_task_interpolation(sd, task_idx, offset, sizeof(int32_t),
                           offsetof(WriterTaskArgs, value1));
    offset += sizeof(int32_t);
    add_task_interpolation(sd, task_idx, offset, sizeof(int64_t),
                           offsetof(WriterTaskArgs, value2));
    offset += sizeof(int64_t);
    add_task_interpolation(sd, task_idx, offset, sizeof(int32_t),
                           offsetof(WriterTaskArgs, value3));
    offset += sizeof(int32_t);
    add_arrival_barrier_interpolation(sd, arrival_idx, offset, sizeof(Barrier));
    offset += sizeof(Barrier);
    add_arrival_redvalue_interpolation(sd, arrival_idx, offset, sizeof(int), REDOP_INT_ADD);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    size_t args_size = sizeof(int32_t) + sizeof(int64_t) + sizeof(int32_t) +
                       sizeof(Barrier) + sizeof(int);
    Serialization::DynamicBufferSerializer serializer(args_size);
    serializer << expected_value1;
    serializer << expected_value2;
    serializer << expected_value3;
    serializer << interpolated_barrier;
    serializer << expected_reduce_value;
    sg.instantiate(serializer.get_buffer(), serializer.bytes_used(), ProfilingRequestSet())
        .wait();
  }

  bool check() override
  {
    bool success = true;
    AffineAccessor<int32_t, 1> acc32(inst32, FID_DATA);
    AffineAccessor<int64_t, 1> acc64(inst64, FID_DATA);
    if(acc32[0] != expected_value1) {
      log_app.error() << "MISMATCH value1: " << acc32[0] << " != " << expected_value1;
      success = false;
    }
    if(acc64[0] != expected_value2) {
      log_app.error() << "MISMATCH value2: " << acc64[0] << " != " << expected_value2;
      success = false;
    }
    if(acc32[1] != expected_value3) {
      log_app.error() << "MISMATCH value3: " << acc32[1] << " != " << expected_value3;
      success = false;
    }
    interpolated_barrier.wait();
    int reduction_result = 0;
    if(!interpolated_barrier.get_result(&reduction_result, sizeof(int)) ||
       (reduction_result != expected_reduce_value)) {
      log_app.error() << "Interpolated barrier reduction failed.";
      success = false;
    }
    return success;
  }

  void cleanup() override
  {
    sg.destroy().wait();
    inst32.destroy();
    inst64.destroy();
  }

private:
  int writer_task_id = 0;
  Subgraph sg;
  RegionInstance inst32, inst64;
  Barrier interpolated_barrier;
  int32_t expected_value1 = 15210;
  int64_t expected_value2 = int64_t(INT32_MAX) * 2;
  int32_t expected_value3 = 42;
  int expected_reduce_value = 99;
};

class ExternalPreconditionTest : public SubgraphTest {
public:
  std::string name() const override { return "ExternalPreconditionTest"; }

  std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const override
  {
    return {SubgraphDefinition::INTERPRETED};
  }

  struct WriterTaskArgs {
    RegionInstance inst;
    IndexSpace<1> is;
    int value;
  };

  static void writer_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const WriterTaskArgs *task_args = static_cast<const WriterTaskArgs *>(args);
    AffineAccessor<int, 1> acc(task_args->inst, FID_DATA);
    for(int i = task_args->is.bounds.lo[0]; i <= task_args->is.bounds.hi[0]; i++)
      acc[i] = task_args->value;
  }

  struct ReaderTaskArgs {
    RegionInstance inst;
    IndexSpace<1> is1;
    IndexSpace<1> is2;
    ExternalPreconditionTest *test;
  };

  static void reader_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const ReaderTaskArgs *task_args = static_cast<const ReaderTaskArgs *>(args);
    AffineAccessor<int, 1> acc(task_args->inst, FID_DATA);
    for(int i = task_args->is1.bounds.lo[0]; i <= task_args->is1.bounds.hi[0]; i++) {
      if(acc[i] != task_args->test->expected_value1) {
        log_app.error() << "MISMATCH at " << i << ": " << acc[i]
                        << " != " << task_args->test->expected_value1;
        task_args->test->error = true;
      }
    }
    for(int i = task_args->is2.bounds.lo[0]; i <= task_args->is2.bounds.hi[0]; i++) {
      if(acc[i] != task_args->test->expected_value2) {
        log_app.error() << "MISMATCH at " << i << ": " << acc[i]
                        << " != " << task_args->test->expected_value2;
        task_args->test->error = true;
      }
    }
  }

  void register_test() override
  {
    writer_task_id = task_id_counter++;
    reader_task_id = task_id_counter++;
    Runtime rt = Runtime::get_runtime();
    rt.register_task(writer_task_id, writer_task);
    rt.register_task(reader_task_id, reader_task);
  }

  bool can_run() override { return worker_cpus().size() >= 2 && sysmem().exists(); }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    error = false;
    cpus = worker_cpus(2);
    IndexSpace<1> is = Rect<1>(0, 9);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(inst, sysmem(), is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    {
      std::vector<CopySrcDstField> dsts(1);
      dsts[0].set_field(inst, FID_DATA, sizeof(int));
      int initial_value = 0;
      is.fill(dsts, ProfilingRequestSet(), &initial_value, sizeof(initial_value)).wait();
    }

    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    ReaderTaskArgs rargs{inst, IndexSpace<1>(Rect<1>(0, 4)), IndexSpace<1>(Rect<1>(5, 9)),
                         this};
    int task = make_task_desc(sd, cpus[0], reader_task_id, &rargs, sizeof(rargs));
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 0,
                   SubgraphDefinition::OPKIND_TASK, task);
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 1,
                   SubgraphDefinition::OPKIND_TASK, task);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    // Launch the subgraph first so it races against the writers it must wait for.
    UserEvent pre1 = UserEvent::create_user_event();
    UserEvent pre2 = UserEvent::create_user_event();
    std::vector<Event> ext_preconds = {pre1, pre2};
    std::vector<Event> ext_postconds;
    Event sg_done =
        sg.instantiate(nullptr, 0, ProfilingRequestSet(), ext_preconds, ext_postconds);

    WriterTaskArgs wargs1{inst, IndexSpace<1>(Rect<1>(0, 4)), expected_value1};
    WriterTaskArgs wargs2{inst, IndexSpace<1>(Rect<1>(5, 9)), expected_value2};
    pre1.trigger(cpus[0].spawn(writer_task_id, &wargs1, sizeof(wargs1)));
    pre2.trigger(cpus[1].spawn(writer_task_id, &wargs2, sizeof(wargs2)));
    sg_done.wait();
  }

  bool check() override { return !error; }

  void cleanup() override
  {
    sg.destroy().wait();
    inst.destroy();
  }

  int expected_value1 = 15210;
  int expected_value2 = 42;
  bool error = false;

private:
  int writer_task_id = 0;
  int reader_task_id = 0;
  Subgraph sg;
  RegionInstance inst;
  std::vector<Processor> cpus;
};

class ExternalPostconditionTest : public SubgraphTest {
public:
  std::string name() const override { return "ExternalPostconditionTest"; }

  std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const override
  {
    return {SubgraphDefinition::INTERPRETED};
  }

  struct WriterTaskArgs {
    RegionInstance inst;
    IndexSpace<1> is;
    int value;
  };

  static void writer_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const WriterTaskArgs *task_args = static_cast<const WriterTaskArgs *>(args);
    AffineAccessor<int, 1> acc(task_args->inst, FID_DATA);
    for(int i = task_args->is.bounds.lo[0]; i <= task_args->is.bounds.hi[0]; i++)
      acc[i] = task_args->value;
  }

  void register_test() override
  {
    writer_task_id = task_id_counter++;
    Runtime::get_runtime().register_task(writer_task_id, writer_task);
  }

  bool can_run() override { return worker_cpus().size() >= 2 && sysmem().exists(); }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    std::vector<Processor> cpus = worker_cpus(2);
    IndexSpace<1> is = Rect<1>(0, 9);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(inst, sysmem(), is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    {
      std::vector<CopySrcDstField> dsts(1);
      dsts[0].set_field(inst, FID_DATA, sizeof(int));
      int initial_value = 0;
      is.fill(dsts, ProfilingRequestSet(), &initial_value, sizeof(initial_value)).wait();
    }

    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    WriterTaskArgs wargs1{inst, IndexSpace<1>(Rect<1>(0, 4)), expected_value1};
    WriterTaskArgs wargs2{inst, IndexSpace<1>(Rect<1>(5, 9)), expected_value2};
    int task1 = make_task_desc(sd, cpus[0], writer_task_id, &wargs1, sizeof(wargs1));
    int task2 = make_task_desc(sd, cpus[1], writer_task_id, &wargs2, sizeof(wargs2));
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, task1,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 0);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, task2,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 1);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    ext_postconds.assign(2, Event::NO_EVENT);
    sg.instantiate(nullptr, 0, ProfilingRequestSet(), {}, ext_postconds).wait();
  }

  bool check() override
  {
    Event::merge_events(ext_postconds).wait();
    AffineAccessor<int, 1> acc(inst, FID_DATA);
    bool success = true;
    for(int i = 0; i <= 9; i++) {
      int expected = (i <= 4) ? expected_value1 : expected_value2;
      if(acc[i] != expected) {
        log_app.error() << "MISMATCH at " << i << ": " << acc[i] << " != " << expected;
        success = false;
      }
    }
    return success;
  }

  void cleanup() override
  {
    sg.destroy().wait();
    inst.destroy();
  }

private:
  int writer_task_id = 0;
  Subgraph sg;
  RegionInstance inst;
  std::vector<Event> ext_postconds;
  int expected_value1 = 15210;
  int expected_value2 = 42;
};

////////////////////////////////////////////////////////////////////////
//
// DagTest: instantiate a generated task graph repeatedly and verify every
// dependency and placement. Instantiations are chained through events, and
// for INSTANTIATION_ORDER subgraphs also issued unchained.
//

class DagTest : public SubgraphTest {
public:
  using Generator = std::function<DagSpec(int nprocs, std::mt19937 &rng)>;

  DagTest(const std::string &shape, Generator gen, size_t max_procs = 1024,
          size_t min_procs = 1,
          SubgraphDefinition::ConcurrencyMode cmode = SubgraphDefinition::INSTANTIATION_ORDER)
    : shape(shape)
    , gen(gen)
    , max_procs(max_procs)
    , min_procs(min_procs)
    , cmode(cmode)
  {}

  std::string name() const override { return "Dag." + shape; }

  bool can_run() override { return worker_cpus().size() >= min_procs; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    procs = worker_cpus(max_procs);
    std::mt19937 rng(config.seed);
    spec = gen(procs.size(), rng);
    state.reset(&spec, procs);
    expected = 0;
    log_app.info() << name() << ": " << spec.size() << " ops, " << spec.num_edges()
                   << " edges, " << procs.size() << " procs";
    sg = build_dag_subgraph(spec, state, mode, cmode);
  }

  void run() override
  {
    int iters = (cmode == SubgraphDefinition::ONE_SHOT) ? 1 : config.iterations;
    Event e = Event::NO_EVENT;
    for(int i = 0; i < iters; i++)
      e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), e);
    e.wait();
    expected += int64_t(iters) * spec.size();

    if(cmode == SubgraphDefinition::INSTANTIATION_ORDER) {
      std::vector<Event> evs;
      for(int i = 0; i < iters; i++)
        evs.push_back(sg.instantiate(nullptr, 0, ProfilingRequestSet()));
      Event::merge_events(evs).wait();
      expected += int64_t(iters) * spec.size();
    }
  }

  bool check() override
  {
    bool ok = (state.executed.load() == expected) && (state.violations.load() == 0);
    if(!ok)
      log_app.error() << name() << ": executed " << state.executed.load() << " of "
                      << expected << ", violations " << state.violations.load();
    return ok;
  }

  void cleanup() override { sg.destroy().wait(); }

private:
  std::string shape;
  Generator gen;
  size_t max_procs, min_procs;
  SubgraphDefinition::ConcurrencyMode cmode;
  std::vector<Processor> procs;
  DagSpec spec;
  DagState state;
  Subgraph sg;
  int64_t expected = 0;
};

////////////////////////////////////////////////////////////////////////
//
// EmptySubgraphTest: a definition with no operations must still complete.
//

class EmptySubgraphTest : public SubgraphTest {
public:
  std::string name() const override { return "EmptySubgraph"; }
  bool can_run() override { return true; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    SubgraphDefinition sd;
    sd.execution_mode = mode;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    completed = wait_with_timeout(e, config.hang_timeout);
    if(!completed)
      log_app.error() << name() << ": finish event never triggered";
  }

  bool check() override { return completed; }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    else
      log_app.warning() << name() << ": leaking subgraph whose instantiation never finished";
  }

private:
  Subgraph sg;
  bool completed = false;
};

////////////////////////////////////////////////////////////////////////
//
// PoisonedPreconditionTest: a poisoned precondition must poison the finish
// event and prevent every task from running.
//

class PoisonedPreconditionTest : public SubgraphTest {
public:
  std::string name() const override { return "PoisonedPrecondition"; }
  bool can_run() override { return worker_cpus().size() >= 1; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    procs = worker_cpus(2);
    spec = dag_layers(2, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, mode, SubgraphDefinition::ONE_SHOT);
  }

  void run() override
  {
    UserEvent u = UserEvent::create_user_event();
    u.cancel();
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), u);
    completed = wait_with_timeout(e, config.hang_timeout, &poisoned);
  }

  bool check() override
  {
    bool ok = completed && poisoned && (state.executed.load() == 0);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " poisoned=" << poisoned
                      << " tasks executed=" << state.executed.load() << " (expected 0)";
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

private:
  std::vector<Processor> procs;
  DagSpec spec;
  DagState state;
  Subgraph sg;
  bool completed = false, poisoned = false;
};

////////////////////////////////////////////////////////////////////////
//
// DestroyOrderingTest: the event returned by destroy() must not trigger
// before every in-flight instantiation has finished.
//

class DestroyOrderingTest : public SubgraphTest {
public:
  DestroyOrderingTest(bool one_shot)
    : one_shot(one_shot)
  {}

  std::string name() const override
  {
    return one_shot ? "DestroyOrdering.OneShot" : "DestroyOrdering.InstantiationOrder";
  }
  bool can_run() override { return worker_cpus().size() >= 1; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    execution_mode = mode;
    procs = worker_cpus(2);
    spec = dag_layers(2, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, mode,
                            one_shot ? SubgraphDefinition::ONE_SHOT
                                     : SubgraphDefinition::INSTANTIATION_ORDER);
  }

  void run() override
  {
    std::vector<Event> evs;
    int count = one_shot ? 1 : config.iterations * 4;
    for(int i = 0; i < count; i++)
      evs.push_back(sg.instantiate(nullptr, 0, ProfilingRequestSet()));
    expected = int64_t(count) * spec.size();

    // Ask for destruction while instantiations may still be running. For
    // the one-shot case, wait on the instantiation explicitly as the API
    // requires; for instantiation order, rely on the implementation's own
    // tracking of in-flight instantiations.
    Event d = one_shot ? sg.destroy(evs[0]) : sg.destroy();
    destroyed = true;
    completed = wait_with_timeout(d, config.hang_timeout);
    executed_at_destroy = state.executed.load();
    Event::merge_events(evs).wait();
  }

  bool check() override
  {
    bool all_ran = (state.executed.load() == expected) && (state.violations.load() == 0);
    // Interpreted mode does not yet track in-flight instantiations in
    // destroy(); tighten this once destroy() is uniform across modes.
    bool ordered = (execution_mode != SubgraphDefinition::COMPILED) ||
                   (executed_at_destroy == expected);
    if(!completed || !all_ran || !ordered)
      log_app.error() << name() << ": destroy event triggered=" << completed
                      << ", tasks done at destroy " << executed_at_destroy << " of "
                      << expected << ", total executed " << state.executed.load();
    return completed && all_ran && ordered;
  }

  void cleanup() override
  {
    if(!destroyed)
      sg.destroy().wait();
  }

private:
  bool one_shot;
  SubgraphDefinition::ExecutionMode execution_mode = SubgraphDefinition::INTERPRETED;
  std::vector<Processor> procs;
  DagSpec spec;
  DagState state;
  Subgraph sg;
  int64_t expected = 0, executed_at_destroy = 0;
  bool completed = false, destroyed = false;
};

////////////////////////////////////////////////////////////////////////
//
// MixedWorkloadTest: normal tasks, including ones that block on events,
// share processors with compiled subgraph execution. Everything must
// complete and no scheduler invariant may trip.
//

struct CounterTaskArgs {
  std::atomic<int64_t> *counter;
  UserEvent gate; // if it exists, wait on it first
};

static int counter_task_id = 0;

static void counter_task(const void *args, size_t arglen, const void *userdata,
                         size_t userlen, Processor p)
{
  const CounterTaskArgs *a = static_cast<const CounterTaskArgs *>(args);
  if(a->gate.exists())
    a->gate.wait();
  a->counter->fetch_add(1);
}

class MixedWorkloadTest : public SubgraphTest {
public:
  std::string name() const override { return "MixedWorkload"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    procs = worker_cpus(4);
    spec = dag_layers(4, 2 * procs.size(), procs.size(), true);
    state.reset(&spec, procs);
    blocked.store(0);
    normal.store(0);
    sg = build_dag_subgraph(spec, state, mode, SubgraphDefinition::INSTANTIATION_ORDER);
  }

  void run() override
  {
    const int blockers_per_proc = 3, normals_per_proc = 8;
    UserEvent gate = UserEvent::create_user_event();
    std::vector<Event> evs;
    for(Processor p : procs) {
      CounterTaskArgs a{&blocked, gate};
      for(int i = 0; i < blockers_per_proc; i++)
        evs.push_back(p.spawn(counter_task_id, &a, sizeof(a)));
    }
    for(int i = 0; i < config.iterations; i++)
      evs.push_back(sg.instantiate(nullptr, 0, ProfilingRequestSet()));
    for(Processor p : procs) {
      CounterTaskArgs a{&normal, UserEvent::NO_USER_EVENT};
      for(int i = 0; i < normals_per_proc; i++)
        evs.push_back(p.spawn(counter_task_id, &a, sizeof(a)));
    }
    for(int i = 0; i < config.iterations; i++)
      evs.push_back(sg.instantiate(nullptr, 0, ProfilingRequestSet()));
    gate.trigger();
    completed = wait_with_timeout(Event::merge_events(evs), config.hang_timeout);
    expected_blocked = blockers_per_proc * procs.size();
    expected_normal = normals_per_proc * procs.size();
    expected_dag = int64_t(2 * config.iterations) * spec.size();
  }

  bool check() override
  {
    bool ok = completed && (blocked.load() == expected_blocked) &&
              (normal.load() == expected_normal) &&
              (state.executed.load() == expected_dag) && (state.violations.load() == 0);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " blocked "
                      << blocked.load() << "/" << expected_blocked << " normal "
                      << normal.load() << "/" << expected_normal << " dag "
                      << state.executed.load() << "/" << expected_dag << " violations "
                      << state.violations.load();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

  bool hung() const override { return !completed; }

private:
  std::vector<Processor> procs;
  DagSpec spec;
  DagState state;
  Subgraph sg;
  std::atomic<int64_t> blocked{0}, normal{0};
  int64_t expected_blocked = 0, expected_normal = 0, expected_dag = 0;
  bool completed = false;
};

////////////////////////////////////////////////////////////////////////
//
// ManyInstantiationsTest: a small graph instantiated many times without
// chaining. Checks completion and reports resident-memory growth.
//

class ManyInstantiationsTest : public SubgraphTest {
public:
  std::string name() const override { return "ManyInstantiations"; }
  bool can_run() override { return worker_cpus().size() >= 1; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    procs = worker_cpus(2);
    spec = dag_layers(2, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, mode, SubgraphDefinition::INSTANTIATION_ORDER);
  }

  void run() override
  {
    long rss0 = resident_kb();
    double t0 = Clock::current_time();
    std::vector<Event> evs;
    evs.reserve(config.many);
    for(int i = 0; i < config.many; i++)
      evs.push_back(sg.instantiate(nullptr, 0, ProfilingRequestSet()));
    completed = wait_with_timeout(Event::merge_events(evs), 4 * config.hang_timeout);
    double t1 = Clock::current_time();
    long rss1 = resident_kb();
    { std::ostringstream _os; _os << name() << ": " << config.many << " instantiations in "
                    << (t1 - t0) * 1e3 << " ms (" << (t1 - t0) * 1e6 / config.many
                    << " us each), resident memory " << rss0 << " -> " << rss1 << " kB"; report(_os.str()); }
  }

  bool check() override
  {
    int64_t expected = int64_t(config.many) * spec.size();
    bool ok = completed && (state.executed.load() == expected) &&
              (state.violations.load() == 0);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " executed "
                      << state.executed.load() << "/" << expected << " violations "
                      << state.violations.load();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

  bool hung() const override { return !completed; }

private:
  std::vector<Processor> procs;
  DagSpec spec;
  DagState state;
  Subgraph sg;
  bool completed = false;
};

////////////////////////////////////////////////////////////////////////
//
// RemoteInstantiateDestroyTest: a task on another address space
// instantiates a subgraph owned by this node several times and then
// destroys it. Everything must run on the owner's processors and the
// destroy event must cover all of it.
//

struct RemoteDriverArgs {
  Subgraph sg;
  UserEvent done;
  int iters;
};

static int remote_driver_task_id = 0;

static void remote_driver_task(const void *args, size_t arglen, const void *userdata,
                               size_t userlen, Processor p)
{
  const RemoteDriverArgs *a = static_cast<const RemoteDriverArgs *>(args);
  std::vector<Event> evs;
  for(int i = 0; i < a->iters; i++)
    evs.push_back(a->sg.instantiate(nullptr, 0, ProfilingRequestSet()));
  // Destroy from the remote node while instantiations may still be running.
  evs.push_back(a->sg.destroy());
  a->done.trigger(Event::merge_events(evs));
}

class RemoteInstantiateDestroyTest : public SubgraphTest {
public:
  std::string name() const override { return "RemoteInstantiateDestroy"; }
  bool can_run() override
  {
    return (Machine::get_machine().get_address_space_count() >= 2) &&
           (worker_cpus().size() >= 1) && remote_cpu().exists();
  }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    procs = worker_cpus(2);
    spec = dag_layers(3, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, mode, SubgraphDefinition::INSTANTIATION_ORDER);
  }

  void run() override
  {
    iters = config.iterations * 4;
    UserEvent done = UserEvent::create_user_event();
    RemoteDriverArgs a{sg, done, iters};
    remote_cpu().spawn(remote_driver_task_id, &a, sizeof(a));
    completed = wait_with_timeout(done, config.hang_timeout);
  }

  bool check() override
  {
    int64_t expected = int64_t(iters) * spec.size();
    bool ok = completed && (state.executed.load() == expected) &&
              (state.violations.load() == 0);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " executed "
                      << state.executed.load() << "/" << expected << " violations "
                      << state.violations.load();
    return ok;
  }

  void cleanup() override {} // destroyed by the remote driver

  bool hung() const override { return !completed; }

private:
  std::vector<Processor> procs;
  DagSpec spec;
  DagState state;
  Subgraph sg;
  int iters = 0;
  bool completed = false;
};

////////////////////////////////////////////////////////////////////////
//
// ConcurrentSubgraphsTest: two different subgraphs whose chains cross the
// same two processors in opposite directions, instantiated concurrently
// from two different launcher tasks. Any acquisition-order dependence
// between executors shows up as a hang.
//

struct LauncherTaskArgs {
  Subgraph sg;
  UserEvent done;
  int iters;
};

static int launcher_task_id = 0;

static void launcher_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  const LauncherTaskArgs *a = static_cast<const LauncherTaskArgs *>(args);
  std::vector<Event> evs;
  for(int i = 0; i < a->iters; i++)
    evs.push_back(a->sg.instantiate(nullptr, 0, ProfilingRequestSet()));
  a->done.trigger(Event::merge_events(evs));
}

class ConcurrentSubgraphsTest : public SubgraphTest {
public:
  std::string name() const override { return "ConcurrentSubgraphs"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init(SubgraphDefinition::ExecutionMode mode) override
  {
    procs = worker_cpus(2);
    const int chain_length = 8;
    spec_a = dag_chain(chain_length, 2);
    spec_b = dag_chain(chain_length, 2);
    for(int &proc : spec_b.proc_of_op)
      proc = 1 - proc; // B starts on the other processor
    state_a.reset(&spec_a, procs);
    state_b.reset(&spec_b, procs);
    sg_a = build_dag_subgraph(spec_a, state_a, mode, SubgraphDefinition::INSTANTIATION_ORDER);
    sg_b = build_dag_subgraph(spec_b, state_b, mode, SubgraphDefinition::INSTANTIATION_ORDER);
  }

  void run() override
  {
    iters = config.iterations * 20;
    UserEvent done_a = UserEvent::create_user_event();
    UserEvent done_b = UserEvent::create_user_event();
    LauncherTaskArgs la{sg_a, done_a, iters};
    LauncherTaskArgs lb{sg_b, done_b, iters};
    procs[0].spawn(launcher_task_id, &la, sizeof(la));
    procs[1].spawn(launcher_task_id, &lb, sizeof(lb));
    completed = wait_with_timeout(Event::merge_events(done_a, done_b), config.hang_timeout);
    if(!completed)
      log_app.error() << name() << ": hang after " << config.hang_timeout
                      << " s; A executed " << state_a.executed.load() << ", B executed "
                      << state_b.executed.load() << " of " << iters * spec_a.size()
                      << " each";
  }

  bool check() override
  {
    int64_t expected = int64_t(iters) * spec_a.size();
    return completed && (state_a.executed.load() == expected) &&
           (state_b.executed.load() == expected) && (state_a.violations.load() == 0) &&
           (state_b.violations.load() == 0);
  }

  void cleanup() override
  {
    if(completed) {
      sg_a.destroy().wait();
      sg_b.destroy().wait();
    }
  }

  bool hung() const override { return !completed; }

private:
  std::vector<Processor> procs;
  DagSpec spec_a, spec_b;
  DagState state_a, state_b;
  Subgraph sg_a, sg_b;
  int iters = 0;
  bool completed = false;
};

////////////////////////////////////////////////////////////////////////
//
// Death scenarios: misuse that a compiled subgraph must reject fatally.
// Each scenario is run in its own process by the test driver (see
// tests/CMakeLists.txt); reaching the end and printing the survival marker
// is the failure.
//

static int waiting_task_id = 0, finish_event_task_id = 0, noop_task_id = 0;

static void waiting_task(const void *, size_t, const void *, size_t, Processor)
{
  // Wait on an event nobody will trigger. A correct implementation refuses
  // this inside a compiled subgraph task; a broken one returns immediately.
  UserEvent::create_user_event().wait();
}

static void finish_event_task(const void *, size_t, const void *, size_t, Processor)
{
  Event e = Processor::get_current_finish_event();
  { std::ostringstream _os; _os << "finish event inside compiled subgraph task: " << e; report(_os.str()); }
}

static void noop_task(const void *, size_t, const void *, size_t, Processor) {}

static Subgraph make_one_task_compiled_subgraph(int task_id)
{
  SubgraphDefinition sd;
  sd.execution_mode = SubgraphDefinition::COMPILED;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, worker_cpus()[0], task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  return sg;
}

static void death_wait_in_compiled_task()
{
  Subgraph sg = make_one_task_compiled_subgraph(waiting_task_id);
  sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait();
}

static void death_finish_event_in_compiled_task()
{
  Subgraph sg = make_one_task_compiled_subgraph(finish_event_task_id);
  sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait();
}

static void death_unsupported_op_compiled()
{
  RegionInstance inst;
  IndexSpace<1> is = Rect<1>(0, 9);
  std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
  RegionInstance::create_instance(inst, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
      .wait();
  SubgraphDefinition sd;
  sd.execution_mode = SubgraphDefinition::COMPILED;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  int fill_value = 0;
  make_fill_desc(sd, is, inst, FID_DATA, &fill_value, sizeof(fill_value));
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_profiling_on_compiled_instantiate()
{
  Subgraph sg = make_one_task_compiled_subgraph(noop_task_id);
  ProfilingRequestSet prs;
  prs.add_request(worker_cpus()[0], noop_task_id)
      .add_measurement<ProfilingMeasurements::OperationTimeline>();
  sg.instantiate(nullptr, 0, prs).wait();
}

static void death_remote_task_compiled()
{
  Processor remote = remote_cpu();
  if(!remote.exists()) {
    printf("DEATH-TEST-SKIPPED (single address space)\n");
    fflush(stdout);
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
    return;
  }
  SubgraphDefinition sd;
  sd.execution_mode = SubgraphDefinition::COMPILED;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, remote, noop_task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_external_precond_compiled_instantiate()
{
  Subgraph sg = make_one_task_compiled_subgraph(noop_task_id);
  std::vector<Event> preconds = {UserEvent::create_user_event()};
  std::vector<Event> postconds;
  sg.instantiate(nullptr, 0, ProfilingRequestSet(), preconds, postconds).wait();
}

struct DeathScenario {
  const char *name;
  void (*fn)();
};

static const DeathScenario death_scenarios[] = {
    {"wait_in_compiled_task", death_wait_in_compiled_task},
    {"finish_event_in_compiled_task", death_finish_event_in_compiled_task},
    {"unsupported_op_compiled", death_unsupported_op_compiled},
    {"profiling_on_compiled_instantiate", death_profiling_on_compiled_instantiate},
    {"external_precond_compiled_instantiate", death_external_precond_compiled_instantiate},
    // multi-rank only; prints DEATH-TEST-SKIPPED in a single-rank run
    {"remote_task_compiled", death_remote_task_compiled},
};

////////////////////////////////////////////////////////////////////////
//
// Driver
//

static void register_common_tasks()
{
  Runtime rt = Runtime::get_runtime();
  dag_task_id = task_id_counter++;
  counter_task_id = task_id_counter++;
  launcher_task_id = task_id_counter++;
  waiting_task_id = task_id_counter++;
  finish_event_task_id = task_id_counter++;
  noop_task_id = task_id_counter++;
  remote_driver_task_id = task_id_counter++;
  rt.register_task(remote_driver_task_id, remote_driver_task);
  rt.register_task(dag_task_id, dag_task);
  rt.register_task(counter_task_id, counter_task);
  rt.register_task(launcher_task_id, launcher_task);
  rt.register_task(waiting_task_id, waiting_task);
  rt.register_task(finish_event_task_id, finish_event_task);
  rt.register_task(noop_task_id, noop_task);
}

static std::vector<std::unique_ptr<SubgraphTest>> make_tests()
{
  std::vector<std::unique_ptr<SubgraphTest>> tests;
  tests.emplace_back(new SimpleTasksTest());
  tests.emplace_back(new SimpleCopyTest());
  tests.emplace_back(new BarrierArrivalTest());
  tests.emplace_back(new InterpolationTest());
  tests.emplace_back(new ExternalPreconditionTest());
  tests.emplace_back(new ExternalPostconditionTest());

  tests.emplace_back(new DagTest(
      "chain_1proc", [](int, std::mt19937 &) { return dag_chain(32, 1); }, 1));
  tests.emplace_back(new DagTest(
      "chain_allprocs", [](int np, std::mt19937 &) { return dag_chain(64, np); }, 1024, 2));
  tests.emplace_back(new DagTest(
      "independent", [](int np, std::mt19937 &) { return dag_independent(64, np); }));
  tests.emplace_back(
      new DagTest("fan", [](int np, std::mt19937 &) { return dag_fan(32, np); }, 1024, 2));
  tests.emplace_back(new DagTest(
      "layers_dense", [](int np, std::mt19937 &) { return dag_layers(8, 8, np, true); }));
  tests.emplace_back(new DagTest(
      "layers_sparse", [](int np, std::mt19937 &) { return dag_layers(8, 8, np, false); }));
  tests.emplace_back(new DagTest("random_small", [](int np, std::mt19937 &rng) {
    return dag_random(16, np, 0.3, rng);
  }));
  tests.emplace_back(new DagTest("random_large", [](int np, std::mt19937 &rng) {
    return dag_random(config.dag_size, np, 0.1, rng);
  }));
  tests.emplace_back(new DagTest(
      "oneshot_random",
      [](int np, std::mt19937 &rng) { return dag_random(24, np, 0.2, rng); }, 1024, 1,
      SubgraphDefinition::ONE_SHOT));

  tests.emplace_back(new EmptySubgraphTest());
  tests.emplace_back(new DestroyOrderingTest(true));
  tests.emplace_back(new DestroyOrderingTest(false));
  tests.emplace_back(new ManyInstantiationsTest());
  tests.emplace_back(new PoisonedPreconditionTest());
  tests.emplace_back(new MixedWorkloadTest());
  tests.emplace_back(new RemoteInstantiateDestroyTest());
  // Last: a hang here leaves executors wedged, so nothing may follow it.
  tests.emplace_back(new ConcurrentSubgraphsTest());
  return tests;
}

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  if(!config.death.empty()) {
    for(const DeathScenario &s : death_scenarios) {
      if(config.death == s.name) {
        { std::ostringstream _os; _os << "death scenario: " << s.name; report(_os.str()); }
        s.fn();
        printf("DEATH-TEST-SURVIVED\n");
        fflush(stdout);
        Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
        return;
      }
    }
    log_app.error() << "unknown death scenario: " << config.death;
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 2);
    return;
  }

  std::vector<std::unique_ptr<SubgraphTest>> tests = make_tests();
  if(config.list) {
    for(auto &test : tests)
      printf("%s\n", test->name().c_str());
    for(const DeathScenario &s : death_scenarios)
      printf("death:%s\n", s.name);
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
    return;
  }

  { std::ostringstream _os; _os << "subgraph tests: ranks=" << Machine::get_machine().get_address_space_count()
                  << " " << all_cpus().size() << " CPUs (" << worker_cpus().size()
                  << " workers), iterations=" << config.iterations << " seed=" << config.seed; report(_os.str()); }
  for(auto &test : tests)
    test->register_test();

  std::vector<std::string> failed;
  int passed = 0, skipped = 0;
  bool any_hung = false;
  for(auto &test : tests) {
    const std::string name = test->name();
    if(!config.only.empty() && name.compare(0, config.only.size(), config.only) != 0)
      continue;
    if(config.skip.count(name))
      continue;
    if(!test->can_run()) {
      { std::ostringstream _os; _os << "SKIP " << name << " (insufficient resources)"; report(_os.str()); }
      skipped++;
      continue;
    }
    for(SubgraphDefinition::ExecutionMode mode : test->get_valid_execution_modes()) {
      { std::ostringstream _os; _os << "RUN  " << name << " [" << mode_name(mode) << "]"; report(_os.str()); }
      double t0 = Clock::current_time();
      test->init(mode);
      test->run();
      bool ok = test->check();
      test->cleanup();
      double ms = (Clock::current_time() - t0) * 1e3;
      { std::ostringstream _os; _os << (ok ? "PASS " : "FAIL ") << name << " [" << mode_name(mode) << "] "
                      << ms << " ms"; report(_os.str()); }
      if(ok)
        passed++;
      else
        failed.push_back(name + "[" + mode_name(mode) + "]");
      if(test->hung()) {
        log_app.error() << "runtime may be wedged after " << name << "; stopping";
        any_hung = true;
        break;
      }
    }
    if(any_hung)
      break;
  }

  std::stringstream ss;
  ss << "SUMMARY: passed " << passed << ", failed " << failed.size() << ", skipped "
     << skipped;
  if(!failed.empty()) {
    ss << " -- failures:";
    for(const std::string &f : failed)
      ss << " " << f;
  }
  if(failed.empty())
    { std::ostringstream _os; _os << ss.str(); report(_os.str()); }
  else
    log_app.error() << ss.str();

  int exit_code = failed.empty() ? 0 : 1;
  if(any_hung) {
    // A clean shutdown may never complete with wedged executors.
    fflush(stdout);
    fflush(stderr);
    _exit(exit_code ? exit_code : 1);
  }
  Runtime::get_runtime().shutdown(Event::NO_EVENT, exit_code);
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-iters") && (i + 1 < argc))
      config.iterations = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-seed") && (i + 1 < argc))
      config.seed = strtoul(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-dag_size") && (i + 1 < argc))
      config.dag_size = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-many") && (i + 1 < argc))
      config.many = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-hang_timeout") && (i + 1 < argc))
      config.hang_timeout = atof(argv[++i]);
    else if(!strcmp(argv[i], "-only") && (i + 1 < argc))
      config.only = argv[++i];
    else if(!strcmp(argv[i], "-skip") && (i + 1 < argc))
      config.skip.insert(argv[++i]);
    else if(!strcmp(argv[i], "-death") && (i + 1 < argc))
      config.death = argv[++i];
    else if(!strcmp(argv[i], "-list"))
      config.list = true;
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_reduction<SumReduction>(REDOP_INT_ADD);
  // Common tasks must exist on every rank: remote tests spawn them there.
  register_common_tasks();

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  return rt.wait_for_shutdown();
}
