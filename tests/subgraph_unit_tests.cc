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
// Tests that need a feature the compiled implementation does not provide yet
// declare it via pending_feature(); they are reported as PENDING and skipped,
// which doubles as the list of remaining gaps. The first CPU processor is
// reserved for the test driver:
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
#include <climits>
#ifdef SUBGRAPH_TESTS_CUDA
#include "realm/cuda/cuda_module.h"
#endif
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

#include "osdep.h" // usleep on every platform

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
  FID_PTR = 101,
  FID_B = 102,
  FID_BIG = 103,
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

// Resident set size in kB, or 0 where /proc is not available (the memory
// check then degenerates to "did not grow", which is still true).
static long resident_kb()
{
#ifdef __linux__
  std::ifstream f("/proc/self/statm");
  long size = 0, resident = 0;
  f >> size >> resident;
  return resident * (sysconf(_SC_PAGESIZE) / 1024);
#else
  return 0;
#endif
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
  virtual void init() {}

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

  // Name of a not-yet-implemented subgraph feature this test needs, or
  // nullptr if the test can run today.
  virtual const char *pending_feature() const { return nullptr; }

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
                                   SubgraphDefinition::ConcurrencyMode cmode)
{
  SubgraphDefinition sd;
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

  void init() override
  {
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

    // The subgraph orders instantiations itself, so the same must work
    // without chaining.
    {
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


  bool can_run() override { return sysmem().exists(); }

  void init() override
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


  bool can_run() override { return true; }

  void init() override
  {
    SubgraphDefinition sd;
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

  void init() override
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

  void init() override
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

  void init() override
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

  void init() override
  {
    procs = worker_cpus(max_procs);
    std::mt19937 rng(config.seed);
    spec = gen(procs.size(), rng);
    state.reset(&spec, procs);
    expected = 0;
    log_app.info() << name() << ": " << spec.size() << " ops, " << spec.num_edges()
                   << " edges, " << procs.size() << " procs";
    sg = build_dag_subgraph(spec, state, cmode);
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

  void init() override
  {
    SubgraphDefinition sd;
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

  void init() override
  {
    procs = worker_cpus(2);
    spec = dag_layers(2, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, SubgraphDefinition::ONE_SHOT);
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

  void init() override
  {
    procs = worker_cpus(2);
    spec = dag_layers(2, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, one_shot ? SubgraphDefinition::ONE_SHOT
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
    bool ordered = (executed_at_destroy == expected);
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

  void init() override
  {
    procs = worker_cpus(4);
    spec = dag_layers(4, 2 * procs.size(), procs.size(), true);
    state.reset(&spec, procs);
    blocked.store(0);
    normal.store(0);
    sg = build_dag_subgraph(spec, state, SubgraphDefinition::INSTANTIATION_ORDER);
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

  void init() override
  {
    procs = worker_cpus(2);
    spec = dag_layers(2, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, SubgraphDefinition::INSTANTIATION_ORDER);
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
// ExternalPoisonTest: a poisoned external precondition skips exactly the
// operations depending on it, poisons exactly the postconditions downstream
// of it, and poisons the finish event.
//

class ExternalPoisonTest : public SubgraphTest {
public:
  std::string name() const override { return "ExternalPoison"; }
  bool can_run() override { return worker_cpus().size() >= 1; }

  void init() override
  {
    Processor cpu = worker_cpus()[0];
    for(auto &c : counts)
      c.store(0);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    CounterTaskArgs a{&counts[0], UserEvent::NO_USER_EVENT};
    CounterTaskArgs b{&counts[1], UserEvent::NO_USER_EVENT};
    CounterTaskArgs c{&counts[2], UserEvent::NO_USER_EVENT};
    int ta = make_task_desc(sd, cpu, counter_task_id, &a, sizeof(a));
    int tb = make_task_desc(sd, cpu, counter_task_id, &b, sizeof(b));
    int tc = make_task_desc(sd, cpu, counter_task_id, &c, sizeof(c));
    // A <- ext 0, B <- ext 1, C <- A and B; postcond 0 <- A, postcond 1 <- B
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 0,
                   SubgraphDefinition::OPKIND_TASK, ta);
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 1,
                   SubgraphDefinition::OPKIND_TASK, tb);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, ta, SubgraphDefinition::OPKIND_TASK,
                   tc);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, tb, SubgraphDefinition::OPKIND_TASK,
                   tc);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, ta,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 0);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, tb,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 1);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    UserEvent bad = UserEvent::create_user_event();
    bad.cancel();
    UserEvent good = UserEvent::create_user_event();
    std::vector<Event> pre = {bad, good};
    std::vector<Event> post(2);
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), pre, post);
    good.trigger();
    completed = wait_with_timeout(e, config.hang_timeout, &finish_poisoned);
    if(completed) {
      post0_done = wait_with_timeout(post[0], config.hang_timeout, &post0_poisoned);
      post1_done = wait_with_timeout(post[1], config.hang_timeout, &post1_poisoned);
    }
  }

  bool check() override
  {
    bool ok = completed && finish_poisoned && post0_done && post0_poisoned && post1_done &&
              !post1_poisoned && (counts[0].load() == 0) && (counts[1].load() == 1) &&
              (counts[2].load() == 0);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " finish_poisoned="
                      << finish_poisoned << " post0=" << post0_done << "/" << post0_poisoned
                      << " post1=" << post1_done << "/" << post1_poisoned << " A=" << counts[0]
                      << " B=" << counts[1] << " C=" << counts[2];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

private:
  Subgraph sg;
  std::atomic<int64_t> counts[3];
  bool completed = false, finish_poisoned = false;
  bool post0_done = false, post0_poisoned = false, post1_done = false, post1_poisoned = false;
};

////////////////////////////////////////////////////////////////////////
//
// Graph priority tests. Each task records the global order in which it ran;
// a graph task may trigger a user event so that competing work becomes
// ready only once the graph is executing on its processors.
//

struct SeqTaskArgs {
  std::atomic<int64_t> *seq; // global sequence counter
  std::atomic<int64_t> *out; // where this task records its sequence number
  UserEvent to_trigger;      // triggered after recording, if it exists
  long spin_ns;
};

static int seq_task_id = 0;

static void seq_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                     Processor p)
{
  const SeqTaskArgs *a = static_cast<const SeqTaskArgs *>(args);
  if(a->spin_ns > 0) {
    long long t0 = Clock::current_time_in_nanoseconds();
    while(Clock::current_time_in_nanoseconds() - t0 < a->spin_ns) {
    }
  }
  a->out->store(a->seq->fetch_add(1));
  if(a->to_trigger.exists())
    a->to_trigger.trigger();
}

// A chain of seq tasks alternating over `procs`; the first task triggers
// `started`. Returns the subgraph; `slots` receives one sequence slot per task.
static Subgraph build_seq_chain(const std::vector<Processor> &procs, int length,
                                long spin_ns, std::atomic<int64_t> *seq,
                                std::vector<std::atomic<int64_t>> &slots,
                                UserEvent started)
{
  slots = std::vector<std::atomic<int64_t>>(length);
  for(auto &slot : slots)
    slot.store(-1);
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  for(int i = 0; i < length; i++) {
    SeqTaskArgs a{seq, &slots[i], (i == 0) ? started : UserEvent::NO_USER_EVENT, spin_ns};
    make_task_desc(sd, procs[i % procs.size()], seq_task_id, &a, sizeof(a));
    if(i > 0)
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, i - 1,
                     SubgraphDefinition::OPKIND_TASK, i);
  }
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  return sg;
}

// Normal tasks of lower, equal and higher priority become ready once an
// instantiation of priority 1 is executing: lower must wait for the graph to
// finish, equal and higher must run while it is executing.
class GraphPriorityTest : public SubgraphTest {
public:
  std::string name() const override { return "GraphPriority.NormalTasks"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init() override
  {
    procs = worker_cpus(2);
    seq.store(0);
    started = UserEvent::create_user_event();
    // odd length: the chain ends on procs[0], where the low-priority task
    // waits, so "after the graph" is exact there
    sg = build_seq_chain(procs, 41, 20000 /*20us*/, &seq, slots, started);
    for(auto &o : outs)
      o.store(-1);
  }

  void run() override
  {
    Event graph_done = sg.instantiate(nullptr, 0, ProfilingRequestSet(), Event::NO_EVENT,
                                      1 /*priority*/);
    // Competing normal tasks, gated on the graph having started.
    SeqTaskArgs low{&seq, &outs[0], UserEvent::NO_USER_EVENT, 0};
    SeqTaskArgs equal{&seq, &outs[1], UserEvent::NO_USER_EVENT, 0};
    SeqTaskArgs high{&seq, &outs[2], UserEvent::NO_USER_EVENT, 0};
    std::vector<Event> evs = {graph_done,
                              procs[0].spawn(seq_task_id, &low, sizeof(low), started, 0),
                              procs[1].spawn(seq_task_id, &equal, sizeof(equal), started, 1),
                              procs[0].spawn(seq_task_id, &high, sizeof(high), started, 2)};
    completed = wait_with_timeout(Event::merge_events(evs), config.hang_timeout);
  }

  bool check() override
  {
    int64_t graph_last = slots.back().load();
    bool ok = completed && (outs[0].load() > graph_last) && (outs[1].load() < graph_last) &&
              (outs[2].load() < graph_last);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " graph_last=" << graph_last
                      << " low=" << outs[0].load() << " equal=" << outs[1].load()
                      << " high=" << outs[2].load();
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
  std::atomic<int64_t> seq{0};
  std::vector<std::atomic<int64_t>> slots;
  std::atomic<int64_t> outs[3];
  UserEvent started;
  Subgraph sg;
  bool completed = false;
};

// A higher-priority instantiation arriving while a lower one is executing
// runs to completion before the lower one runs anything more. The high
// graph's start event is triggered by the low graph's first task; since
// event waiters may run on a background thread, the high graph becomes
// active some time later, so the check starts at its first task. Each
// processor may have dequeued one low task just before the high graph
// arrived, so up to one low task per processor may still interleave.
class GraphPriorityPreemptionTest : public SubgraphTest {
public:
  std::string name() const override { return "GraphPriority.TwoGraphs"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init() override
  {
    procs = worker_cpus(2);
    seq.store(0);
    started = UserEvent::create_user_event();
    sg_low = build_seq_chain(procs, 60, 20000, &seq, slots_low, started);
    sg_high = build_seq_chain(procs, 20, 20000, &seq, slots_high, UserEvent::NO_USER_EVENT);
  }

  void run() override
  {
    Event low_done = sg_low.instantiate(nullptr, 0, ProfilingRequestSet(), Event::NO_EVENT, 0);
    Event high_done = sg_high.instantiate(nullptr, 0, ProfilingRequestSet(), started, 2);
    completed =
        wait_with_timeout(Event::merge_events(low_done, high_done), config.hang_timeout);
  }

  bool check() override
  {
    if(!completed)
      return false;
    int64_t high_min = INT64_MAX, high_max = -1;
    for(auto &s : slots_high) {
      high_min = std::min(high_min, s.load());
      high_max = std::max(high_max, s.load());
    }
    size_t interleaved = 0;
    for(auto &s : slots_low)
      if((s.load() > high_min) && (s.load() < high_max))
        interleaved++;
    bool ok = (high_min > slots_low[0].load()) && (interleaved <= procs.size());
    if(!ok)
      log_app.error() << name() << ": low_first=" << slots_low[0].load() << " high=["
                      << high_min << "," << high_max << "] low tasks interleaved during "
                      << "the high graph: " << interleaved << " (allowed " << procs.size()
                      << ")";
    return ok;
  }

  void cleanup() override
  {
    if(completed) {
      sg_low.destroy().wait();
      sg_high.destroy().wait();
    }
  }

  bool hung() const override { return !completed; }

private:
  std::vector<Processor> procs;
  std::atomic<int64_t> seq{0};
  std::vector<std::atomic<int64_t>> slots_low, slots_high;
  UserEvent started;
  Subgraph sg_low, sg_high;
  bool completed = false;
};

////////////////////////////////////////////////////////////////////////
//
// Tasks inside a subgraph may block on events, query their finish event,
// and be profiled.
//

static bool poll_until(const std::function<bool()> &pred, double seconds)
{
  double deadline = Clock::current_time() + seconds;
  while(!pred()) {
    if(Clock::current_time() > deadline)
      return false;
    usleep(500);
  }
  return true;
}

struct BlockingTaskArgs {
  UserEvent gate;
  std::atomic<int64_t> *seq;
  std::atomic<int64_t> *out;
};
static int blocking_task_id = 0;
static void blocking_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  const BlockingTaskArgs *a = static_cast<const BlockingTaskArgs *>(args);
  a->gate.wait();
  a->out->store(a->seq->fetch_add(1));
}

// A graph task blocks on an event: the processor keeps running the rest of
// the graph (another task on the same processor completes meanwhile) and
// the blocked task finishes once the event triggers.
class BlockingTaskTest : public SubgraphTest {
public:
  std::string name() const override { return "BlockingTask"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init() override
  {
    procs = worker_cpus(2);
    seq.store(0);
    for(auto &o : outs)
      o.store(-1);
    gate = UserEvent::create_user_event();
    x_done = UserEvent::create_user_event();
    y_done = UserEvent::create_user_event();
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    BlockingTaskArgs w{gate, &seq, &outs[0]};
    SeqTaskArgs x{&seq, &outs[1], x_done, 0};
    SeqTaskArgs y{&seq, &outs[2], y_done, 0};
    make_task_desc(sd, procs[0], blocking_task_id, &w, sizeof(w));
    make_task_desc(sd, procs[1], seq_task_id, &x, sizeof(x));
    make_task_desc(sd, procs[0], seq_task_id, &y, sizeof(y)); // same processor as W
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    others_done = wait_with_timeout(Event::merge_events(x_done, y_done), config.hang_timeout);
    gate.trigger();
    completed = wait_with_timeout(e, config.hang_timeout);
  }

  bool check() override
  {
    bool ok = others_done && completed && (outs[0].load() > outs[1].load()) &&
              (outs[0].load() > outs[2].load());
    if(!ok)
      log_app.error() << name() << ": others_done=" << others_done << " completed=" << completed
                      << " W=" << outs[0] << " X=" << outs[1] << " Y=" << outs[2];
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
  std::atomic<int64_t> seq{0};
  std::atomic<int64_t> outs[3];
  UserEvent gate, x_done, y_done;
  Subgraph sg;
  bool others_done = false, completed = false;
};

struct FinishEventTaskArgs {
  Processor target;
  std::atomic<int64_t> *seq;
  std::atomic<int64_t> *out_self;
  std::atomic<int64_t> *out_dep;
  double *stamps; // body start, finish event obtained, spawned, body end
};
static int finish_event_task_id = 0;
static void finish_event_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p)
{
  const FinishEventTaskArgs *a = static_cast<const FinishEventTaskArgs *>(args);
  a->stamps[0] = Clock::current_time();
  Event fe = Processor::get_current_finish_event();
  a->stamps[1] = Clock::current_time();
  SeqTaskArgs dep{a->seq, a->out_dep, UserEvent::NO_USER_EVENT, 0};
  a->target.spawn(seq_task_id, &dep, sizeof(dep), fe);
  a->stamps[2] = Clock::current_time();
  a->out_self->store(a->seq->fetch_add(1));
  a->stamps[3] = Clock::current_time();
}

// A graph task asks for its finish event and launches dependent work on it;
// the dependent work runs after the task returns.
class FinishEventTaskTest : public SubgraphTest {
public:
  std::string name() const override { return "FinishEventTask"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init() override
  {
    procs = worker_cpus(2);
    seq.store(0);
    out_self.store(-1);
    out_dep.store(-1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    FinishEventTaskArgs f{procs[1], &seq, &out_self, &out_dep, stamps};
    make_task_desc(sd, procs[0], finish_event_task_id, &f, sizeof(f));
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    double t0 = Clock::current_time();
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    double t_inst = Clock::current_time();
    completed = wait_with_timeout(e, config.hang_timeout);
    double t1 = Clock::current_time();
    dep_ran = poll_until([&] { return out_dep.load() >= 0; }, config.hang_timeout);
    double t2 = Clock::current_time();
    std::ostringstream os;
    os << name() << ": ms after instantiate: returned " << (t_inst - t0) * 1e3
       << ", body start " << (stamps[0] - t0) * 1e3 << ", finish event "
       << (stamps[1] - t0) * 1e3 << ", spawned " << (stamps[2] - t0) * 1e3 << ", body end "
       << (stamps[3] - t0) * 1e3 << ", graph finished " << (t1 - t0) * 1e3
       << ", dependent seen " << (t2 - t0) * 1e3;
    report(os.str());
  }

  bool check() override
  {
    bool ok = completed && dep_ran && (out_dep.load() > out_self.load());
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " dep_ran=" << dep_ran
                      << " self=" << out_self << " dep=" << out_dep;
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

private:
  std::vector<Processor> procs;
  std::atomic<int64_t> seq{0}, out_self{-1}, out_dep{-1};
  double stamps[4] = {0, 0, 0, 0};
  Subgraph sg;
  bool completed = false, dep_ran = false;
};

class ProfilingTest;
struct ProfPayload {
  ProfilingTest *test;
  Processor expected_proc;
  bool expect_fevent;
  bool expect_status;
};
static int prof_response_task_id = 0;
static void prof_response_task(const void *args, size_t arglen, const void *userdata,
                               size_t userlen, Processor p);

// Definition-time and instantiation-time profiling requests on tasks are
// honored: timeline, processor usage, status and finish event.
class ProfilingTest : public SubgraphTest {
public:
  std::string name() const override { return "Profiling"; }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void handle(const ProfilingResponse &resp, const ProfPayload &pl)
  {
    using namespace ProfilingMeasurements;
    int problems = 0;
    OperationTimeline tl;
    if(resp.get_measurement(tl)) {
      if(!((tl.create_time <= tl.ready_time) && (tl.ready_time <= tl.start_time) &&
           (tl.start_time <= tl.end_time) && (tl.end_time <= tl.complete_time))) {
        log_app.error() << name() << ": timeline out of order " << tl.create_time << " "
                        << tl.ready_time << " " << tl.start_time << " " << tl.end_time << " "
                        << tl.complete_time;
        problems++;
      }
    } else {
      log_app.error() << name() << ": response without timeline";
      problems++;
    }
    OperationProcessorUsage pu;
    if(resp.get_measurement(pu) && (pu.proc != pl.expected_proc)) {
      log_app.error() << name() << ": processor " << pu.proc << " != " << pl.expected_proc;
      problems++;
    }
    if(pl.expect_status) {
      OperationStatus st;
      if(!resp.get_measurement(st) || (st.result != OperationStatus::COMPLETED_SUCCESSFULLY)) {
        log_app.error() << name() << ": missing or unexpected status";
        problems++;
      }
    }
    if(pl.expect_fevent) {
      OperationFinishEvent fe;
      if(!resp.get_measurement(fe) || !fe.finish_event.exists() ||
         !fe.finish_event.has_triggered()) {
        log_app.error() << name() << ": missing or untriggered finish event";
        problems++;
      }
    }
    bad.fetch_add(problems);
    responses.fetch_add(1);
  }

  void init() override
  {
    procs = worker_cpus(2);
    responses.store(0);
    bad.store(0);
    // Responses run as tasks on a worker CPU: the driver CPU is busy polling.
    Processor responder = procs[1];
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    counts.store(0);
    CounterTaskArgs a{&counts, UserEvent::NO_USER_EVENT};
    int t[4];
    for(int i = 0; i < 4; i++) {
      t[i] = make_task_desc(sd, procs[i % 2], counter_task_id, &a, sizeof(a));
      if(i > 0)
        add_dependency(sd, SubgraphDefinition::OPKIND_TASK, t[i - 1],
                       SubgraphDefinition::OPKIND_TASK, t[i]);
    }
    // definition-time request on task 0
    ProfPayload p0{this, procs[0], false, false};
    sd.tasks[t[0]]
        .prs.add_request(responder, prof_response_task_id, &p0, sizeof(p0))
        .add_measurement<ProfilingMeasurements::OperationTimeline>()
        .add_measurement<ProfilingMeasurements::OperationProcessorUsage>();
    // instantiation-time request on task 2
    ProfPayload p2{this, procs[0], true, true};
    ProfilingRequestSet prs2;
    prs2.add_request(responder, prof_response_task_id, &p2, sizeof(p2))
        .add_measurement<ProfilingMeasurements::OperationTimeline>()
        .add_measurement<ProfilingMeasurements::OperationStatus>()
        .add_measurement<ProfilingMeasurements::OperationFinishEvent>();
    iprof.tasks.emplace_back(unsigned(t[2]), prs2);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet(), iprof),
                                  config.hang_timeout);
    got_responses = poll_until([&] { return responses.load() >= 2; }, config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed && got_responses && (responses.load() == 2) && (bad.load() == 0) &&
              (counts.load() == 4);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " responses="
                      << responses.load() << " problems=" << bad.load() << " tasks="
                      << counts.load();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

private:
  std::vector<Processor> procs;
  std::atomic<int64_t> counts{0};
  std::atomic<int> responses{0}, bad{0};
  SubgraphInstantiationProfiling iprof;
  Subgraph sg;
  bool completed = false, got_responses = false;
};

static void prof_response_task(const void *args, size_t arglen, const void *userdata,
                               size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  const ProfPayload *pl = static_cast<const ProfPayload *>(resp.user_data());
  pl->test->handle(resp, *pl);
}


////////////////////////////////////////////////////////////////////////
//
// GPU tasks. Built only with CUDA; the tests skip themselves when the run
// has no GPU. Kernels write to a zero-copy buffer the host can read; they
// wait a while first so that the tests can tell host-side progress from
// device-side completion.
//

#ifdef SUBGRAPH_TESTS_CUDA

extern "C" void subgraph_gpu_spin_add(void *stream, int *dst, const int *src, int add,
                                      long long spin_ns);

static std::vector<Processor> all_gpus()
{
  Machine::ProcessorQuery pq = Machine::ProcessorQuery(Machine::get_machine())
                                   .only_kind(Processor::TOC_PROC)
                                   .local_address_space();
  return std::vector<Processor>(pq.begin(), pq.end());
}

static Memory zcopy_mem()
{
  return Machine::MemoryQuery(Machine::get_machine())
      .only_kind(Memory::Z_COPY_MEM)
      .has_capacity(1 << 20)
      .first();
}

static bool gpu_tests_can_run()
{
  return !all_gpus().empty() && zcopy_mem().exists() && !worker_cpus().empty();
}

// A zero-copy int buffer: host and device see the same addresses.
struct ZcBuffer {
  RegionInstance inst;
  int *ptr = nullptr;

  void create(size_t count)
  {
    std::vector<size_t> field_sizes(1, sizeof(int));
    Rect<1> bounds(Point<1>(0), Point<1>(static_cast<long long>(count) - 1));
    RegionInstance::create_instance(inst, zcopy_mem(), IndexSpace<1>(bounds), field_sizes,
                                    0 /*SOA*/, ProfilingRequestSet())
        .wait();
    AffineAccessor<int, 1> acc(inst, 0);
    ptr = acc.ptr(Point<1>(0));
    for(size_t i = 0; i < count; i++)
      ptr[i] = 0;
  }
  void destroy()
  {
    if(inst.exists())
      inst.destroy();
    inst = RegionInstance::NO_INST;
    ptr = nullptr;
  }
};

struct GpuSpinArgs {
  int *dst;
  const int *src;
  int add;
  long long spin_ns;
  double *stamp; // host time at which the task function ran, if not null
};

static int gpu_deferred_task_id = 0; // DeferredEffectsProperty
static int gpu_plain_task_id = 0;    // the same function without the property
static int gpu_stream_task_id = 0;   // stream-aware prototype (implicitly deferred)
static int gpu_fevent_task_id = 0;   // deferred, asks for its finish event
static int read_int_task_id = 0;     // CPU: copies *src to an atomic
static int gpu_fb_write_task_id = 0; // deferred, writes frame-buffer memory
static int gpu_ctxsync_task_id = 0;  // deferred but asks for a context sync (death scenario)
static void gpu_fb_write_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p);
static void gpu_ctxsync_task(const void *args, size_t arglen, const void *userdata,
                             size_t userlen, Processor p);

static void gpu_spin_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  const GpuSpinArgs *a = static_cast<const GpuSpinArgs *>(args);
  if(a->stamp)
    *a->stamp = Clock::current_time();
  subgraph_gpu_spin_add(Cuda::get_task_cuda_stream(), a->dst, a->src, a->add, a->spin_ns);
}

static void gpu_spin_stream_task(const void *args, size_t arglen, const void *userdata,
                                 size_t userlen, Processor p, CUstream_st *stream)
{
  const GpuSpinArgs *a = static_cast<const GpuSpinArgs *>(args);
  if(a->stamp)
    *a->stamp = Clock::current_time();
  subgraph_gpu_spin_add(stream, a->dst, a->src, a->add, a->spin_ns);
}

struct ReadIntArgs {
  const int *src;
  std::atomic<int64_t> *out;
};

static void read_int_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  const ReadIntArgs *a = static_cast<const ReadIntArgs *>(args);
  a->out->store(*a->src);
}

struct GpuFinishEventArgs {
  GpuSpinArgs spin;
  Processor target;
  std::atomic<int64_t> *out;
};

static void gpu_fevent_task(const void *args, size_t arglen, const void *userdata,
                            size_t userlen, Processor p)
{
  const GpuFinishEventArgs *a = static_cast<const GpuFinishEventArgs *>(args);
  gpu_spin_task(&a->spin, sizeof(a->spin), userdata, userlen, p);
  // the finish event must cover the kernel, not just this function
  ReadIntArgs r{a->spin.dst, a->out};
  a->target.spawn(read_int_task_id, &r, sizeof(r), Processor::get_current_finish_event());
}

static void register_gpu_tasks()
{
  gpu_deferred_task_id = task_id_counter++;
  gpu_plain_task_id = task_id_counter++;
  gpu_stream_task_id = task_id_counter++;
  gpu_fevent_task_id = task_id_counter++;
  read_int_task_id = task_id_counter++;
  gpu_fb_write_task_id = task_id_counter++;
  gpu_ctxsync_task_id = task_id_counter++;
  Runtime::get_runtime().register_task(read_int_task_id, read_int_task);
  if(all_gpus().empty())
    return;
  {
    CodeDescriptor fbw(gpu_fb_write_task);
    fbw.add_property(new DeferredEffectsProperty);
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                     gpu_fb_write_task_id, fbw, ProfilingRequestSet())
        .wait();
    CodeDescriptor cs(gpu_ctxsync_task);
    cs.add_property(new DeferredEffectsProperty);
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                     gpu_ctxsync_task_id, cs, ProfilingRequestSet())
        .wait();
  }
  CodeDescriptor deferred(gpu_spin_task);
  deferred.add_property(new DeferredEffectsProperty);
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   gpu_deferred_task_id, deferred, ProfilingRequestSet())
      .wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   gpu_plain_task_id, CodeDescriptor(gpu_spin_task),
                                   ProfilingRequestSet())
      .wait();
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   gpu_stream_task_id, CodeDescriptor(gpu_spin_stream_task),
                                   ProfilingRequestSet())
      .wait();
  CodeDescriptor fevent(gpu_fevent_task);
  fevent.add_property(new DeferredEffectsProperty);
  Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                   gpu_fevent_task_id, fevent, ProfilingRequestSet())
      .wait();
}

// A chain of GPU tasks on one GPU, each kernel waiting `spin_ns` and then
// writing buf[i] = buf[i-1] + 1. With deferred effects (property or
// stream-aware prototype) the task functions all run while the first kernel
// is still waiting and the device orders the kernels; without, each task
// waits for the previous kernel to complete. Either way the values must
// come out right and the finish event must wait for the last kernel.
class GpuChainTest : public SubgraphTest {
public:
  GpuChainTest(const char *_name, int *_task_id, bool _expect_ahead)
    : test_name(_name)
    , task_id(_task_id)
    , expect_ahead(_expect_ahead)
  {}
  std::string name() const override { return test_name; }
  bool can_run() override { return gpu_tests_can_run(); }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(N);
    stamps.assign(N, 0.0);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    for(int i = 0; i < N; i++) {
      GpuSpinArgs a{buf.ptr + i, (i > 0) ? buf.ptr + (i - 1) : nullptr, 1, spin_ns,
                    &stamps[i]};
      int t = make_task_desc(sd, gpu, *task_id, &a, sizeof(a));
      if(i > 0)
        add_dependency(sd, SubgraphDefinition::OPKIND_TASK, t - 1,
                       SubgraphDefinition::OPKIND_TASK, t);
    }
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    // once to warm up (module loading, streams), then the measured run
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
    if(!completed)
      return;
    for(int i = 0; i < N; i++)
      buf.ptr[i] = 0;
    t0 = Clock::current_time();
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
    t1 = Clock::current_time();
  }

  bool check() override
  {
    if(!completed)
      return false;
    const double total_spin = N * spin_ns * 1e-9;
    const double bodies = stamps[N - 1] - stamps[0];
    const double finish = t1 - t0;
    bool values_ok = true;
    for(int i = 0; i < N; i++)
      values_ok = values_ok && (buf.ptr[i] == i + 1);
    // the finish event waited for the device
    bool finish_ok = finish >= 0.5 * total_spin;
    // deferred: the host enqueued everything while the first kernel waited;
    // otherwise each task waited for the previous kernel
    bool ahead_ok = expect_ahead ? (bodies < 0.5 * total_spin)
                                 : (bodies >= 0.5 * (N - 1) * spin_ns * 1e-9);
    bool ok = values_ok && finish_ok && ahead_ok;
    std::ostringstream os;
    os << name() << ": last value " << buf.ptr[N - 1] << " (want " << N
       << "), task functions spread over " << bodies * 1e3 << " ms, finished after "
       << finish * 1e3 << " ms (" << N << " kernels of " << spin_ns / 1000 << " us)";
    if(ok)
      report(os.str());
    else
      log_app.error() << os.str();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 12;
  static constexpr long long spin_ns = 200000;
  std::string test_name;
  int *task_id;
  bool expect_ahead;
  Processor gpu;
  ZcBuffer buf;
  std::vector<double> stamps;
  Subgraph sg;
  double t0 = 0, t1 = 0;
  bool completed = false;
};

// A deferred GPU task followed by a CPU task and an external postcondition:
// both must see the kernel's result, i.e. wait for the device, not for the
// task function.
class GpuToCpuTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.ToCpu"; }
  bool can_run() override { return gpu_tests_can_run(); }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(1);
    seen.store(-1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    GpuSpinArgs a{buf.ptr, nullptr, 42, 1000000 /*1 ms*/, nullptr};
    int g = make_task_desc(sd, gpu, gpu_deferred_task_id, &a, sizeof(a));
    ReadIntArgs r{buf.ptr, &seen};
    int c = make_task_desc(sd, worker_cpus()[0], read_int_task_id, &r, sizeof(r));
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, g, SubgraphDefinition::OPKIND_TASK, c);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, g,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 0);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    std::vector<Event> preconds, postconds(1); // filled in by instantiate
    double t0 = Clock::current_time();
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), preconds, postconds);
    postcond_ok = wait_with_timeout(postconds[0], config.hang_timeout);
    postcond_delay = Clock::current_time() - t0;
    value_at_postcond = buf.ptr[0];
    completed = wait_with_timeout(e, config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed && postcond_ok && (seen.load() == 42) && (value_at_postcond == 42) &&
              (postcond_delay >= 0.5e-3);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " postcond=" << postcond_ok
                      << " after " << postcond_delay * 1e3 << " ms, value then "
                      << value_at_postcond << ", CPU task saw " << seen.load();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  Processor gpu;
  ZcBuffer buf;
  std::atomic<int64_t> seen{-1};
  Subgraph sg;
  int value_at_postcond = -1;
  double postcond_delay = 0;
  bool completed = false, postcond_ok = false;
};

// A deferred GPU task asks for its finish event and spawns a CPU task on
// it: the CPU task must see the kernel's result.
class GpuFinishEventTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.FinishEvent"; }
  bool can_run() override { return gpu_tests_can_run(); }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(1);
    seen.store(-1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    GpuFinishEventArgs a{{buf.ptr, nullptr, 7, 1000000 /*1 ms*/, nullptr}, worker_cpus()[0],
                         &seen};
    make_task_desc(sd, gpu, gpu_fevent_task_id, &a, sizeof(a));
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
    dep_ran = poll_until([&] { return seen.load() >= 0; }, config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed && dep_ran && (seen.load() == 7);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " dep_ran=" << dep_ran
                      << " saw " << seen.load();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  Processor gpu;
  ZcBuffer buf;
  std::atomic<int64_t> seen{-1};
  Subgraph sg;
  bool completed = false, dep_ran = false;
};

// Profiling a GPU task: the timeline's completion comes after its end by at
// least the kernel's duration, and the processor is the GPU.
class GpuProfilingTest;
struct GpuProfPayload {
  GpuProfilingTest *test;
};
static void gpu_prof_response_task(const void *args, size_t arglen, const void *userdata,
                                   size_t userlen, Processor p);
static int gpu_prof_response_task_id = 0;

class GpuProfilingTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.Profiling"; }
  bool can_run() override { return gpu_tests_can_run(); }

  void handle(const ProfilingResponse &resp)
  {
    using namespace ProfilingMeasurements;
    OperationTimeline tl;
    OperationProcessorUsage pu;
    if(resp.get_measurement(tl)) {
      end_to_complete = (tl.complete_time - tl.end_time) * 1e-9;
      ordered = (tl.ready_time <= tl.start_time) && (tl.start_time <= tl.end_time) &&
                (tl.end_time <= tl.complete_time);
    }
    if(resp.get_measurement(pu))
      proc_ok = (pu.proc == gpu);
    responses.fetch_add(1);
  }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(1);
    responses.store(0);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    GpuSpinArgs a{buf.ptr, nullptr, 1, 2000000 /*2 ms*/, nullptr};
    int t = make_task_desc(sd, gpu, gpu_deferred_task_id, &a, sizeof(a));
    GpuProfPayload pl{this};
    sd.tasks[t]
        .prs.add_request(worker_cpus()[0], gpu_prof_response_task_id, &pl, sizeof(pl))
        .add_measurement<ProfilingMeasurements::OperationTimeline>()
        .add_measurement<ProfilingMeasurements::OperationProcessorUsage>();
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
    got_response = poll_until([&] { return responses.load() >= 1; }, config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed && got_response && ordered && proc_ok && (end_to_complete >= 1e-3) &&
              (buf.ptr[0] == 1);
    std::ostringstream os;
    os << name() << ": completed=" << completed << " response=" << got_response
       << " ordered=" << ordered << " proc_ok=" << proc_ok << " end->complete "
       << end_to_complete * 1e3 << " ms (kernel 2 ms)";
    if(ok)
      report(os.str());
    else
      log_app.error() << os.str();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  Processor gpu;
  ZcBuffer buf;
  std::atomic<int> responses{0};
  Subgraph sg;
  double end_to_complete = 0;
  bool ordered = false, proc_ok = false, completed = false, got_response = false;
};

static void gpu_prof_response_task(const void *args, size_t arglen, const void *userdata,
                                   size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  static_cast<const GpuProfPayload *>(resp.user_data())->test->handle(resp);
}

// Many replays of a small deferred chain in instantiation order, each adding
// to the same cell: exercises token and event recycling.
class GpuReplayTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.Replay"; }
  bool can_run() override { return gpu_tests_can_run(); }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    for(int i = 0; i < chain; i++) {
      GpuSpinArgs a{buf.ptr, buf.ptr, 1, 0, nullptr};
      int t = make_task_desc(sd, gpu, gpu_deferred_task_id, &a, sizeof(a));
      if(i > 0)
        add_dependency(sd, SubgraphDefinition::OPKIND_TASK, t - 1,
                       SubgraphDefinition::OPKIND_TASK, t);
    }
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    double t0 = Clock::current_time();
    Event last = Event::NO_EVENT;
    for(int i = 0; i < replays; i++)
      last = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    completed = wait_with_timeout(last, config.hang_timeout);
    elapsed = Clock::current_time() - t0;
  }

  bool check() override
  {
    bool ok = completed && (buf.ptr[0] == chain * replays);
    std::ostringstream os;
    os << name() << ": " << replays << " replays of " << chain << " kernels in "
       << elapsed * 1e3 << " ms (" << elapsed * 1e6 / (chain * replays)
       << " us per kernel), value " << buf.ptr[0] << " (want " << chain * replays << ")";
    if(ok)
      report(os.str());
    else
      log_app.error() << os.str();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int chain = 4, replays = 500;
  Processor gpu;
  ZcBuffer buf;
  Subgraph sg;
  double elapsed = 0;
  bool completed = false;
};

// GPU and CPU tasks alternating in a chain through zero-copy memory: a GPU
// task writes buf[i] from buf[i-1]; the CPU task after it does the same on
// the host. Both directions of dependency must carry the data.
struct HostAddArgs {
  int *dst;
  const int *src;
};
static int host_add_task_id = 0;
static void host_add_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
{
  const HostAddArgs *a = static_cast<const HostAddArgs *>(args);
  *a->dst = *a->src + 1;
}

class GpuMixedChainTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.MixedChain"; }
  bool can_run() override { return gpu_tests_can_run(); }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(N);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    for(int i = 0; i < N; i++) {
      int t;
      if(i % 2 == 0) {
        GpuSpinArgs a{buf.ptr + i, (i > 0) ? buf.ptr + (i - 1) : nullptr, 1, 100000, nullptr};
        t = make_task_desc(sd, gpu, gpu_deferred_task_id, &a, sizeof(a));
      } else {
        HostAddArgs a{buf.ptr + i, buf.ptr + (i - 1)};
        t = make_task_desc(sd, worker_cpus()[0], host_add_task_id, &a, sizeof(a));
      }
      if(i > 0)
        add_dependency(sd, SubgraphDefinition::OPKIND_TASK, t - 1,
                       SubgraphDefinition::OPKIND_TASK, t);
    }
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed;
    for(int i = 0; i < N; i++)
      ok = ok && (buf.ptr[i] == i + 1);
    if(!ok) {
      std::ostringstream os;
      os << name() << ": completed=" << completed << " values";
      for(int i = 0; i < N; i++)
        os << " " << buf.ptr[i];
      log_app.error() << os.str();
    }
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 10;
  Processor gpu;
  ZcBuffer buf;
  Subgraph sg;
  bool completed = false;
};


// Edges between tasks on different GPUs take the completion path: each
// task waits for the previous kernel to finish before its function runs.
class GpuCrossChainTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.CrossGpuChain"; }
  bool can_run() override { return gpu_tests_can_run() && (all_gpus().size() >= 2); }

  void init() override
  {
    gpus = all_gpus();
    buf.create(N);
    stamps.assign(N, 0.0);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    for(int i = 0; i < N; i++) {
      GpuSpinArgs a{buf.ptr + i, (i > 0) ? buf.ptr + (i - 1) : nullptr, 1, spin_ns,
                    &stamps[i]};
      int t = make_task_desc(sd, gpus[i % 2], gpu_deferred_task_id, &a, sizeof(a));
      if(i > 0)
        add_dependency(sd, SubgraphDefinition::OPKIND_TASK, t - 1,
                       SubgraphDefinition::OPKIND_TASK, t);
    }
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    // warm up both GPUs, then measure
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
    if(!completed)
      return;
    for(int i = 0; i < N; i++)
      buf.ptr[i] = 0;
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
  }

  bool check() override
  {
    if(!completed)
      return false;
    bool values_ok = true;
    for(int i = 0; i < N; i++)
      values_ok = values_ok && (buf.ptr[i] == i + 1);
    const double spread = stamps[N - 1] - stamps[0];
    const bool waited = spread >= 0.5 * (N - 1) * spin_ns * 1e-9;
    bool ok = values_ok && waited;
    std::ostringstream os;
    os << name() << ": last value " << buf.ptr[N - 1] << " (want " << N
       << "), task functions spread over " << spread * 1e3 << " ms (" << N << " kernels of "
       << spin_ns / 1000 << " us alternating between 2 GPUs)";
    if(ok)
      report(os.str());
    else
      log_app.error() << os.str();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 8;
  static constexpr long long spin_ns = 100000;
  std::vector<Processor> gpus;
  ZcBuffer buf;
  std::vector<double> stamps;
  Subgraph sg;
  bool completed = false;
};

// A deferred GPU task writes frame-buffer memory, a copy moves it to system
// memory, a CPU task reads it: the copy must wait for the kernel.
struct FbWriteArgs {
  RegionInstance inst;
  int value;
  long long spin_ns;
};
static void gpu_fb_write_task(const void *args, size_t arglen, const void *userdata,
                              size_t userlen, Processor p)
{
  const FbWriteArgs *a = static_cast<const FbWriteArgs *>(args);
  AffineAccessor<int, 1> acc(a->inst, FID_DATA);
  subgraph_gpu_spin_add(Cuda::get_task_cuda_stream(), acc.ptr(Point<1>(0)), nullptr, a->value,
                        a->spin_ns);
}

class GpuTaskThenCopyTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.TaskThenCopy"; }
  bool can_run() override
  {
    if(!gpu_tests_can_run() || !sysmem().exists())
      return false;
    return Machine::MemoryQuery(Machine::get_machine())
        .only_kind(Memory::GPU_FB_MEM)
        .best_affinity_to(all_gpus()[0])
        .first()
        .exists();
  }

  void init() override
  {
    gpu = all_gpus()[0];
    Memory fb = Machine::MemoryQuery(Machine::get_machine())
                    .only_kind(Memory::GPU_FB_MEM)
                    .best_affinity_to(gpu)
                    .first();
    IndexSpace<1> is = Rect<1>(0, 0);
    std::map<FieldID, size_t> sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(fb_inst, fb, is, sizes, 0, ProfilingRequestSet()).wait();
    RegionInstance::create_instance(sys_inst, sysmem(), is, sizes, 0, ProfilingRequestSet())
        .wait();
    AffineAccessor<int, 1> acc(sys_inst, FID_DATA);
    acc[0] = -1;
    sys_ptr = acc.ptr(Point<1>(0));
    seen.store(-1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    FbWriteArgs w{fb_inst, 77, 1000000 /*1 ms*/};
    int g = make_task_desc(sd, gpu, gpu_fb_write_task_id, &w, sizeof(w));
    int c = make_copy_desc(sd, is, fb_inst, sys_inst, FID_DATA, sizeof(int));
    ReadIntArgs r{sys_ptr, &seen};
    int t = make_task_desc(sd, worker_cpus()[0], read_int_task_id, &r, sizeof(r));
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, g, SubgraphDefinition::OPKIND_COPY, c);
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, c, SubgraphDefinition::OPKIND_TASK, t);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed && (seen.load() == 77) && (*sys_ptr == 77);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " seen=" << seen.load()
                      << " sysmem=" << *sys_ptr;
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    fb_inst.destroy();
    sys_inst.destroy();
  }

  bool hung() const override { return !completed; }

private:
  Processor gpu;
  RegionInstance fb_inst, sys_inst;
  int *sys_ptr = nullptr;
  std::atomic<int64_t> seen{-1};
  Subgraph sg;
  bool completed = false;
};

// A poisoned input skips a GPU task and poisons the finish event.
class GpuPoisonTest : public SubgraphTest {
public:
  std::string name() const override { return "Gpu.Poison"; }
  bool can_run() override { return gpu_tests_can_run(); }

  void init() override
  {
    gpu = all_gpus()[0];
    buf.create(1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    GpuSpinArgs a{buf.ptr, nullptr, 42, 0, nullptr};
    int g = make_task_desc(sd, gpu, gpu_deferred_task_id, &a, sizeof(a));
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 0,
                   SubgraphDefinition::OPKIND_TASK, g);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    UserEvent bad = UserEvent::create_user_event();
    bad.cancel();
    std::vector<Event> pre = {bad}, post;
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet(), pre, post),
                                  config.hang_timeout, &poisoned);
    usleep(2000); // a kernel that ran anyway would have written by now
  }

  bool check() override
  {
    bool ok = completed && poisoned && (buf.ptr[0] == 0);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " poisoned=" << poisoned
                      << " value=" << buf.ptr[0];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    buf.destroy();
  }

  bool hung() const override { return !completed; }

private:
  Processor gpu;
  ZcBuffer buf;
  Subgraph sg;
  bool completed = false, poisoned = false;
};

// For the death scenario: a deferred-effects task that breaks its promise.
static void gpu_ctxsync_task(const void *args, size_t arglen, const void *userdata,
                             size_t userlen, Processor p)
{
  const GpuSpinArgs *a = static_cast<const GpuSpinArgs *>(args);
  Cuda::set_task_ctxsync_required(true);
  subgraph_gpu_spin_add(Cuda::get_task_cuda_stream(), a->dst, a->src, a->add, a->spin_ns);
}

static void death_deferred_task_requests_ctxsync_impl()
{
  if(!gpu_tests_can_run()) {
    printf("DEATH-TEST-SKIPPED\n");
    return;
  }
  ZcBuffer buf;
  buf.create(1);
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  GpuSpinArgs a{buf.ptr, nullptr, 1, 0, nullptr};
  make_task_desc(sd, all_gpus()[0], gpu_ctxsync_task_id, &a, sizeof(a));
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait();
}

#endif // SUBGRAPH_TESTS_CUDA


////////////////////////////////////////////////////////////////////////
//
// Copies: compiled once into a transfer plan, replayed per instantiation.
//

// Every replay writes a different value, copies it and reads it back: the
// shared plan must move the right data every time, and the reader must see
// it (the copy completes asynchronously through the DMA system).
class CopyReplayTest : public SubgraphTest {
public:
  std::string name() const override { return "CopyReplay"; }

  struct WriterArgs {
    RegionInstance inst;
    int value;
  };
  struct ReaderArgs {
    RegionInstance inst;
    std::atomic<int64_t> *slots;
    int slot;
  };

  static void writer_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const WriterArgs *a = static_cast<const WriterArgs *>(args);
    AffineAccessor<int, 1> acc(a->inst, FID_DATA);
    for(int i = 0; i < N; i++)
      acc[i] = a->value + i;
  }

  static void reader_task(const void *args, size_t arglen, const void *userdata,
                          size_t userlen, Processor p)
  {
    const ReaderArgs *a = static_cast<const ReaderArgs *>(args);
    AffineAccessor<int, 1> acc(a->inst, FID_DATA);
    bool ok = true;
    for(int i = 0; i < N; i++)
      ok = ok && (acc[i] == acc[0] + i);
    a->slots[a->slot].store(ok ? acc[0] : -2);
  }

  void register_test() override
  {
    writer_task_id = task_id_counter++;
    reader_task_id = task_id_counter++;
    Runtime::get_runtime().register_task(writer_task_id, writer_task);
    Runtime::get_runtime().register_task(reader_task_id, reader_task);
  }

  bool can_run() override { return (worker_cpus().size() >= 1) && sysmem().exists(); }

  void init() override
  {
    Processor cpu = worker_cpus()[0];
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(src, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(dst, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    slots = std::vector<std::atomic<int64_t>>(replays);
    for(auto &s : slots)
      s.store(-1);

    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    WriterArgs w{src, 0};
    ReaderArgs r{dst, slots.data(), 0};
    int tw = make_task_desc(sd, cpu, writer_task_id, &w, sizeof(w));
    int c = make_copy_desc(sd, is, src, dst, FID_DATA, sizeof(int));
    sd.copies[c].priority = 1; // added to the instantiation's priority
    int tr = make_task_desc(sd, cpu, reader_task_id, &r, sizeof(r));
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, tw, SubgraphDefinition::OPKIND_COPY,
                   c);
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, c, SubgraphDefinition::OPKIND_TASK,
                   tr);
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, c,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 0);
    // instantiation args: {int value; int slot}
    SubgraphDefinition::Interpolation iv, is_;
    iv.offset = 0;
    iv.bytes = sizeof(int);
    iv.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
    iv.target_index = tw;
    iv.target_offset = offsetof(WriterArgs, value);
    iv.redop_id = 0;
    sd.interpolations.push_back(iv);
    is_.offset = sizeof(int);
    is_.bytes = sizeof(int);
    is_.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
    is_.target_index = tr;
    is_.target_offset = offsetof(ReaderArgs, slot);
    is_.redop_id = 0;
    sd.interpolations.push_back(is_);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    double t0 = Clock::current_time();
    Event last = Event::NO_EVENT;
    std::vector<Event> preconds;
    posts.resize(replays);
    for(int i = 0; i < replays; i++) {
      int args[2] = {1000 + i, i};
      std::vector<Event> post(1);
      last = sg.instantiate(args, sizeof(args), ProfilingRequestSet(), preconds, post);
      posts[i] = post[0];
    }
    completed = wait_with_timeout(last, config.hang_timeout);
    elapsed = Clock::current_time() - t0;
  }

  bool check() override
  {
    bool ok = completed;
    int bad = 0;
    for(int i = 0; i < replays; i++)
      if(slots[i].load() != 1000 + i)
        bad++;
    for(int i = 0; i < replays; i++)
      ok = ok && posts[i].has_triggered();
    ok = ok && (bad == 0);
    std::ostringstream os;
    os << name() << ": " << replays << " replays in " << elapsed * 1e3 << " ms ("
       << elapsed * 1e6 / replays << " us each), " << bad << " wrong";
    if(ok)
      report(os.str());
    else
      log_app.error() << os.str() << " completed=" << completed;
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    src.destroy();
    dst.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 64, replays = 50;
  int writer_task_id = 0, reader_task_id = 0;
  RegionInstance src, dst;
  std::vector<std::atomic<int64_t>> slots;
  std::vector<Event> posts;
  Subgraph sg;
  double elapsed = 0;
  bool completed = false;
};

// Profiling requests on copies, from the definition and from the
// instantiation, are answered by Realm's transfer machinery.
class CopyProfilingTest;
struct CopyProfPayload {
  CopyProfilingTest *test;
  int which;
};
static int copy_prof_response_task_id = 0;
static void copy_prof_response_task(const void *args, size_t arglen, const void *userdata,
                                    size_t userlen, Processor p);

class CopyProfilingTest : public SubgraphTest {
public:
  std::string name() const override { return "CopyProfiling"; }
  bool can_run() override { return (worker_cpus().size() >= 1) && sysmem().exists(); }

  void handle(const ProfilingResponse &resp, int which)
  {
    using namespace ProfilingMeasurements;
    int problems = 0;
    OperationTimeline tl;
    if(!resp.get_measurement(tl) || !(tl.start_time <= tl.end_time) ||
       !(tl.end_time <= tl.complete_time))
      problems++;
    if(which == 0) {
      OperationMemoryUsage mu;
      if(!resp.get_measurement(mu) || (mu.target != sysmem()) || (mu.size != N * sizeof(int)))
        problems++;
    } else {
      OperationCopyInfo ci;
      if(!resp.get_measurement(ci) || ci.inst_info.empty())
        problems++;
    }
    bad.fetch_add(problems);
    responses.fetch_add(1);
  }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(src, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(dst, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    responses.store(0);
    bad.store(0);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    int v = 3;
    int f = make_fill_desc(sd, is, src, FID_DATA, &v, sizeof(v));
    int c = make_copy_desc(sd, is, src, dst, FID_DATA, sizeof(int));
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, f, SubgraphDefinition::OPKIND_COPY,
                   c);
    CopyProfPayload p0{this, 0};
    sd.copies[c]
        .prs.add_request(worker_cpus()[0], copy_prof_response_task_id, &p0, sizeof(p0))
        .add_measurement<ProfilingMeasurements::OperationTimeline>()
        .add_measurement<ProfilingMeasurements::OperationMemoryUsage>();
    CopyProfPayload p1{this, 1};
    ProfilingRequestSet prs1;
    prs1.add_request(worker_cpus()[0], copy_prof_response_task_id, &p1, sizeof(p1))
        .add_measurement<ProfilingMeasurements::OperationTimeline>()
        .add_measurement<ProfilingMeasurements::OperationCopyInfo>();
    iprof.copies.emplace_back(unsigned(f), prs1);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet(), iprof),
                                  config.hang_timeout);
    got_responses = poll_until([&] { return responses.load() >= 2; }, config.hang_timeout);
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc(dst, FID_DATA);
    bool data_ok = true;
    for(int i = 0; i < N; i++)
      data_ok = data_ok && (acc[i] == 3);
    bool ok = completed && got_responses && (responses.load() == 2) && (bad.load() == 0) &&
              data_ok;
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " responses="
                      << responses.load() << " problems=" << bad.load()
                      << " data_ok=" << data_ok;
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    src.destroy();
    dst.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 32;
  RegionInstance src, dst;
  std::atomic<int> responses{0}, bad{0};
  SubgraphInstantiationProfiling iprof;
  Subgraph sg;
  bool completed = false, got_responses = false;
};

static void copy_prof_response_task(const void *args, size_t arglen, const void *userdata,
                                    size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  const CopyProfPayload *pl = static_cast<const CopyProfPayload *>(resp.user_data());
  pl->test->handle(resp, pl->which);
}

// A copy behind a poisoned precondition is skipped (its destination keeps
// its old contents) while an independent copy still runs.
class CopyPoisonTest : public SubgraphTest {
public:
  std::string name() const override { return "CopyPoison"; }
  bool can_run() override { return sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(a, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(b, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    int zero = 0;
    std::vector<CopySrcDstField> da(1), db(1);
    da[0].set_field(a, FID_DATA, sizeof(int));
    db[0].set_field(b, FID_DATA, sizeof(int));
    is.fill(da, ProfilingRequestSet(), &zero, sizeof(zero)).wait();
    is.fill(db, ProfilingRequestSet(), &zero, sizeof(zero)).wait();
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    int va = 11, vb = 22;
    int fa = make_fill_desc(sd, is, a, FID_DATA, &va, sizeof(va));
    int fb = make_fill_desc(sd, is, b, FID_DATA, &vb, sizeof(vb));
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 0,
                   SubgraphDefinition::OPKIND_COPY, fa);
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, fa,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 0);
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, fb,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 1);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    UserEvent bad = UserEvent::create_user_event();
    bad.cancel();
    std::vector<Event> pre = {bad};
    std::vector<Event> post(2);
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), pre, post);
    completed = wait_with_timeout(e, config.hang_timeout, &finish_poisoned);
    if(completed) {
      post0_done = wait_with_timeout(post[0], config.hang_timeout, &post0_poisoned);
      post1_done = wait_with_timeout(post[1], config.hang_timeout, &post1_poisoned);
    }
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc_a(a, FID_DATA), acc_b(b, FID_DATA);
    bool ok = completed && finish_poisoned && post0_done && post0_poisoned && post1_done &&
              !post1_poisoned && (acc_a[0] == 0) && (acc_b[0] == 22);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " finish_poisoned="
                      << finish_poisoned << " post0=" << post0_done << "/" << post0_poisoned
                      << " post1=" << post1_done << "/" << post1_poisoned << " a=" << acc_a[0]
                      << " b=" << acc_b[0];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    a.destroy();
    b.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 8;
  RegionInstance a, b;
  Subgraph sg;
  bool completed = false, finish_poisoned = false;
  bool post0_done = false, post0_poisoned = false, post1_done = false, post1_poisoned = false;
};

// A gather through a typed indirection: dst[i] = src[idx[i]].
class IndirectCopyTest : public SubgraphTest {
public:
  std::string name() const override { return "IndirectCopy"; }
  bool can_run() override { return sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> data_sizes = {{FID_DATA, sizeof(int)}};
    std::map<FieldID, size_t> ptr_sizes = {{FID_PTR, sizeof(Point<1>)}};
    RegionInstance::create_instance(src, sysmem(), is, data_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(dst, sysmem(), is, data_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(idx, sysmem(), is, ptr_sizes, 0, ProfilingRequestSet())
        .wait();
    {
      AffineAccessor<int, 1> acc_src(src, FID_DATA), acc_dst(dst, FID_DATA);
      AffineAccessor<Point<1>, 1> acc_idx(idx, FID_PTR);
      for(int i = 0; i < N; i++) {
        acc_src[i] = 10 * i;
        acc_dst[i] = -1;
        acc_idx[i] = Point<1>(N - 1 - i);
      }
    }
    CopyIndirection<1, int>::Unstructured<1, int> ind(
        idx, std::vector<IndexSpace<1>>(1, is), std::vector<RegionInstance>(1, src), FID_PTR);
    ind.next_indirection = nullptr;
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    SubgraphDefinition::CopyDesc cd;
    cd.space = is;
    cd.srcs.resize(1);
    cd.srcs[0].set_indirect(0, FID_DATA, sizeof(int));
    cd.dsts.resize(1);
    cd.dsts[0].set_field(dst, FID_DATA, sizeof(int));
    cd.add_indirection<1, int>(&ind);
    sd.copies.push_back(cd);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
    // `ind` may go away now: the plan took what it needed
  }

  void run() override
  {
    Event e = Event::NO_EVENT;
    for(int i = 0; i < 3; i++)
      e = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    completed = wait_with_timeout(e, config.hang_timeout);
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc(dst, FID_DATA);
    bool ok = completed;
    for(int i = 0; i < N; i++)
      ok = ok && (acc[i] == 10 * (N - 1 - i));
    if(!ok) {
      std::ostringstream os;
      os << name() << ": completed=" << completed << " dst";
      for(int i = 0; i < N; i++)
        os << " " << acc[i];
      log_app.error() << os.str();
    }
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    src.destroy();
    dst.destroy();
    idx.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 16;
  RegionInstance src, dst, idx;
  Subgraph sg;
  bool completed = false;
};

// Multi-rank: copies to and from an instance on another node create their
// transfer descriptors remotely on every replay.
static Memory remote_sysmem()
{
  AddressSpace here = Processor::get_executing_processor().address_space();
  Machine::MemoryQuery mq =
      Machine::MemoryQuery(Machine::get_machine()).only_kind(Memory::SYSTEM_MEM);
  for(Memory m : mq)
    if((m.address_space() != here) && (m.capacity() >= (1 << 20)))
      return m;
  return Memory::NO_MEMORY;
}

class RemoteCopyTest : public SubgraphTest {
public:
  std::string name() const override { return "RemoteCopy"; }
  bool can_run() override { return sysmem().exists() && remote_sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(a, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(b, sysmem(), is, field_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(r, remote_sysmem(), is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    {
      AffineAccessor<int, 1> acc_a(a, FID_DATA), acc_b(b, FID_DATA);
      for(int i = 0; i < N; i++) {
        acc_a[i] = 7 * i + 1;
        acc_b[i] = 0;
      }
    }
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    int c1 = make_copy_desc(sd, is, a, r, FID_DATA, sizeof(int));
    int c2 = make_copy_desc(sd, is, r, b, FID_DATA, sizeof(int));
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, c1, SubgraphDefinition::OPKIND_COPY,
                   c2);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    double t0 = Clock::current_time();
    Event e = Event::NO_EVENT;
    for(int i = 0; i < replays; i++)
      e = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    completed = wait_with_timeout(e, config.hang_timeout);
    elapsed = Clock::current_time() - t0;
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc(b, FID_DATA);
    bool ok = completed;
    for(int i = 0; i < N; i++)
      ok = ok && (acc[i] == 7 * i + 1);
    std::ostringstream os;
    os << name() << ": " << replays << " round trips in " << elapsed * 1e3 << " ms ("
       << elapsed * 1e6 / replays << " us each)";
    if(ok)
      report(os.str());
    else
      log_app.error() << os.str() << " completed=" << completed << " b[0]=" << acc[0];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    a.destroy();
    b.destroy();
    r.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 256, replays = 20;
  RegionInstance a, b, r;
  Subgraph sg;
  double elapsed = 0;
  bool completed = false;
};


////////////////////////////////////////////////////////////////////////
//
// Second batch: priority with external inputs, remote instantiation with
// the full payload, copy variants, poison across operation kinds.
//

// A high-priority graph waits on an external precondition produced by
// lower-priority work (a normal task or a lower-priority graph) on one of
// its processors. The compile-time analysis of which inputs each processor
// depends on keeps that processor open until the input has triggered, so
// the producer runs and the graph completes instead of deadlocking.
class GraphPriorityInputTest : public SubgraphTest {
public:
  explicit GraphPriorityInputTest(bool _producer_is_graph)
    : producer_is_graph(_producer_is_graph)
  {}
  std::string name() const override
  {
    return producer_is_graph ? "GraphPriority.InputFromGraph" : "GraphPriority.InputFromTask";
  }
  bool can_run() override { return worker_cpus().size() >= 2; }

  void init() override
  {
    procs = worker_cpus(2);
    seq.store(0);
    slots = std::vector<std::atomic<int64_t>>(N);
    for(auto &s : slots)
      s.store(-1);
    producer_slot.store(-1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    for(int i = 0; i < N; i++) {
      SeqTaskArgs a{&seq, &slots[i], UserEvent::NO_USER_EVENT, 20000};
      make_task_desc(sd, procs[i % 2], seq_task_id, &a, sizeof(a));
      if(i > 0)
        add_dependency(sd, SubgraphDefinition::OPKIND_TASK, i - 1,
                       SubgraphDefinition::OPKIND_TASK, i);
    }
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 0,
                   SubgraphDefinition::OPKIND_TASK, 0);
    Subgraph::create_subgraph(sg_high, sd, ProfilingRequestSet()).wait();
    if(producer_is_graph) {
      // the event to trigger is only known at run time: interpolate it
      SubgraphDefinition pd;
      pd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
      SeqTaskArgs a{&seq, &producer_slot, UserEvent::NO_USER_EVENT, 2000000 /*2 ms*/};
      int t = make_task_desc(pd, procs[0], seq_task_id, &a, sizeof(a));
      SubgraphDefinition::Interpolation ip;
      ip.offset = 0;
      ip.bytes = sizeof(UserEvent);
      ip.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
      ip.target_index = t;
      ip.target_offset = offsetof(SeqTaskArgs, to_trigger);
      ip.redop_id = 0;
      pd.interpolations.push_back(ip);
      Subgraph::create_subgraph(sg_low, pd, ProfilingRequestSet()).wait();
    }
  }

  void run() override
  {
    UserEvent input = UserEvent::create_user_event();
    std::vector<Event> pre = {input}, post;
    Event high = sg_high.instantiate(nullptr, 0, ProfilingRequestSet(), pre, post,
                                     Event::NO_EVENT, 2 /*priority*/);
    usleep(2000); // let the high graph become active on both processors
    Event low;
    if(producer_is_graph) {
      low = sg_low.instantiate(&input, sizeof(input), ProfilingRequestSet(), Event::NO_EVENT,
                               0 /*priority*/);
    } else {
      SeqTaskArgs a{&seq, &producer_slot, input, 2000000 /*2 ms*/};
      low = procs[0].spawn(seq_task_id, &a, sizeof(a), Event::NO_EVENT, 0 /*priority*/);
    }
    completed = wait_with_timeout(Event::merge_events(high, low), config.hang_timeout);
  }

  bool check() override
  {
    bool ok = completed && (producer_slot.load() >= 0);
    for(auto &s : slots)
      ok = ok && (s.load() > producer_slot.load());
    if(!ok)
      log_app.error() << name() << ": completed=" << completed
                      << " producer=" << producer_slot.load()
                      << " first graph task=" << slots[0].load();
    return ok;
  }

  void cleanup() override
  {
    if(completed) {
      sg_high.destroy().wait();
      if(producer_is_graph)
        sg_low.destroy().wait();
    }
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 6;
  bool producer_is_graph;
  std::vector<Processor> procs;
  std::atomic<int64_t> seq{0}, producer_slot{-1};
  std::vector<std::atomic<int64_t>> slots;
  Subgraph sg_high, sg_low;
  bool completed = false;
};

// Remote instantiation with everything in the payload: interpolated
// arguments, a precondition event from the remote node, a postcondition the
// remote node waits on, and instantiation-time profiling answered on the
// remote node.
struct StoreArgs {
  std::atomic<int64_t> *slot;
  int64_t value;
};
static int store_value_task_id = 0;
static void store_value_task(const void *args, size_t arglen, const void *userdata,
                             size_t userlen, Processor p)
{
  const StoreArgs *a = static_cast<const StoreArgs *>(args);
  a->slot->store(a->value);
}

static std::atomic<int> remote_prof_responses{0}; // on the node that runs the driver
static int remote_prof_response_task_id = 0;
static void remote_prof_response_task(const void *args, size_t arglen, const void *userdata,
                                      size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  ProfilingMeasurements::OperationTimeline tl;
  if(resp.get_measurement(tl))
    remote_prof_responses.fetch_add(1);
}

struct RemoteFullDriverArgs {
  Subgraph sg;
  UserEvent done;
  int64_t value;
};
static int remote_full_driver_task_id = 0;
static void remote_full_driver_task(const void *args, size_t arglen, const void *userdata,
                                    size_t userlen, Processor p)
{
  const RemoteFullDriverArgs *a = static_cast<const RemoteFullDriverArgs *>(args);
  remote_prof_responses.store(0);
  // the response runs as a task on this node, on a processor other than the
  // one this driver is busy polling on
  Processor responder = Processor::NO_PROC;
  for(Processor c : all_cpus())
    if(c != p) {
      responder = c;
      break;
    }
  if(!responder.exists()) {
    a->done.cancel();
    return;
  }
  UserEvent pre = UserEvent::create_user_event();
  std::vector<Event> preconds = {pre};
  std::vector<Event> posts(1);
  ProfilingRequestSet prs;
  prs.add_request(responder, remote_prof_response_task_id, nullptr, 0)
      .add_measurement<ProfilingMeasurements::OperationTimeline>();
  SubgraphInstantiationProfiling iprof;
  iprof.tasks.emplace_back(0u, prs);
  int64_t v = a->value;
  Event e = a->sg.instantiate(&v, sizeof(v), ProfilingRequestSet(), iprof, preconds, posts);
  pre.trigger();
  e.wait();
  posts[0].wait();
  double deadline = Clock::current_time() + 10.0;
  while((remote_prof_responses.load() < 1) && (Clock::current_time() < deadline))
    usleep(500);
  if(remote_prof_responses.load() == 1)
    a->done.trigger();
  else
    a->done.cancel();
}

class RemoteFullInstantiateTest : public SubgraphTest {
public:
  std::string name() const override { return "RemoteFullInstantiate"; }
  bool can_run() override
  {
    return (Machine::get_machine().get_address_space_count() >= 2) &&
           (worker_cpus().size() >= 1) && remote_cpu().exists();
  }

  void init() override
  {
    slot.store(-1);
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    StoreArgs a{&slot, 0};
    int t = make_task_desc(sd, worker_cpus()[0], store_value_task_id, &a, sizeof(a));
    add_dependency(sd, SubgraphDefinition::OPKIND_EXT_PRECOND, 0,
                   SubgraphDefinition::OPKIND_TASK, t);
    add_dependency(sd, SubgraphDefinition::OPKIND_TASK, t,
                   SubgraphDefinition::OPKIND_EXT_POSTCOND, 0);
    SubgraphDefinition::Interpolation ip;
    ip.offset = 0;
    ip.bytes = sizeof(int64_t);
    ip.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
    ip.target_index = t;
    ip.target_offset = offsetof(StoreArgs, value);
    ip.redop_id = 0;
    sd.interpolations.push_back(ip);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    UserEvent done = UserEvent::create_user_event();
    RemoteFullDriverArgs d{sg, done, 4242};
    remote_cpu().spawn(remote_full_driver_task_id, &d, sizeof(d));
    completed = wait_with_timeout(done, config.hang_timeout, &poisoned);
  }

  bool check() override
  {
    bool ok = completed && !poisoned && (slot.load() == 4242);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " poisoned=" << poisoned
                      << " (profiling response missing on the remote node) slot="
                      << slot.load();
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
  }

  bool hung() const override { return !completed; }

private:
  std::atomic<int64_t> slot{-1};
  Subgraph sg;
  bool completed = false, poisoned = false;
};

// Two fields moved by one copy operation.
class CopyMultiFieldTest : public SubgraphTest {
public:
  std::string name() const override { return "CopyMultiField"; }
  bool can_run() override { return sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> sizes = {{FID_DATA, sizeof(int)}, {FID_B, sizeof(int)}};
    RegionInstance::create_instance(src, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
    RegionInstance::create_instance(dst, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
    AffineAccessor<int, 1> sa(src, FID_DATA), sb(src, FID_B), da(dst, FID_DATA), db(dst, FID_B);
    for(int i = 0; i < N; i++) {
      sa[i] = i;
      sb[i] = 100 + i;
      da[i] = db[i] = -1;
    }
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    SubgraphDefinition::CopyDesc cd;
    cd.space = is;
    cd.srcs.resize(2);
    cd.dsts.resize(2);
    cd.srcs[0].set_field(src, FID_DATA, sizeof(int));
    cd.dsts[0].set_field(dst, FID_DATA, sizeof(int));
    cd.srcs[1].set_field(src, FID_B, sizeof(int));
    cd.dsts[1].set_field(dst, FID_B, sizeof(int));
    sd.copies.push_back(cd);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
  }

  bool check() override
  {
    AffineAccessor<int, 1> da(dst, FID_DATA), db(dst, FID_B);
    bool ok = completed;
    for(int i = 0; i < N; i++)
      ok = ok && (da[i] == i) && (db[i] == 100 + i);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " dst[0]=" << da[0] << "/"
                      << db[0];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    src.destroy();
    dst.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 16;
  RegionInstance src, dst;
  Subgraph sg;
  bool completed = false;
};

// A scatter through a typed indirection: dst[idx[i]] = src[i].
class CopyScatterTest : public SubgraphTest {
public:
  std::string name() const override { return "CopyScatter"; }
  bool can_run() override { return sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> data_sizes = {{FID_DATA, sizeof(int)}};
    std::map<FieldID, size_t> ptr_sizes = {{FID_PTR, sizeof(Point<1>)}};
    RegionInstance::create_instance(src, sysmem(), is, data_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(dst, sysmem(), is, data_sizes, 0, ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(idx, sysmem(), is, ptr_sizes, 0, ProfilingRequestSet())
        .wait();
    {
      AffineAccessor<int, 1> acc_src(src, FID_DATA), acc_dst(dst, FID_DATA);
      AffineAccessor<Point<1>, 1> acc_idx(idx, FID_PTR);
      for(int i = 0; i < N; i++) {
        acc_src[i] = 10 * i;
        acc_dst[i] = -1;
        acc_idx[i] = Point<1>(N - 1 - i);
      }
    }
    CopyIndirection<1, int>::Unstructured<1, int> ind(
        idx, std::vector<IndexSpace<1>>(1, is), std::vector<RegionInstance>(1, dst), FID_PTR);
    ind.next_indirection = nullptr;
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    SubgraphDefinition::CopyDesc cd;
    cd.space = is;
    cd.srcs.resize(1);
    cd.srcs[0].set_field(src, FID_DATA, sizeof(int));
    cd.dsts.resize(1);
    cd.dsts[0].set_indirect(0, FID_DATA, sizeof(int));
    cd.add_indirection<1, int>(&ind);
    sd.copies.push_back(cd);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc(dst, FID_DATA);
    bool ok = completed;
    for(int i = 0; i < N; i++)
      ok = ok && (acc[N - 1 - i] == 10 * i);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " dst[N-1]=" << acc[N - 1];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    src.destroy();
    dst.destroy();
    idx.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 16;
  RegionInstance src, dst, idx;
  Subgraph sg;
  bool completed = false;
};

// A fill whose value is larger than the inline fill buffer of a copy field.
struct BigValue {
  int v[6];
};
class CopyLargeFillTest : public SubgraphTest {
public:
  std::string name() const override { return "CopyLargeFill"; }
  bool can_run() override { return sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> sizes = {{FID_BIG, sizeof(BigValue)}};
    RegionInstance::create_instance(dst, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    BigValue fill = {{1, 2, 3, 4, 5, 6}};
    SubgraphDefinition::CopyDesc cd;
    cd.space = is;
    cd.srcs.resize(1);
    cd.srcs[0].set_fill(&fill, sizeof(fill));
    cd.dsts.resize(1);
    cd.dsts[0].set_field(dst, FID_BIG, sizeof(BigValue));
    sd.copies.push_back(cd);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    completed = wait_with_timeout(sg.instantiate(nullptr, 0, ProfilingRequestSet()),
                                  config.hang_timeout);
  }

  bool check() override
  {
    AffineAccessor<BigValue, 1> acc(dst, FID_BIG);
    bool ok = completed;
    for(int i = 0; i < N; i++)
      for(int k = 0; k < 6; k++)
        ok = ok && (acc[i].v[k] == k + 1);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " dst[0]=" << acc[0].v[0]
                      << ".." << acc[0].v[5];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    dst.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 8;
  RegionInstance dst;
  Subgraph sg;
  bool completed = false;
};

// Multi-rank: a copy whose endpoints are both on another node, followed by
// a copy back so the result can be checked here.
class RemoteCopyRemoteEndpointsTest : public SubgraphTest {
public:
  std::string name() const override { return "RemoteCopyRemoteEndpoints"; }
  bool can_run() override { return sysmem().exists() && remote_sysmem().exists(); }

  void init() override
  {
    IndexSpace<1> is = Rect<1>(0, N - 1);
    std::map<FieldID, size_t> sizes = {{FID_DATA, sizeof(int)}};
    Memory rm = remote_sysmem();
    RegionInstance::create_instance(ra, rm, is, sizes, 0, ProfilingRequestSet()).wait();
    RegionInstance::create_instance(rb, rm, is, sizes, 0, ProfilingRequestSet()).wait();
    RegionInstance::create_instance(local, sysmem(), is, sizes, 0, ProfilingRequestSet())
        .wait();
    int five = 5, zero = 0;
    std::vector<CopySrcDstField> fa(1), fl(1);
    fa[0].set_field(ra, FID_DATA, sizeof(int));
    fl[0].set_field(local, FID_DATA, sizeof(int));
    is.fill(fa, ProfilingRequestSet(), &five, sizeof(five)).wait();
    is.fill(fl, ProfilingRequestSet(), &zero, sizeof(zero)).wait();
    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
    int c1 = make_copy_desc(sd, is, ra, rb, FID_DATA, sizeof(int));
    int c2 = make_copy_desc(sd, is, rb, local, FID_DATA, sizeof(int));
    add_dependency(sd, SubgraphDefinition::OPKIND_COPY, c1, SubgraphDefinition::OPKIND_COPY,
                   c2);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    Event e = Event::NO_EVENT;
    for(int i = 0; i < 5; i++)
      e = sg.instantiate(nullptr, 0, ProfilingRequestSet());
    completed = wait_with_timeout(e, config.hang_timeout);
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc(local, FID_DATA);
    bool ok = completed;
    for(int i = 0; i < N; i++)
      ok = ok && (acc[i] == 5);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " local[0]=" << acc[0];
    return ok;
  }

  void cleanup() override
  {
    if(completed)
      sg.destroy().wait();
    ra.destroy();
    rb.destroy();
    local.destroy();
  }

  bool hung() const override { return !completed; }

private:
  static constexpr int N = 64;
  RegionInstance ra, rb, local;
  Subgraph sg;
  bool completed = false;
};

// A poisoned input skips a fill, the arrival behind it and a profiled task
// (whose status comes back CANCELLED), poisons a postcondition with one
// poisoned source among two, and leaves an unrelated task and postcondition
// alone.
struct StatusPayload {
  std::atomic<int> *result;
};
static int status_response_task_id = 0;
static void status_response_task(const void *args, size_t arglen, const void *userdata,
                                 size_t userlen, Processor p)
{
  ProfilingResponse resp(args, arglen);
  const StatusPayload *pl = static_cast<const StatusPayload *>(resp.user_data());
  ProfilingMeasurements::OperationStatus st;
  pl->result->store(resp.get_measurement(st) ? int(st.result) : -1);
}

class PoisonKindsTest : public SubgraphTest {
public:
  std::string name() const override { return "PoisonKinds"; }
  bool can_run() override { return (worker_cpus().size() >= 1) && sysmem().exists(); }

  void init() override
  {
    Processor cpu = worker_cpus()[0];
    IndexSpace<1> is = Rect<1>(0, 7);
    std::map<FieldID, size_t> sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(dst, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
    int zero = 0;
    std::vector<CopySrcDstField> f(1);
    f[0].set_field(dst, FID_DATA, sizeof(int));
    is.fill(f, ProfilingRequestSet(), &zero, sizeof(zero)).wait();
    barrier = Barrier::create_barrier(1);
    counts.store(0);
    status.store(INT_MIN);

    SubgraphDefinition sd;
    sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
    int eleven = 11;
    int fill = make_fill_desc(sd, is, dst, FID_DATA, &eleven, sizeof(eleven));
    SubgraphDefinition::ArrivalDesc ad;
    ad.barrier = barrier;
    ad.count = 1;
    sd.arrivals.push_back(ad);
    int arr = int(sd.arrivals.size() - 1);
    CounterTaskArgs c{&counts, UserEvent::NO_USER_EVENT};
    int clean = make_task_desc(sd, cpu, counter_task_id, &c, sizeof(c));
    int skipped = make_task_desc(sd, cpu, counter_task_id, &c, sizeof(c));
    StatusPayload pl{&status};
    sd.tasks[skipped]
        .prs.add_request(cpu, status_response_task_id, &pl, sizeof(pl))
        .add_measurement<ProfilingMeasurements::OperationStatus>();
    typedef SubgraphDefinition D;
    add_dependency(sd, D::OPKIND_EXT_PRECOND, 0, D::OPKIND_COPY, fill);
    add_dependency(sd, D::OPKIND_EXT_PRECOND, 0, D::OPKIND_TASK, skipped);
    add_dependency(sd, D::OPKIND_COPY, fill, D::OPKIND_ARRIVAL, arr);
    add_dependency(sd, D::OPKIND_COPY, fill, D::OPKIND_EXT_POSTCOND, 0);
    add_dependency(sd, D::OPKIND_TASK, clean, D::OPKIND_EXT_POSTCOND, 0);
    add_dependency(sd, D::OPKIND_TASK, clean, D::OPKIND_EXT_POSTCOND, 1);
    Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  }

  void run() override
  {
    UserEvent bad = UserEvent::create_user_event();
    bad.cancel();
    std::vector<Event> pre = {bad};
    std::vector<Event> post(2);
    Event e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), pre, post);
    completed = wait_with_timeout(e, config.hang_timeout, &finish_poisoned);
    if(completed) {
      post0_done = wait_with_timeout(post[0], config.hang_timeout, &post0_poisoned);
      post1_done = wait_with_timeout(post[1], config.hang_timeout, &post1_poisoned);
      got_status = poll_until([&] { return status.load() != INT_MIN; }, config.hang_timeout);
    }
  }

  bool check() override
  {
    AffineAccessor<int, 1> acc(dst, FID_DATA);
    const int cancelled = int(ProfilingMeasurements::OperationStatus::CANCELLED);
    bool ok = completed && finish_poisoned && post0_done && post0_poisoned && post1_done &&
              !post1_poisoned && (acc[0] == 0) && !barrier.has_triggered() &&
              (counts.load() == 1) && got_status && (status.load() == cancelled);
    if(!ok)
      log_app.error() << name() << ": completed=" << completed << " finish_poisoned="
                      << finish_poisoned << " post0=" << post0_done << "/" << post0_poisoned
                      << " post1=" << post1_done << "/" << post1_poisoned
                      << " dst=" << acc[0] << " barrier_triggered=" << barrier.has_triggered()
                      << " tasks=" << counts.load() << " status=" << status.load()
                      << " (cancelled=" << cancelled << ")";
    return ok;
  }

  void cleanup() override
  {
    barrier.arrive(); // let the skipped arrival's generation complete
    if(completed)
      sg.destroy().wait();
    dst.destroy();
  }

  bool hung() const override { return !completed; }

private:
  RegionInstance dst;
  Barrier barrier;
  std::atomic<int64_t> counts{0};
  std::atomic<int> status{INT_MIN};
  Subgraph sg;
  bool completed = false, finish_poisoned = false, got_status = false;
  bool post0_done = false, post0_poisoned = false, post1_done = false, post1_poisoned = false;
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

  void init() override
  {
    procs = worker_cpus(2);
    spec = dag_layers(3, 2, procs.size(), true);
    state.reset(&spec, procs);
    sg = build_dag_subgraph(spec, state, SubgraphDefinition::INSTANTIATION_ORDER);
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

  void init() override
  {
    procs = worker_cpus(2);
    const int chain_length = 8;
    spec_a = dag_chain(chain_length, 2);
    spec_b = dag_chain(chain_length, 2);
    for(int &proc : spec_b.proc_of_op)
      proc = 1 - proc; // B starts on the other processor
    state_a.reset(&spec_a, procs);
    state_b.reset(&spec_b, procs);
    sg_a = build_dag_subgraph(spec_a, state_a, SubgraphDefinition::INSTANTIATION_ORDER);
    sg_b = build_dag_subgraph(spec_b, state_b, SubgraphDefinition::INSTANTIATION_ORDER);
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

static int noop_task_id = 0;

static void noop_task(const void *, size_t, const void *, size_t, Processor) {}

static Subgraph make_one_task_subgraph(int task_id)
{
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, worker_cpus()[0], task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  return sg;
}

static void death_unsupported_op_compiled()
{
  // nested instantiations are not implemented: compile must refuse them
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, worker_cpus()[0], noop_task_id, nullptr, 0);
  SubgraphDefinition::InstantiationDesc inner;
  inner.subgraph = make_one_task_subgraph(noop_task_id);
  sd.instantiations.push_back(inner);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_profiling_on_compiled_instantiate()
{
  Subgraph sg = make_one_task_subgraph(noop_task_id);
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
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, remote, noop_task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_concurrent_mode_unsupported()
{
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::CONCURRENT;
  make_task_desc(sd, worker_cpus()[0], noop_task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_dependency_cycle()
{
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  int a = make_task_desc(sd, worker_cpus()[0], noop_task_id, nullptr, 0);
  int b = make_task_desc(sd, worker_cpus()[0], noop_task_id, nullptr, 0);
  add_dependency(sd, SubgraphDefinition::OPKIND_TASK, a, SubgraphDefinition::OPKIND_TASK, b);
  add_dependency(sd, SubgraphDefinition::OPKIND_TASK, b, SubgraphDefinition::OPKIND_TASK, a);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_indirection_type_mismatch()
{
  // a 2-D indirection on a 1-D copy: compile must refuse it
  IndexSpace<1> is = Rect<1>(0, 7);
  std::map<FieldID, size_t> sizes = {{FID_DATA, sizeof(int)}};
  RegionInstance src, dst, idx;
  RegionInstance::create_instance(src, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
  RegionInstance::create_instance(dst, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
  RegionInstance::create_instance(idx, sysmem(), is, sizes, 0, ProfilingRequestSet()).wait();
  CopyIndirection<2, int>::Unstructured<1, int> ind(
      idx, std::vector<IndexSpace<1>>(1, is), std::vector<RegionInstance>(1, src), FID_DATA);
  ind.next_indirection = nullptr;
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  SubgraphDefinition::CopyDesc cd;
  cd.space = is;
  cd.srcs.resize(1);
  cd.srcs[0].set_indirect(0, FID_DATA, sizeof(int));
  cd.dsts.resize(1);
  cd.dsts[0].set_field(dst, FID_DATA, sizeof(int));
  cd.add_indirection<2, int>(&ind);
  sd.copies.push_back(cd);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_unregistered_task()
{
  // a task id nobody registered: compile must refuse it
  make_one_task_subgraph(task_id_counter + 1000);
}

static void death_external_precond_compiled_instantiate()
{
  Subgraph sg = make_one_task_subgraph(noop_task_id);
  std::vector<Event> preconds = {UserEvent::create_user_event()};
  std::vector<Event> postconds;
  sg.instantiate(nullptr, 0, ProfilingRequestSet(), preconds, postconds).wait();
}


static void death_interpolation_out_of_range()
{
  // 8 bytes written at offset 4 of an 8-byte argument block
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  int64_t pad = 0;
  int t = make_task_desc(sd, worker_cpus()[0], noop_task_id, &pad, sizeof(pad));
  SubgraphDefinition::Interpolation ip;
  ip.offset = 0;
  ip.bytes = 8;
  ip.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  ip.target_index = t;
  ip.target_offset = 4;
  ip.redop_id = 0;
  sd.interpolations.push_back(ip);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_redop_size_mismatch()
{
  // the int add reduction expects 4 bytes, the interpolation provides 8
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  int64_t pad = 0;
  int t = make_task_desc(sd, worker_cpus()[0], noop_task_id, &pad, sizeof(pad));
  SubgraphDefinition::Interpolation ip;
  ip.offset = 0;
  ip.bytes = 8;
  ip.target_kind = SubgraphDefinition::Interpolation::TARGET_TASK_ARGS;
  ip.target_index = t;
  ip.target_offset = 0;
  ip.redop_id = REDOP_INT_ADD;
  sd.interpolations.push_back(ip);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_deferred_creation()
{
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, worker_cpus()[0], noop_task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet(), UserEvent::create_user_event());
}

static void death_task_on_utility_proc()
{
  Processor util = Machine::ProcessorQuery(Machine::get_machine())
                       .only_kind(Processor::UTIL_PROC)
                       .local_address_space()
                       .first();
  if(!util.exists()) {
    printf("DEATH-TEST-SKIPPED (no utility processor; run with -ll:util 1)\n");
    return;
  }
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::ONE_SHOT;
  make_task_desc(sd, util, noop_task_id, nullptr, 0);
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
}

static void death_destroy_twice()
{
  Subgraph sg = make_one_task_subgraph(noop_task_id);
  sg.destroy().wait();
  sg.destroy();
}

static void death_instantiate_after_destroy()
{
  Subgraph sg = make_one_task_subgraph(noop_task_id);
  sg.destroy().wait();
  sg.instantiate(nullptr, 0, ProfilingRequestSet());
}

static void death_deferred_task_requests_ctxsync()
{
#ifdef SUBGRAPH_TESTS_CUDA
  death_deferred_task_requests_ctxsync_impl();
#else
  printf("DEATH-TEST-SKIPPED (built without CUDA)\n");
#endif
}

struct DeathScenario {
  const char *name;
  void (*fn)();
};

static const DeathScenario death_scenarios[] = {
    {"unsupported_op_compiled", death_unsupported_op_compiled},
    {"profiling_on_compiled_instantiate", death_profiling_on_compiled_instantiate},
    {"external_precond_compiled_instantiate", death_external_precond_compiled_instantiate},
    {"concurrent_mode_unsupported", death_concurrent_mode_unsupported},
    {"dependency_cycle", death_dependency_cycle},
    {"unregistered_task", death_unregistered_task},
    {"indirection_type_mismatch", death_indirection_type_mismatch},
    {"interpolation_out_of_range", death_interpolation_out_of_range},
    {"redop_size_mismatch", death_redop_size_mismatch},
    {"deferred_creation", death_deferred_creation},
    {"task_on_utility_proc", death_task_on_utility_proc},
    {"destroy_twice", death_destroy_twice},
    {"instantiate_after_destroy", death_instantiate_after_destroy},
    // GPU only; prints DEATH-TEST-SKIPPED without CUDA or a GPU
    {"deferred_task_requests_ctxsync", death_deferred_task_requests_ctxsync},
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
  noop_task_id = task_id_counter++;
  remote_driver_task_id = task_id_counter++;
  seq_task_id = task_id_counter++;
  blocking_task_id = task_id_counter++;
  finish_event_task_id = task_id_counter++;
  prof_response_task_id = task_id_counter++;
  rt.register_task(remote_driver_task_id, remote_driver_task);
  rt.register_task(seq_task_id, seq_task);
  rt.register_task(blocking_task_id, blocking_task);
  rt.register_task(finish_event_task_id, finish_event_task);
  rt.register_task(prof_response_task_id, prof_response_task);
  rt.register_task(dag_task_id, dag_task);
  rt.register_task(counter_task_id, counter_task);
  rt.register_task(launcher_task_id, launcher_task);
  rt.register_task(noop_task_id, noop_task);
  copy_prof_response_task_id = task_id_counter++;
  rt.register_task(copy_prof_response_task_id, copy_prof_response_task);
  store_value_task_id = task_id_counter++;
  remote_prof_response_task_id = task_id_counter++;
  remote_full_driver_task_id = task_id_counter++;
  status_response_task_id = task_id_counter++;
  rt.register_task(store_value_task_id, store_value_task);
  rt.register_task(remote_prof_response_task_id, remote_prof_response_task);
  rt.register_task(remote_full_driver_task_id, remote_full_driver_task);
  rt.register_task(status_response_task_id, status_response_task);
#ifdef SUBGRAPH_TESTS_CUDA
  host_add_task_id = task_id_counter++;
  gpu_prof_response_task_id = task_id_counter++;
  rt.register_task(host_add_task_id, host_add_task);
  rt.register_task(gpu_prof_response_task_id, gpu_prof_response_task);
  register_gpu_tasks();
#endif
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
  tests.emplace_back(new ExternalPoisonTest());
  tests.emplace_back(new MixedWorkloadTest());
  tests.emplace_back(new BlockingTaskTest());
  tests.emplace_back(new FinishEventTaskTest());
  tests.emplace_back(new ProfilingTest());
  tests.emplace_back(new GraphPriorityTest());
  tests.emplace_back(new GraphPriorityPreemptionTest());
  tests.emplace_back(new CopyReplayTest());
  tests.emplace_back(new CopyProfilingTest());
  tests.emplace_back(new CopyPoisonTest());
  tests.emplace_back(new IndirectCopyTest());
  tests.emplace_back(new RemoteCopyTest());
  tests.emplace_back(new GraphPriorityInputTest(false));
  tests.emplace_back(new GraphPriorityInputTest(true));
  tests.emplace_back(new CopyMultiFieldTest());
  tests.emplace_back(new CopyScatterTest());
  tests.emplace_back(new CopyLargeFillTest());
  tests.emplace_back(new PoisonKindsTest());
  tests.emplace_back(new RemoteFullInstantiateTest());
  tests.emplace_back(new RemoteCopyRemoteEndpointsTest());
#ifdef SUBGRAPH_TESTS_CUDA
  tests.emplace_back(new GpuChainTest("Gpu.DeferredChain", &gpu_deferred_task_id, true));
  tests.emplace_back(new GpuChainTest("Gpu.StreamAwareChain", &gpu_stream_task_id, true));
  tests.emplace_back(new GpuChainTest("Gpu.PlainChain", &gpu_plain_task_id, false));
  tests.emplace_back(new GpuToCpuTest());
  tests.emplace_back(new GpuFinishEventTest());
  tests.emplace_back(new GpuProfilingTest());
  tests.emplace_back(new GpuReplayTest());
  tests.emplace_back(new GpuMixedChainTest());
  tests.emplace_back(new GpuCrossChainTest());
  tests.emplace_back(new GpuTaskThenCopyTest());
  tests.emplace_back(new GpuPoisonTest());
#endif
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
  int passed = 0, skipped = 0, pending = 0;
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
    if(const char *feature = test->pending_feature()) {
      { std::ostringstream _os; _os << "PENDING " << name << ": needs " << feature; report(_os.str()); }
      pending++;
      continue;
    }
    { std::ostringstream _os; _os << "RUN  " << name; report(_os.str()); }
    double t0 = Clock::current_time();
    test->init();
    test->run();
    bool ok = test->check();
    test->cleanup();
    double ms = (Clock::current_time() - t0) * 1e3;
    { std::ostringstream _os; _os << (ok ? "PASS " : "FAIL ") << name << " " << ms << " ms"; report(_os.str()); }
    if(ok)
      passed++;
    else
      failed.push_back(name);
    if(test->hung()) {
      log_app.error() << "runtime may be wedged after " << name << "; stopping";
      any_hung = true;
    }
    if(any_hung)
      break;
  }

  std::stringstream ss;
  ss << "SUMMARY: passed " << passed << ", failed " << failed.size() << ", skipped "
     << skipped << ", pending " << pending;
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
