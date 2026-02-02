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

#include "realm.h"
#include "realm/indexspace.h"
#include "realm/subgraph.h"

#include <vector>

using namespace Realm;

Logger log_app("app");

// This file contains unit tests for Realm subgraphs. It is
// set up as a unit test instead of an integration test because
// checking features about subgraphs requires the entire Realm
// runtime to execute tasks, which can't be easily mocked.

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
};

// A counter for assigning task IDs to tests so that we
// don't need a gigantic enum with all of the task IDs.
static int32_t task_id_counter = TOP_LEVEL_TASK + 1;

enum
{
  FID_DATA = 100,
};

// A counter for tests that may want to perform reduction copies.
static int32_t redop_id_counter = 100;

// Parent class for subgraph unit tests.
class SubgraphTest {
public:
  virtual ~SubgraphTest() = default;

  // Register all metadata that will be used in the test.
  virtual void register_test() {};

  // Make sure that the requisite machine resources are
  // available for the test.
  virtual bool can_run() { return false; };

  // Initialize any state for the test.
  virtual void init() {};

  // Start the test. Run is responsible for not having any
  // pending work left after it returns.
  virtual void run() {};

  // Do any verification of the test results. This may be
  // needed if the checking can only be done after the subgraphs
  // launched by the test have completed.
  virtual bool check() { return true; };

  virtual void cleanup() {};

  virtual std::string name() const = 0;
};

// Helper utilities to simplify subgraph definitions.
int make_task_desc(SubgraphDefinition &sd, Processor proc, int task_id, void *args,
                   size_t args_size)
{
  SubgraphDefinition::TaskDesc td;
  td.proc = proc;
  td.task_id = task_id;
  td.args.set(args, args_size);
  sd.tasks.push_back(td);
  return sd.tasks.size() - 1;
}

int make_copy_desc(SubgraphDefinition &sd, IndexSpace<1> space, RegionInstance src,
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

int make_fill_desc(SubgraphDefinition &sd, IndexSpace<1> space, RegionInstance inst,
                   FieldID field_id, void *fill_value, size_t fill_value_size)
{
  SubgraphDefinition::CopyDesc cd;
  cd.space = space;
  cd.srcs.resize(1);
  cd.srcs[0].set_fill(fill_value, fill_value_size);
  cd.dsts.resize(1);
  cd.dsts[0].set_field(inst, FID_DATA, sizeof(int));
  sd.copies.push_back(cd);
  return sd.copies.size() - 1;
}

int make_reduction_copy_desc(SubgraphDefinition &sd, IndexSpace<1> space,
                             RegionInstance src, RegionInstance dst, FieldID field_id,
                             int redop_id)
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

void add_dependency(SubgraphDefinition &sd, SubgraphDefinition::OpKind src_op_kind,
                    int src_op_index, SubgraphDefinition::OpKind tgt_op_kind,
                    int tgt_op_index)
{
  SubgraphDefinition::Dependency dep;
  dep.src_op_kind = src_op_kind;
  dep.src_op_index = src_op_index;
  dep.tgt_op_kind = tgt_op_kind;
  dep.tgt_op_index = tgt_op_index;
  // TODO (rohany): Allow for specifying ports.
  sd.dependencies.push_back(dep);
}

// SimpleTasksTests is a simple subgraph that only contains tasks.
class SimpleTasksTest : public SubgraphTest {
public:
  SimpleTasksTest() {}
  ~SimpleTasksTest() {}

  std::string name() const override { return "SimpleTasksTest"; }

  struct WriterTaskArgs {
    WriterTaskArgs(RegionInstance inst, IndexSpace<1> is, int level,
                   SimpleTasksTest *test)
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

  bool can_run() override
  {
    // Require 2 CPUs.
    return Machine::ProcessorQuery(Machine::get_machine())
               .only_kind(Processor::LOC_PROC)
               .count() >= 2;
  }

  void init() override
  {
    // Get the CPUs and the system memory.
    Machine::ProcessorQuery pq =
        Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
    std::vector<Processor> cpus(pq.begin(), pq.end());
    Memory sysmem = Machine::MemoryQuery(Machine::get_machine())
                        .only_kind(Memory::Kind::SYSTEM_MEM)
                        .first();
    assert(sysmem.exists());

    // Make some data.
    IndexSpace<1> is = Rect<1>(0, 9);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(inst, sysmem, is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(inst.exists());

    // Define the subgraph. The structure of the subgraph is:
    // 4 layers, where the layers alternate between a writer and a reader. The
    // region instance of size 10 is split across the two CPUs, who will each
    // write into half. The readers will then check the values written by
    // each writer.
    {
      SubgraphDefinition sd;
      // TODO (rohany): Control the concurrency mode of the subgraph.
      sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
      WriterTaskArgs warg1(inst, IndexSpace<1>(Rect<1>(0, 4)), 0, this);
      WriterTaskArgs warg2(inst, IndexSpace<1>(Rect<1>(5, 9)), 0, this);
      ReaderTaskArgs rarg(inst, IndexSpace<1>(Rect<1>(0, 9)), 0, this);

      int w0_1 = make_task_desc(sd, cpus[0], writer_task_id, &warg1, sizeof(warg1));
      int w0_2 = make_task_desc(sd, cpus[1], writer_task_id, &warg2, sizeof(warg2));
      int r0_1 = make_task_desc(sd, cpus[0], reader_task_id, &rarg, sizeof(rarg));
      int r0_2 = make_task_desc(sd, cpus[1], reader_task_id, &rarg, sizeof(rarg));

      warg1.level = 1;
      warg2.level = 1;
      rarg.level = 1;

      int w1_1 = make_task_desc(sd, cpus[0], writer_task_id, &warg1, sizeof(warg1));
      int w1_2 = make_task_desc(sd, cpus[1], writer_task_id, &warg2, sizeof(warg2));
      int r1_1 = make_task_desc(sd, cpus[0], reader_task_id, &rarg, sizeof(rarg));
      int r1_2 = make_task_desc(sd, cpus[1], reader_task_id, &rarg, sizeof(rarg));

      // Connect both writers at level 0 to both readers at level 0.
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w0_1,
                     SubgraphDefinition::OPKIND_TASK, r0_1);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w0_1,
                     SubgraphDefinition::OPKIND_TASK, r0_2);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w0_2,
                     SubgraphDefinition::OPKIND_TASK, r0_1);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w0_2,
                     SubgraphDefinition::OPKIND_TASK, r0_2);

      // Connect each reader at level 0 to both writers at level 1.
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, r0_1,
                     SubgraphDefinition::OPKIND_TASK, w1_1);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, r0_1,
                     SubgraphDefinition::OPKIND_TASK, w1_2);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, r0_2,
                     SubgraphDefinition::OPKIND_TASK, w1_1);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, r0_2,
                     SubgraphDefinition::OPKIND_TASK, w1_2);

      // Connect both writers at level 1 to both readers at level 1.
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w1_1,
                     SubgraphDefinition::OPKIND_TASK, r1_1);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w1_1,
                     SubgraphDefinition::OPKIND_TASK, r1_2);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w1_2,
                     SubgraphDefinition::OPKIND_TASK, r1_1);
      add_dependency(sd, SubgraphDefinition::OPKIND_TASK, w1_2,
                     SubgraphDefinition::OPKIND_TASK, r1_2);
      Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
    }
  }

  void run() override
  {
    // Invoke the subgraph once.
    sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait();
  }

  void cleanup() override
  {
    sg.destroy();
    inst.destroy();
  }

  bool check() override { return !error; }

private:
  int writer_task_id = 0;
  int reader_task_id = 0;
  Subgraph sg;
  RegionInstance inst;
  bool error = false;
};

// SimpleCopyTest is a simple subgraph that contains the three
// kinds of copies that Realm supports.
class SimpleCopyTest : public SubgraphTest {
public:
  SimpleCopyTest() {}
  ~SimpleCopyTest() {}
  std::string name() const override { return "SimpleCopyTest"; }

  // Simple integer addition reduction operation
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

  void register_test() override
  {
    redop_id = redop_id_counter++;
    Runtime rt = Runtime::get_runtime();
    rt.register_reduction<SumReduction>(redop_id);
  }

  bool can_run() override
  {
    // Just need a system memory.
    Memory sysmem = Machine::MemoryQuery(Machine::get_machine())
                        .only_kind(Memory::Kind::SYSTEM_MEM)
                        .first();
    return sysmem.exists();
  }

  void init() override
  {
    Memory sysmem = Machine::MemoryQuery(Machine::get_machine())
                        .only_kind(Memory::Kind::SYSTEM_MEM)
                        .first();
    assert(sysmem.exists());

    // Create 2 regions with 10 elements.
    // In inst:
    // Elements 0-4: will be filled.
    // Elements 5-9: will be reduced.
    IndexSpace<1> is = Rect<1>(0, 9);
    std::map<FieldID, size_t> field_sizes = {{FID_DATA, sizeof(int)}};
    RegionInstance::create_instance(inst, sysmem, is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(inst.exists());
    // In copy_dst_inst:
    // Elements 0-4 will be copied from inst.
    // The remaining elements will be filled to test the sub-piece copy.
    RegionInstance::create_instance(copy_dst_inst, sysmem, is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(copy_dst_inst.exists());
    {
      int fill_value = -1;
      std::vector<CopySrcDstField> dsts;
      dsts.resize(1);
      dsts[0].set_field(copy_dst_inst, FID_DATA, sizeof(int));
      is.fill(dsts, ProfilingRequestSet(), &fill_value, sizeof(fill_value)).wait();
    }

    // Create an external instance that will be used as part of the reduction source.
    IndexSpace<1> red_src_is = Rect<1>(5, 9);
    RegionInstance::create_instance(red_src_inst, sysmem, red_src_is, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    assert(red_src_inst.exists());
    {
      int fill_value = 42;
      std::vector<CopySrcDstField> dsts;
      dsts.resize(1);
      dsts[0].set_field(red_src_inst, FID_DATA, sizeof(int));
      red_src_is.fill(dsts, ProfilingRequestSet(), &fill_value, sizeof(fill_value))
          .wait();
    }

    {
      SubgraphDefinition sd;
      sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;

      // Fill elements 0-4 with value 15210.
      int fill_value = 15210;
      int f1 = make_fill_desc(sd, IndexSpace<1>(Rect<1>(0, 4)), inst, FID_DATA,
                              &fill_value, sizeof(fill_value));

      // Initialize the target of the reduction with value 5.
      fill_value = 5;
      int f2 = make_fill_desc(sd, IndexSpace<1>(Rect<1>(5, 9)), inst, FID_DATA,
                              &fill_value, sizeof(fill_value));

      // For the range 5-9, do a copy-reduction.
      int r1 = make_reduction_copy_desc(sd, IndexSpace<1>(Rect<1>(5, 9)), red_src_inst,
                                        inst, FID_DATA, redop_id);

      // Copy between inst and copy_dst_inst.
      int c1 = make_copy_desc(sd, IndexSpace<1>(Rect<1>(0, 4)), inst, copy_dst_inst,
                              FID_DATA, sizeof(int));

      // The copy depends on the first fill.
      add_dependency(sd, SubgraphDefinition::OPKIND_COPY, f1,
                     SubgraphDefinition::OPKIND_COPY, c1);
      // The reductions depend on the second fill.
      add_dependency(sd, SubgraphDefinition::OPKIND_COPY, f2,
                     SubgraphDefinition::OPKIND_COPY, r1);
      Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
    }
  }

  void run() override { sg.instantiate(nullptr, 0, ProfilingRequestSet()).wait(); }

  bool check() override
  {
    AffineAccessor<int, 1> acc_inst(inst, FID_DATA);
    AffineAccessor<int, 1> acc_copy_dst_inst(copy_dst_inst, FID_DATA);
    bool success = true;

    // [0,4] of inst was filled to 15210.
    for(int i = 0; i <= 4; i++) {
      int expected = 15210;
      int actual = acc_inst[i];
      if(actual != expected) {
        log_app.error() << "MISMATCH at " << i << ": " << actual << " != " << expected;
        success = false;
      }
    }

    // [5,9] of inst was filled to 5, and [5,9] of red_src_inst was filled to 42.
    for(int i = 5; i <= 9; i++) {
      int expected = 5 + 42;
      int actual = acc_inst[i];
      if(actual != expected) {
        log_app.error() << "MISMATCH at " << i << ": " << actual << " != " << expected;
        success = false;
      }
    }

    // [0,4] of copy_dst_inst was copied from [0,4] of inst.
    for(int i = 0; i <= 4; i++) {
      int expected = 15210;
      int actual = acc_copy_dst_inst[i];
      if(actual != expected) {
        log_app.error() << "MISMATCH at " << i << ": " << actual << " != " << expected;
        success = false;
      }
    }
    // [5,9] of copy_dst_inst was filled to -1.
    for(int i = 5; i <= 9; i++) {
      int expected = -1;
      int actual = acc_copy_dst_inst[i];
      if(actual != expected) {
        log_app.error() << "MISMATCH at " << i << ": " << actual << " != " << expected;
        success = false;
      }
    }

    return success;
  }

  void cleanup() override
  {
    sg.destroy();
    inst.destroy();
    copy_dst_inst.destroy();
    red_src_inst.destroy();
  }

private:
  ReductionOpID redop_id;
  Subgraph sg;
  RegionInstance inst, red_src_inst, copy_dst_inst;
};

// TODO (rohany): Some more tests to write:
// * A test with dependencies between tasks and copies.
// * A test with interpolations.
// * A test with external pre/post-conditions.
// * A test with barrier arrivals.
// * A test with chained subgraphs.
// * A test that instantiation order subgraphs are respected? (maybe not).

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  std::vector<std::unique_ptr<SubgraphTest>> tests;
  tests.emplace_back(new SimpleTasksTest());
  tests.emplace_back(new SimpleCopyTest());

  log_app.info() << "Beginning subgraph tests.";
  std::vector<std::string> failed_tests;

  for(auto &test : tests) {
    test->register_test();
    if(!test->can_run()) {
      log_app.info() << "Test " << test->name() << " cannot run.";
      continue;
    }

    log_app.info() << "Running test " << test->name() << ".";
    test->init();
    test->run();
    if(!test->check()) {
      failed_tests.push_back(test->name());
    }
    test->cleanup();
  }

  if(failed_tests.empty()) {
    log_app.info() << "All subgraph tests complete.";
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
  } else {
    std::stringstream ss;
    ss << "Subgraph tests failed: ";
    for(size_t i = 0; i < failed_tests.size(); ++i) {
      if(i > 0)
        ss << ", ";
      ss << failed_tests[i];
    }
    ss << ".";
    log_app.error() << ss.str();
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 1);
  }
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  // Register the top-level task.
  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  // Kick off the top-level task.
  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);

  return rt.wait_for_shutdown();
}