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
#include "realm/event.h"
#include "realm/indexspace.h"
#include "realm/profiling.h"
#include "realm/serialize.h"
#include "realm/subgraph.h"

#include <cstdint>
#include <memory>
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

// Field IDs.
enum
{
  FID_DATA = 100,
};

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
  virtual void init(SubgraphDefinition::ExecutionMode mode) {};

  // Start the test. Run is responsible for not having any
  // pending work left after it returns.
  virtual void run() {};

  // Do any verification of the test results. This may be
  // needed if the checking can only be done after the subgraphs
  // launched by the test have completed.
  virtual bool check() { return true; };

  virtual void cleanup() {};

  virtual std::vector<SubgraphDefinition::ExecutionMode> get_valid_execution_modes() const
  {
    // Unless overridden, by default, all execution modes should be valid.
    return {SubgraphDefinition::INTERPRETED, SubgraphDefinition::COMPILED};
  }

  virtual std::string name() const = 0;
};

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

void add_dependency(SubgraphDefinition &sd, SubgraphDefinition::OpKind src_op_kind,
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

  void init(SubgraphDefinition::ExecutionMode mode) override
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
      sd.execution_mode = mode;
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
    sg.destroy().wait();
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

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  std::vector<std::unique_ptr<SubgraphTest>> tests;
  tests.emplace_back(new SimpleTasksTest());

  log_app.info() << "Beginning subgraph tests.";
  std::vector<std::string> failed_tests;

  for(auto &test : tests) {
    test->register_test();
    if(!test->can_run()) {
      log_app.info() << "Test " << test->name() << " cannot run.";
      continue;
    }

    for(auto mode : test->get_valid_execution_modes()) {
      log_app.info() << "Running test " << test->name() << " (mode=" << mode << ").";
      test->init(mode);
      test->run();
      if(!test->check()) {
        failed_tests.push_back(test->name());
      }
      test->cleanup();
    }
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