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

#include <unistd.h>

#include "realm.h"
#include "realm/cmdline.h"

using namespace Realm;

#include "../realm_ubench_common.h"

Logger log_app("app");

enum
{
  BENCHMARK_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  DUMMY_TASK,
};

enum TestFlags
{
  FAN_TEST = 1 << 1,
  CHAIN_TEST = 1 << 2
};

namespace TestConfig {
  uint64_t enabled_tests = 0;
  int num_samples = 2;
  size_t size = 32;
  int output_configs = 1;
}; // namespace TestConfig

struct DummyTaskArgs {
  size_t id;
};

struct Node {
  size_t id;
  Event current_event;
  std::vector<Node *> dependencies;
  Node()
    : id(0)
    , current_event(Event::NO_EVENT)
  {}
  Node(size_t id, Event e = Event::NO_EVENT)
    : id(id)
    , current_event(e)
  {}
  void add_dependency(Node *node) { dependencies.push_back(node); }
};

class Graph {
public:
  virtual ~Graph() 
  {
    subgraph.destroy();
  }

  Graph(const char *name)
    : name(name)
  {}

  virtual void create() = 0;

  virtual size_t num_tasks() = 0;

  Event run_as_task_graph(Event &trigger_event, std::vector<Processor> &procs)
  {
    Event pre_condition_event = trigger_event;
    int idx = 0;
    for(std::vector<Node>::iterator it = nodes.begin(); it != nodes.end(); it++) {
      Processor p = procs[idx % procs.size()];
      if(it->dependencies.size() == 1) {
        pre_condition_event = it->dependencies[0]->current_event;
      } else if(it->dependencies.size() > 1) {
        std::vector<Event> events(it->dependencies.size(), Event::NO_EVENT);
        for(size_t i = 0; i < it->dependencies.size(); i++) {
          events[i] = it->dependencies[i]->current_event;
        }
        pre_condition_event = Event::merge_events(events);
      }

      DummyTaskArgs args = {it->id};
      it->current_event =
          p.spawn(DUMMY_TASK, &args, sizeof(DummyTaskArgs), pre_condition_event);
      idx++;
    }
    Event end_event = nodes[nodes.size() - 1].current_event;
    return end_event;
  }

  Event run_as_subgraph(Event &trigger_event)
  {
    Event end_event = subgraph.instantiate(NULL, 0, ProfilingRequestSet(), trigger_event);
    return end_event;
  }

  void build_subgraph(std::vector<Processor> &procs)
  {
    log_app.info("Start creating the subgraph");
    sd.tasks.resize(nodes.size());
    int idx = 0;
    sd.dependencies.clear();
    for(std::vector<Node>::iterator it = nodes.begin(); it != nodes.end(); it++) {
      // build sd
      Processor p = procs[idx % procs.size()];
      sd.tasks[idx].proc = p;
      sd.tasks[idx].task_id = DUMMY_TASK;
      DummyTaskArgs args = {it->id};
      sd.tasks[idx].args.set(&args, sizeof(DummyTaskArgs));

      // build dependency
      if(idx == 0) {
        assert(it->dependencies.size() == 0);
      } else {
        for(std::vector<Node *>::iterator dp_it = it->dependencies.begin();
            dp_it != it->dependencies.end(); dp_it++) {
          SubgraphDefinition::Dependency dependency;
          dependency.src_op_kind = SubgraphDefinition::OPKIND_TASK;
          dependency.src_op_index = (*dp_it)->id;
          dependency.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
          dependency.tgt_op_index = it->id;
          sd.dependencies.push_back(dependency);
          // log_app.print("src %d, dst %d", dependency.src_op_index,
          // dependency.tgt_op_index);
        }
      }
      idx++;
    }
    Subgraph::create_subgraph(subgraph, sd, ProfilingRequestSet()).wait();
    log_app.info("Done with building subgraph, num_dependencies:%zu", sd.dependencies.size());
  }

public:
  std::vector<Node> nodes;
  Subgraph subgraph;
  SubgraphDefinition sd;
  std::string name;
};

// The shape of a FanGraph with size 4 is:
// 0 -> 1 -> 5, 0 -> 2 -> 5, 0 -> 3 -> 5, 0 -> 4 -> 5
class FanGraph : public Graph {
public:
  FanGraph(const char *name, size_t fs)
    : Graph(name)
    , fan_size(fs)
  {}
  /*virtual*/
  void create()
  {
    nodes.resize(fan_size + 2);
    nodes[0] = Node(0);
    nodes[fan_size + 1] = Node(fan_size + 1);
    for(size_t i = 0; i < fan_size; i++) {
      nodes[i + 1] = Node(i + 1);
      nodes[i + 1].add_dependency(&nodes[0]);
      nodes[fan_size + 1].add_dependency(&nodes[i + 1]);
    }
  }
  /*virtual*/
  size_t num_tasks()
  {
    return fan_size + 2;
  }

public:
  size_t fan_size;
};

// The shape of a ChainGraph with size 4 is:
// 0 -> 1 -> 2 -> 3 -> 4
class ChainGraph : public Graph {
public:
  ChainGraph(const char *name, size_t cs)
    : Graph(name)
    , chain_size(cs)
  {}
  /*virtual*/
  void create()
  {
    nodes.resize(chain_size);
    nodes[0] = Node(0);
    for(size_t i = 1; i < chain_size; i++) {
      nodes[i] = Node(i);
      nodes[i].add_dependency(&nodes[i - 1]);
    }
  }
  /*virtual*/
  size_t num_tasks()
  {
    return chain_size;
  }

public:
  size_t chain_size;
};

void benchmark_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  // output configuration
  if (TestConfig::output_configs) {
    output_machine_config();
    std::cout << "BENCHMARK_CONFIGURATION {enabled_tests:" << std::hex << TestConfig::enabled_tests << std::dec
              << ", task graph size:" << TestConfig::size
              << ", num_samples:" << TestConfig::num_samples << "}" << std::endl;
  }

  std::vector<Processor> all_procs;
  Machine::ProcessorQuery processors_to_test =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  all_procs.assign(processors_to_test.begin(), processors_to_test.end());

  std::vector<Graph *> graphs;
  if(TestConfig::enabled_tests & FAN_TEST) {
    FanGraph *fan_graph = new FanGraph("fan", TestConfig::size);
    fan_graph->create();
    graphs.push_back(fan_graph);
  }
  if(TestConfig::enabled_tests & CHAIN_TEST) {
    ChainGraph *chain_graph = new ChainGraph("chain", TestConfig::size);
    chain_graph->create();
    graphs.push_back(chain_graph);
  }

  for(size_t graph_idx = 0; graph_idx < graphs.size(); graph_idx++) {
    Graph *graph = graphs[graph_idx];
    {
      log_app.print("Warmup task graph:%s", graph->name.c_str());
      UserEvent trigger_event = UserEvent::create_user_event();
      Event end_event = graph->run_as_task_graph(trigger_event, all_procs);
      trigger_event.trigger();
      end_event.wait();
    }

    {
      log_app.print("Start task graph:%s", graph->name.c_str());
      UserEvent trigger_event = UserEvent::create_user_event();
      Event end_event = trigger_event;
      for(int i = 0; i < TestConfig::num_samples; i++) {
        end_event = graph->run_as_task_graph(end_event, all_procs);
      }
      double start, stop;
      start = Realm::Clock::current_time_in_microseconds();
      trigger_event.trigger();
      end_event.wait();
      stop = Realm::Clock::current_time_in_microseconds();
      double throughput = graph->num_tasks() * TestConfig::num_samples / ((stop - start) / 1000000.0);
#ifdef BENCHMARK_USE_JSON_FORMAT
      printf("RESULT {graph:taskgaph, throughput:%.2f, unit:+tasks/s}\n", throughput);
#else
      printf("RESULT taskgraph throughput=/%.2f, +tasks/s\n", throughput);
#endif
    }

    {
      log_app.print("Start subgraph:%s", graph->name.c_str());
      UserEvent trigger_event = UserEvent::create_user_event();
      Event end_event = trigger_event;
      graph->build_subgraph(all_procs);
      for(int i = 0; i < TestConfig::num_samples; i++) {
        end_event = graph->run_as_subgraph(end_event);
      }
      double start, stop;
      start = Realm::Clock::current_time_in_microseconds();
      trigger_event.trigger();
      end_event.wait();
      stop = Realm::Clock::current_time_in_microseconds();
      double throughput = graph->num_tasks() * TestConfig::num_samples / ((stop - start) / 1000000.0);
#ifdef BENCHMARK_USE_JSON_FORMAT
      printf("RESULT {graph:subgraph, throughput:%.2f, unit:+tasks/s}\n", throughput);
#else
      printf("RESULT subgraph throughput=/%.2f, +tasks/s\n", throughput);
#endif
    }
  }

  for(size_t i = 0; i < graphs.size(); i++) {
    delete graphs[i];
  }
}

void dummy_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                Processor p)
{
  // Do nothing
  // const DummyTaskArgs& dummy_task_args = *(const DummyTaskArgs *)args;
  // log_app.print("id: %zu", dummy_task_args.id);
  // sleep(1);
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, (char ***)&argv);

  std::vector<std::string> enabled_tests;

  CommandLineParser cp;
  cp.add_option_int("-config", TestConfig::output_configs);
  cp.add_option_int("-size", TestConfig::size);
  cp.add_option_int("-s", TestConfig::num_samples);
  cp.add_option_stringlist("-t", enabled_tests);
  bool ok = cp.parse_command_line(argc, const_cast<const char **>(argv));
  assert(ok);

  if(enabled_tests.size() == 0) {
    TestConfig::enabled_tests = ~0ULL;
  } else {
    for(size_t i = 0; i < enabled_tests.size(); i++) {
      if(enabled_tests[i] == "FAN")
        TestConfig::enabled_tests |= (uint64_t)FAN_TEST;
      else if(enabled_tests[i] == "CHAIN")
        TestConfig::enabled_tests |= (uint64_t)CHAIN_TEST;
      else
        abort();
    }
  }

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());

  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, BENCHMARK_TASK,
                                   CodeDescriptor(benchmark_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, DUMMY_TASK,
                                   CodeDescriptor(dummy_task), ProfilingRequestSet())
      .external_wait();

  Event e = rt.collective_spawn(p, BENCHMARK_TASK, 0, 0);

  rt.shutdown(e);

  int ret = rt.wait_for_shutdown();

  return ret;
}
