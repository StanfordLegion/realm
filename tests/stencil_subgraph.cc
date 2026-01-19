#include "realm.h"

using namespace Realm;

Logger log_app("app");

// Task IDs, some IDs are reserved so start at first available number
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE+0,
  DUMMY_TASK,
};

void dummy_task(const void *args, size_t arglen,
                const void *userdata, size_t userlen, Processor p) {
}


void top_level_task(const void *args, size_t arglen,
                    const void *userdata, size_t userlen, Processor p) {
  log_app.print() << "Realm subgraphs test";
  auto pq = Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  std::vector<Processor> cpus(pq.begin(), pq.end());

  size_t width = cpus.size();
  size_t steps = 50;

  // Generate a subgraph with a stencil pattern. 50 steps, width=width.
  SubgraphDefinition sd;
  // Add all the tasks.
  std::map<std::pair<size_t, size_t>, size_t> point_to_tasks;
  for (size_t i = 0; i < steps; i++) {
    for (size_t j = 0; j < width; j++) {
      SubgraphDefinition::TaskDesc task;
      task.task_id = DUMMY_TASK;
      task.proc = cpus[j];
      point_to_tasks[{i, j}] = sd.tasks.size();
      sd.tasks.push_back(task);
    }
  }
  // Add the stencil dependencies.
  for (size_t i = 1; i < steps; i++) {
    for (size_t j = 0; j < width; j++) {
      auto tgt_task = point_to_tasks[{i, j}];
      // j-1, j, j+1
      if (j > 0) {
        SubgraphDefinition::Dependency dep;
        dep.src_op_kind = SubgraphDefinition::OPKIND_TASK;
        dep.src_op_index = point_to_tasks.at({i - 1, j - 1});
        dep.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
        dep.tgt_op_index = tgt_task;
        sd.dependencies.push_back(dep);
      }
      {
        SubgraphDefinition::Dependency dep;
        dep.src_op_kind = SubgraphDefinition::OPKIND_TASK;
        dep.src_op_index = point_to_tasks.at({i - 1, j});
        dep.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
        dep.tgt_op_index = tgt_task;
        sd.dependencies.push_back(dep);
      }
      if (j < width - 1) {
        SubgraphDefinition::Dependency dep;
        dep.src_op_kind = SubgraphDefinition::OPKIND_TASK;
        dep.src_op_index = point_to_tasks.at({i - 1, j + 1});
        dep.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
        dep.tgt_op_index = tgt_task;
        sd.dependencies.push_back(dep);
      }
    }
  }
  Subgraph graph;
  Subgraph::create_subgraph(graph, sd, ProfilingRequestSet()).wait();
  graph.instantiate(nullptr, 0, ProfilingRequestSet()).wait();

  Runtime::get_runtime().shutdown();
}

int main(int argc, char **argv)
{
  Runtime rt;

  rt.init(&argc, &argv);

#if 0
//  for(int i = 1; i < argc; i++) {
//    if(!strcmp(argv[i], "-b")) {
//      buffer_size = strtoll(argv[++i], 0, 10);
//      continue;
//    }
//  }
#endif

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(DUMMY_TASK, dummy_task);

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
