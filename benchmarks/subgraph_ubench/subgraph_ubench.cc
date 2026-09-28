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

// Subgraph micro-benchmark.
//
// Replays one task graph many times three ways and reports the cost per
// instantiation and per task:
//   spawn        plain Processor::spawn with event dependencies (baseline)
//   interpreted  SubgraphDefinition::INTERPRETED
//   compiled     SubgraphDefinition::COMPILED
//
// The first CPU runs the driver; the graph's tasks use the remaining CPUs.
//
//   -shape chain|independent|fan|layers|random   graph shape (default chain)
//   -n N            operations for chain/independent/fan/random (default 64)
//   -layers L -width W   for the layers shape (default 8 x 8, dense)
//   -prob P         edge probability for random (default 0.1)
//   -p P            processors to use (default: all workers)
//   -k K            distinct subgraphs kept active concurrently (default 1)
//   -iters I        measured instantiations per subgraph (default 1000)
//   -warmup W       unmeasured instantiations (default 100)
//   -work NS        busy-wait per task in nanoseconds (default 0)
//   -mode M         spawn|interpreted|compiled|all (default all)
//   -seed S         seed for the random shape
//
// Output lines start with RESULT and are key=value so they can be grepped
// and tabulated directly.

#include "realm.h"
#include "realm/subgraph.h"
#include "realm/timers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  WORK_TASK,
};

struct BenchConfig {
  std::string shape = "chain";
  int n = 64;
  int layers = 8, width = 8;
  double prob = 0.1;
  int p = 0;
  int k = 1;
  int iters = 1000;
  int warmup = 100;
  long work_ns = 0;
  std::string mode = "all";
  unsigned seed = 1;
};
static BenchConfig cfg;

struct WorkArgs {
  long spin_ns;
};

static void work_task(const void *args, size_t arglen, const void *userdata,
                      size_t userlen, Processor p)
{
  const WorkArgs *a = static_cast<const WorkArgs *>(args);
  if(a->spin_ns > 0) {
    long long t0 = Clock::current_time_in_nanoseconds();
    while(Clock::current_time_in_nanoseconds() - t0 < a->spin_ns) {
    }
  }
}

////////////////////////////////////////////////////////////////////////
//
// Graph shapes
//

struct Dag {
  std::vector<int> proc_of_op;
  std::vector<std::vector<int>> preds;
  std::vector<int> sinks; // operations with no successors

  size_t size() const { return proc_of_op.size(); }
  size_t num_edges() const
  {
    size_t e = 0;
    for(const std::vector<int> &p : preds)
      e += p.size();
    return e;
  }
  int add_op(int proc, std::vector<int> ps = {})
  {
    proc_of_op.push_back(proc);
    preds.push_back(std::move(ps));
    return proc_of_op.size() - 1;
  }
  void finalize()
  {
    std::vector<bool> has_succ(size(), false);
    for(const std::vector<int> &ps : preds)
      for(int p : ps)
        has_succ[p] = true;
    sinks.clear();
    for(size_t i = 0; i < size(); i++)
      if(!has_succ[i])
        sinks.push_back(int(i));
  }
};

static Dag make_dag(int nprocs)
{
  Dag d;
  std::mt19937 rng(cfg.seed);
  if(cfg.shape == "chain") {
    for(int i = 0; i < cfg.n; i++)
      d.add_op(i % nprocs, (i > 0) ? std::vector<int>{i - 1} : std::vector<int>{});
  } else if(cfg.shape == "independent") {
    for(int i = 0; i < cfg.n; i++)
      d.add_op(i % nprocs);
  } else if(cfg.shape == "fan") {
    int root = d.add_op(0);
    std::vector<int> middle;
    for(int i = 0; i < cfg.n; i++)
      middle.push_back(d.add_op((i + 1) % nprocs, {root}));
    d.add_op(0, middle);
  } else if(cfg.shape == "layers") {
    for(int l = 0; l < cfg.layers; l++)
      for(int w = 0; w < cfg.width; w++) {
        std::vector<int> ps;
        if(l > 0)
          for(int pw = 0; pw < cfg.width; pw++)
            ps.push_back((l - 1) * cfg.width + pw);
        d.add_op((l * cfg.width + w) % nprocs, ps);
      }
  } else if(cfg.shape == "random") {
    std::uniform_int_distribution<int> pick(0, nprocs - 1);
    std::bernoulli_distribution edge(cfg.prob);
    for(int i = 0; i < cfg.n; i++) {
      std::vector<int> ps;
      for(int j = 0; j < i; j++)
        if(edge(rng))
          ps.push_back(j);
      d.add_op(pick(rng), ps);
    }
  } else {
    fprintf(stderr, "unknown shape '%s'\n", cfg.shape.c_str());
    exit(1);
  }
  d.finalize();
  return d;
}

////////////////////////////////////////////////////////////////////////
//
// Replay strategies
//

static Subgraph build_subgraph(const Dag &dag, const std::vector<Processor> &procs,
                               SubgraphDefinition::ExecutionMode mode)
{
  SubgraphDefinition sd;
  sd.execution_mode = mode;
  sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
  WorkArgs wa{cfg.work_ns};
  for(size_t i = 0; i < dag.size(); i++) {
    SubgraphDefinition::TaskDesc td;
    td.proc = procs[dag.proc_of_op[i]];
    td.task_id = WORK_TASK;
    td.args.set(&wa, sizeof(wa));
    sd.tasks.push_back(td);
  }
  for(size_t i = 0; i < dag.size(); i++)
    for(int pred : dag.preds[i]) {
      SubgraphDefinition::Dependency dep;
      dep.src_op_kind = SubgraphDefinition::OPKIND_TASK;
      dep.src_op_index = pred;
      dep.tgt_op_kind = SubgraphDefinition::OPKIND_TASK;
      dep.tgt_op_index = i;
      sd.dependencies.push_back(dep);
    }
  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  return sg;
}

// One replay of the graph with plain spawns. Roots wait on `chain`.
static Event replay_spawn(const Dag &dag, const std::vector<Processor> &procs, Event chain,
                          std::vector<Event> &events, std::vector<Event> &scratch)
{
  WorkArgs wa{cfg.work_ns};
  events.resize(dag.size());
  for(size_t i = 0; i < dag.size(); i++) {
    Event pre;
    const std::vector<int> &ps = dag.preds[i];
    if(ps.empty()) {
      pre = chain;
    } else if(ps.size() == 1) {
      pre = events[ps[0]];
    } else {
      scratch.clear();
      for(int p : ps)
        scratch.push_back(events[p]);
      pre = Event::merge_events(scratch);
    }
    events[i] = procs[dag.proc_of_op[i]].spawn(WORK_TASK, &wa, sizeof(wa), pre);
  }
  if(dag.sinks.size() == 1)
    return events[dag.sinks[0]];
  scratch.clear();
  for(int s : dag.sinks)
    scratch.push_back(events[s]);
  return Event::merge_events(scratch);
}

struct Result {
  double issue_us; // time for the issuing call(s) to return, per instantiation
  double inst_us;  // end-to-end wall time, per instantiation
};

static void print_result(const char *mode, bool chained, const Dag &dag, int nprocs,
                         const Result &r)
{
  double ns_per_task = r.inst_us * 1e3 / dag.size();
  printf("RESULT shape=%s n=%zu edges=%zu p=%d k=%d work_ns=%ld mode=%s chained=%d iters=%d "
         "issue_us=%.3f inst_us=%.3f ns_per_task=%.1f\n",
         cfg.shape.c_str(), dag.size(), dag.num_edges(), nprocs, cfg.k, cfg.work_ns, mode,
         chained ? 1 : 0, cfg.iters, r.issue_us, r.inst_us, ns_per_task);
  fflush(stdout);
}

static Result measure_spawn(const Dag &dag, const std::vector<Processor> &procs)
{
  std::vector<Event> events, scratch;
  Event chain = Event::NO_EVENT;
  for(int i = 0; i < cfg.warmup; i++)
    chain = replay_spawn(dag, procs, chain, events, scratch);
  chain.wait();

  chain = Event::NO_EVENT;
  double t0 = Clock::current_time();
  for(int i = 0; i < cfg.iters; i++)
    chain = replay_spawn(dag, procs, chain, events, scratch);
  double t1 = Clock::current_time();
  chain.wait();
  double t2 = Clock::current_time();
  Result r;
  r.issue_us = (t1 - t0) * 1e6 / cfg.iters;
  r.inst_us = (t2 - t0) * 1e6 / cfg.iters;
  return r;
}

static Result measure_subgraphs(std::vector<Subgraph> &sgs, bool chained)
{
  const int k = sgs.size();
  std::vector<Event> last(k, Event::NO_EVENT);
  for(int i = 0; i < cfg.warmup; i++)
    for(int j = 0; j < k; j++)
      last[j] = sgs[j].instantiate(nullptr, 0, ProfilingRequestSet(),
                                   chained ? last[j] : Event::NO_EVENT);
  Event::merge_events(last).wait();

  std::fill(last.begin(), last.end(), Event::NO_EVENT);
  double t0 = Clock::current_time();
  for(int i = 0; i < cfg.iters; i++)
    for(int j = 0; j < k; j++)
      last[j] = sgs[j].instantiate(nullptr, 0, ProfilingRequestSet(),
                                   chained ? last[j] : Event::NO_EVENT);
  double t1 = Clock::current_time();
  Event::merge_events(last).wait();
  double t2 = Clock::current_time();
  Result r;
  r.issue_us = (t1 - t0) * 1e6 / (double(cfg.iters) * k);
  r.inst_us = (t2 - t0) * 1e6 / (double(cfg.iters) * k);
  return r;
}

////////////////////////////////////////////////////////////////////////
//
// Driver
//

void top_level_task(const void *args, size_t arglen, const void *userdata, size_t userlen,
                    Processor p)
{
  Machine::ProcessorQuery pq =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  std::vector<Processor> cpus(pq.begin(), pq.end());
  if(cpus.size() < 2) {
    fprintf(stderr, "need at least 2 CPUs (-ll:cpu N): one driver plus workers\n");
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 1);
    return;
  }
  std::vector<Processor> procs(cpus.begin() + 1, cpus.end());
  if((cfg.p > 0) && (size_t(cfg.p) < procs.size()))
    procs.resize(cfg.p);
  const int nprocs = procs.size();

  Dag dag = make_dag(nprocs);
  printf("subgraph_ubench: shape=%s ops=%zu edges=%zu procs=%d k=%d iters=%d warmup=%d "
         "work_ns=%ld\n",
         cfg.shape.c_str(), dag.size(), dag.num_edges(), nprocs, cfg.k, cfg.iters,
         cfg.warmup, cfg.work_ns);
  fflush(stdout);

  const bool all = (cfg.mode == "all");
  if(all || (cfg.mode == "spawn")) {
    Result r = measure_spawn(dag, procs);
    print_result("spawn", true, dag, nprocs, r);
  }
  for(int m = 0; m < 2; m++) {
    SubgraphDefinition::ExecutionMode mode =
        (m == 0) ? SubgraphDefinition::INTERPRETED : SubgraphDefinition::COMPILED;
    const char *name = (m == 0) ? "interpreted" : "compiled";
    if(!all && (cfg.mode != name))
      continue;
    std::vector<Subgraph> sgs;
    for(int j = 0; j < cfg.k; j++)
      sgs.push_back(build_subgraph(dag, procs, mode));
    for(int chained = 1; chained >= 0; chained--) {
      Result r = measure_subgraphs(sgs, chained != 0);
      print_result(name, chained != 0, dag, nprocs, r);
    }
    std::vector<Event> done;
    for(Subgraph &sg : sgs)
      done.push_back(sg.destroy());
    Event::merge_events(done).wait();
  }

  Runtime::get_runtime().shutdown(Event::NO_EVENT, 0);
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-shape") && (i + 1 < argc))
      cfg.shape = argv[++i];
    else if(!strcmp(argv[i], "-n") && (i + 1 < argc))
      cfg.n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-layers") && (i + 1 < argc))
      cfg.layers = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-width") && (i + 1 < argc))
      cfg.width = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-prob") && (i + 1 < argc))
      cfg.prob = atof(argv[++i]);
    else if(!strcmp(argv[i], "-p") && (i + 1 < argc))
      cfg.p = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-k") && (i + 1 < argc))
      cfg.k = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-iters") && (i + 1 < argc))
      cfg.iters = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-warmup") && (i + 1 < argc))
      cfg.warmup = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-work") && (i + 1 < argc))
      cfg.work_ns = atol(argv[++i]);
    else if(!strcmp(argv[i], "-mode") && (i + 1 < argc))
      cfg.mode = argv[++i];
    else if(!strcmp(argv[i], "-seed") && (i + 1 < argc))
      cfg.seed = strtoul(argv[++i], 0, 10);
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(WORK_TASK, work_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, TOP_LEVEL_TASK, 0, 0);
  return rt.wait_for_shutdown();
}
