/* Copyright 2025 Stanford University, NVIDIA Corporation
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

// A 2-D 5-point stencil over px x py tiles, each with a halo, as an
// end-to-end check and benchmark of compiled subgraphs: every step moves
// halos between neighbors with copies, runs a stencil task per tile and
// then an increment task per tile. The same step sequence runs either as
// individually issued copies and task spawns ("direct") or as replays of
// one compiled subgraph covering `-sgsteps` steps ("subgraph"), and the
// result is checked against a serial computation.
//
//   -nx N -ny N     grid size (default 100 x 100)
//   -px P -py P     tiles (default 2 x 2); tiles go to GPUs if there are
//                   any, otherwise to CPUs, round robin
//   -steps S        steps per run (default 50)
//   -sgsteps S      steps per subgraph instantiation (default 10)
//   -mode M         direct, subgraph or both (default both)
//   -rounds R       runs per mode; the first is a warm-up (default 2)
//   -no-check       skip the verification

#include "stencil_subgraph.h"
#include "realm/subgraph.h"

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>

using namespace Realm;

Logger log_app("app");

enum
{
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 0,
  STENCIL_TASK,
  INCREMENT_TASK,
  VERIFY_TASK,
};

struct VerifyArgs {
  RegionInstance golden = RegionInstance::NO_INST;
  RegionInstance computed = RegionInstance::NO_INST;
  IndexSpace<2> space = IndexSpace<2>();
  std::atomic<long> *mismatches = nullptr;
};

struct TLTArgs {
  int64_t nx = 100;
  int64_t ny = 100;
  int64_t px = 2;
  int64_t py = 2;
  int64_t steps = 50;
  int64_t sgsteps = 10;
  int rounds = 2;
  bool direct = true, subgraph = true;
  bool verify = true;
};

typedef std::pair<int64_t, int64_t> Tile;
template <typename T>
using TileMap = std::map<Tile, T>;

struct Grid {
  int64_t nx, ny, px, py;
  TileMap<Processor> procs;
  TileMap<IndexSpace<2>> local, north, south, west, east;
  TileMap<RegionInstance> inst;
};

void stencil_task(const void *_args, size_t arglen, const void *userdata, size_t userlen,
                  Processor p)
{
  const StencilArgs *args = static_cast<const StencilArgs *>(_args);
  AffineAccessor<float, 2> input(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> output(args->buffer, FID_OUTPUT);
  Rect<2> b = args->local_space.bounds;
  for(int64_t i = b.lo[0]; i <= b.hi[0]; i++)
    for(int64_t j = b.lo[1]; j <= b.hi[1]; j++) {
      float center = input[Point<2>(i, j)];
      float north = (i > 0) ? input[Point<2>(i - 1, j)] : 0.0f;
      float south = (i < args->hx - 1) ? input[Point<2>(i + 1, j)] : 0.0f;
      float west = (j > 0) ? input[Point<2>(i, j - 1)] : 0.0f;
      float east = (j < args->hy - 1) ? input[Point<2>(i, j + 1)] : 0.0f;
      output[Point<2>(i, j)] = (center + north + south + west + east) / 5.f;
    }
}

void increment_task(const void *_args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p)
{
  const IncrementArgs *args = static_cast<const IncrementArgs *>(_args);
  AffineAccessor<float, 2> input(args->buffer, FID_INPUT);
  AffineAccessor<float, 2> output(args->buffer, FID_OUTPUT);
  Rect<2> b = args->local_space.bounds;
  for(int64_t i = b.lo[0]; i <= b.hi[0]; i++)
    for(int64_t j = b.lo[1]; j <= b.hi[1]; j++)
      input[Point<2>(i, j)] = output[Point<2>(i, j)] + 1.0f;
}

void verify_task(const void *_args, size_t arglen, const void *userdata, size_t userlen,
                 Processor p)
{
  const VerifyArgs *args = static_cast<const VerifyArgs *>(_args);
  AffineAccessor<float, 2> computed(args->computed, FID_INPUT);
  AffineAccessor<float, 2> golden(args->golden, FID_INPUT);
  Rect<2> b = args->space.bounds;
  long bad = 0;
  for(int64_t i = b.lo[0]; i <= b.hi[0]; i++)
    for(int64_t j = b.lo[1]; j <= b.hi[1]; j++)
      if(computed[Point<2>(i, j)] != golden[Point<2>(i, j)]) {
        if(bad < 5)
          printf("MISMATCH (%lld,%lld): found %.8f expected %.8f\n", (long long)i,
                 (long long)j, computed[Point<2>(i, j)], golden[Point<2>(i, j)]);
        bad++;
      }
  args->mismatches->store(bad);
}

// ---- direct issue: a copy or spawn per operation, events for dependencies

void run_stencil_direct(const Grid &g, int64_t steps)
{
  TileMap<Event> done, in_north, in_south, in_east, in_west;
  for(auto &kv : g.procs) {
    done[kv.first] = Event::NO_EVENT;
    in_north[kv.first] = in_south[kv.first] = in_east[kv.first] = in_west[kv.first] =
        Event::NO_EVENT;
  }
  std::vector<Event> deps;
  for(int64_t step = 0; step < steps; step++) {
    // halo copies from each neighbor's input into ours
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        std::vector<CopySrcDstField> src(1), dst(1);
        Event cur = done[t];
        dst[0].set_field(g.inst.at(t), FID_INPUT, sizeof(float));
        if(i > 0) {
          src[0].set_field(g.inst.at(Tile(i - 1, j)), FID_INPUT, sizeof(float));
          in_north[t] = g.north.at(t).copy(src, dst, ProfilingRequestSet(),
                                           Event::merge_events(cur, done[Tile(i - 1, j)]));
        }
        if(i < g.px - 1) {
          src[0].set_field(g.inst.at(Tile(i + 1, j)), FID_INPUT, sizeof(float));
          in_south[t] = g.south.at(t).copy(src, dst, ProfilingRequestSet(),
                                           Event::merge_events(cur, done[Tile(i + 1, j)]));
        }
        if(j > 0) {
          src[0].set_field(g.inst.at(Tile(i, j - 1)), FID_INPUT, sizeof(float));
          in_west[t] = g.west.at(t).copy(src, dst, ProfilingRequestSet(),
                                         Event::merge_events(cur, done[Tile(i, j - 1)]));
        }
        if(j < g.py - 1) {
          src[0].set_field(g.inst.at(Tile(i, j + 1)), FID_INPUT, sizeof(float));
          in_east[t] = g.east.at(t).copy(src, dst, ProfilingRequestSet(),
                                         Event::merge_events(cur, done[Tile(i, j + 1)]));
        }
      }
    // stencil tasks
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        deps.assign({done[t], in_north[t], in_south[t], in_east[t], in_west[t]});
        StencilArgs args;
        args.buffer = g.inst.at(t);
        args.local_space = g.local.at(t);
        args.hx = g.nx;
        args.hy = g.ny;
        done[t] = g.procs.at(t).spawn(STENCIL_TASK, &args, sizeof(args),
                                      Event::merge_events(deps));
      }
    // increment tasks: wait for the copies that read our input as well
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        deps.assign(1, done[t]);
        if(i > 0)
          deps.push_back(in_south[Tile(i - 1, j)]);
        if(i < g.px - 1)
          deps.push_back(in_north[Tile(i + 1, j)]);
        if(j > 0)
          deps.push_back(in_east[Tile(i, j - 1)]);
        if(j < g.py - 1)
          deps.push_back(in_west[Tile(i, j + 1)]);
        IncrementArgs args;
        args.buffer = g.inst.at(t);
        args.local_space = g.local.at(t);
        done[t] = g.procs.at(t).spawn(INCREMENT_TASK, &args, sizeof(args),
                                      Event::merge_events(deps));
      }
  }
  std::vector<Event> all;
  for(auto &kv : done)
    all.push_back(kv.second);
  Event::merge_events(all).wait();
}

// ---- compiled: one subgraph covering sgsteps steps

Subgraph compile_stencil(const Grid &g, int64_t sgsteps)
{
  typedef std::tuple<int64_t, int64_t, int64_t> Key; // i, j, step
  SubgraphDefinition sd;
  sd.concurrency_mode = SubgraphDefinition::INSTANTIATION_ORDER;
  std::map<Key, unsigned> stencil, increment, cn, cs, cw, ce;

  auto add_copy = [&](const IndexSpace<2> &space, RegionInstance from, RegionInstance to) {
    SubgraphDefinition::CopyDesc copy;
    copy.space = space;
    copy.srcs.resize(1);
    copy.srcs[0].set_field(from, FID_INPUT, sizeof(float));
    copy.dsts.resize(1);
    copy.dsts[0].set_field(to, FID_INPUT, sizeof(float));
    sd.copies.push_back(copy);
    return unsigned(sd.copies.size() - 1);
  };
  auto dep = [&](SubgraphDefinition::OpKind sk, unsigned si, SubgraphDefinition::OpKind tk,
                 unsigned ti) {
    SubgraphDefinition::Dependency d;
    d.src_op_kind = sk;
    d.src_op_index = si;
    d.tgt_op_kind = tk;
    d.tgt_op_index = ti;
    sd.dependencies.push_back(d);
  };
  const SubgraphDefinition::OpKind TASK = SubgraphDefinition::OPKIND_TASK;
  const SubgraphDefinition::OpKind COPY = SubgraphDefinition::OPKIND_COPY;

  for(int64_t step = 0; step < sgsteps; step++) {
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        if(i > 0)
          cn[Key(i, j, step)] = add_copy(g.north.at(t), g.inst.at(Tile(i - 1, j)), g.inst.at(t));
        if(i < g.px - 1)
          cs[Key(i, j, step)] = add_copy(g.south.at(t), g.inst.at(Tile(i + 1, j)), g.inst.at(t));
        if(j > 0)
          cw[Key(i, j, step)] = add_copy(g.west.at(t), g.inst.at(Tile(i, j - 1)), g.inst.at(t));
        if(j < g.py - 1)
          ce[Key(i, j, step)] = add_copy(g.east.at(t), g.inst.at(Tile(i, j + 1)), g.inst.at(t));
      }
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        SubgraphDefinition::TaskDesc task;
        task.proc = g.procs.at(t);
        task.task_id = STENCIL_TASK;
        StencilArgs args;
        args.buffer = g.inst.at(t);
        args.local_space = g.local.at(t);
        args.hx = g.nx;
        args.hy = g.ny;
        task.args = ByteArray(&args, sizeof(args));
        stencil[Key(i, j, step)] = unsigned(sd.tasks.size());
        sd.tasks.push_back(task);
      }
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        SubgraphDefinition::TaskDesc task;
        task.proc = g.procs.at(t);
        task.task_id = INCREMENT_TASK;
        IncrementArgs args;
        args.buffer = g.inst.at(t);
        args.local_space = g.local.at(t);
        task.args = ByteArray(&args, sizeof(args));
        increment[Key(i, j, step)] = unsigned(sd.tasks.size());
        sd.tasks.push_back(task);
      }
  }

  for(int64_t step = 0; step < sgsteps; step++)
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Key k(i, j, step);
        // copies of this step follow the previous step's increments of
        // both tiles involved
        if(step > 0) {
          Key prev(i, j, step - 1);
          if(i > 0) {
            dep(TASK, increment[prev], COPY, cn[k]);
            dep(TASK, increment[Key(i - 1, j, step - 1)], COPY, cn[k]);
          }
          if(i < g.px - 1) {
            dep(TASK, increment[prev], COPY, cs[k]);
            dep(TASK, increment[Key(i + 1, j, step - 1)], COPY, cs[k]);
          }
          if(j > 0) {
            dep(TASK, increment[prev], COPY, cw[k]);
            dep(TASK, increment[Key(i, j - 1, step - 1)], COPY, cw[k]);
          }
          if(j < g.py - 1) {
            dep(TASK, increment[prev], COPY, ce[k]);
            dep(TASK, increment[Key(i, j + 1, step - 1)], COPY, ce[k]);
          }
          dep(TASK, increment[prev], TASK, stencil[k]);
        }
        // the stencil follows its incoming halo copies
        if(i > 0)
          dep(COPY, cn[k], TASK, stencil[k]);
        if(i < g.px - 1)
          dep(COPY, cs[k], TASK, stencil[k]);
        if(j > 0)
          dep(COPY, cw[k], TASK, stencil[k]);
        if(j < g.py - 1)
          dep(COPY, ce[k], TASK, stencil[k]);
        // the increment follows the stencil and the copies reading our input
        dep(TASK, stencil[k], TASK, increment[k]);
        if(i > 0)
          dep(COPY, cs[Key(i - 1, j, step)], TASK, increment[k]);
        if(i < g.px - 1)
          dep(COPY, cn[Key(i + 1, j, step)], TASK, increment[k]);
        if(j > 0)
          dep(COPY, ce[Key(i, j - 1, step)], TASK, increment[k]);
        if(j < g.py - 1)
          dep(COPY, cw[Key(i, j + 1, step)], TASK, increment[k]);
      }

  Subgraph sg;
  Subgraph::create_subgraph(sg, sd, ProfilingRequestSet()).wait();
  printf("compiled: %zu tasks, %zu copies, %zu dependencies per %lld steps\n",
         sd.tasks.size(), sd.copies.size(), sd.dependencies.size(), (long long)sgsteps);
  return sg;
}

void run_stencil_subgraph(Subgraph sg, int64_t steps, int64_t sgsteps)
{
  Event e = Event::NO_EVENT;
  for(int64_t step = 0; step < steps; step += sgsteps)
    e = sg.instantiate(nullptr, 0, ProfilingRequestSet(), e);
  e.wait();
}

void top_level_task(const void *_args, size_t arglen, const void *userdata,
                    size_t userlen, Processor p)
{
  const TLTArgs *args = static_cast<const TLTArgs *>(_args);
  Grid g;
  g.nx = args->nx;
  g.ny = args->ny;
  g.px = args->px;
  g.py = args->py;
  if((g.nx % g.px != 0) || (g.ny % g.py != 0)) {
    fprintf(stderr, "grid %lld x %lld is not divisible into %lld x %lld tiles\n",
            (long long)g.nx, (long long)g.ny, (long long)g.px, (long long)g.py);
    Runtime::get_runtime().shutdown(Event::NO_EVENT, 1);
    return;
  }

  Machine::ProcessorQuery cq =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::LOC_PROC);
  std::vector<Processor> cpus(cq.begin(), cq.end());
  Machine::ProcessorQuery gq =
      Machine::ProcessorQuery(Machine::get_machine()).only_kind(Processor::TOC_PROC);
  std::vector<Processor> gpus(gq.begin(), gq.end());
  // the driver CPU stays out of the tile set when there are enough CPUs
  std::vector<Processor> workers = gpus.empty() ? cpus : gpus;
  if(gpus.empty() && (workers.size() > 1))
    workers.erase(workers.begin());

  TileMap<Memory> mems;
  for(int64_t i = 0; i < g.px; i++)
    for(int64_t j = 0; j < g.py; j++) {
      Tile t(i, j);
      g.procs[t] = workers[(i * g.py + j) % workers.size()];
      Machine::MemoryQuery mq(Machine::get_machine());
      mq.only_kind(gpus.empty() ? Memory::SYSTEM_MEM : Memory::GPU_FB_MEM)
          .has_capacity(1)
          .best_affinity_to(g.procs[t]);
      mems[t] = mq.first();
      if(!mems[t].exists()) {
        fprintf(stderr, "no memory for processor %llx\n", (unsigned long long)g.procs[t].id);
        Runtime::get_runtime().shutdown(Event::NO_EVENT, 1);
        return;
      }
    }

  const int64_t lx = g.nx / g.px, ly = g.ny / g.py;
  TileMap<IndexSpace<2>> bloated;
  for(int64_t i = 0; i < g.px; i++)
    for(int64_t j = 0; j < g.py; j++) {
      Tile t(i, j);
      int64_t x0 = i * lx, x1 = (i + 1) * lx, y0 = j * ly, y1 = (j + 1) * ly;
      g.local[t] = Rect<2>(Point<2>(x0, y0), Point<2>(x1 - 1, y1 - 1));
      bloated[t] = Rect<2>(Point<2>((i == 0) ? x0 : x0 - 1, (j == 0) ? y0 : y0 - 1),
                           Point<2>((i == g.px - 1) ? x1 - 1 : x1,
                                    (j == g.py - 1) ? y1 - 1 : y1));
    }

  std::map<FieldID, size_t> field_sizes = {{FID_INPUT, sizeof(float)},
                                           {FID_OUTPUT, sizeof(float)}};
  {
    Event e = Event::NO_EVENT;
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        e = RegionInstance::create_instance(g.inst[t], mems[t], bloated[t], field_sizes, 0,
                                            ProfilingRequestSet(), e);
        std::vector<CopySrcDstField> f(1);
        f[0].set_field(g.inst[t], FID_INPUT, sizeof(float));
        float zero = 0;
        e = g.local[t].fill(f, ProfilingRequestSet(), &zero, sizeof(zero), e);
      }
    e.wait();
  }
  {
    Event e = Event::NO_EVENT;
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        if(i > 0)
          e = IndexSpace<2>::compute_intersection(bloated[t], g.local[Tile(i - 1, j)],
                                                  g.north[t], ProfilingRequestSet(), e);
        if(j > 0)
          e = IndexSpace<2>::compute_intersection(bloated[t], g.local[Tile(i, j - 1)],
                                                  g.west[t], ProfilingRequestSet(), e);
        if(i < g.px - 1)
          e = IndexSpace<2>::compute_intersection(bloated[t], g.local[Tile(i + 1, j)],
                                                  g.south[t], ProfilingRequestSet(), e);
        if(j < g.py - 1)
          e = IndexSpace<2>::compute_intersection(bloated[t], g.local[Tile(i, j + 1)],
                                                  g.east[t], ProfilingRequestSet(), e);
      }
    e.wait();
  }

  printf("stencil_subgraph: %lld x %lld grid, %lld x %lld tiles on %zu %s, %lld steps per "
         "run, %d rounds per mode (first is warm-up), ranks=%u\n",
         (long long)g.nx, (long long)g.ny, (long long)g.px, (long long)g.py, workers.size(),
         gpus.empty() ? "CPUs" : "GPUs", (long long)args->steps, args->rounds,
         Machine::get_machine().get_address_space_count());
  fflush(stdout);

  int64_t total_steps = 0;
  auto report = [&](const char *mode, double seconds, int round) {
    printf("RESULT mode=%s round=%d nx=%lld ny=%lld px=%lld py=%lld steps=%lld time_ms=%.3f "
           "us_per_step=%.2f\n",
           mode, round, (long long)g.nx, (long long)g.ny, (long long)g.px, (long long)g.py,
           (long long)args->steps, seconds * 1e3, seconds * 1e6 / args->steps);
    fflush(stdout);
  };
  if(args->direct) {
    for(int r = 0; r < args->rounds; r++) {
      double t0 = Clock::current_time();
      run_stencil_direct(g, args->steps);
      report("direct", Clock::current_time() - t0, r);
      total_steps += args->steps;
    }
  }
  if(args->subgraph) {
    int64_t sgsteps = args->sgsteps;
    if(args->steps % sgsteps != 0) {
      fprintf(stderr, "steps (%lld) must be a multiple of sgsteps (%lld)\n",
              (long long)args->steps, (long long)sgsteps);
      Runtime::get_runtime().shutdown(Event::NO_EVENT, 1);
      return;
    }
    double tc = Clock::current_time();
    Subgraph sg = compile_stencil(g, sgsteps);
    printf("compile time %.3f ms\n", (Clock::current_time() - tc) * 1e3);
    for(int r = 0; r < args->rounds; r++) {
      double t0 = Clock::current_time();
      run_stencil_subgraph(sg, args->steps, sgsteps);
      report("subgraph", Clock::current_time() - t0, r);
      total_steps += args->steps;
    }
    sg.destroy().wait();
  }

  int exit_code = 0;
  if(args->verify) {
    Memory sysmem = Machine::MemoryQuery(Machine::get_machine())
                        .only_kind(Memory::SYSTEM_MEM)
                        .has_affinity_to(p)
                        .first();
    IndexSpace<2> full = Rect<2>(Point<2>(0, 0), Point<2>(g.nx - 1, g.ny - 1));
    RegionInstance golden, computed;
    RegionInstance::create_instance(golden, sysmem, full, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    RegionInstance::create_instance(computed, sysmem, full, field_sizes, 0,
                                    ProfilingRequestSet())
        .wait();
    {
      AffineAccessor<float, 2> input(golden, FID_INPUT);
      AffineAccessor<float, 2> output(golden, FID_OUTPUT);
      for(int64_t i = 0; i < g.nx; i++)
        for(int64_t j = 0; j < g.ny; j++)
          input[Point<2>(i, j)] = 0;
      for(int64_t step = 0; step < total_steps; step++) {
        for(int64_t i = 0; i < g.nx; i++)
          for(int64_t j = 0; j < g.ny; j++) {
            float center = input[Point<2>(i, j)];
            float north = (i > 0) ? input[Point<2>(i - 1, j)] : 0.0f;
            float south = (i < g.nx - 1) ? input[Point<2>(i + 1, j)] : 0.0f;
            float west = (j > 0) ? input[Point<2>(i, j - 1)] : 0.0f;
            float east = (j < g.ny - 1) ? input[Point<2>(i, j + 1)] : 0.0f;
            output[Point<2>(i, j)] = (center + north + south + west + east) / 5.f;
          }
        for(int64_t i = 0; i < g.nx; i++)
          for(int64_t j = 0; j < g.ny; j++)
            input[Point<2>(i, j)] = output[Point<2>(i, j)] + 1.0f;
      }
    }
    for(int64_t i = 0; i < g.px; i++)
      for(int64_t j = 0; j < g.py; j++) {
        Tile t(i, j);
        std::vector<CopySrcDstField> src(1), dst(1);
        src[0].set_field(g.inst[t], FID_INPUT, sizeof(float));
        dst[0].set_field(computed, FID_INPUT, sizeof(float));
        g.local[t].copy(src, dst, ProfilingRequestSet()).wait();
      }
    std::atomic<long> mismatches(0);
    VerifyArgs v;
    v.golden = golden;
    v.computed = computed;
    v.space = full;
    v.mismatches = &mismatches;
    p.spawn(VERIFY_TASK, &v, sizeof(v)).wait();
    if(mismatches.load() == 0) {
      printf("VERIFY PASS (%lld steps)\n", (long long)total_steps);
    } else {
      printf("VERIFY FAIL: %ld mismatches\n", mismatches.load());
      exit_code = 1;
    }
    golden.destroy();
    computed.destroy();
  }
  for(auto &kv : g.inst)
    kv.second.destroy();
  Runtime::get_runtime().shutdown(Event::NO_EVENT, exit_code);
}

int main(int argc, char **argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  TLTArgs args;
  for(int i = 1; i < argc; i++) {
    if(!strcmp(argv[i], "-nx") && (i + 1 < argc))
      args.nx = strtoll(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-ny") && (i + 1 < argc))
      args.ny = strtoll(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-px") && (i + 1 < argc))
      args.px = strtoll(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-py") && (i + 1 < argc))
      args.py = strtoll(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-steps") && (i + 1 < argc))
      args.steps = strtoll(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-sgsteps") && (i + 1 < argc))
      args.sgsteps = strtoll(argv[++i], 0, 10);
    else if(!strcmp(argv[i], "-rounds") && (i + 1 < argc))
      args.rounds = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-mode") && (i + 1 < argc)) {
      const char *m = argv[++i];
      args.direct = !strcmp(m, "direct") || !strcmp(m, "both");
      args.subgraph = !strcmp(m, "subgraph") || !strcmp(m, "both");
    } else if(!strcmp(argv[i], "-subgraph")) {
      args.direct = false;
      args.subgraph = true;
    } else if(!strcmp(argv[i], "-no-check"))
      args.verify = false;
  }

  rt.register_task(TOP_LEVEL_TASK, top_level_task);
  rt.register_task(VERIFY_TASK, verify_task);
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, STENCIL_TASK,
                                   CodeDescriptor(stencil_task), ProfilingRequestSet())
      .external_wait();
  Processor::register_task_by_kind(Processor::LOC_PROC, false /*!global*/, INCREMENT_TASK,
                                   CodeDescriptor(increment_task), ProfilingRequestSet())
      .external_wait();
#ifdef REALM_USE_CUDA
  {
    // the GPU variants put all their work on the task's stream
    CodeDescriptor stencil_gpu(stencil_task_gpu);
    stencil_gpu.add_property(new DeferredEffectsProperty);
    CodeDescriptor increment_gpu(increment_task_gpu);
    increment_gpu.add_property(new DeferredEffectsProperty);
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/, STENCIL_TASK,
                                     stencil_gpu, ProfilingRequestSet())
        .external_wait();
    Processor::register_task_by_kind(Processor::TOC_PROC, false /*!global*/,
                                     INCREMENT_TASK, increment_gpu, ProfilingRequestSet())
        .external_wait();
  }
#endif

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                    .only_kind(Processor::LOC_PROC)
                    .first();
  assert(p.exists());
  rt.collective_spawn(p, TOP_LEVEL_TASK, &args, sizeof(args));
  return rt.wait_for_shutdown();
}
